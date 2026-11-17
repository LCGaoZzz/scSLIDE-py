"""Landmark-level object construction and landmark-to-query UMAP projection (F01, F11).

Implements native Python counterparts of:
- ``BuildLandmarkObject`` (:func:`build_landmark_object`, :class:`LandmarkObject` — F01)
- ``RunAndProjectUMAP`` (:func:`run_and_project_umap` — F11)

Backend policy (WP3 strict transparency contract):
- ``umap_backend`` must be one of ``'auto'``, ``'umap-learn'``, or ``'native'``.
- Under ``umap_backend='auto'`` with ``allow_fallback=False`` (default), ``umap-learn``
  is used when installed and an actionable ``ImportError`` is raised when missing.
- Under ``umap_backend='auto'`` with ``allow_fallback=True``, **only** a missing
  ``umap-learn`` import (``ImportError``) falls back to ``'native'`` with an explicit
  ``UserWarning`` and ``fallback_reason`` recorded in output provenance.
- Runtime, numerical, or invalid-input errors inside ``umap-learn`` or ``'native'``
  are NEVER caught and silently swapped to another algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh
from scipy.special import betainc

from ._validation import validate_bijective_ids, validate_numeric_matrix_values, validate_open_backed_anndata
from .sample import SampleLevelObject, log_normalize


VALID_UMAP_BACKENDS = ("auto", "umap-learn", "native")


def _umap_fuzzy_simplicial_graph(
    nn_idx_0based: np.ndarray,
    nn_dist: np.ndarray,
    n_vertices: int,
) -> sp.csr_matrix:
    """Construct fuzzy simplicial set membership strengths matching UMAP / uwot over precomputed kNN."""
    n, k = nn_idx_0based.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nn_idx_0based.reshape(-1).astype(np.int64)
    dists = np.asarray(nn_dist, dtype=np.float64).reshape(-1)

    target = float(np.log2(max(k, 2)))
    rho = np.zeros(n, dtype=np.float64)
    sigma = np.ones(n, dtype=np.float64)
    for i in range(n):
        di = nn_dist[i]
        pos = di[di > 0]
        rho[i] = float(pos[0]) if pos.size > 0 else 0.0
        lo, hi = 1e-10, 1000.0
        mid = 1.0
        for _ in range(48):
            psum = float(np.sum(np.exp(-np.maximum(0.0, di - rho[i]) / mid)))
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = 0.5 * (lo + hi)
            else:
                lo = mid
                mid = 2.0 * mid if hi == 1000.0 else 0.5 * (lo + hi)
        sigma[i] = max(mid, 1e-8)

    adj_d = np.maximum(0.0, dists - np.repeat(rho, k))
    vals = np.exp(-adj_d / np.repeat(sigma, k))
    non_self = rows != cols
    p_dir = sp.coo_matrix((vals[non_self], (rows[non_self], cols[non_self])), shape=(n_vertices, n_vertices)).tocsr()
    prod = p_dir.multiply(p_dir.T)
    sym = p_dir + p_dir.T - prod
    sym.eliminate_zeros()
    return sym.tocsr()


def _fit_umap_from_precomputed_knn(
    nn_idx_1based: np.ndarray,
    nn_dist: np.ndarray,
    *,
    n_components: int = 2,
    min_dist: float = 0.3,
    spread: float = 1.0,
    n_epochs: int = 200,
    seed: int = 42,
    umap_backend: str = "auto",
    allow_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit 2-D/k-D UMAP layout over a precomputed kNN graph with strict explicit backend semantics."""
    if umap_backend not in VALID_UMAP_BACKENDS:
        raise ValueError(f"umap_backend must be one of {VALID_UMAP_BACKENDS}, got {umap_backend!r}")
    if int(n_components) < 1:
        raise ValueError("n_components must be >= 1")
    if not np.isfinite(min_dist) or float(min_dist) <= 0:
        raise ValueError("min_dist must be positive and finite")
    if not np.isfinite(spread) or float(spread) <= 0:
        raise ValueError("spread must be positive and finite")
    if int(n_epochs) < 1:
        raise ValueError("n_epochs must be >= 1")

    idx_raw = np.asarray(nn_idx_1based)
    if idx_raw.ndim != 2 or not np.issubdtype(idx_raw.dtype, np.integer):
        raise ValueError("nn_idx_1based must be a 2-D integer array")
    n_obs, k = idx_raw.shape
    if n_obs < 2 or k < 1:
        raise ValueError("UMAP requires at least 2 observations and k >= 1 neighbor")
    if np.any(idx_raw < 1) or np.any(idx_raw > n_obs):
        raise ValueError(f"Landmark neighbor indices must be 1-based integers in 1..{n_obs}")

    dists = np.asarray(nn_dist, dtype=np.float64)
    if dists.shape != idx_raw.shape:
        raise ValueError(f"nn_dist shape {dists.shape} must match nn_idx shape {idx_raw.shape}")
    validate_numeric_matrix_values(dists, name="nn_dist", non_negative=True)
    idx0 = idx_raw.astype(np.int64) - 1

    effective_backend: str
    fallback_reason: str | None = None
    umap_mod = None
    find_ab_fn = None
    simplicial_emb_fn = None

    if umap_backend in ("umap-learn", "auto"):
        try:
            import umap as _umap_pkg
            from umap.umap_ import find_ab_params as _fab, simplicial_set_embedding as _sse

            umap_mod = _umap_pkg
            find_ab_fn = _fab
            simplicial_emb_fn = _sse
            effective_backend = "umap-learn"
        except ImportError as exc:
            if umap_backend == "umap-learn" or not allow_fallback:
                raise ImportError(
                    "UMAP backend 'umap-learn' is not installed in the active Python environment. "
                    "Install optional extra `pip install scslide-py[ann]` (or `pip install umap-learn`), "
                    "or explicitly opt into the built-in optimizer via `umap_backend='native'` "
                    "(or `umap_backend='auto', allow_fallback=True`)."
                ) from exc
            effective_backend = "native"
            fallback_reason = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"Optional dependency 'umap-learn' is not installed ({fallback_reason}); "
                "using scSLIDE native force-directed UMAP optimizer because allow_fallback=True.",
                UserWarning,
                stacklevel=3,
            )
    else:
        effective_backend = "native"

    # Execute selected backend OUTSIDE any broad try/except so runtime/numerical errors propagate!
    if effective_backend == "umap-learn":
        assert umap_mod is not None and find_ab_fn is not None and simplicial_emb_fn is not None
        graph = _umap_fuzzy_simplicial_graph(idx0, dists, n_obs)
        a_param, b_param = find_ab_fn(float(spread), float(min_dist))
        rng_rs = np.random.RandomState(int(seed))
        emb = simplicial_emb_fn(
            data=np.zeros((n_obs, max(2, int(n_components))), dtype=np.float32),
            graph=graph,
            n_components=int(n_components),
            initial_alpha=1.0,
            a=float(a_param),
            b=float(b_param),
            gamma=1.0,
            negative_sample_rate=5,
            n_epochs=int(n_epochs),
            init="spectral",
            random_state=rng_rs,
            metric="euclidean",
            metric_kwds={},
            densmap=False,
            densmap_kwds={},
            output_dens=False,
        )
        coords = np.asarray(emb[0] if isinstance(emb, tuple) else emb, dtype=np.float64)
        validate_numeric_matrix_values(coords, name="umap-learn coordinates")
        info = {
            "backend_requested": umap_backend,
            "backend_effective": "umap-learn",
            "backend_version": getattr(umap_mod, "__version__", "unknown"),
            "allow_fallback": bool(allow_fallback),
            "fallback_reason": None,
            "init_effective": "spectral",
            "n_components": int(n_components),
            "min_dist": float(min_dist),
            "spread": float(spread),
            "n_epochs": int(n_epochs),
            "seed": int(seed),
            "a": float(a_param),
            "b": float(b_param),
        }
        return coords, info

    # Explicit native force-directed UMAP cross-entropy SGD optimizer
    from scipy.optimize import curve_fit

    xv = np.linspace(0.0, float(spread) * 3.0, 300)
    yv = np.where(xv < float(min_dist), 1.0, np.exp(-(xv - float(min_dist)) / float(spread)))
    (a_param, b_param), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * (x ** (2.0 * b))), xv, yv, p0=[1.5, 0.9])

    graph = _umap_fuzzy_simplicial_graph(idx0, dists, n_obs)
    deg = np.asarray(graph.sum(axis=1), dtype=np.float64).ravel()
    inv_sqrt_d = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    norm_adj = sp.diags(inv_sqrt_d) @ graph @ sp.diags(inv_sqrt_d)
    rng = np.random.default_rng(int(seed))
    init_effective = "random_small_n"
    if n_obs > int(n_components) + 2 and graph.nnz > 0:
        try:
            v0 = rng.normal(size=n_obs)
            vals, vecs = eigsh(norm_adj, k=int(n_components) + 1, which="LA", v0=v0)
            ord_e = np.argsort(vals)[::-1]
            vecs = vecs[:, ord_e[1 : int(n_components) + 1]]
            for c in range(vecs.shape[1]):
                if vecs[int(np.argmax(np.abs(vecs[:, c]))), c] < 0:
                    vecs[:, c] = -vecs[:, c]
            coords = 10.0 * (vecs / np.maximum(np.max(np.abs(vecs)), 1e-8)) + rng.normal(
                scale=1e-4, size=(n_obs, int(n_components))
            )
            init_effective = "spectral"
        except (ArpackNoConvergence, ArpackError):
            coords = rng.normal(scale=4.0, size=(n_obs, int(n_components)))
            init_effective = "random_arpack_nonconvergence"
    else:
        coords = rng.normal(scale=4.0, size=(n_obs, int(n_components)))

    coo = graph.tocoo()
    head = coo.row.astype(np.int64)
    tail = coo.col.astype(np.int64)
    weights = coo.data.astype(np.float64)
    max_w = float(weights.max()) if weights.size > 0 else 1.0
    epochs_per_sample = np.where(weights > 0, max_w / np.maximum(weights, 1e-8), 1e8)
    epoch_of_next = epochs_per_sample.copy()
    neg_rate = 5

    for ep in range(1, int(n_epochs) + 1):
        alpha = 1.0 * (1.0 - float(ep - 1) / float(n_epochs))
        active = np.flatnonzero(epoch_of_next <= ep)
        if active.size == 0:
            continue
        h = head[active]
        t = tail[active]
        diff = coords[h] - coords[t]
        dist_sq = np.sum(diff * diff, axis=1) + 1e-6
        grad_coeff = (-2.0 * a_param * b_param * (dist_sq ** (b_param - 1.0))) / (1.0 + a_param * (dist_sq ** b_param))
        grad = np.clip(grad_coeff[:, None] * diff, -4.0, 4.0) * alpha
        np.add.at(coords, h, grad)
        np.add.at(coords, t, -grad)

        neg_t = rng.integers(0, n_obs, size=(active.size, neg_rate))
        diff_neg = coords[h][:, None, :] - coords[neg_t]
        dist_neg_sq = np.sum(diff_neg * diff_neg, axis=2) + 1e-4
        rep_coeff = (2.0 * b_param) / ((0.001 + dist_neg_sq) * (1.0 + a_param * (dist_neg_sq ** b_param)))
        rep_grad = np.clip(rep_coeff[:, :, None] * diff_neg, -4.0, 4.0) * alpha
        np.add.at(coords, h, rep_grad.sum(axis=1))

        epoch_of_next[active] += epochs_per_sample[active]

    validate_numeric_matrix_values(coords, name="native UMAP coordinates")
    info = {
        "backend_requested": umap_backend,
        "backend_effective": "native",
        "backend_version": f"scslide-native/scipy-{scipy.__version__}",
        "allow_fallback": bool(allow_fallback),
        "fallback_reason": fallback_reason,
        "init_effective": init_effective,
        "n_components": int(n_components),
        "min_dist": float(min_dist),
        "spread": float(spread),
        "n_epochs": int(n_epochs),
        "seed": int(seed),
        "a": float(a_param),
        "b": float(b_param),
    }
    return coords, info


def _project_umap_via_landmark_knn(
    landmark_umap: np.ndarray,
    query_nn_idx_1based: np.ndarray,
    query_nn_dist: np.ndarray,
    *,
    k_nn: int = 5,
    min_dist: float = 0.3,
    spread: float = 1.0,
    n_epochs: int = 60,
    seed: int = 42,
) -> np.ndarray:
    """Project all query cells onto a fitted landmark UMAP using their query->landmark kNN graph (F11)."""
    from scipy.optimize import curve_fit

    k_use = min(int(k_nn), query_nn_idx_1based.shape[1])
    idx0 = np.asarray(query_nn_idx_1based[:, :k_use], dtype=np.int64) - 1
    dists = np.asarray(query_nn_dist[:, :k_use], dtype=np.float64)
    validate_numeric_matrix_values(dists, name="query_nn_dist", non_negative=True)
    n_query = len(idx0)
    n_landmarks, n_comp = landmark_umap.shape
    if np.any(idx0 < 0) or np.any(idx0 >= n_landmarks):
        raise ValueError(f"query_nn_idx_1based contains indices outside 1..{n_landmarks}")

    rho = np.zeros(n_query, dtype=np.float64)
    pos_mask = dists > 0
    for i in range(n_query):
        p = dists[i, pos_mask[i]]
        rho[i] = float(p[0]) if p.size > 0 else 0.0
    scale = np.maximum(np.mean(np.maximum(0.0, dists - rho[:, None]), axis=1), 1e-4)
    w = np.exp(-np.maximum(0.0, dists - rho[:, None]) / scale[:, None])
    w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    proj = np.einsum("ik,ikd->id", w_norm, landmark_umap[idx0])
    exact_self = dists[:, 0] <= 1e-12
    proj[exact_self] = landmark_umap[idx0[exact_self, 0]]

    if n_epochs <= 0:
        return proj

    xv = np.linspace(0.0, spread * 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    (a_param, b_param), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * (x ** (2.0 * b))), xv, yv, p0=[1.5, 0.9])

    rng = np.random.default_rng(seed)
    non_exact = np.flatnonzero(~exact_self)
    if non_exact.size == 0:
        return proj

    for ep in range(1, n_epochs + 1):
        alpha = 0.35 * (1.0 - float(ep - 1) / float(n_epochs))
        q_cur = proj[non_exact]
        lm_anc = landmark_umap[idx0[non_exact]]
        diff = q_cur[:, None, :] - lm_anc
        d2 = np.sum(diff * diff, axis=2) + 1e-6
        att_c = (-2.0 * a_param * b_param * (d2 ** (b_param - 1.0))) / (1.0 + a_param * (d2 ** b_param))
        att_step = np.sum((w_norm[non_exact] * att_c)[:, :, None] * diff, axis=1)

        neg_idx = rng.integers(0, n_landmarks, size=(non_exact.size, 3))
        diff_neg = q_cur[:, None, :] - landmark_umap[neg_idx]
        d2_neg = np.sum(diff_neg * diff_neg, axis=2) + 1e-3
        rep_c = (2.0 * b_param) / ((0.01 + d2_neg) * (1.0 + a_param * (d2_neg ** b_param)))
        rep_step = 0.15 * np.sum(rep_c[:, :, None] * diff_neg, axis=1)

        proj[non_exact] = q_cur + alpha * np.clip(att_step + rep_step, -4.0, 4.0)

    return proj


@dataclass
class LandmarkObject:
    """Landmark-level object produced by :func:`build_landmark_object` (F01 ``BuildLandmarkObject``).

    Rows / observations are **landmarks** (not samples and not all query cells),
    columns / features are **samples** (transposed log-normalized LMC densities),
    ``umap`` holds the 2-D ``wnn.umap`` coordinates on the landmark-only neighbor subgraph,
    and ``metadata`` holds single-cell landmark annotations plus ``cor.coef_<feature>``
    and ``cor.pval_<feature>`` columns against sample-level trajectories/phenotypes.
    """

    density_by_sample: pd.DataFrame
    umap: pd.DataFrame
    metadata: pd.DataFrame
    nn_idx_landmark: np.ndarray
    nn_dist_landmark: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_anndata(self) -> Any:
        from .anndata import _json_safe_value, _require_anndata

        ad = _require_anndata()
        adata = ad.AnnData(
            X=self.density_by_sample.to_numpy(dtype=np.float64),
            obs=self.metadata.loc[self.density_by_sample.index].copy(),
            var=pd.DataFrame(index=self.density_by_sample.columns.astype(str)),
        )
        adata.obsm["X_wnn_umap"] = self.umap.loc[self.density_by_sample.index].to_numpy(dtype=np.float64)
        adata.obsm["scslide_landmark_nn_idx"] = self.nn_idx_landmark.copy()
        adata.obsm["scslide_landmark_nn_dist"] = self.nn_dist_landmark.copy()
        safe_attrs = _json_safe_value(self.attrs)
        adata.uns["scslide"] = {
            "schema_version": "landmark-level-2",
            "observation_unit": "landmark",
            "variable_unit": "sample",
            "attrs_json": json.dumps(safe_attrs, ensure_ascii=False, sort_keys=True),
        }
        return adata


def _pearson_cor_with_na_omit(mat_rows_by_samples: np.ndarray, vec_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match R ``cor.test(exp_data[i, ], feature_data[[TRAJ]], na.action='na.omit')`` with 0/1 NA fallback."""
    valid = np.isfinite(vec_samples)
    y = vec_samples[valid]
    n = int(len(y))
    n_rows = mat_rows_by_samples.shape[0]
    if n < 3 or np.std(y, ddof=1) == 0:
        return np.zeros(n_rows, dtype=np.float64), np.ones(n_rows, dtype=np.float64)

    x = mat_rows_by_samples[:, valid]
    yc = y - float(y.mean())
    xc = x - x.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(xc * xc, axis=1) * float(np.sum(yc * yc)))
    r = np.zeros(n_rows, dtype=np.float64)
    ok = denom > 0
    r[ok] = np.clip((xc[ok] @ yc) / denom[ok], -1.0, 1.0)

    df = max(n - 2, 1)
    p = np.ones(n_rows, dtype=np.float64)
    perf = ok & (np.abs(r) >= 1.0 - 1e-15)
    p[perf] = 0.0
    mid = ok & ~perf
    if np.any(mid):
        r2 = r[mid] ** 2
        t2 = df * r2 / np.maximum(1.0 - r2, 1e-300)
        p[mid] = betainc(0.5 * df, 0.5, df / (df + t2))
    return r, p


def build_landmark_object(
    cell_adata_or_nn: Any,
    sample_obj: SampleLevelObject | Any,
    *,
    features_to_test: Sequence[str],
    nn_name: str = "weighted.nn",
    landmark_assay_name: str = "LANDMARK",
    rename_sample: Sequence[str] | None = None,
    k_nn: int = 20,
    min_dist: float = 0.3,
    spread: float = 1.0,
    n_epochs: int = 200,
    scale_factor: float = 10_000.0,
    seed: int = 42,
    umap_backend: str = "auto",
    allow_fallback: bool = False,
    cell_ids: Sequence[str] | None = None,
    landmark_cell_ids: Sequence[str] | None = None,
    nn_dist: np.ndarray | None = None,
    cell_metadata: pd.DataFrame | None = None,
) -> LandmarkObject:
    """Native implementation of ``BuildLandmarkObject`` (F01, ``R/visualization.R``)."""
    from .anndata import load_rectangular_neighbors, sample_object_from_anndata

    if not isinstance(sample_obj, SampleLevelObject):
        sample_obj = sample_object_from_anndata(sample_obj)

    if len(features_to_test) == 0:
        raise ValueError("features_to_test must name at least one sample-level trajectory or metadata column")

    norm_lm_by_s = log_normalize(sample_obj.counts, scale_factor=scale_factor)
    assert isinstance(norm_lm_by_s, pd.DataFrame)
    if rename_sample is not None:
        rs_arr, _ = validate_bijective_ids(rename_sample, axis_name="rename_sample")
        if len(rs_arr) != norm_lm_by_s.shape[1]:
            raise ValueError("rename_sample length must match number of samples")
        norm_lm_by_s = norm_lm_by_s.copy()
        norm_lm_by_s.columns = pd.Index(rs_arr.astype(str))

    if hasattr(cell_adata_or_nn, "obs") and hasattr(cell_adata_or_nn, "obsm"):
        validate_open_backed_anndata(cell_adata_or_nn, caller="build_landmark_object")
        nn_idx_all, nn_dist_all, q_ids_arr, lm_ids_reg = load_rectangular_neighbors(cell_adata_or_nn, nn_name)
        q_ids = [str(x) for x in q_ids_arr]
        sketches = cell_adata_or_nn.uns.get("scslide", {}).get("sketches", {})
        if landmark_assay_name in sketches and "cell_ids" in sketches[landmark_assay_name]:
            lm_orig_ids = [str(x) for x in sketches[landmark_assay_name]["cell_ids"]]
        else:
            lm_orig_ids = lm_ids_reg
        c_meta = cell_adata_or_nn.obs[:].copy() if getattr(cell_adata_or_nn, "isbacked", False) else cell_adata_or_nn.obs.copy()
    else:
        nn_idx_all = np.asarray(cell_adata_or_nn, dtype=np.int64)
        if cell_ids is None or landmark_cell_ids is None:
            raise ValueError("When passing raw nn_idx array, cell_ids and landmark_cell_ids are required")
        q_ids = [str(x) for x in cell_ids]
        lm_orig_ids = [str(x) for x in landmark_cell_ids]
        nn_dist_all = np.zeros_like(nn_idx_all, dtype=np.float64) if nn_dist is None else np.asarray(nn_dist, dtype=np.float64)
        c_meta = cell_metadata

    if nn_dist_all is None:
        nn_dist_all = np.ones_like(nn_idx_all, dtype=np.float64)

    if len(lm_orig_ids) != norm_lm_by_s.shape[0]:
        raise ValueError(
            f"Landmark count mismatch: single-cell landmark set has {len(lm_orig_ids)} cells, "
            f"but sample_obj has {norm_lm_by_s.shape[0]} rows"
        )

    q_lookup = {cid: pos for pos, cid in enumerate(q_ids)}
    missing_lm = [cid for cid in lm_orig_ids if cid not in q_lookup]
    if missing_lm:
        raise ValueError(f"Landmark cell IDs missing from NN query_ids: {missing_lm[:5]}")
    slct_row = np.array([q_lookup[cid] for cid in lm_orig_ids], dtype=np.int64)

    k_use = min(int(k_nn), nn_idx_all.shape[1])
    lm_nn_idx = nn_idx_all[slct_row, :k_use]
    lm_nn_dist = nn_dist_all[slct_row, :k_use]

    umap_xy, umap_info = _fit_umap_from_precomputed_knn(
        lm_nn_idx,
        lm_nn_dist,
        n_components=2,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        seed=seed,
        umap_backend=umap_backend,
        allow_fallback=allow_fallback,
    )
    umap_df = pd.DataFrame(umap_xy, index=norm_lm_by_s.index, columns=["UMAP_1", "UMAP_2"])

    if c_meta is not None:
        lm_meta = c_meta.loc[lm_orig_ids].copy()
        lm_meta.index = norm_lm_by_s.index
    else:
        lm_meta = pd.DataFrame(index=norm_lm_by_s.index)
    lm_meta["canonical_landmark_id"] = lm_orig_ids

    if sample_obj.metadata is None:
        raise ValueError("sample_obj.metadata is required to evaluate features_to_test")
    s_meta = sample_obj.metadata.loc[sample_obj.counts.columns]
    exp_arr = norm_lm_by_s.to_numpy(dtype=np.float64)

    for feat in features_to_test:
        if feat in s_meta.columns:
            vec = pd.to_numeric(s_meta[feat], errors="coerce").to_numpy(dtype=np.float64)
        else:
            found = None
            for edf in sample_obj.embeddings.values():
                if feat in edf.columns:
                    found = edf.loc[sample_obj.counts.columns, feat].to_numpy(dtype=np.float64)
                    break
            if found is None:
                raise ValueError(f"Feature/trajectory {feat!r} not found in sample_obj.metadata or embeddings")
            vec = found
        r_vec, p_vec = _pearson_cor_with_na_omit(exp_arr, vec)
        lm_meta[f"cor.coef_{feat}"] = r_vec
        lm_meta[f"cor.pval_{feat}"] = p_vec

    return LandmarkObject(
        density_by_sample=norm_lm_by_s,
        umap=umap_df,
        metadata=lm_meta,
        nn_idx_landmark=lm_nn_idx,
        nn_dist_landmark=lm_nn_dist,
        attrs={
            "nn_name": nn_name,
            "k_nn": int(k_use),
            "min_dist": float(min_dist),
            "spread": float(spread),
            "features_tested": list(features_to_test),
            "umap_info": umap_info,
        },
    )


def run_and_project_umap(
    adata_or_nn: Any,
    *,
    nn_name: str,
    k_nn: int = 5,
    landmark_assay_name: str = "LANDMARK",
    reduction_name: str = "sketched.umap",
    min_dist: float = 0.3,
    spread: float = 1.0,
    n_epochs: int = 200,
    projection_epochs: int = 60,
    seed: int = 42,
    umap_backend: str = "auto",
    allow_fallback: bool = False,
    cell_ids: Sequence[str] | None = None,
    landmark_cell_ids: Sequence[str] | None = None,
    nn_dist: np.ndarray | None = None,
) -> Any | dict[str, Any]:
    """Native implementation of ``RunAndProjectUMAP`` (F11, ``R/visualization.R``).

    Fits UMAP on the landmark-only neighbor subgraph (``reduction_name``) with explicit
    backend provenance and projects all query cells onto the same UMAP coordinate system
    (``proj.<reduction_name>``).
    """
    from .anndata import _json_safe_value, load_rectangular_neighbors

    is_adata = hasattr(adata_or_nn, "obs") and hasattr(adata_or_nn, "obsm")
    if is_adata:
        adata = adata_or_nn
        validate_open_backed_anndata(adata, caller="run_and_project_umap")
        nn_idx_all, nn_dist_all, q_ids_arr, lm_ids_reg = load_rectangular_neighbors(adata, nn_name)
        q_ids = [str(x) for x in q_ids_arr]
        sketches = adata.uns.get("scslide", {}).get("sketches", {})
        if landmark_assay_name in sketches and "cell_ids" in sketches[landmark_assay_name]:
            lm_ids = [str(x) for x in sketches[landmark_assay_name]["cell_ids"]]
        else:
            lm_ids = lm_ids_reg
    else:
        nn_idx_all = np.asarray(adata_or_nn, dtype=np.int64)
        if cell_ids is None or landmark_cell_ids is None:
            raise ValueError("cell_ids and landmark_cell_ids are required when passing a raw nn_idx array")
        q_ids = [str(x) for x in cell_ids]
        lm_ids = [str(x) for x in landmark_cell_ids]
        nn_dist_all = np.ones_like(nn_idx_all, dtype=np.float64) if nn_dist is None else np.asarray(nn_dist, dtype=np.float64)

    if nn_dist_all is None:
        nn_dist_all = np.ones_like(nn_idx_all, dtype=np.float64)

    k_use = min(int(k_nn), nn_idx_all.shape[1])
    if k_use < 1:
        raise ValueError("k_nn must be >= 1")
    q_lookup = {cid: pos for pos, cid in enumerate(q_ids)}
    missing_lm = [cid for cid in lm_ids if cid not in q_lookup]
    if missing_lm:
        raise ValueError(f"Landmark cell IDs missing from query_ids: {missing_lm[:5]}")
    slct_row = np.array([q_lookup[cid] for cid in lm_ids], dtype=np.int64)

    lm_umap, umap_info = _fit_umap_from_precomputed_knn(
        nn_idx_all[slct_row, :k_use],
        nn_dist_all[slct_row, :k_use],
        n_components=2,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        seed=seed,
        umap_backend=umap_backend,
        allow_fallback=allow_fallback,
    )
    proj_umap = _project_umap_via_landmark_knn(
        lm_umap,
        nn_idx_all[:, :k_use],
        nn_dist_all[:, :k_use],
        k_nn=k_use,
        min_dist=min_dist,
        spread=spread,
        n_epochs=projection_epochs,
        seed=seed,
    )

    umap_info["k_nn"] = int(k_use)
    umap_info["projection_epochs"] = int(projection_epochs)
    umap_info["reduction_name"] = str(reduction_name)

    sketch_df = pd.DataFrame(lm_umap, index=lm_ids, columns=["UMAP_1", "UMAP_2"])
    proj_df = pd.DataFrame(proj_umap, index=q_ids, columns=["UMAP_1", "UMAP_2"])

    if is_adata:
        if not getattr(adata, "isbacked", False):
            full_sketch = np.full((adata.n_obs, 2), np.nan, dtype=np.float64)
            full_sketch[slct_row] = lm_umap
            adata.obsm[f"X_{reduction_name}"] = full_sketch
            adata.obsm[f"X_proj.{reduction_name}"] = proj_umap
        sc_uns = dict(adata.uns.get("scslide", {}))
        umap_reg = dict(sc_uns.get("umap", {}))
        umap_reg[str(reduction_name)] = _json_safe_value(umap_info)
        sc_uns["umap"] = umap_reg
        adata.uns["scslide"] = sc_uns
        return adata

    return {reduction_name: sketch_df, f"proj.{reduction_name}": proj_df, "umap_info": umap_info}
