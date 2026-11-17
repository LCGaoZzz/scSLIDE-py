"""Single-modality and multi-modal Weighted Nearest Neighbor (WNN) projection (F03, F04).

Implements exact native counterparts of:
- ``FindsmNN`` (:func:`find_sm_nn` and backward-compatible :func:`landmark_neighbors`)
- ``FindmmNN`` (:func:`find_mm_nn` = ``Seurat:::FindModalityWeights`` + ``Seurat:::MultiModalNN``)

Supports both fixed modality weights (``fix_wnn_weights=(0.5, 0.5)``) and dynamically
learned modality weights (``fix_wnn_weights=None``), exact Euclidean kNN via ``cKDTree``
or optional ``pynndescent`` approximate kNN for million-cell workloads, tie diagnostics,
and both ``Seurat v6`` and ``legacy`` (Seurat v5 ``self.query`` col-1 drop) profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
from collections.abc import Sequence
import warnings

import numpy as np
from scipy.spatial import cKDTree

from ._validation import validate_bijective_ids, validate_numeric_matrix_values


@dataclass
class NeighborResult:
    """Structured query-to-landmark nearest-neighbor object matching Seurat ``Neighbor``."""

    nn_idx: np.ndarray              # 1-based (n_query, k) integer landmark indices
    nn_dist: np.ndarray             # (n_query, k) distances
    query_ids: list[str]
    landmark_ids: list[str]
    modality_weights: list[np.ndarray] | None = None
    sigma_list: list[np.ndarray] | None = None
    nearest_dist: list[np.ndarray] | None = None
    tie_boundary_queries: int = 0
    params: dict[str, Any] = field(default_factory=dict)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Match Seurat ``L2Norm``: divide each row by its L2 norm; zero-norm rows become all-zero."""
    arr = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    pos = norms[:, 0] > 0
    out[pos] = arr[pos] / norms[pos]
    return out


def _resolve_nn_backend_version(method: str) -> str:
    import importlib.metadata as md
    import scipy

    if method == "exact":
        return f"scipy.spatial.cKDTree/{scipy.__version__}"
    if method == "pynndescent":
        try:
            return f"pynndescent/{md.version('pynndescent')}"
        except md.PackageNotFoundError:
            return "pynndescent/unknown"
    if method == "annoy":
        try:
            return f"annoy/{md.version('annoy')}"
        except md.PackageNotFoundError:
            return "annoy/unknown"
    return method


def _detect_cgroup_cpu_quota(
    cgroup_v2_max_path: Path = Path("/sys/fs/cgroup/cpu.max"),
    cgroup_v1_quota_path: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    cgroup_v1_period_path: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Read finite quotas at conventional cgroup mounts and process ancestors.

    A root-only read misses limits in systemd/user slices. Inspect the process
    membership and its ancestors as well; the tightest visible quota applies.
    CPU quotas constrain runtime, not the number of live BLAS/native threads.
    """
    v2_paths = {cgroup_v2_max_path}
    v1_paths = {(cgroup_v1_quota_path, cgroup_v1_period_path)}
    try:
        for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, member = line.split(":", 2)
            rel = Path(member.lstrip("/"))
            # A container may expose membership outside its mounted namespace.
            if ".." in rel.parts:
                continue
            roots = [rel, *rel.parents]
            if hierarchy == "0" and controllers == "":
                v2_paths.update(cgroup_v2_max_path.parent / p / "cpu.max" for p in roots)
            if "cpu" in controllers.split(","):
                v1_paths.update(
                    (cgroup_v1_quota_path.parent / p / "cpu.cfs_quota_us",
                     cgroup_v1_period_path.parent / p / "cpu.cfs_period_us")
                    for p in roots
                )
    except (OSError, ValueError):
        pass
    limits = []
    for path in v2_paths:
        try:
            parts = path.read_text(encoding="utf-8").strip().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    limits.append(max(1, quota // period))
        except (OSError, ValueError):
            continue
    for quota_path, period_path in v1_paths:
        try:
            quota = int(quota_path.read_text(encoding="utf-8").strip())
            period = int(period_path.read_text(encoding="utf-8").strip())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            continue
    return min(limits) if limits else None


def resolve_effective_workers(
    n_jobs: int | None = 1,
    *,
    available_cpus_override: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve ``n_jobs`` into a positive worker count bounded by OS CPU affinity & cgroup quota.

    Unlike ``scipy.spatial.cKDTree.query(..., workers=-1)``, which queries host-wide
    ``os.cpu_count()`` and ignores ``sched_setaffinity`` / cgroup limits, negative
    ``n_jobs`` values here are resolved strictly against the process's permitted CPU
    quota (``n_jobs=-1`` -> all permitted CPUs; ``n_jobs=-2`` -> all permitted CPUs minus 1)
    and passed to native backends as an explicit positive integer ``>= 1``.
    """
    if isinstance(n_jobs, bool) or (n_jobs is not None and not isinstance(n_jobs, (int, np.integer))):
        raise ValueError(f"n_jobs must be a non-zero integer or None, got {n_jobs!r}")
    req = 1 if n_jobs is None else int(n_jobs)
    if req == 0:
        raise ValueError("n_jobs cannot be 0; pass 1 for single-worker or -1 for all available affinity/cgroup CPUs")

    os_cpus = max(1, int(os.cpu_count() or 1))
    if available_cpus_override is not None:
        avail = max(1, int(available_cpus_override))
    else:
        if hasattr(os, "sched_getaffinity"):
            try:
                aff_cpus = max(1, len(os.sched_getaffinity(0)))
            except OSError:
                aff_cpus = os_cpus
        else:
            aff_cpus = os_cpus
        cg_cpus = _detect_cgroup_cpu_quota()
        avail = min(aff_cpus, cg_cpus) if cg_cpus is not None else aff_cpus

    if req > 0:
        eff = req
    else:
        eff = max(1, avail + 1 + req)

    info = {
        "workers_requested": req,
        "workers_effective": int(eff),
        "available_cpus": int(avail),
        "os_cpu_count": int(os_cpus),
        "exceeds_available_cpus": bool(eff > avail),
    }
    return int(eff), info


def _knn_query(
    reference: np.ndarray,
    query: np.ndarray,
    k: int,
    *,
    method: str = "exact",
    n_jobs: int | None = 1,
    seed: int = 42,
    annoy_n_trees: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Query ``k`` Euclidean nearest neighbors in ``reference`` for each row in ``query``.

    Returns ``(zero_based_indices, distances, boundary_tie_count)``.
    """
    eff_workers, _ = resolve_effective_workers(n_jobs)
    n_ref = len(reference)
    k_eff = min(int(k), n_ref)
    if method == "exact":
        # Check k + 1 when possible to detect exact distance ties at the k-th boundary
        k_probe = min(k_eff + 1, n_ref)
        tree = cKDTree(reference)
        dists, idxs = tree.query(query, k=k_probe, workers=eff_workers)
        if k_probe == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        if k_probe > k_eff:
            ties = int(np.sum(np.abs(dists[:, k_eff - 1] - dists[:, k_eff]) <= 1e-12))
            # Deterministic tie-breaking by smaller landmark index when exact tie occurs at boundary
            tie_rows = np.flatnonzero(np.abs(dists[:, k_eff - 1] - dists[:, k_eff]) <= 1e-12)
            for r in tie_rows:
                row_d = np.linalg.norm(reference - query[r : r + 1], axis=1)
                # Stable sort breaks exact distance ties by ascending landmark index (0..n_ref-1)
                ord_r = np.argsort(np.round(row_d, 13), kind="mergesort")[:k_eff]
                idxs[r, :k_eff] = ord_r
                dists[r, :k_eff] = row_d[ord_r]
            dists = dists[:, :k_eff]
            idxs = idxs[:, :k_eff]
        else:
            ties = 0
        return np.asarray(idxs, dtype=np.int64), np.asarray(dists, dtype=np.float64), ties

    if method == "pynndescent":
        try:
            import pynndescent
        except ImportError as exc:
            raise ImportError(
                "nn_method='pynndescent' requires optional package 'pynndescent'. "
                "Install via `pip install scslide-py[ann]` or use `nn_method='exact'`."
            ) from exc

        index = pynndescent.NNDescent(
            reference.astype(np.float32),
            metric="euclidean",
            n_neighbors=min(max(30, k_eff), n_ref),
            random_state=seed,
            n_jobs=eff_workers,
        )
        idxs, dists = index.query(query.astype(np.float32), k=k_eff)
        return np.asarray(idxs, dtype=np.int64), np.asarray(dists, dtype=np.float64), 0

    if method == "annoy":
        try:
            from annoy import AnnoyIndex
        except ImportError as exc:
            raise ImportError(
                "nn_method='annoy' requires optional package 'annoy' (not included in `[ann]`). "
                "Install via `pip install \"scslide-py[annoy]\"` (or `pip install annoy`) or use `nn_method='exact'`."
            ) from exc

        d_dim = int(reference.shape[1])
        ann_idx = AnnoyIndex(d_dim, "euclidean")
        ann_idx.set_seed(int(seed))
        ref32 = np.asarray(reference, dtype=np.float32)
        for i in range(n_ref):
            ann_idx.add_item(i, ref32[i])
        ann_idx.build(int(annoy_n_trees), n_jobs=eff_workers)
        qry32 = np.asarray(query, dtype=np.float32)
        idxs = np.zeros((len(qry32), k_eff), dtype=np.int64)
        dists = np.zeros((len(qry32), k_eff), dtype=np.float64)
        for q_i in range(len(qry32)):
            ni, nd = ann_idx.get_nns_by_vector(qry32[q_i], k_eff, include_distances=True)
            idxs[q_i, : len(ni)] = ni
            dists[q_i, : len(nd)] = nd
        return idxs, dists, 0

    raise ValueError(f"Unsupported neighbor search method {method!r}; choose 'exact', 'annoy', or 'pynndescent'")


def landmark_neighbors(
    reference: np.ndarray,
    query: np.ndarray | None = None,
    *,
    k: int = 5,
    l2_normalize: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Return one-based landmark indices for each query row (preserves v0.1.1 API)."""
    ref = np.asarray(reference, dtype=np.float64)
    qry = ref if query is None else np.asarray(query, dtype=np.float64)
    if ref.ndim != 2 or qry.ndim != 2 or ref.shape[1] != qry.shape[1]:
        raise ValueError("reference and query must be 2-D with equal feature dimensions")
    validate_numeric_matrix_values(ref, name="reference")
    validate_numeric_matrix_values(qry, name="query")
    if not 1 <= k <= len(ref):
        raise ValueError("k must be between 1 and number of reference rows")
    if l2_normalize:
        ref = _l2_normalize_rows(ref)
        qry = _l2_normalize_rows(qry)
    idxs, _, _ = _knn_query(ref, qry, k=k, method="exact", n_jobs=n_jobs)
    return idxs.astype(np.int64) + 1


def find_sm_nn(
    query_embedding: np.ndarray,
    landmark_indices_or_embedding: np.ndarray,
    *,
    dims: Sequence[int] | None = None,
    k_nn: int = 20,
    l2_norm: bool = True,
    query_ids: Sequence[str] | None = None,
    landmark_ids: Sequence[str] | None = None,
    nn_method: str = "exact",
    annoy_n_trees: int = 50,
    strict_upstream_k_check: bool = False,
    n_jobs: int = 1,
    seed: int = 42,
) -> NeighborResult:
    """Native implementation of ``FindsmNN`` (F04).

    ``landmark_indices_or_embedding`` may either be a 1-D integer index array selecting
    landmark rows from ``query_embedding``, or a 2-D ``(n_landmarks, d)`` landmark embedding.
    """
    k_nn = int(k_nn)
    if strict_upstream_k_check and k_nn <= 2:
        raise ValueError("k_nn must be an integer > 2 under strict upstream FindsmNN check")
    if k_nn < 1:
        raise ValueError("k_nn must be >= 1")

    q_emb = np.asarray(query_embedding, dtype=np.float64)
    validate_numeric_matrix_values(q_emb, name="query_embedding")
    if dims is not None:
        q_emb = q_emb[:, list(dims)]

    lm_arg = np.asarray(landmark_indices_or_embedding)
    if lm_arg.ndim == 1 and np.issubdtype(lm_arg.dtype, np.integer):
        ref_emb = q_emb[lm_arg]
        if landmark_ids is None and query_ids is not None:
            q_list = [str(x) for x in query_ids]
            landmark_ids = [q_list[int(i)] for i in lm_arg]
    else:
        ref_emb = np.asarray(lm_arg, dtype=np.float64)
        validate_numeric_matrix_values(ref_emb, name="landmark_embedding")
        if dims is not None and ref_emb.shape[1] != q_emb.shape[1]:
            ref_emb = ref_emb[:, list(dims)]

    if ref_emb.ndim != 2 or ref_emb.shape[1] != q_emb.shape[1]:
        raise ValueError("Landmark and query embeddings must have matching column dimensions")
    if k_nn > len(ref_emb):
        raise ValueError(f"k_nn ({k_nn}) exceeds number of landmarks ({len(ref_emb)})")

    q_ids = [str(x) for x in query_ids] if query_ids is not None else [f"cell_{i}" for i in range(len(q_emb))]
    lm_ids = [str(x) for x in landmark_ids] if landmark_ids is not None else [f"LM_{i}" for i in range(len(ref_emb))]
    validate_bijective_ids(q_ids, axis_name="query_ids")
    validate_bijective_ids(lm_ids, axis_name="landmark_ids")

    q_use = _l2_normalize_rows(q_emb) if l2_norm else q_emb
    ref_use = _l2_normalize_rows(ref_emb) if l2_norm else ref_emb

    eff_workers, w_info = resolve_effective_workers(n_jobs)
    idx0, dists, ties = _knn_query(
        ref_use, q_use, k=k_nn, method=nn_method, n_jobs=eff_workers, seed=seed, annoy_n_trees=annoy_n_trees
    )
    return NeighborResult(
        nn_idx=idx0 + 1,
        nn_dist=dists,
        query_ids=q_ids,
        landmark_ids=lm_ids,
        tie_boundary_queries=ties,
        params={
            "modality": "single",
            "k_nn": k_nn,
            "l2_norm": bool(l2_norm),
            "nn_method": nn_method,
            "backend_requested": nn_method,
            "backend_effective": nn_method,
            "backend_version": _resolve_nn_backend_version(nn_method),
            "annoy_n_trees": int(annoy_n_trees) if nn_method == "annoy" else None,
            "workers_requested": w_info["workers_requested"],
            "workers_effective": w_info["workers_effective"],
            "build_workers_effective": 1 if nn_method == "exact" else w_info["workers_effective"],
            "query_workers_effective": 1 if nn_method == "annoy" else w_info["workers_effective"],
            "available_cpus": w_info["available_cpus"],
            "os_cpu_count": w_info["os_cpu_count"],
            "exceeds_available_cpus": w_info["exceeds_available_cpus"],
        },
    )


def compute_modality_weights(
    embeddings_list: Sequence[np.ndarray],
    *,
    k_nn: int = 20,
    sigma_idx: int | None = None,
    l2_norm: bool = True,
    sd_scale: float = 1.0,
    fix_wnn_weights: Sequence[float] | None = None,
    cross_constant: float = 1e-4,
    smooth: bool = False,
    nn_method: str = "exact",
    annoy_n_trees: int = 50,
    n_jobs: int = 1,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Exact native implementation of ``Seurat:::FindModalityWeights(..., snn.far.nn = FALSE)``.

    Returns ``(modality_weights, sigma_list, nearest_dist)`` over all ``N`` cells.
    Matches Seurat v6 ``clustering.R:1229-1542``:
    - Even when ``fix_wnn_weights`` is supplied, L2 normalization, per-modality ``k``-NN,
      ``nearest_dist`` (2nd neighbor = 1st non-self neighbor), and ``sigma_list``
      (distance to ``sigma_idx``-th neighbor minus ``nearest_dist``) are computed on the
      full dataset.
    - When ``fix_wnn_weights is None``, computes within-modality and cross-modality kNN
      imputation distances, exponential kernels, clipped ratios ``MinMax(score, 0, 200)``,
      and softmax weights across modalities.
    """
    n_mod = len(embeddings_list)
    if n_mod < 2:
        raise ValueError("compute_modality_weights requires at least 2 modalities")

    norm_embs: list[np.ndarray] = []
    n_cells = len(embeddings_list[0])
    for m_idx, emb in enumerate(embeddings_list):
        arr = np.asarray(emb, dtype=np.float64)
        validate_numeric_matrix_values(arr, name=f"modality_{m_idx} embedding")
        if len(arr) != n_cells:
            raise ValueError("All modality embeddings must have the same number of cells")
        norm_embs.append(_l2_normalize_rows(arr) if l2_norm else arr.copy())

    k_eff = min(int(k_nn), n_cells)
    if k_eff < 2:
        raise ValueError("Need at least 2 cells for modality bandwidth estimation")
    sig_col = k_eff if sigma_idx is None else min(int(sigma_idx), n_cells)
    k_search = max(k_eff, sig_col)

    nn_indices: list[np.ndarray] = []
    nearest_dist: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []

    for m_idx in range(n_mod):
        idx0, dists, _ = _knn_query(
            norm_embs[m_idx],
            norm_embs[m_idx],
            k=k_search,
            method=nn_method,
            n_jobs=n_jobs,
            seed=seed,
            annoy_n_trees=annoy_n_trees,
        )
        # Ensure column 0 is self (distance 0) when exact ties occur with identical cell embeddings
        rows_ar = np.arange(n_cells)
        not_self_first = idx0[:, 0] != rows_ar
        if np.any(not_self_first):
            for r in np.flatnonzero(not_self_first):
                pos = np.flatnonzero(idx0[r] == r)
                if pos.size > 0:
                    p0 = int(pos[0])
                    idx0[r, 1 : p0 + 1] = idx0[r, 0:p0]
                    dists[r, 1 : p0 + 1] = dists[r, 0:p0]
                idx0[r, 0] = r
                dists[r, 0] = 0.0

        nn_indices.append(idx0[:, :k_eff])
        nd = dists[:, 1].copy()
        nearest_dist.append(nd)
        sig = np.maximum((dists[:, sig_col - 1] - nd) * float(sd_scale), 1e-12)
        sigma_list.append(sig)

    if fix_wnn_weights is not None:
        if len(fix_wnn_weights) != n_mod:
            raise ValueError(f"fix_wnn_weights length ({len(fix_wnn_weights)}) must match number of modalities ({n_mod})")
        weights = [np.full(n_cells, float(w), dtype=np.float64) for w in fix_wnn_weights]
        return weights, sigma_list, nearest_dist

    # Learned modality weights (Hao et al. Cell 2021):
    # PredictAssay averages neighbor embeddings EXCLUDING self (col 0) when self-query!
    within_kernel: list[np.ndarray] = []
    for r in range(n_mod):
        emb_r = norm_embs[r]
        nbr_r = nn_indices[r][:, 1:] if k_eff > 1 else nn_indices[r]
        pred_within = emb_r[nbr_r].mean(axis=1)
        d_within = np.maximum(0.0, np.linalg.norm(emb_r - pred_within, axis=1) - nearest_dist[r])
        within_kernel.append(np.exp(-d_within / sigma_list[r]))

    modality_scores: list[np.ndarray] = []
    for r in range(n_mod):
        emb_r = norm_embs[r]
        other_mods = [r2 for r2 in range(n_mod) if r2 != r]
        score_cols = []
        for r2 in other_mods:
            nbr_r2 = nn_indices[r2][:, 1:] if k_eff > 1 else nn_indices[r2]
            pred_cross = emb_r[nbr_r2].mean(axis=1)
            d_cross = np.maximum(0.0, np.linalg.norm(emb_r - pred_cross, axis=1) - nearest_dist[r])
            k_cross = np.exp(-d_cross / sigma_list[r])
            ratio = within_kernel[r] / (k_cross + float(cross_constant))
            score_cols.append(np.clip(ratio, 0.0, 200.0))
        score_mat_r = np.column_stack(score_cols)
        if smooth and k_eff > 1:
            nbr_r = nn_indices[r][:, 1:]
            score_mat_r = score_mat_r[nbr_r].mean(axis=1)
        modality_scores.append(score_mat_r)

    # Numerically stabilized softmax matching rowSums(exp(score_m)) / rowSums(exp(cbind(...)))
    all_scores = np.hstack(modality_scores)
    max_s = np.max(all_scores, axis=1, keepdims=True)
    denom = np.sum(np.exp(all_scores - max_s), axis=1)
    weights = [
        np.sum(np.exp(sm_r - max_s), axis=1) / denom
        for sm_r in modality_scores
    ]
    return weights, sigma_list, nearest_dist


def find_mm_nn(
    query_embeddings: Sequence[np.ndarray],
    landmark_indices: Sequence[int] | np.ndarray,
    *,
    dims_list: Sequence[Sequence[int] | None] | None = None,
    k_nn: int = 20,
    knn_range: int = 200,
    l2_norm: bool = True,
    fix_wnn_weights: Sequence[float] | None = None,
    kernel_power: float = 1.0,
    query_ids: Sequence[str] | None = None,
    landmark_ids: Sequence[str] | None = None,
    nn_method: str = "exact",
    annoy_n_trees: int = 50,
    reference_profile: str = "v6",
    batch_size: int = 8192,
    n_jobs: int = 1,
    seed: int = 42,
) -> NeighborResult:
    """Native implementation of ``FindmmNN`` (F03: ``FindModalityWeights`` + ``MultiModalNN``).

    Matches ``Seurat v6`` (``reference_profile='v6'``, keeping all ``1..knn_range`` landmark
    candidates when ``query != sketch_obj``) and also supports ``reference_profile='legacy_v5_bug'``
    (reproducing Seurat v5's unconditional ``Indices[, -1]`` drop for diagnostic parity checks).
    """
    k_nn_in = int(k_nn)
    if k_nn_in <= 2:
        raise ValueError("k_nn must be an integer > 2")
    if reference_profile not in ("v6", "legacy", "legacy_v5_bug"):
        raise ValueError("reference_profile must be 'v6', 'legacy', or 'legacy_v5_bug'")

    n_mod = len(query_embeddings)
    if n_mod < 2:
        raise ValueError("find_mm_nn requires at least 2 modality embeddings")

    sliced_embs: list[np.ndarray] = []
    n_query = len(query_embeddings[0])
    for m_idx, emb in enumerate(query_embeddings):
        arr = np.asarray(emb, dtype=np.float64)
        if dims_list is not None and dims_list[m_idx] is not None:
            arr = arr[:, list(dims_list[m_idx])]
        validate_numeric_matrix_values(arr, name=f"modality_{m_idx}")
        if len(arr) != n_query:
            raise ValueError("All query modality embeddings must have equal row counts")
        sliced_embs.append(arr)

    lm_idx = np.asarray(landmark_indices, dtype=np.int64)
    n_landmarks = len(lm_idx)
    if n_landmarks == 0 or np.any(lm_idx < 0) or np.any(lm_idx >= n_query):
        raise ValueError("landmark_indices must be valid 0-based indices into query_embeddings")

    # Upstream FindmmNN silently raises k.nn < 20 up to 20 inside FindModalityWeights & MultiModalNN!
    k_eff = max(20, k_nn_in)
    if k_eff > n_landmarks:
        k_eff = n_landmarks

    knn_range_eff = int(knn_range)
    if knn_range_eff > n_landmarks:
        warnings.warn(
            f"knn_range ({knn_range_eff}) exceeds number of landmarks ({n_landmarks}); clamping to {n_landmarks}",
            UserWarning,
            stacklevel=2,
        )
        knn_range_eff = n_landmarks

    eff_workers, w_info = resolve_effective_workers(n_jobs)
    weights, sigma_list, nearest_dist = compute_modality_weights(
        sliced_embs,
        k_nn=k_eff,
        l2_norm=l2_norm,
        fix_wnn_weights=fix_wnn_weights,
        nn_method=nn_method,
        annoy_n_trees=annoy_n_trees,
        n_jobs=eff_workers,
        seed=seed,
    )

    q_norm = [_l2_normalize_rows(e) if l2_norm else e for e in sliced_embs]
    ref_norm = [e[lm_idx] for e in q_norm]

    # Candidate landmark search per modality (query -> landmarks)
    cand_per_mod: list[np.ndarray] = []
    for m_idx in range(n_mod):
        idx_m, _, _ = _knn_query(
            ref_norm[m_idx],
            q_norm[m_idx],
            k=knn_range_eff,
            method=nn_method,
            n_jobs=eff_workers,
            seed=seed,
            annoy_n_trees=annoy_n_trees,
        )
        if reference_profile == "legacy_v5_bug" and knn_range_eff > 1:
            idx_m = idx_m[:, 1:]
        cand_per_mod.append(idx_m)

    k_out = min(k_eff, cand_per_mod[0].shape[1] * n_mod, n_landmarks)
    select_nn = np.zeros((n_query, k_out), dtype=np.int64)
    select_sim = np.zeros((n_query, k_out), dtype=np.float64)
    tie_count = 0

    # Vectorized batched evaluation over concatenated candidate indices (M * knn_range_eff per cell)
    # Deduplicating per row while preserving modality-appearance order matches R base::union!
    for start in range(0, n_query, batch_size):
        stop = min(n_query, start + batch_size)
        b_len = stop - start
        concat_cand = np.hstack([c[start:stop] for c in cand_per_mod])  # (b_len, M * R)
        max_c = concat_cand.shape[1]

        # Compute joint similarity for all M * R positions simultaneously in 3D vectorized BLAS/broadcasting!
        joint_sim = np.zeros((b_len, max_c), dtype=np.float64)
        for m_idx in range(n_mod):
            qm = q_norm[m_idx][start:stop]                      # (b_len, d_m)
            rm_cand = ref_norm[m_idx][concat_cand]              # (b_len, max_c, d_m)
            diff = rm_cand - qm[:, None, :]
            d_raw = np.linalg.norm(diff, axis=2)                # (b_len, max_c)
            d_adj = np.maximum(0.0, d_raw - nearest_dist[m_idx][start:stop, None])
            sig = sigma_list[m_idx][start:stop, None]
            kern = np.exp(-((d_adj / sig) ** float(kernel_power)))
            joint_sim += weights[m_idx][start:stop, None] * kern

        # Mask duplicate landmark IDs across modalities so each unique candidate appears once (at its first occurrence, matching R Reduce(union))
        for i in range(b_len):
            row_c = concat_cand[i]
            _, first_pos = np.unique(row_c, return_index=True)
            first_pos.sort()  # preserves R base::union appearance order!
            uniq_c = row_c[first_pos]
            uniq_s = joint_sim[i, first_pos]
            ord_s = np.argsort(-uniq_s, kind="mergesort")
            if len(ord_s) > k_out and abs(uniq_s[ord_s[k_out - 1]] - uniq_s[ord_s[k_out]]) <= 1e-12:
                tie_count += 1
            top = ord_s[:k_out]
            select_nn[start + i, : len(top)] = uniq_c[top] + 1
            select_sim[start + i, : len(top)] = uniq_s[top]

    select_dist = np.sqrt(np.clip((1.0 - select_sim) / 2.0, 0.0, 1.0))

    q_ids = [str(x) for x in query_ids] if query_ids is not None else [f"cell_{i}" for i in range(n_query)]
    lm_ids = (
        [str(x) for x in landmark_ids]
        if landmark_ids is not None
        else [q_ids[int(i)] for i in lm_idx]
    )
    validate_bijective_ids(q_ids, axis_name="query_ids")
    validate_bijective_ids(lm_ids, axis_name="landmark_ids")

    return NeighborResult(
        nn_idx=select_nn,
        nn_dist=select_dist,
        query_ids=q_ids,
        landmark_ids=lm_ids,
        modality_weights=weights,
        sigma_list=sigma_list,
        nearest_dist=nearest_dist,
        tie_boundary_queries=tie_count,
        params={
            "modality": "multi",
            "k_nn_requested": k_nn_in,
            "k_nn_stored": int(k_out),
            "knn_range": int(knn_range_eff),
            "l2_norm": bool(l2_norm),
            "fix_wnn_weights": list(fix_wnn_weights) if fix_wnn_weights is not None else None,
            "nn_method": nn_method,
            "backend_requested": nn_method,
            "backend_effective": nn_method,
            "backend_version": _resolve_nn_backend_version(nn_method),
            "annoy_n_trees": int(annoy_n_trees) if nn_method == "annoy" else None,
            "reference_profile": reference_profile,
            "workers_requested": w_info["workers_requested"],
            "workers_effective": w_info["workers_effective"],
            "build_workers_effective": 1 if nn_method == "exact" else w_info["workers_effective"],
            "query_workers_effective": 1 if nn_method == "annoy" else w_info["workers_effective"],
            "available_cpus": w_info["available_cpus"],
            "os_cpu_count": w_info["os_cpu_count"],
            "exceeds_available_cpus": w_info["exceeds_available_cpus"],
        },
    )
