"""Sample-level Diffusion Map, Principal Curve, and spectral subspace comparison (F12, B12).

Implements:
- :func:`diffusion_style_embedding` (preserves v0.1.1 symmetric diagnostic API)
- :func:`run_diffusion_map` (F12: exact ``destiny::DiffusionMap`` locally-scaled density-normalized
  Markov transition operator + ``RunDiffusionMap`` eigenvalue weighting)
- :func:`fit_principal_curve` (exact Hastie & Stuetzle 1989 / ``princurve::principal_curve``
  iterative orthogonal polyline projection + cubic smoothing spline refit)
- :func:`compare_spectral_embeddings` (B12: mathematically legal sign-flip and degenerate-eigenspace
  orthogonal alignment without arbitrary full-space Procrustes rotation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq
from sklearn.metrics import pairwise_distances

from ._validation import validate_bijective_ids, validate_numeric_matrix_values, validate_open_backed_anndata
from .sample import SampleLevelObject


def diffusion_style_embedding(
    x: np.ndarray | pd.DataFrame,
    *,
    n_components: int = 2,
    metric: str = "cosine",
    weight_by_eigenvalue: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a symmetric diffusion-style embedding from a sample matrix (preserves v0.1.1 API)."""
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError("x must have at least two samples")
    validate_numeric_matrix_values(data, name="x")
    d = pairwise_distances(data, metric=metric)
    scale = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    kernel = np.exp(-(d * d) / (2.0 * scale * scale))
    np.fill_diagonal(kernel, 0.0)
    degree = kernel.sum(axis=1)
    inv = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized = inv @ kernel @ inv
    vals, vecs = np.linalg.eigh(normalized)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    n_eff = min(int(n_components), len(data) - 1)
    vals = vals[1 : n_eff + 1]
    vecs = vecs[:, 1 : n_eff + 1]
    # Deterministic sign convention on each column
    for j in range(vecs.shape[1]):
        if vecs[int(np.argmax(np.abs(vecs[:, j]))), j] < 0:
            vecs[:, j] = -vecs[:, j]
    if weight_by_eigenvalue:
        embedding = vecs * np.sqrt(np.maximum(vals, 0.0))
    else:
        embedding = vecs
    return embedding, vals


@dataclass
class DiffusionMapResult:
    """Result of :func:`run_diffusion_map` matching ``RunDiffusionMap`` + ``destiny::DiffusionMap``."""

    embeddings: pd.DataFrame       # (n_samples, ncomp), columns DC_1..DC_k
    eigenvalues: np.ndarray        # (ncomp,) non-trivial eigenvalues lambda_1..lambda_k
    eigenvectors: pd.DataFrame     # unweighted eigenvectors
    sigmas: np.ndarray             # local or global kernel bandwidths per sample
    transition_sym: np.ndarray     # symmetric conjugate transition operator D^{-1/2} K_alpha D^{-1/2}
    n_connected_components: int
    params: dict[str, Any] = field(default_factory=dict)


def _pairwise_destiny_distance(data: np.ndarray, metric: str, *, cosine_mode: str = "covertree_cor") -> np.ndarray:
    """Compute sample-by-sample distance matching ``destiny`` conventions (euclidean, cosine, rankcor).

    In ``destiny`` with ``knn.covertree`` (loaded by official scSLIDE vignettes),
    ``distance='cosine'`` computes row-centered Pearson correlation distance ``1 - cor(x_i, x_j)``
    (``cosine_mode='covertree_cor'``), whereas ``method='hnsw'`` computes uncentered ``1 - cos(x_i, x_j)``
    (``cosine_mode='uncentered'``).
    """
    m = metric.lower()
    if m in ("euclidean", "l2"):
        return pairwise_distances(data, metric="euclidean")
    if m == "cosine":
        x = data - data.mean(axis=1, keepdims=True) if cosine_mode == "covertree_cor" else data
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe = np.where(norms > 0, x / np.maximum(norms, 1e-300), 0.0)
        sim = np.clip(safe @ safe.T, -1.0, 1.0)
        d = np.maximum(0.0, 1.0 - sim)
        np.fill_diagonal(d, 0.0)
        return d
    if m in ("rankcor", "spearman"):
        ranks = np.apply_along_axis(lambda row: pd.Series(row).rank(method="average").to_numpy(), 1, data)
        rc = ranks - ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(rc, axis=1, keepdims=True)
        safe = np.where(norms > 0, rc / np.maximum(norms, 1e-300), 0.0)
        sim = np.clip(safe @ safe.T, -1.0, 1.0)
        d = np.maximum(0.0, 1.0 - sim)
        np.fill_diagonal(d, 0.0)
        return d
    raise ValueError(f"Unsupported DiffusionMap distance metric {metric!r}; choose 'cosine', 'euclidean', or 'rankcor'")


def run_diffusion_map(
    object_or_matrix: Any,
    *,
    ncomp: int = 10,
    metric: str = "cosine",
    cosine_mode: str = "covertree_cor",
    eigen_which: str = "LM",
    distance_matrix: np.ndarray | pd.DataFrame | None = None,
    sigma: str | float | np.ndarray = "local",
    k_local: tuple[int, int] = (5, 7),
    density_norm: bool = True,
    n_eigs_knn: int | None = None,
    weight_by_var: bool = True,
    layer: str = "scale.data",
    scale_max: float | None = 10.0,
    features: Sequence[str] | None = None,
    reduction_name: str = "DiffMap",
    reduction_key: str = "DC_",
    seed_use: int | None = 101,
) -> DiffusionMapResult | Any:
    """Native Python implementation of ``RunDiffusionMap`` (F12, ``destiny::DiffusionMap`` formulation).

    Follows Haghverdi, Buettner & Theis (Bioinformatics 2015) as implemented in ``destiny``:
    1. Pairwise distance ``D`` (default ``metric='cosine'`` on centered Pearson residuals).
    2. Local bandwidth ``sigma_i`` = half the distance to the ``k``-th nearest neighbor
       (default ``sigma='local'``, matching ``destiny`` local scale ``d_{i,(k)} / 2`` with
       ``k = min(max(5, round(0.05 * n)), n - 1)`` clipped to ``k_local`` when ``n`` allows),
       or global scalar ``sigma``.
    3. Locally scaled anisotropic Gaussian kernel:
       ``K_{ij} = sqrt(2 sigma_i sigma_j / (sigma_i^2 + sigma_j^2)) * exp(-D_{ij}^2 / (sigma_i^2 + sigma_j^2))``
       with zero diagonal ``K_{ii} = 0`` (matching ``destiny`` no-self-loop transition kernel).
    4. Density normalization (``density_norm=True``): ``q_i = sum_j K_{ij}``, ``K_alpha = D_q^{-1} K D_q^{-1}``.
    5. Symmetric Markov operator ``M = D_rot^{-1/2} K_alpha D_rot^{-1/2}`` where ``d_rot_i = sum_j K_alpha_{ij}``:
       compute top ``ncomp + 1`` eigenpairs ``(lambda, phi)``, verify connected components,
       discard trivial stationary eigenvector ``phi_0`` (``lambda_0 ~ 1``), normalize right
       eigenvectors ``psi_k = phi_k`` (matching ``destiny`` L2-orthonormal symmetric eigenvectors
       in ``@eigenvectors``), and scale by ``sqrt(max(lambda_k, 0))`` when ``weight_by_var=True``.
    """
    is_sample_obj = isinstance(object_or_matrix, SampleLevelObject)
    is_adata = hasattr(object_or_matrix, "obs") and hasattr(object_or_matrix, "var_names") and hasattr(object_or_matrix, "obsm")

    if distance_matrix is not None:
        d_mat = distance_matrix.to_numpy(dtype=np.float64) if isinstance(distance_matrix, pd.DataFrame) else np.asarray(distance_matrix, dtype=np.float64)
        if d_mat.ndim != 2 or d_mat.shape[0] != d_mat.shape[1] or d_mat.shape[0] < 3:
            raise ValueError("distance_matrix must be a square (n x n) matrix with n >= 3")
        validate_numeric_matrix_values(d_mat, name="distance_matrix", non_negative=True)
        if isinstance(distance_matrix, pd.DataFrame):
            sample_ids = [str(x) for x in distance_matrix.index]
        elif is_sample_obj:
            sample_ids = list(object_or_matrix.counts.columns)
        elif is_adata:
            sample_ids = [str(x) for x in object_or_matrix.obs_names]
        else:
            sample_ids = [f"sample_{i}" for i in range(d_mat.shape[0])]
        n_samples = len(sample_ids)
    else:
        if is_sample_obj:
            s_obj: SampleLevelObject = object_or_matrix
            sample_ids = list(s_obj.counts.columns)
            eff_smax = s_obj.attrs.get("scale_max", scale_max) if "scale_max" in s_obj.attrs else scale_max
            if layer == "scale.data" and s_obj.residuals is not None:
                feat_df = s_obj.centered_residuals(scale_max=eff_smax)
            elif layer == "residuals" and s_obj.residuals is not None:
                feat_df = s_obj.residuals
            elif layer in ("normalized", "data") and s_obj.normalized is not None:
                feat_df = s_obj.normalized
            elif s_obj.residuals is not None:
                feat_df = s_obj.centered_residuals(scale_max=eff_smax)
            else:
                feat_df = s_obj.counts.astype(np.float64)
            if features is not None:
                feat_df = feat_df.loc[[f for f in features if f in feat_df.index]]
            data_s_by_f = feat_df.T.to_numpy(dtype=np.float64)
        elif is_adata:
            adata = object_or_matrix
            validate_open_backed_anndata(adata, caller="run_diffusion_map")
            sample_ids = [str(x) for x in adata.obs_names]
            if layer in adata.layers:
                raw_m = adata.layers[layer]
            elif layer == "scale.data" and "residuals" in adata.layers:
                r_m = np.asarray(adata.layers["residuals"][:], dtype=np.float64)
                cent_m = r_m - r_m.mean(axis=0, keepdims=True)
                raw_m = np.minimum(cent_m, float(scale_max)) if scale_max is not None else cent_m
            else:
                raw_m = adata.X
            data_s_by_f = raw_m.toarray() if sp.issparse(raw_m) else np.asarray(raw_m[:], dtype=np.float64)
            if features is not None:
                var_map = {str(v): i for i, v in enumerate(adata.var_names)}
                cols = [var_map[str(f)] for f in features if str(f) in var_map]
                data_s_by_f = data_s_by_f[:, cols]
        elif isinstance(object_or_matrix, pd.DataFrame):
            # Support samples x features DataFrame directly
            sample_ids = [str(x) for x in object_or_matrix.index]
            data_s_by_f = object_or_matrix.to_numpy(dtype=np.float64)
        else:
            data_s_by_f = np.asarray(object_or_matrix, dtype=np.float64)
            sample_ids = [f"sample_{i}" for i in range(data_s_by_f.shape[0])]

        validate_bijective_ids(sample_ids, axis_name="sample_ids")
        validate_numeric_matrix_values(data_s_by_f, name="diffusion input")
        n_samples = len(sample_ids)
        if n_samples < 3:
            raise ValueError("RunDiffusionMap requires at least 3 samples")
        d_mat = _pairwise_destiny_distance(data_s_by_f, metric=metric, cosine_mode=cosine_mode)

    # Match destiny::DiffusionMap default k = find_dm_k(n - 1) and n_local = seq(to = min(k, 7), length.out = min(k, 3))
    k_dm = n_samples - 1 if n_eigs_knn is None else min(max(2, int(n_eigs_knn)), n_samples - 1)
    sorted_d = np.sort(d_mat, axis=1)
    if isinstance(sigma, str):
        if sigma == "local":
            if k_local == (5, 7):
                to_idx = min(k_dm, 7)
                len_loc = min(k_dm, 3)
                # 1-based non-self neighbor ranks match sorted_d[:, rank] directly since sorted_d[:, 0] == 0 (self)
                ranks_1based = np.arange(to_idx - len_loc + 1, to_idx + 1, dtype=np.int64)
            else:
                k_lo = min(max(1, int(k_local[0])), k_dm)
                k_hi = min(max(k_lo, int(k_local[1])), k_dm)
                ranks_1based = np.arange(k_lo, k_hi + 1, dtype=np.int64)
            local_d = sorted_d[:, ranks_1based].mean(axis=1) / 2.0
            pos_d = d_mat[d_mat > 0]
            fallback = float(np.median(pos_d)) / 2.0 if pos_d.size > 0 else 1.0
            sigmas = np.where(local_d > 0, local_d, fallback)
        elif sigma == "global":
            pos_d = d_mat[d_mat > 0]
            s_val = float(np.median(pos_d)) / np.sqrt(2.0) if pos_d.size > 0 else 1.0
            sigmas = np.full(n_samples, max(s_val, 1e-8), dtype=np.float64)
        else:
            raise ValueError("sigma string must be 'local' or 'global'")
    elif np.ndim(sigma) == 0:
        s_val = float(sigma)
        if not np.isfinite(s_val) or s_val <= 0:
            raise ValueError("scalar sigma must be positive and finite")
        sigmas = np.full(n_samples, s_val, dtype=np.float64)
    else:
        sigmas = np.asarray(sigma, dtype=np.float64)
        if sigmas.shape != (n_samples,) or np.any(sigmas <= 0):
            raise ValueError("vector sigma must be positive of length n_samples")

    sig_sq_sum = (sigmas[:, None] ** 2) + (sigmas[None, :] ** 2)
    prefactor = np.sqrt((2.0 * sigmas[:, None] * sigmas[None, :]) / np.maximum(sig_sq_sum, 1e-300))
    k_mat = prefactor * np.exp(-(d_mat ** 2) / np.maximum(sig_sq_sum, 1e-300))
    np.fill_diagonal(k_mat, 0.0)

    if n_eigs_knn is not None and int(n_eigs_knn) < n_samples - 1:
        k_keep = max(2, int(n_eigs_knn))
        mask = np.zeros_like(k_mat, dtype=bool)
        nbrs = np.argsort(d_mat, axis=1)[:, 1 : k_keep + 1]
        for i in range(n_samples):
            mask[i, nbrs[i]] = True
        mask = mask | mask.T
        k_mat = np.where(mask, k_mat, 0.0)

    if density_norm:
        # Match destiny::DiffusionMap line 79: d <- rowSums(trans_p, na.rm = TRUE) + 1
        q_dens = k_mat.sum(axis=1) + 1.0
        inv_q = 1.0 / np.maximum(q_dens, 1e-300)
        k_alpha = (k_mat * inv_q[:, None]) * inv_q[None, :]
    else:
        k_alpha = k_mat

    d_rot = k_alpha.sum(axis=1)
    inv_sqrt_d = np.where(d_rot > 0, 1.0 / np.sqrt(np.maximum(d_rot, 1e-300)), 0.0)
    m_sym = (k_alpha * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]
    m_sym = 0.5 * (m_sym + m_sym.T)

    vals_all, vecs_all = la.eigh(m_sym)
    # Count connected components (unit eigenvalues >= 1 - 1e-9)
    n_conn = int(np.sum(vals_all >= 1.0 - 1e-9))
    n_out = min(int(ncomp), n_samples - 1)
    if n_out < 1:
        raise ValueError("ncomp must be >= 1")

    if eigen_which.upper() == "LM":
        # Match destiny:::eig_decomp -> RSpectra::eigs(M, n_eigs + 1) on dgCMatrix:
        # selects the (n_out + 1) eigenpairs of largest magnitude |lambda|, then orders
        # them by algebraic value descending (so stationary lambda_0 ~ 1 is index 0).
        mag_top = np.argsort(np.abs(vals_all))[::-1][: n_out + 1]
        alg_ord = mag_top[np.argsort(vals_all[mag_top])[::-1]]
        vals_sel = vals_all[alg_ord]
        vecs_sel = vecs_all[:, alg_ord]
    elif eigen_which.upper() == "LA":
        ord_desc = np.argsort(vals_all)[::-1][: n_out + 1]
        vals_sel = vals_all[ord_desc]
        vecs_sel = vecs_all[:, ord_desc]
    else:
        raise ValueError("eigen_which must be 'LM' or 'LA'")

    # Drop stationary mode 0
    eigvals = vals_sel[1 : n_out + 1].copy()
    eigvecs = vecs_sel[:, 1 : n_out + 1].copy()

    # Deterministic gauge sign: largest-magnitude coordinate positive (tie-broken by first index)
    for j in range(eigvecs.shape[1]):
        idx_max = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[idx_max, j] < 0:
            eigvecs[:, j] = -eigvecs[:, j]

    if weight_by_var:
        scale_ev = np.sqrt(np.maximum(eigvals, 0.0))
        coords = eigvecs * scale_ev[None, :]
    else:
        coords = eigvecs.copy()

    cols = [f"{reduction_key}{j + 1}" for j in range(n_out)]
    emb_df = pd.DataFrame(coords, index=sample_ids, columns=cols)
    vec_df = pd.DataFrame(eigvecs, index=sample_ids, columns=cols)

    dm_res = DiffusionMapResult(
        embeddings=emb_df,
        eigenvalues=eigvals,
        eigenvectors=vec_df,
        sigmas=sigmas,
        transition_sym=m_sym,
        n_connected_components=max(n_conn, 1),
        params={
            "metric": metric,
            "sigma": sigma if isinstance(sigma, (str, float, int)) else "custom_vector",
            "density_norm": bool(density_norm),
            "weight_by_var": bool(weight_by_var),
            "ncomp": int(n_out),
            "seed_use": seed_use,
        },
    )

    if is_sample_obj:
        s_obj.embeddings[reduction_name] = emb_df
        s_obj.attrs[f"{reduction_name}_eigenvalues"] = eigvals.tolist()
        s_obj.attrs[f"{reduction_name}_n_components_connected"] = dm_res.n_connected_components
        return dm_res

    if is_adata:
        adata.obsm[f"X_{reduction_name.lower()}"] = coords
        adata.obsm[f"X_{reduction_name}"] = coords
        sc_uns = dict(adata.uns.get("scslide", {}))
        sc_uns[reduction_name] = {
            "eigenvalues": eigvals.tolist(),
            "n_connected_components": dm_res.n_connected_components,
            "params": dm_res.params,
        }
        adata.uns["scslide"] = sc_uns
        return dm_res

    return dm_res


# ---------------------------------------------------------------------------
# Principal Curve (exact Hastie & Stuetzle 1989 / princurve::principal_curve)
# ---------------------------------------------------------------------------

@dataclass
class PrincipalCurveResult:
    """Result of :func:`fit_principal_curve` matching ``princurve::principal_curve``."""

    s: np.ndarray              # (n, p) projected curve coordinates per sample
    ord: np.ndarray            # 0-based ordering indices along lambda
    ord_1based: np.ndarray     # 1-based ordering indices matching R princurve$ord
    lambda_: np.ndarray        # (n,) arc-length pseudotime along the curve
    dist: float                # sum of squared orthogonal projection distances
    converged: bool
    num_iterations: int
    trace_dist: list[float] = field(default_factory=list)
    curve_dense: np.ndarray | None = None  # (n_dense, p) smooth spline evaluated on dense arc-length grid


def project_to_curve(x: np.ndarray, s: np.ndarray, *, stretch: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Orthogonal projection of points ``x`` onto ordered polyline ``s`` (matching ``princurve::project_to_curve``)."""
    pts = np.asarray(x, dtype=np.float64)
    curve = np.asarray(s, dtype=np.float64).copy()
    if len(curve) < 2:
        raise ValueError("Curve matrix 's' must contain at least 2 points")

    # Extend endpoints linearly by stretch factor matching R princurve C++ project_to_curve
    if stretch > 0:
        n_c = len(curve)
        diff1 = curve[0] - curve[1]
        diff2 = curve[n_c - 1] - curve[n_c - 2]
        curve[0] = curve[0] + float(stretch) * diff1
        curve[n_c - 1] = curve[n_c - 1] + float(stretch) * diff2

    seg_vecs = curve[1:] - curve[:-1]                        # (M, d)
    seg_lens_sq = np.sum(seg_vecs * seg_vecs, axis=1)        # (M,)
    seg_lens = np.sqrt(seg_lens_sq)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])   # (M + 1,)

    # Vectorized point-to-segment projection: pts (N, 1, d), curve[:-1] (1, M, d)
    w = pts[:, None, :] - curve[None, :-1, :]                # (N, M, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(
            seg_lens_sq[None, :] > 1e-24,
            np.einsum("nmd,md->nm", w, seg_vecs) / np.maximum(seg_lens_sq[None, :], 1e-24),
            0.0,
        )
    t_clamped = np.clip(t, 0.0, 1.0)                         # (N, M)
    proj_all = curve[None, :-1, :] + t_clamped[:, :, None] * seg_vecs[None, :, :]  # (N, M, d)
    d2_all = np.sum((pts[:, None, :] - proj_all) ** 2, axis=2)                     # (N, M)

    best_seg = np.argmin(d2_all, axis=1)
    rows = np.arange(len(pts))
    s_new = proj_all[rows, best_seg, :]
    lam_raw = cum_arc[best_seg] + t_clamped[rows, best_seg] * seg_lens[best_seg]
    lam = lam_raw - float(lam_raw.min())
    ord_idx = np.argsort(lam, kind="mergesort")
    total_dist = float(np.sum(d2_all[rows, best_seg]))
    return s_new, ord_idx, lam, total_dist


def _smooth_coord_along_lambda(
    lam: np.ndarray,
    y: np.ndarray,
    *,
    df: float = 5.0,
) -> tuple[np.ndarray, Any]:
    """Natural cubic smoothing spline matching R ``princurve`` ``stats::smooth.spline(..., df=5)``.

    Uses the exact Green & Silverman (1994) generalized-eigenvalue trace equation
    ``tr(S(lam_pen)) = 2 + sum_k 1 / (1 + lam_pen * mu_k) = df`` solved via Brent's method,
    evaluating the resulting Reinsch spline on sorted ``lam`` (matching ``princurve``'s
    ``smoother_functions$smooth_spline``).
    """
    ord_l = np.argsort(lam, kind="mergesort")
    xs = np.asarray(lam[ord_l], dtype=np.float64)
    ys = np.asarray(y[ord_l], dtype=np.float64)

    uniq_x, inv = np.unique(xs, return_inverse=True)
    n_u = len(uniq_x)
    if n_u < 4 or float(df) <= 2.0:
        slope, intercept = np.polyfit(xs, ys, 1) if n_u >= 2 else (0.0, float(np.mean(ys)))
        fn_lin = lambda z, a=intercept, b=slope: a + b * np.asarray(z, dtype=np.float64)
        return fn_lin(xs), fn_lin

    weights = np.bincount(inv).astype(np.float64)
    y_mean = np.bincount(inv, weights=ys) / weights
    if float(np.var(ys)) <= 1e-24:
        mu_y = float(np.mean(ys))
        fn_c = lambda z, c=mu_y: np.full_like(np.asarray(z, dtype=np.float64), c)
        return fn_c(xs), fn_c

    target_df = min(max(2.05, float(df)), float(n_u) - 0.05)
    h = np.diff(uniq_x)
    q_mat = np.zeros((n_u, n_u - 2), dtype=np.float64)
    idx = np.arange(n_u - 2)
    q_mat[idx, idx] = 1.0 / h[:-1]
    q_mat[idx + 1, idx] = -(1.0 / h[:-1] + 1.0 / h[1:])
    q_mat[idx + 2, idx] = 1.0 / h[1:]
    r_mat = np.zeros((n_u - 2, n_u - 2), dtype=np.float64)
    r_mat[idx, idx] = (h[:-1] + h[1:]) / 3.0
    if n_u > 3:
        r_mat[idx[:-1], idx[1:]] = h[1:-1] / 6.0
        r_mat[idx[1:], idx[:-1]] = h[1:-1] / 6.0

    q_tilde = q_mat * (1.0 / np.sqrt(weights))[:, None]
    l_chol = np.linalg.cholesky(r_mat)
    l_inv_qt = np.linalg.solve(l_chol, q_tilde.T)
    mu = np.maximum(np.linalg.eigvalsh(l_inv_qt @ l_inv_qt.T), 0.0)

    def _df_residual(log10_pen: float) -> float:
        pen = 10.0 ** log10_pen
        return float(2.0 + np.sum(1.0 / (1.0 + pen * mu)) - target_df)

    lo, hi = -18.0, 12.0
    while _df_residual(lo) < 0.0 and lo > -60.0:
        lo -= 4.0
    while _df_residual(hi) > 0.0 and hi < 60.0:
        hi += 4.0
    opt_pen = 10.0 ** brentq(_df_residual, lo, hi, xtol=1e-11)
    sp_fit = make_smoothing_spline(uniq_x, y_mean, w=weights, lam=opt_pen)
    return np.asarray(sp_fit(xs), dtype=np.float64), sp_fit


def fit_principal_curve(
    x: np.ndarray | pd.DataFrame,
    *,
    start: np.ndarray | None = None,
    thresh: float = 0.001,
    maxit: int = 20,
    stretch: float = 2.0,
    df: float = 5.0,
    n_dense: int = 200,
    anchor_Positive_cor_with: np.ndarray | None = None,
) -> PrincipalCurveResult:
    """Fit a principal curve matching ``princurve::principal_curve`` (Hastie & Stuetzle 1989)."""
    pts = x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float64)
    validate_numeric_matrix_values(pts, name="principal_curve input")
    n, d = pts.shape
    if n < 4 or d < 2:
        raise ValueError("fit_principal_curve requires at least 4 observations and 2 dimensions")

    dist_old = float(np.sum(np.var(pts, axis=0, ddof=1)))
    if start is None:
        xbar = pts.mean(axis=0)
        xstar = pts - xbar[None, :]
        u, sv, vt = la.svd(xstar, full_matrices=False)
        v1 = vt[0].copy()
        u1 = u[:, 0].copy()
        if u1[int(np.argmax(np.abs(u1)))] < 0:
            u1 = -u1
            v1 = -v1
        lam = u1 * sv[0]
        ord_idx = np.argsort(lam, kind="mergesort")
        s_cur = xbar[None, :] + np.outer(lam, v1)
        dist_cur = float(np.sum(sv[1:] ** 2))
    else:
        start_arr = np.asarray(start, dtype=np.float64)
        s_cur, ord_idx, lam, dist_cur = project_to_curve(pts, start_arr, stretch=stretch)

    trace_d = [dist_cur]
    has_converged = abs(dist_old - dist_cur) <= thresh * max(dist_old, 1e-12)
    it = 0
    splines: list[Any] | None = None

    while not has_converged and it < int(maxit):
        it += 1
        s_sorted = np.empty_like(pts)
        splines = []
        for j in range(d):
            s_sorted[:, j], sp_j = _smooth_coord_along_lambda(lam, pts[:, j], df=df)
            splines.append(sp_j)
        dist_old = dist_cur
        s_cur, ord_idx, lam, dist_cur = project_to_curve(pts, s_sorted, stretch=stretch)
        trace_d.append(dist_cur)
        has_converged = abs(dist_old - dist_cur) <= thresh * max(dist_old, 1e-12)

    # Evaluate smooth C^2 curve along dense arc-length grid for publication plotting
    if splines is not None and int(n_dense) >= 2:
        # Refit once on final converged lambda so dense grid aligns with final arc-length
        dense_grid = np.linspace(float(lam.min()), float(lam.max()), int(n_dense))
        curve_dense = np.empty((int(n_dense), d), dtype=np.float64)
        for j in range(d):
            _, sp_final = _smooth_coord_along_lambda(lam, pts[:, j], df=df)
            curve_dense[:, j] = np.asarray(sp_final(dense_grid), dtype=np.float64)
    else:
        curve_dense = s_cur[ord_idx].copy()

    if anchor_Positive_cor_with is not None:
        ref_v = np.asarray(anchor_Positive_cor_with, dtype=np.float64).ravel()
        valid = np.isfinite(ref_v)
        if np.sum(valid) >= 3 and np.std(ref_v[valid]) > 0 and np.std(lam[valid]) > 0:
            if np.corrcoef(lam[valid], ref_v[valid])[0, 1] < 0:
                lam = float(lam.max()) - lam
                ord_idx = ord_idx[::-1].copy()
                curve_dense = curve_dense[::-1].copy()

    return PrincipalCurveResult(
        s=s_cur,
        ord=ord_idx,
        ord_1based=ord_idx + 1,
        lambda_=lam,
        dist=dist_cur,
        converged=bool(has_converged),
        num_iterations=int(it),
        trace_dist=trace_d,
        curve_dense=curve_dense,
    )


# ---------------------------------------------------------------------------
# B12: Spectral identifiability comparator (sign flips & degenerate blocks ONLY)
# ---------------------------------------------------------------------------

def compare_spectral_embeddings(
    ref_embeddings: np.ndarray | pd.DataFrame,
    cand_embeddings: np.ndarray | pd.DataFrame,
    ref_eigenvalues: Sequence[float],
    cand_eigenvalues: Sequence[float],
    *,
    degeneracy_gap_tol: float = 1e-6,
    eigenvalue_atol: float = 1e-6,
    subspace_atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare spectral embeddings allowing ONLY mathematically legal gauge transformations (B12).

    - Simple eigenvalues (spectral gap ``> degeneracy_gap_tol``): ONLY ``+/-1`` sign flips are allowed.
    - Genuinely degenerate eigenvalue clusters (``|lambda_i - lambda_j| <= degeneracy_gap_tol``):
      orthogonal rotation ``Q in O(m)`` is allowed ONLY within that ``m``-dimensional cluster,
      never mixing axes across distinct eigenvalues!
    - Unequal spectra (``max |lambda_ref - lambda_cand| > eigenvalue_atol``) are explicitly flagged
      and never masked by rotation.
    """
    if isinstance(ref_embeddings, pd.DataFrame) and isinstance(cand_embeddings, pd.DataFrame):
        validate_bijective_ids(ref_embeddings.index, axis_name="ref_embeddings.index")
        validate_bijective_ids(cand_embeddings.index, axis_name="cand_embeddings.index")
        if set(ref_embeddings.index.astype(str)) != set(cand_embeddings.index.astype(str)):
            raise ValueError("Reference and candidate embedding sample ID sets differ")
        ref_arr = ref_embeddings.to_numpy(dtype=np.float64)
        cand_df = cand_embeddings.copy()
        cand_df.index = cand_df.index.astype(str)
        cand_arr = cand_df.loc[ref_embeddings.index.astype(str)].to_numpy(dtype=np.float64)
    else:
        ref_arr = np.asarray(ref_embeddings, dtype=np.float64)
        cand_arr = np.asarray(cand_embeddings, dtype=np.float64)

    if ref_arr.shape != cand_arr.shape:
        raise ValueError(f"Embedding shapes differ: {ref_arr.shape} vs {cand_arr.shape}")

    ev_ref = np.asarray(ref_eigenvalues, dtype=np.float64)
    ev_cand = np.asarray(cand_eigenvalues, dtype=np.float64)
    if ev_ref.shape != ev_cand.shape or len(ev_ref) != ref_arr.shape[1]:
        raise ValueError("Eigenvalue vector lengths must match embedding column count")

    max_ev_diff = float(np.max(np.abs(ev_ref - ev_cand)))
    spectra_match = bool(max_ev_diff <= eigenvalue_atol)

    # Partition dimensions into isolated simple modes vs genuinely degenerate blocks based on ref spectrum
    k = len(ev_ref)
    blocks: list[list[int]] = []
    cur_block = [0]
    for j in range(1, k):
        if abs(ev_ref[j] - ev_ref[j - 1]) <= degeneracy_gap_tol:
            cur_block.append(j)
        else:
            blocks.append(cur_block)
            cur_block = [j]
    blocks.append(cur_block)

    aligned_cand = np.empty_like(cand_arr)
    block_transforms: list[dict[str, Any]] = []
    for blk in blocks:
        r_sub = ref_arr[:, blk]
        c_sub = cand_arr[:, blk]
        if len(blk) == 1:
            j = blk[0]
            dot_val = float(np.dot(r_sub[:, 0], c_sub[:, 0]))
            sgn = 1.0 if dot_val >= 0.0 else -1.0
            aligned_cand[:, j] = sgn * c_sub[:, 0]
            block_transforms.append({"dims": blk, "type": "sign_flip", "sign": int(sgn)})
        else:
            # Orthogonal Procrustes strictly restricted to the degenerate eigenspace block
            u_p, _, vt_p = la.svd(c_sub.T @ r_sub, full_matrices=False)
            q_orth = u_p @ vt_p
            aligned_cand[:, blk] = c_sub @ q_orth
            block_transforms.append({"dims": blk, "type": "degenerate_orthogonal", "size": len(blk)})

    diff = ref_arr - aligned_cand
    max_abs_err = float(np.max(np.abs(diff)))
    frob_err = float(la.norm(diff))
    ref_norm = float(la.norm(ref_arr))
    rel_frob = frob_err / ref_norm if ref_norm > 0 else frob_err

    # Pairwise distance preservation check (gauge-invariant across all orthogonal transforms)
    d_ref = pairwise_distances(ref_arr, metric="euclidean")
    d_cand = pairwise_distances(cand_arr, metric="euclidean")
    pairwise_dist_max_err = float(np.max(np.abs(d_ref - d_cand)))

    equivalent = bool(spectra_match and max_abs_err <= subspace_atol)
    return {
        "equivalent_under_legal_gauge": equivalent,
        "spectra_match": spectra_match,
        "max_eigenvalue_abs_diff": max_ev_diff,
        "max_coordinate_abs_err_after_legal_gauge": max_abs_err,
        "relative_frobenius_err_after_legal_gauge": rel_frob,
        "pairwise_distance_max_abs_err": pairwise_dist_max_err,
        "blocks": block_transforms,
        "aligned_candidate": aligned_cand,
    }
