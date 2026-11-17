"""Donor-level pseudobulk aggregation and Gamma-Poisson Trajectory DE testing (F16, B13, H7).

Implements exact native Python counterparts of:
- ``AggregateExpression`` (:func:`aggregate_pseudobulk`)
- ``scSLIDE:::filterByExpr`` & ``cpm`` (:func:`cpm`, :func:`filter_by_expr`)
- ``Seurat::FoldChange`` along trajectory quantiles (:func:`trajectory_fold_change`)
- ``glmGamPoi::glm_gp(..., size_factors=FALSE)`` + ``glmGamPoi::test_de(contrast='Traj')``
  (:func:`fit_gamma_poisson_glm`, :func:`traj_de_test`)

Also provides explicit method-separation modes for B13 negative controls
(``test_mode='wald'`` vs ``'ql_f'``, ``library_size_mode='covariate'`` vs ``'offset'``)
and exploratory donor-level permutation / split-sample diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any
from collections.abc import Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, f as f_dist, norm, spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ._validation import (
    StreamingMatrix,
    validate_bijective_ids,
    validate_numeric_matrix_values,
    validate_open_backed_anndata,
)


# ---------------------------------------------------------------------------
# Pseudobulk aggregation (Donor x CellType raw count sum)
# ---------------------------------------------------------------------------

def aggregate_pseudobulk(
    adata: Any,
    *,
    group_by: Sequence[str] = ("Donor", "azimuth_label"),
    layer: str | None = "counts",
    min_cells_per_pseudobulk: int = 1,
    require_integer_counts: bool = False,
) -> Any:
    """Aggregate single-cell raw UMI counts into ``donor x cell_type`` pseudobulk ``AnnData``.

    Matches ``Seurat::AggregateExpression(..., return.seurat = TRUE)``:
    - Sums raw counts per unique combination of ``group_by`` columns without densifying ``X``.
    - Computes ``nCount_RNA``, ``nFeature_RNA``, ``log_count = log10(nCount_RNA)``, and
      standard log-normalized ``layers['data']`` (``log1p(counts / nCount_RNA * 10000)``)
      for downstream ``FoldChange``.
    """
    from .anndata import _require_anndata

    ad = _require_anndata()
    validate_open_backed_anndata(adata, caller="aggregate_pseudobulk")
    if len(group_by) < 1:
        raise ValueError("group_by must specify at least one obs column")
    for col in group_by:
        if col not in adata.obs.columns:
            raise ValueError(f"Column {col!r} not found in adata.obs")

    if layer is not None and layer in adata.layers:
        raw_mat = adata.layers[layer]
    elif adata.raw is not None and layer == "raw":
        raw_mat = adata.raw.X
    else:
        raw_mat = adata.X

    obs_sub = adata.obs[list(group_by)].copy()
    valid_rows = np.flatnonzero(~obs_sub.isna().any(axis=1).to_numpy())
    if valid_rows.size == 0:
        raise ValueError("No cells have non-missing values across group_by columns")

    obs_valid = obs_sub.iloc[valid_rows]
    # Build composite key matching Seurat AggregateExpression ('_' joined with '_' -> '-')
    comp_keys = [
        "_".join(str(obs_valid.iloc[i, j]).replace("_", "-") for j in range(len(group_by)))
        for i in range(len(obs_valid))
    ]
    uniq_keys: list[str] = []
    key_to_pb: dict[str, int] = {}
    pb_idx_per_cell = np.empty(len(valid_rows), dtype=np.int64)
    first_row_for_pb: list[int] = []

    for pos, k_str in enumerate(comp_keys):
        if k_str not in key_to_pb:
            key_to_pb[k_str] = len(uniq_keys)
            uniq_keys.append(k_str)
            first_row_for_pb.append(pos)
        pb_idx_per_cell[pos] = key_to_pb[k_str]

    n_pb = len(uniq_keys)
    sm = StreamingMatrix(raw_mat, row_indices=valid_rows)
    pb_counts = np.zeros((n_pb, sm.n_vars), dtype=np.float64)
    cell_counts = np.bincount(pb_idx_per_cell, minlength=n_pb)

    for start, stop, blk in sm.iter_row_chunks():
        indicator = sp.csr_matrix(
            (np.ones(stop - start, dtype=np.float64), (pb_idx_per_cell[start:stop], np.arange(stop - start))),
            shape=(n_pb, stop - start),
        )
        prod = indicator @ blk
        pb_counts += prod.toarray() if sp.issparse(prod) else np.asarray(prod, dtype=np.float64)

    validate_numeric_matrix_values(
        pb_counts, name="pseudobulk counts", non_negative=True, require_integer=require_integer_counts
    )

    keep_pb = np.flatnonzero(cell_counts >= int(min_cells_per_pseudobulk))
    pb_counts = pb_counts[keep_pb]
    pb_obs = obs_valid.iloc[[first_row_for_pb[i] for i in keep_pb]].copy()
    pb_obs.index = pd.Index([uniq_keys[i] for i in keep_pb])
    pb_obs["n_cells_aggregated"] = cell_counts[keep_pb].astype(np.int64)
    pb_obs["nCount_RNA"] = pb_counts.sum(axis=1)
    pb_obs["nFeature_RNA"] = (pb_counts > 0).sum(axis=1).astype(np.int64)
    pb_obs["log_count"] = np.where(pb_obs["nCount_RNA"] > 0, np.log10(np.maximum(pb_obs["nCount_RNA"], 1.0)), 0.0)

    totals = pb_obs["nCount_RNA"].to_numpy(dtype=np.float64)[:, None]
    norm_data = np.log1p(np.divide(pb_counts, totals, out=np.zeros_like(pb_counts), where=totals > 0) * 10_000.0)

    var_df = pd.DataFrame(index=adata.var_names.astype(str))
    out = ad.AnnData(X=pb_counts.copy(), obs=pb_obs, var=var_df)
    out.layers["counts"] = pb_counts.copy()
    out.layers["data"] = norm_data
    out.uns["scslide"] = {
        "schema_version": "pseudobulk-1",
        "observation_unit": "pseudobulk_donor_celltype",
        "variable_unit": "gene",
        "group_by": list(group_by),
    }
    return out


# ---------------------------------------------------------------------------
# Exact filterByExpr and trajectory quantile FoldChange
# ---------------------------------------------------------------------------

def cpm(
    counts_genes_by_samples: np.ndarray,
    *,
    lib_size: np.ndarray | None = None,
    log: bool = False,
    prior_count: float = 2.0,
) -> np.ndarray:
    """Exact translation of ``scSLIDE:::cpm`` (``genes x samples``)."""
    y = np.asarray(counts_genes_by_samples, dtype=np.float64)
    ls = y.sum(axis=0) if lib_size is None else np.asarray(lib_size, dtype=np.float64)
    safe_ls = np.maximum(ls, 1e-300)
    if log:
        return np.log2((y + float(prior_count)) / safe_ls[None, :] * 1e6)
    return (y / safe_ls[None, :]) * 1e6


def filter_by_expr(
    counts_genes_by_samples: np.ndarray,
    *,
    design: np.ndarray | None = None,
    group: Sequence[Any] | None = None,
    lib_size: np.ndarray | None = None,
    min_count: float = 10.0,
    min_total_count: float = 15.0,
    large_n: int = 10,
    min_prop: float = 0.7,
) -> np.ndarray:
    """Exact translation of ``scSLIDE:::filterByExpr`` (``R/differential_expression.R:363-392``)."""
    y = np.asarray(counts_genes_by_samples, dtype=np.float64)
    validate_numeric_matrix_values(y, name="counts", non_negative=True)
    ls = y.sum(axis=0) if lib_size is None else np.asarray(lib_size, dtype=np.float64)

    if group is None:
        if design is None:
            min_sample_size = float(y.shape[1])
        else:
            q_mat, _ = la.qr(np.asarray(design, dtype=np.float64), mode="economic")
            h = np.sum(q_mat * q_mat, axis=1)
            min_sample_size = 1.0 / max(float(np.max(h)), 1e-12)
    else:
        _, cnts = np.unique(np.asarray(list(group), dtype=object), return_counts=True)
        cnts = cnts[cnts > 0]
        min_sample_size = float(cnts.min()) if cnts.size > 0 else float(y.shape[1])

    if min_sample_size > large_n:
        min_sample_size = float(large_n) + (min_sample_size - float(large_n)) * float(min_prop)

    med_ls = float(np.median(ls))
    if med_ls <= 0:
        return np.zeros(y.shape[0], dtype=bool)
    cpm_cutoff = float(min_count) / med_ls * 1e6
    cpm_mat = cpm(y, lib_size=ls, log=False)
    tol = 1e-14
    keep_cpm = np.sum(cpm_mat >= cpm_cutoff, axis=1) >= (min_sample_size - tol)
    keep_tot = y.sum(axis=1) >= (float(min_total_count) - tol)
    return keep_cpm & keep_tot


def trajectory_fold_change(
    log_norm_genes_by_samples: np.ndarray,
    traj_values: np.ndarray,
    *,
    prob_break_point: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    pseudocount_use: float = 1.0,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Exact translation of ``TrajDETest.Assay`` quantile ``Seurat::FoldChange``."""
    if len(prob_break_point) != 2:
        raise ValueError("prob_break_point must have length 2")
    data = np.asarray(log_norm_genes_by_samples, dtype=np.float64)
    traj = np.asarray(traj_values, dtype=np.float64).ravel()
    if data.shape[1] != len(traj):
        raise ValueError("Sample count in log_norm_genes_by_samples must match traj_values length")

    # Match R quantile(..., type = 7) default
    q1, q2 = np.quantile(traj, prob_break_point, method="linear")
    bottom_mask = traj <= q1
    top_mask = traj > q2
    if not np.any(bottom_mask) or not np.any(top_mask):
        raise ValueError("Trajectory break points produced an empty top or bottom sample group")

    x_top = data[:, top_mask]
    x_bot = data[:, bottom_mask]

    # Seurat FoldChange mean.fxn in TrajDETest.Assay: log2((rowSums(expm1(x)) + pseudocount.use) / NCOL(x))
    mean_top = np.log2((np.sum(np.expm1(x_top), axis=1) + float(pseudocount_use)) / float(x_top.shape[1]))
    mean_bot = np.log2((np.sum(np.expm1(x_bot), axis=1) + float(pseudocount_use)) / float(x_bot.shape[1]))
    avg_log2fc = mean_top - mean_bot

    # Seurat FoldChange pct.1 & pct.2 rounded to 3 decimal places
    pct_1 = np.round(np.sum(x_top > 0, axis=1) / float(x_top.shape[1]), 3)
    pct_2 = np.round(np.sum(x_bot > 0, axis=1) / float(x_bot.shape[1]), 3)

    idx = pd.Index([str(f) for f in feature_names]) if feature_names is not None else pd.RangeIndex(data.shape[0])
    return pd.DataFrame(
        {"avg_log2FC": avg_log2fc, "pct.1": pct_1, "pct.2": pct_2},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Exact Gamma-Poisson GLM + Empirical Bayes QL F-test (glmGamPoi native core)
# ---------------------------------------------------------------------------

def _nb_deviance_rowwise(y: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Exact Negative Binomial unit deviance ``2 * sum_i [y log(y/mu) - (y + 1/theta) log((1 + theta y)/(1 + theta mu))]``."""
    mu_safe = np.maximum(mu, 1e-300)
    th = np.asarray(theta, dtype=np.float64)[:, None]
    pos_y = y > 0
    term1 = np.where(pos_y, y * np.log(np.maximum(y, 1e-300) / mu_safe), 0.0)
    # Poisson limit when theta <= 1e-12
    pois_dev = 2.0 * np.sum(term1 - (y - mu_safe), axis=1)

    nb_mask = (th[:, 0] > 1e-12)
    out = pois_dev.copy()
    if np.any(nb_mask):
        ym = y[nb_mask]
        mm = mu_safe[nb_mask]
        tm = th[nb_mask]
        inv_t = 1.0 / tm
        t1 = np.where(ym > 0, ym * np.log(np.maximum(ym, 1e-300) / mm), 0.0)
        t2 = (ym + inv_t) * np.log((1.0 + tm * ym) / (1.0 + tm * mm))
        out[nb_mask] = 2.0 * np.sum(t1 - t2, axis=1)
    return np.maximum(out, 0.0)


def _fit_nb_betas_irls(
    y_genes_by_samples: np.ndarray,
    design: np.ndarray,
    dispersions: np.ndarray,
    *,
    offset_matrix: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Damped Newton / Fisher-scoring IRLS for ``G`` Negative Binomial GLMs with log link."""
    n_genes, n_samples = y_genes_by_samples.shape
    p = design.shape[1]
    off = np.zeros_like(y_genes_by_samples) if offset_matrix is None else np.asarray(offset_matrix, dtype=np.float64)

    if beta_init is None:
        log_y = np.log(np.maximum(y_genes_by_samples, 0.5)) - off
        # OLS initial guess across all genes simultaneously: (X^T X)^{-1} X^T log_y^T
        beta = la.lstsq(design, log_y.T)[0].T.copy()
    else:
        beta = np.asarray(beta_init, dtype=np.float64).copy()

    eta = np.clip(beta @ design.T + off, -25.0, 25.0)
    mu = np.exp(eta)
    dev = _nb_deviance_rowwise(y_genes_by_samples, mu, dispersions)

    # Active mask converges gene-by-gene with step-halving line search on NB deviance
    active = np.ones(n_genes, dtype=bool)
    eye_p = 1e-10 * np.eye(p, dtype=np.float64)

    for _ in range(max_iter):
        idx_act = np.flatnonzero(active)
        if idx_act.size == 0:
            break

        y_a = y_genes_by_samples[idx_act]
        mu_a = mu[idx_act]
        th_a = dispersions[idx_act][:, None]
        w_a = mu_a / (1.0 + th_a * mu_a)                      # (G_a, N) Fisher weights
        score_a = (y_a - mu_a) / (1.0 + th_a * mu_a)          # (G_a, N) d logL / d eta

        # Batched 3D Cholesky / solve over (G_a, p, p) Fisher information X^T diag(w_g) X!
        grad = score_a @ design                               # (G_a, p)
        fim = np.einsum("gn,ni,nj->gij", w_a, design, design) + eye_p[None, :, :]
        try:
            delta = np.linalg.solve(fim, grad[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            delta = np.array([la.lstsq(fim[g], grad[g])[0] for g in range(len(idx_act))])

        # Step-halving backtracking where deviance fails to improve
        step = np.ones(len(idx_act), dtype=np.float64)
        b_old = beta[idx_act]
        d_old = dev[idx_act]
        off_a = off[idx_act]

        for _bt in range(10):
            b_try = b_old + step[:, None] * delta
            eta_try = np.clip(b_try @ design.T + off_a, -25.0, 25.0)
            mu_try = np.exp(eta_try)
            dev_try = _nb_deviance_rowwise(y_a, mu_try, dispersions[idx_act])
            bad = (dev_try > d_old + 1e-7 * np.maximum(1.0, d_old)) | ~np.isfinite(dev_try)
            if not np.any(bad):
                break
            step[bad] *= 0.5

        b_new = b_old + step[:, None] * delta
        eta_new = np.clip(b_new @ design.T + off_a, -25.0, 25.0)
        mu_new = np.exp(eta_new)
        dev_new = _nb_deviance_rowwise(y_a, mu_new, dispersions[idx_act])

        # Retain old iterate if even tiny step worsened deviance
        worse = dev_new > d_old + 1e-6 * np.maximum(1.0, d_old)
        b_new[worse] = b_old[worse]
        mu_new[worse] = mu_a[worse]
        dev_new[worse] = d_old[worse]

        rel_change = np.abs(d_old - dev_new) / np.maximum(0.1, d_old)
        step_norm = np.max(np.abs(b_new - b_old), axis=1)

        beta[idx_act] = b_new
        mu[idx_act] = mu_new
        dev[idx_act] = dev_new
        converged_now = (rel_change < tol) & (step_norm < 1e-6)
        active[idx_act[converged_now]] = False

    return beta, mu, dev


def _cox_reid_nb_loglik_neg(
    log_theta: float,
    y_row: np.ndarray,
    mu_row: np.ndarray,
    design: np.ndarray,
    do_cr: bool = True,
) -> float:
    theta = math.exp(log_theta)
    r = 1.0 / theta
    pos = y_row > 0
    ll = 0.0
    if np.any(pos):
        yp = y_row[pos]
        ll += float(np.sum(gammaln(yp + r) - gammaln(r)))
    ll += float(np.sum(r * np.log(r / (r + mu_row)) + y_row * np.log(np.maximum(mu_row, 1e-300) / (r + mu_row))))
    if do_cr:
        w = mu_row / (1.0 + theta * mu_row)
        xt_wx = design.T @ (w[:, None] * design)
        sign, logdet = np.linalg.slogdet(xt_wx)
        if sign > 0:
            ll -= 0.5 * float(logdet)
    return -ll


def _estimate_overdispersion_mle(
    y_genes_by_samples: np.ndarray,
    mu_genes_by_samples: np.ndarray,
    design: np.ndarray,
    *,
    do_cox_reid: bool = True,
) -> np.ndarray:
    """Per-gene Cox-Reid adjusted profile likelihood MLE of NB overdispersion ``theta`` (matching ``glmGamPoi::overdispersion_mle``)."""
    n_genes = y_genes_by_samples.shape[0]
    out = np.zeros(n_genes, dtype=np.float64)

    for g in range(n_genes):
        yv = y_genes_by_samples[g]
        mv = mu_genes_by_samples[g]
        if np.all(yv == 0) or float(mv.mean()) < 1e-8:
            out[g] = 0.0
            continue
        # Compare boundary theta -> 0 (Poisson) vs bounded 1-D Brent optimum over log(theta) in [-18, 6]
        res = minimize_scalar(
            _cox_reid_nb_loglik_neg,
            bounds=(-18.0, 6.0),
            args=(yv, mv, design, do_cox_reid),
            method="bounded",
            options={"xatol": 1e-5},
        )
        f_zero = _cox_reid_nb_loglik_neg(-22.0, yv, mv, design, do_cox_reid)
        if f_zero <= float(res.fun):
            out[g] = 0.0
        else:
            out[g] = float(math.exp(res.x))
    return out


def _weighted_median_no_interp(values: np.ndarray, weights: np.ndarray) -> float:
    ord_i = np.argsort(values, kind="mergesort")
    v_s = values[ord_i]
    w_s = weights[ord_i]
    cum_w = np.cumsum(w_s)
    half = 0.5 * float(cum_w[-1])
    idx = int(np.searchsorted(cum_w, half, side="left"))
    return float(v_s[min(idx, len(v_s) - 1)])


def _loc_median_fit(x: np.ndarray, y: np.ndarray, *, fraction: float = 0.1, min_npoints: int = 100) -> np.ndarray:
    """Exact translation of ``glmGamPoi:::loc_median_fit``."""
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    npoints = min(n, max(1, min_npoints, int(round(n * fraction))))
    ord_x = np.argsort(x, kind="mergesort")
    ordered_y = y[ord_x]
    half_pts = npoints // 2
    start = half_pts
    end = n - 1 - half_pts

    if end < start:
        w_all = norm.pdf(np.linspace(-3.0, 3.0, n))
        wm = _weighted_median_no_interp(y, w_all)
        return np.full(n, wm, dtype=np.float64)

    weights = norm.pdf(np.linspace(-3.0, 3.0, 2 * half_pts + 1))
    res_ord = np.empty(n, dtype=np.float64)
    for idx in range(start, end + 1):
        sel = ordered_y[idx - half_pts : idx + half_pts + 1]
        res_ord[idx] = _weighted_median_no_interp(sel, weights)
    res_ord[:start] = res_ord[start]
    res_ord[end + 1 :] = res_ord[end]

    inv_ord = np.empty_like(ord_x)
    inv_ord[ord_x] = np.arange(n)
    return res_ord[inv_ord]


def _natural_cubic_spline_basis_df4(x: np.ndarray, knots_internal: tuple[float, float]) -> np.ndarray:
    """Natural cubic spline basis with ``df=4, intercept=TRUE`` and 2 internal knots (spans identical space to ``splines::ns(..., df=4, knots=k, intercept=TRUE)``)."""
    k_min, k_max = float(np.min(x)), float(np.max(x))
    k1, k2 = float(knots_internal[0]), float(knots_internal[1])
    if k_max - k_min < 1e-12:
        return np.ones((len(x), 1), dtype=np.float64)

    def d_fn(u: np.ndarray, knot: float) -> np.ndarray:
        denom = max(k_max - knot, 1e-12)
        return (np.maximum(0.0, u - knot) ** 3 - np.maximum(0.0, u - k_max) ** 3) / denom

    d_kmin = d_fn(x, k_min)
    n3 = d_kmin - d_fn(x, k2)
    n4 = d_fn(x, k1) - d_fn(x, k2)
    raw = np.column_stack([np.ones_like(x), x, n3, n4])
    # Orthonormalize columns via QR for numerical conditioning during BFGS/Nelder-Mead MLE
    q_basis, _ = la.qr(raw, mode="economic")
    return q_basis


def _variance_prior_glmgampoi(
    ql_disp: np.ndarray,
    df: float,
    gene_means: np.ndarray,
    *,
    abundance_trend: bool = True,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Exact translation of ``glmGamPoi:::variance_prior`` (scaled F prior MLE)."""
    valid = np.isfinite(ql_disp) & (ql_disp > 0)
    if not np.any(valid):
        return np.ones_like(ql_disp), np.inf, ql_disp.copy()

    s2_sub = np.maximum(ql_disp[valid], 1e-8)
    if np.allclose(s2_sub, 1.0):
        return np.ones_like(ql_disp), np.inf, ql_disp.copy()

    def neg_ll_const(par: np.ndarray) -> float:
        lv0, ldf0 = float(par[0]), float(np.clip(par[1], -12.0, 16.0))
        v0 = math.exp(np.clip(lv0, -15.0, 15.0))
        df0 = math.exp(ldf0)
        lp = f_dist.logpdf(s2_sub / v0, dfn=df, dfd=df0) - math.log(v0)
        val = -float(np.sum(lp))
        return val if np.isfinite(val) else 1e30

    opt0 = minimize(neg_ll_const, x0=np.array([0.0, 0.0]), method="Nelder-Mead", options={"maxiter": 2000})
    var0_sub = np.full(len(s2_sub), math.exp(float(np.clip(opt0.x[0], -15.0, 15.0))), dtype=np.float64)
    df0 = math.exp(float(np.clip(opt0.x[1], -12.0, 16.0)))

    cov_sub = gene_means[valid]
    not_zero = cov_sub > 1e-8
    if abundance_trend and len(s2_sub) >= 10 and int(np.sum(not_zero)) >= 10:
        log_cov = np.log(cov_sub[not_zero])
        lo, hi = float(np.min(log_cov)), float(np.max(log_cov))
        if hi - lo > 1e-8:
            knots = (lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0)
            basis = _natural_cubic_spline_basis_df4(log_cov, knots)
            k_b = basis.shape[1]
            init_beta = la.lstsq(basis, np.log(s2_sub[not_zero]))[0]
            s2_nz = s2_sub[not_zero]

            def neg_ll_trend(par: np.ndarray) -> float:
                eta_v = np.clip(basis @ par[:k_b], -15.0, 15.0)
                v0_vec = np.exp(eta_v)
                df0_t = math.exp(float(np.clip(par[k_b], -12.0, 16.0)))
                lp = f_dist.logpdf(s2_nz / v0_vec, dfn=df, dfd=df0_t) - eta_v
                val = -float(np.sum(lp))
                return val if np.isfinite(val) else 1e30

            opt_t = minimize(
                neg_ll_trend,
                x0=np.concatenate([init_beta, [math.log(max(df0, 1e-3))]]),
                method="BFGS",
                options={"maxiter": 1000},
            )
            if np.isfinite(opt_t.fun) and opt_t.fun <= opt0.fun + 1e-3:
                var0_sub[not_zero] = np.exp(np.clip(basis @ opt_t.x[:k_b], -15.0, 15.0))
                df0 = math.exp(float(np.clip(opt_t.x[k_b], -12.0, 16.0)))

    var0_full = np.full_like(ql_disp, np.nan)
    var_post = np.full_like(ql_disp, np.nan)
    var0_full[valid] = var0_sub
    var_post[valid] = (df0 * var0_sub + df * s2_sub) / (df0 + df)
    return var0_full, float(df0), var_post


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Exact R ``p.adjust(p, method = 'BH')`` preserving ``NaN`` entries."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = len(valid)
    if m == 0:
        return out
    pv = p[valid]
    ord_desc = np.argsort(-pv, kind="mergesort")
    ranks_desc = np.arange(m, 0, -1, dtype=np.float64)
    adj_desc = np.minimum(1.0, np.minimum.accumulate((float(m) / ranks_desc) * pv[ord_desc]))
    inv = np.empty_like(ord_desc)
    inv[ord_desc] = np.arange(m)
    out[valid] = adj_desc[inv]
    return out


def benjamini_yekutieli(pvals: np.ndarray) -> np.ndarray:
    """Exact R ``p.adjust(p, method = 'BY')`` valid under arbitrary dependence across genes."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = len(valid)
    if m == 0:
        return out
    c_m = float(np.sum(1.0 / np.arange(1, m + 1, dtype=np.float64)))
    pv = p[valid]
    ord_desc = np.argsort(-pv, kind="mergesort")
    ranks_desc = np.arange(m, 0, -1, dtype=np.float64)
    adj_desc = np.minimum(1.0, np.minimum.accumulate((float(m) * c_m / ranks_desc) * pv[ord_desc]))
    inv = np.empty_like(ord_desc)
    inv[ord_desc] = np.arange(m)
    out[valid] = adj_desc[inv]
    return out


@dataclass
class GammaPoissonFit:
    """Fitted native ``glmGamPoi`` model state."""

    beta: np.ndarray
    mu: np.ndarray
    deviances: np.ndarray
    overdispersions: np.ndarray
    dispersion_trend: np.ndarray
    ql_disp_shrunken: np.ndarray
    ql_df0: float
    design: np.ndarray
    design_columns: list[str]
    offset: np.ndarray
    feature_names: list[str]
    sample_names: list[str]


def fit_gamma_poisson_glm(
    counts_genes_by_samples: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
    *,
    offset: np.ndarray | float = 0.0,
    overdispersion_shrinkage: bool = True,
    do_cox_reid_adjustment: bool = True,
) -> GammaPoissonFit:
    """Native Python implementation of ``glmGamPoi::glm_gp(..., size_factors=FALSE)``."""
    if isinstance(counts_genes_by_samples, pd.DataFrame):
        feat_names = [str(x) for x in counts_genes_by_samples.index]
        samp_names = [str(x) for x in counts_genes_by_samples.columns]
        y_mat = counts_genes_by_samples.to_numpy(dtype=np.float64)
    else:
        y_mat = np.asarray(counts_genes_by_samples, dtype=np.float64)
        feat_names = [f"gene_{i}" for i in range(y_mat.shape[0])]
        samp_names = [f"sample_{j}" for j in range(y_mat.shape[1])]

    validate_numeric_matrix_values(y_mat, name="counts", non_negative=True)
    if isinstance(design, pd.DataFrame):
        des_cols = [str(c) for c in design.columns]
        x_des = design.to_numpy(dtype=np.float64)
    else:
        x_des = np.asarray(design, dtype=np.float64)
        des_cols = [f"Coef_{j + 1}" for j in range(x_des.shape[1])]

    n_genes, n_samples = y_mat.shape
    p = x_des.shape[1]
    if x_des.shape[0] != n_samples:
        raise ValueError(f"Design matrix row count ({x_des.shape[0]}) must match sample count ({n_samples})")
    if n_samples <= p:
        raise ValueError(
            f"Sample count ({n_samples}) must exceed design matrix column count ({p}) to estimate NB dispersion"
        )
    rank_x = int(np.linalg.matrix_rank(x_des))
    if rank_x < p:
        raise ValueError(f"Design matrix is rank-deficient (rank {rank_x} < {p} columns); check collinear covariates")

    off_mat = (
        np.full_like(y_mat, float(offset))
        if np.ndim(offset) == 0
        else np.broadcast_to(np.asarray(offset, dtype=np.float64), y_mat.shape).copy()
    )

    # 1. Rough method-of-moments initial dispersion + initial Fisher scoring beta
    row_means = np.maximum(y_mat.mean(axis=1), 1e-8)
    row_vars = np.var(y_mat, axis=1, ddof=1)
    disp_init = np.clip((row_vars - row_means) / np.maximum(row_means ** 2, 1e-8), 1e-4, 10.0)

    beta_1, mu_1, _ = _fit_nb_betas_irls(y_mat, x_des, disp_init, offset_matrix=off_mat)

    # 2. Cox-Reid adjusted profile likelihood overdispersion MLE
    disp_mle = _estimate_overdispersion_mle(y_mat, mu_1, x_des, do_cox_reid=do_cox_reid_adjustment)

    # 3. Overdispersion trend + Empirical Bayes quasi-likelihood dispersion shrinkage
    df_res = float(n_samples - p)
    gene_means = mu_1.mean(axis=1)
    if overdispersion_shrinkage and n_genes >= 3:
        disp_trend = _loc_median_fit(gene_means, disp_mle)
        disp_trend = np.maximum(disp_trend, 0.0)
        ql_disp = (1.0 + gene_means * disp_mle) / np.maximum(1.0 + gene_means * disp_trend, 1e-12)
        _, ql_df0, ql_shrunk = _variance_prior_glmgampoi(
            ql_disp, df=df_res, gene_means=gene_means, abundance_trend=(n_genes >= 100)
        )
    else:
        disp_trend = disp_mle.copy()
        ql_shrunk = np.ones(n_genes, dtype=np.float64)
        ql_df0 = np.inf

    # 4. Refit beta using trended dispersion (matching glmGamPoi:::glm_gp_impl lines 105-115)
    beta_final, mu_final, dev_final = _fit_nb_betas_irls(
        y_mat, x_des, disp_trend, offset_matrix=off_mat, beta_init=beta_1
    )

    return GammaPoissonFit(
        beta=beta_final,
        mu=mu_final,
        deviances=dev_final,
        overdispersions=disp_mle,
        dispersion_trend=disp_trend,
        ql_disp_shrunken=ql_shrunk,
        ql_df0=ql_df0,
        design=x_des,
        design_columns=des_cols,
        offset=off_mat,
        feature_names=feat_names,
        sample_names=samp_names,
    )


def _build_traj_design(
    traj_series: pd.Series,
    latent_df: pd.DataFrame | None,
) -> tuple[np.ndarray, list[str]]:
    """Build R ``model.matrix(~ ., data = cbind(Traj = traj.var, latent.vars))`` with intercept."""
    if traj_series.isna().any() or not np.all(np.isfinite(pd.to_numeric(traj_series, errors="coerce"))):
        raise ValueError("traj_var contains missing or non-finite values among tested samples")

    cols = ["(Intercept)", "Traj"]
    blocks = [np.ones((len(traj_series), 1), dtype=np.float64), traj_series.to_numpy(dtype=np.float64).reshape(-1, 1)]

    if latent_df is not None and latent_df.shape[1] > 0:
        if latent_df.isna().any().any():
            bad_cols = list(latent_df.columns[latent_df.isna().any()])
            raise ValueError(f"latent_vars contains NA values in columns {bad_cols}; subset samples first")
        for col in latent_df.columns:
            s = latent_df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
                vals = s.to_numpy(dtype=np.float64).reshape(-1, 1)
                if not np.all(np.isfinite(vals)):
                    raise ValueError(f"Numeric covariate {col!r} contains non-finite values")
                blocks.append(vals)
                cols.append(str(col))
            else:
                if isinstance(s.dtype, pd.CategoricalDtype):
                    levels = [lvl for lvl in s.cat.categories]
                    codes = s.cat.codes.to_numpy()
                else:
                    str_v = s.astype(str).to_numpy()
                    levels = sorted(pd.unique(str_v).tolist())
                    lmap = {lv: i for i, lv in enumerate(levels)}
                    codes = np.array([lmap[v] for v in str_v], dtype=int)
                if len(levels) < 2:
                    raise ValueError(f"Categorical covariate {col!r} has only 1 level among tested samples (collinear with intercept)")
                for j in range(1, len(levels)):
                    ind = (codes == j).astype(np.float64).reshape(-1, 1)
                    blocks.append(ind)
                    cols.append(f"{col}{levels[j]}")

    mat = np.hstack(blocks)
    return mat, cols


def traj_de_test(
    object_or_counts: Any,
    traj_var: str | Sequence[float] | pd.Series | pd.DataFrame,
    *,
    latent_vars: Sequence[str] | pd.DataFrame | None = None,
    samples: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    layer: str = "counts",
    data_layer: str = "data",
    logfc_threshold: float = 0.0,
    prob_break_point: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    min_pct: float = 0.1,
    min_count: float = 10.0,
    pseudocount_use: float = 1.0,
    complete_results: bool = False,
    apply_assay_filters: bool | None = None,
    library_size_mode: str = "covariate",
    test_mode: str = "ql_f",
) -> pd.DataFrame:
    """Native Python implementation of ``TrajDETest`` (F16, ``R/differential_expression.R``).

    Parameters
    ----------
    library_size_mode
        ``'covariate'`` (default, faithful upstream scSLIDE contract: keeps ``log_count``
        in ``latent_vars`` as a free regression covariate with ``size_factors=FALSE``) or
        ``'offset'`` (explicit model alternative for B13/T5-vs-T6 testing: removes
        ``log_count`` from covariates and sets ``offset = log(library_size)``).
    test_mode
        ``'ql_f'`` (default, faithful ``glmGamPoi::test_de`` empirical-Bayes quasi-likelihood
        deviance F-test) or ``'wald'`` (asymptotic NB Fisher-information Wald z-test for B13 comparison).
    """
    if library_size_mode not in ("covariate", "offset"):
        raise ValueError("library_size_mode must be 'covariate' or 'offset'")
    if test_mode not in ("ql_f", "wald"):
        raise ValueError("test_mode must be 'ql_f' or 'wald'")

    is_adata = hasattr(object_or_counts, "obs") and hasattr(object_or_counts, "var_names")
    if is_adata:
        adata = object_or_counts
        validate_open_backed_anndata(adata, caller="traj_de_test")
        if apply_assay_filters is None:
            apply_assay_filters = True
        all_samples = [str(s) for s in adata.obs_names]
        use_samples = [str(s) for s in samples] if samples is not None else all_samples
        s_lookup = {s: i for i, s in enumerate(all_samples)}
        missing_s = [s for s in use_samples if s not in s_lookup]
        if missing_s:
            raise ValueError(f"Requested samples not found in AnnData: {missing_s[:5]}")
        row_idx = np.array([s_lookup[s] for s in use_samples], dtype=np.int64)

        all_genes = [str(g) for g in adata.var_names]
        use_genes = [str(g) for g in features] if features is not None else all_genes
        g_lookup = {g: i for i, g in enumerate(all_genes)}
        col_idx = np.array([g_lookup[g] for g in use_genes if g in g_lookup], dtype=np.int64)
        gene_names = [all_genes[i] for i in col_idx]

        raw_c = adata.layers[layer] if layer in adata.layers else adata.X
        sub_c = raw_c[row_idx, :][:, col_idx]
        counts_g_by_s = (sub_c.toarray() if sp.issparse(sub_c) else np.asarray(sub_c, dtype=np.float64)).T

        if data_layer in adata.layers:
            raw_d = adata.layers[data_layer][row_idx, :][:, col_idx]
            norm_g_by_s = (raw_d.toarray() if sp.issparse(raw_d) else np.asarray(raw_d, dtype=np.float64)).T
        else:
            ls = counts_g_by_s.sum(axis=0, keepdims=True)
            norm_g_by_s = np.log1p(np.divide(counts_g_by_s, ls, out=np.zeros_like(counts_g_by_s), where=ls > 0) * 10_000.0)

        obs_sub = adata.obs.iloc[row_idx].copy()
        if isinstance(traj_var, str):
            if traj_var not in obs_sub.columns:
                raise ValueError(f"traj_var column {traj_var!r} not found in AnnData.obs")
            traj_s = pd.to_numeric(obs_sub[traj_var], errors="coerce")
        elif isinstance(traj_var, pd.DataFrame):
            traj_s = pd.to_numeric(traj_var.loc[use_samples].iloc[:, 0], errors="coerce")
        elif isinstance(traj_var, pd.Series):
            traj_s = pd.to_numeric(traj_var.loc[use_samples], errors="coerce")
        else:
            traj_s = pd.Series(np.asarray(traj_var, dtype=np.float64), index=use_samples)

        if latent_vars is None:
            lat_df = None
        elif isinstance(latent_vars, pd.DataFrame):
            lat_df = latent_vars.loc[use_samples].copy()
        else:
            lat_cols = [str(c) for c in latent_vars]
            miss_c = [c for c in lat_cols if c not in obs_sub.columns]
            if miss_c:
                raise ValueError(f"latent_vars columns not found in AnnData.obs: {miss_c}")
            lat_df = obs_sub[lat_cols].copy()
    else:
        # Matrix / DataFrame default branch (genes x samples)
        if apply_assay_filters is None:
            apply_assay_filters = False
        if isinstance(object_or_counts, pd.DataFrame):
            gene_names = [str(g) for g in object_or_counts.index]
            use_samples = [str(s) for s in object_or_counts.columns]
            counts_g_by_s = object_or_counts.to_numpy(dtype=np.float64)
        else:
            counts_g_by_s = np.asarray(object_or_counts, dtype=np.float64)
            gene_names = [f"gene_{i}" for i in range(counts_g_by_s.shape[0])]
            use_samples = [f"sample_{j}" for j in range(counts_g_by_s.shape[1])]
        if features is not None:
            f_set = set(str(f) for f in features)
            keep_f = [i for i, g in enumerate(gene_names) if g in f_set]
            counts_g_by_s = counts_g_by_s[keep_f]
            gene_names = [gene_names[i] for i in keep_f]
        ls = counts_g_by_s.sum(axis=0, keepdims=True)
        norm_g_by_s = np.log1p(np.divide(counts_g_by_s, ls, out=np.zeros_like(counts_g_by_s), where=ls > 0) * 10_000.0)
        if isinstance(traj_var, pd.DataFrame):
            traj_s = pd.to_numeric(traj_var.iloc[:, 0], errors="coerce")
        elif isinstance(traj_var, pd.Series):
            traj_s = pd.to_numeric(traj_var, errors="coerce")
        else:
            traj_s = pd.Series(np.asarray(traj_var, dtype=np.float64), index=use_samples)
        lat_df = latent_vars.copy() if isinstance(latent_vars, pd.DataFrame) else None

    fc_df: pd.DataFrame | None = None
    if apply_assay_filters:
        keep_expr = filter_by_expr(counts_g_by_s, min_count=min_count, min_prop=min_pct)
        keep_idx = np.flatnonzero(keep_expr)
        if keep_idx.size == 0:
            raise ValueError("No genes passed filterByExpr in TrajDETest")
        counts_g_by_s = counts_g_by_s[keep_idx]
        norm_g_by_s = norm_g_by_s[keep_idx]
        gene_names = [gene_names[i] for i in keep_idx]

        fc_df = trajectory_fold_change(
            norm_g_by_s,
            traj_s.to_numpy(dtype=np.float64),
            prob_break_point=prob_break_point,
            pseudocount_use=pseudocount_use,
            feature_names=gene_names,
        )
        if float(logfc_threshold) != 0.0:
            keep_fc = np.flatnonzero(np.abs(fc_df["avg_log2FC"].to_numpy()) >= float(logfc_threshold))
            if keep_fc.size == 0:
                raise ValueError(f"No genes passed logfc_threshold={logfc_threshold}")
            counts_g_by_s = counts_g_by_s[keep_fc]
            gene_names = [gene_names[i] for i in keep_fc]
            fc_df = fc_df.iloc[keep_fc].copy()

    # Handle library_size_mode ('covariate' faithful vs 'offset' B13 comparator)
    offset_arg: np.ndarray | float = 0.0
    lat_use = lat_df.copy() if lat_df is not None else None
    if library_size_mode == "offset":
        if lat_use is not None and "log_count" in lat_use.columns:
            # Convert log10(nCount_RNA) to natural-log offset matching NB log link
            offset_vec = lat_use["log_count"].to_numpy(dtype=np.float64) * np.log(10.0)
            lat_use = lat_use.drop(columns=["log_count"])
        else:
            lib_sz = np.maximum(counts_g_by_s.sum(axis=0), 1.0)
            offset_vec = np.log(lib_sz)
        offset_arg = np.broadcast_to(offset_vec[None, :], counts_g_by_s.shape).copy()

    design_mat, design_cols = _build_traj_design(traj_s, lat_use)
    fit = fit_gamma_poisson_glm(
        pd.DataFrame(counts_g_by_s, index=gene_names, columns=use_samples),
        pd.DataFrame(design_mat, index=use_samples, columns=design_cols),
        offset=offset_arg,
        overdispersion_shrinkage=True,
    )

    traj_col_idx = design_cols.index("Traj")
    lfc_glm = np.clip(fit.beta[:, traj_col_idx] / np.log(2.0), -10.0, 10.0)

    if test_mode == "ql_f":
        # Reduced design removes column 'Traj' (equivalent to QR orthogonal complement of e_Traj)
        keep_cols = [j for j in range(len(design_cols)) if j != traj_col_idx]
        red_design = design_mat[:, keep_cols]
        b_init_red = fit.beta[:, keep_cols]
        _, _, dev_red = _fit_nb_betas_irls(
            counts_g_by_s, red_design, fit.dispersion_trend, offset_matrix=fit.offset, beta_init=b_init_red
        )
        lr = np.maximum(0.0, dev_red - fit.deviances)
        df_test = 1.0
        if np.isfinite(fit.ql_df0):
            df_fit = fit.ql_df0 + float(counts_g_by_s.shape[1] - len(design_cols))
            f_stat = (lr / df_test) / np.maximum(fit.ql_disp_shrunken, 1e-12)
            p_traj = f_dist.sf(f_stat, dfn=df_test, dfd=df_fit)
        else:
            f_stat = lr / df_test
            df_fit = np.inf
            p_traj = chi2.sf(lr, df=df_test)
    else:
        # Asymptotic NB Fisher-information Wald test (B13 negative control)
        w = fit.mu / (1.0 + fit.dispersion_trend[:, None] * fit.mu)
        fim = np.einsum("gn,ni,nj->gij", w, design_mat, design_mat) + 1e-10 * np.eye(len(design_cols))[None, :, :]
        cov_b = np.linalg.inv(fim)
        se_traj = np.sqrt(np.maximum(cov_b[:, traj_col_idx, traj_col_idx], 1e-30))
        z_stat = fit.beta[:, traj_col_idx] / se_traj
        f_stat = z_stat ** 2
        df_test = 1.0
        df_fit = np.inf
        p_traj = 2.0 * norm.sf(np.abs(z_stat))

    adj_pval = benjamini_hochberg(p_traj)

    res = pd.DataFrame(index=gene_names)
    for j, cname in enumerate(design_cols):
        res[f"beta_{cname}"] = fit.beta[:, j]
    res["name"] = gene_names
    res["p_Traj"] = p_traj
    res["adj_pval"] = adj_pval
    res["f_statistic"] = f_stat
    res["df1"] = df_test
    res["df2"] = df_fit
    res["lfc"] = lfc_glm
    res["overdispersion"] = fit.overdispersions
    res["dispersion_trend"] = fit.dispersion_trend
    res["ql_disp_shrunken"] = fit.ql_disp_shrunken
    res["gene_ID"] = gene_names

    if fc_df is not None:
        fc_aligned = fc_df.loc[gene_names]
        res["avg_log2FC"] = fc_aligned["avg_log2FC"].to_numpy()
        res["pct.1"] = fc_aligned["pct.1"].to_numpy()
        res["pct.2"] = fc_aligned["pct.2"].to_numpy()

    if is_adata:
        res = res.sort_values("p_Traj", kind="mergesort")
        res.attrs["inference_tier"] = "descriptive_post_selection_r_compatible"
        if not complete_results and fc_df is not None:
            out_short = res[["gene_ID", "beta_Traj", "p_Traj", "adj_pval", "avg_log2FC", "pct.1", "pct.2"]].copy()
            out_short.attrs["inference_tier"] = "descriptive_post_selection_r_compatible"
            return out_short
    res.attrs["inference_tier"] = "descriptive_post_selection_r_compatible"
    return res


# ---------------------------------------------------------------------------
# WP2: Subject-vs-draw validation, grouped CV, unified permutation & OOF DE
# ---------------------------------------------------------------------------

@dataclass
class SubjectMappingResult:
    """Validated mapping from sample/draw IDs (e.g. blood draws) to biological subjects."""

    table: pd.DataFrame
    sample_to_subject: dict[str, str]
    n_samples: int
    n_subjects: int
    primary_samples: list[str]
    multi_draw_subjects: dict[str, list[str]]
    invariant_conflicts: dict[str, dict[str, list[Any]]]
    discordant_columns: dict[str, dict[str, list[Any]]]
    sha256: str


def resolve_subject_mapping(
    metadata: pd.DataFrame,
    *,
    sample_col: str = "Donor",
    subject_col: str | None = None,
    subject_id_sep: str | None = None,
    invariant_cols: Sequence[str] = (),
    check_discordant_cols: Sequence[str] = (),
    time_col: str | None = None,
    strict_invariant_check: bool = True,
) -> SubjectMappingResult:
    """Build and audit a ``sample_id -> subject_id`` map from cell- or sample-level metadata."""
    if sample_col in metadata.columns:
        df_s = metadata.loc[metadata[sample_col].notna()].drop_duplicates(subset=[sample_col]).copy()
        df_s[sample_col] = df_s[sample_col].astype(str)
    else:
        validate_bijective_ids(metadata.index, axis_name="metadata.index")
        df_s = metadata.copy()
        df_s[sample_col] = df_s.index.astype(str)

    validate_bijective_ids(df_s[sample_col], axis_name=sample_col)

    if subject_col is not None:
        if subject_col not in df_s.columns:
            raise ValueError(f"subject_col {subject_col!r} not found in metadata")
        if df_s[subject_col].isna().any():
            raise ValueError(f"subject_col {subject_col!r} contains missing values")
        df_s["subject_id"] = df_s[subject_col].astype(str)
    elif subject_id_sep is not None:
        df_s["subject_id"] = [str(x).split(subject_id_sep)[0] for x in df_s[sample_col]]
        if any(s == "" for s in df_s["subject_id"]):
            raise ValueError(f"Empty subject prefix produced when splitting {sample_col!r} on {subject_id_sep!r}")
    else:
        df_s["subject_id"] = df_s[sample_col].astype(str)

    sort_keys = ["subject_id"]
    if time_col is not None and time_col in df_s.columns:
        df_s["_time_sort"] = pd.to_numeric(df_s[time_col], errors="coerce")
        sort_keys.append("_time_sort")
    sort_keys.append(sample_col)
    df_s = df_s.sort_values(sort_keys, na_position="first").reset_index(drop=True)
    if "_time_sort" in df_s.columns:
        df_s = df_s.drop(columns=["_time_sort"])

    df_s["draw_order_within_subject"] = df_s.groupby("subject_id").cumcount() + 1
    df_s["n_draws_for_subject"] = df_s.groupby("subject_id")[sample_col].transform("count")
    df_s["is_primary_baseline_draw"] = df_s["draw_order_within_subject"] == 1

    inv_conflicts: dict[str, dict[str, list[Any]]] = {}
    for col in invariant_cols:
        if col in df_s.columns:
            bad: dict[str, list[Any]] = {}
            for sid, sub in df_s.groupby("subject_id", sort=True):
                vals = pd.unique(sub[col].dropna())
                if len(vals) > 1:
                    bad[str(sid)] = [v.item() if isinstance(v, np.generic) else v for v in vals]
            if bad:
                inv_conflicts[str(col)] = bad
    if strict_invariant_check and inv_conflicts:
        raise ValueError(f"Subject mapping violated invariant subject metadata columns: {inv_conflicts}")

    disc_cols: dict[str, dict[str, list[Any]]] = {}
    for col in check_discordant_cols:
        if col in df_s.columns:
            d_bad: dict[str, list[Any]] = {}
            for sid, sub in df_s.groupby("subject_id", sort=True):
                vals = pd.unique(sub[col].dropna().astype(str))
                if len(vals) > 1:
                    d_bad[str(sid)] = vals.tolist()
            if d_bad:
                disc_cols[str(col)] = d_bad

    multi: dict[str, list[str]] = {}
    for sid, sub in df_s.groupby("subject_id", sort=True):
        if len(sub) > 1:
            multi[str(sid)] = sub[sample_col].astype(str).tolist()

    s_to_subj = dict(zip(df_s[sample_col].astype(str), df_s["subject_id"].astype(str)))
    prim = df_s.loc[df_s["is_primary_baseline_draw"], sample_col].astype(str).tolist()
    payload = json.dumps(
        [{"sample": k, "subject": s_to_subj[k]} for k in sorted(s_to_subj)],
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    return SubjectMappingResult(
        table=df_s,
        sample_to_subject=s_to_subj,
        n_samples=int(len(df_s)),
        n_subjects=int(df_s["subject_id"].nunique()),
        primary_samples=prim,
        multi_draw_subjects=multi,
        invariant_conflicts=inv_conflicts,
        discordant_columns=disc_cols,
        sha256=digest,
    )


def stratified_subject_kfold(
    sample_metadata: pd.DataFrame,
    *,
    y_col: str,
    subject_col: str = "subject_id",
    sample_col: str | None = None,
    n_splits: int = 4,
    n_bins_continuous: int = 4,
) -> list[tuple[list[str], list[str]]]:
    """Partition samples into ``n_splits`` stratified folds keeping all draws of each subject together."""
    df = sample_metadata.copy()
    if sample_col is not None and sample_col in df.columns:
        s_ids = df[sample_col].astype(str).tolist()
    else:
        s_ids = df.index.astype(str).tolist()
    validate_bijective_ids(s_ids, axis_name="sample_ids")

    if subject_col not in df.columns:
        raise ValueError(f"subject_col {subject_col!r} not found in sample_metadata")
    if y_col not in df.columns:
        raise ValueError(f"y_col {y_col!r} not found in sample_metadata")

    groups = df[subject_col].astype(str).to_numpy()
    n_uniq_subj = len(np.unique(groups))
    if int(n_splits) < 2 or int(n_splits) > n_uniq_subj:
        raise ValueError(f"n_splits ({n_splits}) must be between 2 and number of unique subjects ({n_uniq_subj})")

    y_raw = df[y_col]
    if pd.api.types.is_numeric_dtype(y_raw) and not isinstance(y_raw.dtype, pd.CategoricalDtype):
        q = min(int(n_bins_continuous), max(2, n_uniq_subj // int(n_splits)))
        strat = pd.qcut(y_raw.rank(method="first"), q=q, labels=False).to_numpy()
    else:
        strat = y_raw.astype(str).to_numpy()

    _, strat_cnt = np.unique(strat, return_counts=True)
    splitter = (
        StratifiedGroupKFold(n_splits=int(n_splits))
        if int(np.max(strat_cnt)) >= int(n_splits)
        else GroupKFold(n_splits=int(n_splits))
    )
    folds: list[tuple[list[str], list[str]]] = []
    for tr_idx, te_idx in splitter.split(s_ids, strat, groups=groups):
        tr_subj = set(groups[tr_idx])
        te_subj = set(groups[te_idx])
        if not tr_subj.isdisjoint(te_subj):
            raise RuntimeError("Internal StratifiedGroupKFold error: subject leaked across train/test")
        folds.append(([s_ids[i] for i in tr_idx], [s_ids[i] for i in te_idx]))
    return folds


def binomial_thin_counts(
    counts_or_adata: Any,
    *,
    prop: float = 0.5,
    layer: str = "counts",
    seed: int = 42,
) -> tuple[Any, Any]:
    """Sparse-native ``O(nnz)`` Binomial count splitting ``X = X^{(1)} + X^{(2)}`` (data thinning)."""
    if not (0.0 < float(prop) < 1.0):
        raise ValueError("prop must lie strictly in (0, 1)")
    rng = np.random.default_rng(int(seed))

    is_adata = hasattr(counts_or_adata, "obs") and hasattr(counts_or_adata, "var_names")
    if is_adata:
        from .anndata import _require_anndata

        ad = _require_anndata()
        validate_open_backed_anndata(counts_or_adata, caller="binomial_thin_counts")
        src = counts_or_adata.layers[layer] if layer in counts_or_adata.layers else counts_or_adata.X
        c1, c2 = binomial_thin_counts(src, prop=prop, seed=seed)
        ad1 = ad.AnnData(X=c1.copy(), obs=counts_or_adata.obs.copy(), var=counts_or_adata.var.copy())
        ad2 = ad.AnnData(X=c2.copy(), obs=counts_or_adata.obs.copy(), var=counts_or_adata.var.copy())
        ad1.layers["counts"] = c1
        ad2.layers["counts"] = c2
        for k in counts_or_adata.obsm.keys():
            ad1.obsm[k] = counts_or_adata.obsm[k].copy()
            ad2.obsm[k] = counts_or_adata.obsm[k].copy()
        return ad1, ad2

    if sp.issparse(counts_or_adata):
        csr = counts_or_adata.tocsr()
        validate_numeric_matrix_values(csr, name="counts", non_negative=True, require_integer=True)
        d1 = rng.binomial(np.round(csr.data).astype(np.int64), float(prop)).astype(csr.data.dtype)
        m1 = csr.copy()
        m1.data = d1
        m1.eliminate_zeros()
        m2 = (csr - m1).tocsr()
        m2.eliminate_zeros()
        return m1, m2

    arr = np.asarray(counts_or_adata)
    validate_numeric_matrix_values(arr, name="counts", non_negative=True, require_integer=True)
    a1 = rng.binomial(np.round(arr).astype(np.int64), float(prop)).astype(np.float64)
    a2 = arr.astype(np.float64) - a1
    if isinstance(counts_or_adata, pd.DataFrame):
        return (
            pd.DataFrame(a1, index=counts_or_adata.index, columns=counts_or_adata.columns),
            pd.DataFrame(a2, index=counts_or_adata.index, columns=counts_or_adata.columns),
        )
    return a1, a2


@dataclass(frozen=True)
class InferencePipelineSpec:
    """Frozen, hash-attested hyperparameter specification shared by observed and permuted runs."""

    ncomp_pls: int = 10
    pls_function: str = "cppls"
    add_hvg: bool = True
    top_prop: float | None = 0.01
    cor_cutoff: float | None = None
    k_nn: int = 5
    knn_range: int = 150
    fix_wnn_weights: tuple[float, float] | None = (0.5, 0.5)
    nn_method: str = "exact"
    ncomp_diffusion: int = 3
    traj_component: str = "DC_1"
    scale_max_sample: float | None = 10.0
    test_mode: str = "ql_f"
    library_size_mode: str = "covariate"

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def traj_de_permutation_test(
    object_or_counts: Any,
    traj_var: str | Sequence[float] | pd.Series | pd.DataFrame,
    *,
    latent_vars: Sequence[str] | pd.DataFrame | None = None,
    samples: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    layer: str = "counts",
    data_layer: str = "data",
    n_permutations: int = 199,
    seed: int = 42,
    apply_assay_filters: bool | None = None,
    library_size_mode: str = "covariate",
    test_mode: str = "ql_f",
    trajectory_provenance: str = "external",
) -> pd.DataFrame:
    """Exploratory permutation of covariate-residualized trajectory values.

    This permutes the regressor residuals, not expression residuals. External
    trajectory provenance alone does not establish residual exchangeability,
    independence of repeated samples, or validity of gene-wise multiplicity
    correction. Both provenance modes are exploratory; the historical numerical
    procedure is retained for reproducibility.
    """
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be >= 1")
    if trajectory_provenance not in ("external", "in_sample_learned"):
        raise ValueError("trajectory_provenance must be 'external' or 'in_sample_learned'")

    obs_df = traj_de_test(
        object_or_counts,
        traj_var,
        latent_vars=latent_vars,
        samples=samples,
        features=features,
        layer=layer,
        data_layer=data_layer,
        complete_results=True,
        apply_assay_filters=apply_assay_filters,
        library_size_mode=library_size_mode,
        test_mode=test_mode,
    )
    gene_universe = list(obs_df.index.astype(str))
    obs_f = obs_df["f_statistic"].to_numpy(dtype=np.float64)

    is_adata = hasattr(object_or_counts, "obs") and hasattr(object_or_counts, "var_names")
    if is_adata:
        all_s = [str(s) for s in object_or_counts.obs_names]
        use_s = [str(s) for s in samples] if samples is not None else all_s
        obs_sub = object_or_counts.obs.loc[use_s].copy()
        t_vec = (
            pd.to_numeric(obs_sub[traj_var], errors="coerce").to_numpy(dtype=np.float64)
            if isinstance(traj_var, str)
            else pd.Series(traj_var).loc[use_s].to_numpy(dtype=np.float64)
            if isinstance(traj_var, pd.Series)
            else np.asarray(traj_var, dtype=np.float64).ravel()
        )
        lat_df = (
            None
            if latent_vars is None
            else latent_vars.loc[use_s].copy()
            if isinstance(latent_vars, pd.DataFrame)
            else obs_sub[[str(c) for c in latent_vars]].copy()
        )
        g_map = {str(g): i for i, g in enumerate(object_or_counts.var_names)}
        g_idx = np.array([g_map[g] for g in gene_universe], dtype=np.int64)
        s_map = {s: i for i, s in enumerate(all_s)}
        r_idx = np.array([s_map[s] for s in use_s], dtype=np.int64)
        raw_c = object_or_counts.layers[layer] if layer in object_or_counts.layers else object_or_counts.X
        sub_c = raw_c[r_idx, :][:, g_idx]
        counts_fixed = pd.DataFrame(
            (sub_c.toarray() if sp.issparse(sub_c) else np.asarray(sub_c, dtype=np.float64)).T,
            index=gene_universe,
            columns=use_s,
        )
    else:
        counts_fixed = (
            object_or_counts.loc[gene_universe].copy()
            if isinstance(object_or_counts, pd.DataFrame)
            else pd.DataFrame(np.asarray(object_or_counts, dtype=np.float64), index=gene_universe)
        )
        use_s = list(counts_fixed.columns.astype(str))
        t_vec = (
            traj_var.loc[use_s].to_numpy(dtype=np.float64)
            if isinstance(traj_var, pd.Series)
            else np.asarray(traj_var, dtype=np.float64).ravel()
        )
        lat_df = latent_vars.loc[use_s].copy() if isinstance(latent_vars, pd.DataFrame) else None

    full_des, full_cols = _build_traj_design(pd.Series(t_vec, index=use_s), lat_df)
    nuis_cols = [j for j, c in enumerate(full_cols) if c != "Traj"]
    z_nuis = full_des[:, nuis_cols]
    gamma_hat = la.lstsq(z_nuis, t_vec)[0]
    t_hat = z_nuis @ gamma_hat
    t_resid = t_vec - t_hat

    rng = np.random.default_rng(int(seed))
    b_tot = int(n_permutations)
    exceed = np.zeros(len(gene_universe), dtype=np.int64)
    valid_b = np.zeros(len(gene_universe), dtype=np.int64)

    for _ in range(b_tot):
        t_perm = t_hat + rng.permutation(t_resid)
        perm_de = traj_de_test(
            counts_fixed,
            pd.Series(t_perm, index=use_s),
            latent_vars=lat_df,
            apply_assay_filters=False,
            library_size_mode=library_size_mode,
            test_mode=test_mode,
        ).reindex(gene_universe)
        pf = perm_de["f_statistic"].to_numpy(dtype=np.float64)
        ok = np.isfinite(pf) & np.isfinite(obs_f)
        valid_b[ok] += 1
        exceed[ok] += (pf[ok] >= obs_f[ok] - 1e-15).astype(np.int64)

    emp_p = np.where(valid_b > 0, (1.0 + exceed) / (valid_b + 1.0), np.nan)
    mc_se = np.where(valid_b > 0, np.sqrt(emp_p * (1.0 - emp_p) / (valid_b + 1.0)), np.nan)

    out = obs_df.copy()
    out["emp_p_gene"] = emp_p
    out["emp_p_mc_se"] = mc_se
    out["emp_n_valid_perms"] = valid_b
    out["emp_min_attainable_p"] = np.where(valid_b > 0, 1.0 / (valid_b + 1.0), np.nan)
    out["emp_q_bh"] = benjamini_hochberg(emp_p)
    out["emp_q_by"] = benjamini_yekutieli(emp_p)
    out = out.sort_values(["emp_p_gene", "p_Traj"], kind="mergesort")
    if trajectory_provenance == "in_sample_learned":
        warnings.warn(
            "traj_de_permutation_test was invoked on an in-sample learned trajectory "
            "(trajectory_provenance='in_sample_learned'). Freedman-Lane residual permutation "
            "does not remove post-selection endogeneity when X has shared donor co-expression; "
            "results are tagged exploratory.",
            UserWarning,
            stacklevel=2,
        )
        out.attrs["inference_tier"] = "exploratory_insample_traj_residual_permutation"
    else:
        warnings.warn(
            "External trajectory provenance does not establish exchangeability of residuals "
            "or gene-level FDR control. This regressor-residual permutation is exploratory.",
            UserWarning,
            stacklevel=2,
        )
        out.attrs["inference_tier"] = "exploratory_external_traj_residual_permutation"
    out.attrs["trajectory_provenance"] = trajectory_provenance
    out.attrs["confirmatory_calibration_established"] = False
    out.attrs["null_hypothesis"] = (
        "Intended H0_g: beta_Traj_g == 0 | latent_vars; residual exchangeability and multiplicity validity unverified"
    )
    return out


def run_unified_permutation_inference(
    pipeline_fn: Callable[[dict[str, Any], InferencePipelineSpec, np.random.Generator], pd.DataFrame],
    subject_labels: Mapping[str, Any] | pd.Series,
    *,
    spec: InferencePipelineSpec | None = None,
    n_permutations: int = 49,
    alpha: float = 0.05,
    seed: int = 20261114,
    missing_gene_policy: str = "nan",
    null_aggregation: str = "gene_wise",
) -> dict[str, Any]:
    """Execute observed AND subject-permuted runs through ONE identical ``pipeline_fn`` & ``spec``."""
    if missing_gene_policy not in ("nan", "conservative_one", "error"):
        raise ValueError("missing_gene_policy must be 'nan', 'conservative_one', or 'error'")
    if null_aggregation not in ("gene_wise", "pooled_diagnostic"):
        raise ValueError("null_aggregation must be 'gene_wise' or 'pooled_diagnostic'")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be >= 1")

    spec_use = spec or InferencePipelineSpec()
    spec_hash = spec_use.sha256()
    subj_dict = {str(k): v for k, v in subject_labels.items()}
    subj_keys = sorted(subj_dict.keys())
    if len(subj_keys) < 4:
        raise ValueError("At least 4 distinct subjects are required for subject-level permutation")
    orig_vals = np.array([subj_dict[k] for k in subj_keys], dtype=object)

    master_rng = np.random.default_rng(int(seed))
    obs_seed = int(master_rng.integers(0, 2**31 - 1))
    obs_de = pipeline_fn(dict(subj_dict), spec_use, np.random.default_rng(obs_seed)).copy()
    if "gene_ID" in obs_de.columns:
        obs_de.index = pd.Index(obs_de["gene_ID"].astype(str))
    gene_universe = list(obs_de.index.astype(str))
    validate_bijective_ids(gene_universe, axis_name="test_gene_universe")
    gene_universe_hash = hashlib.sha256("\n".join(gene_universe).encode("utf-8")).hexdigest()
    obs_f = obs_de["f_statistic"].to_numpy(dtype=np.float64)

    b_tot = int(n_permutations)
    perm_f_mat = np.full((b_tot, len(gene_universe)), np.nan, dtype=np.float64)
    finite_perm_mat = np.zeros((b_tot, len(gene_universe)), dtype=bool)
    draw_records: list[dict[str, Any]] = []
    nom_p_rates: list[float] = []
    nom_bh_counts: list[int] = []

    for b in range(b_tot):
        b_seed = int(master_rng.integers(0, 2**31 - 1))
        rng_b = np.random.default_rng(b_seed)
        shuf_vals = rng_b.permutation(orig_vals)
        perm_map = {k: shuf_vals[i] for i, k in enumerate(subj_keys)}
        perm_hash = hashlib.sha256(
            json.dumps([(k, str(perm_map[k])) for k in subj_keys]).encode("utf-8")
        ).hexdigest()

        pde = pipeline_fn(perm_map, spec_use, np.random.default_rng(b_seed + 1)).copy()
        if "gene_ID" in pde.columns:
            pde.index = pd.Index(pde["gene_ID"].astype(str))

        pv_all = pde["p_Traj"].to_numpy(dtype=np.float64)
        qv_all = pde["adj_pval"].to_numpy(dtype=np.float64)
        fin_pv = np.isfinite(pv_all)
        if np.any(fin_pv):
            nom_p_rates.append(float(np.mean(pv_all[fin_pv] <= float(alpha))))
            nom_bh_counts.append(int(np.sum(qv_all[fin_pv] <= float(alpha))))

        aligned = pde.reindex(gene_universe)
        raw_f = aligned["f_statistic"].to_numpy(dtype=np.float64)
        fin_row = np.isfinite(raw_f)
        n_missing = int(np.sum(~fin_row))
        if n_missing > 0 and missing_gene_policy == "error":
            raise ValueError(f"Permutation {b} omitted {n_missing} genes from the declared test_gene_universe")

        finite_perm_mat[b] = fin_row
        f_row = np.where(fin_row, raw_f, np.inf) if missing_gene_policy == "conservative_one" else raw_f
        perm_f_mat[b] = f_row

        draw_records.append(
            {
                "perm_index": b,
                "rng_seed": b_seed,
                "spec_sha256": spec_hash,
                "subject_perm_sha256": perm_hash,
                "n_genes_returned": int(len(pde)),
                "n_genes_missing_from_universe": n_missing,
                "nominal_p_le_alpha_fraction": float(np.mean(pv_all[fin_pv] <= float(alpha))) if np.any(fin_pv) else None,
            }
        )

    # Missing draws deliberately count as exceedances under conservative_one.
    obs_fin = np.isfinite(obs_f)
    valid_null = ~np.isnan(perm_f_mat) if missing_gene_policy == "conservative_one" else np.isfinite(perm_f_mat)
    valid_mask = valid_null & obs_fin[None, :]
    valid_b = np.sum(valid_mask, axis=0)
    finite_b = np.sum(finite_perm_mat & obs_fin[None, :], axis=0)
    missing_b = np.sum(~finite_perm_mat, axis=0)
    exceed = np.sum(valid_mask & (perm_f_mat >= obs_f[None, :] - 1e-15), axis=0)
    emp_p_gw = np.where(valid_b > 0, (1.0 + exceed) / (valid_b + 1.0), np.nan)
    mc_se_gw = np.where(valid_b > 0, np.sqrt(emp_p_gw * (1.0 - emp_p_gw) / (valid_b + 1.0)), np.nan)

    cal_df = obs_de.copy()
    cal_df["emp_p_gene_wise"] = emp_p_gw
    cal_df["emp_p_mc_se"] = mc_se_gw
    cal_df["emp_n_valid_perms"] = valid_b
    cal_df["emp_n_finite_perms"] = finite_b
    cal_df["emp_n_missing_perms"] = missing_b
    cal_df["emp_missing_perm_rate"] = missing_b / float(b_tot)
    cal_df["emp_q_bh_gene_wise"] = benjamini_hochberg(emp_p_gw)
    cal_df["emp_q_by_gene_wise"] = benjamini_yekutieli(emp_p_gw)

    calibration_status = "UNIFIED_GENE_WISE_PHENOTYPE_NULL"
    if null_aggregation == "pooled_diagnostic":
        warnings.warn(
            "null_aggregation='pooled_diagnostic' pools F-statistics across heterogeneous genes "
            "and is NOT calibrated for gene-level FDR; use gene_wise columns for exploratory phenotype-null p-values.",
            UserWarning,
            stacklevel=2,
        )
        calibration_status = "UNVERIFIED_DIAGNOSTIC_POOLED_NULL"
        fin_pool = np.sort(perm_f_mat[finite_perm_mat])
        if fin_pool.size > 0:
            exc_pl = len(fin_pool) - np.searchsorted(fin_pool, obs_f, side="left")
            p_pl = (1.0 + exc_pl) / (len(fin_pool) + 1.0)
            cal_df["emp_p_pooled_diagnostic"] = p_pl
            cal_df["emp_q_pooled_diagnostic"] = benjamini_hochberg(p_pl)

    cal_df.attrs["inference_tier"] = "exploratory_subject_phenotype_permutation"
    cal_df.attrs["confirmatory_calibration_established"] = False
    cal_df.attrs["missing_gene_policy"] = missing_gene_policy

    return {
        "inference_tier": "exploratory_subject_phenotype_permutation",
        "confirmatory_calibration_established": False,
        "missing_gene_policy": missing_gene_policy,
        "spec_sha256": spec_hash,
        "test_gene_universe_sha256": gene_universe_hash,
        "n_subjects": len(subj_keys),
        "n_genes_universe": len(gene_universe),
        "n_permutations": b_tot,
        "min_attainable_gene_wise_p": round(1.0 / (b_tot + 1.0), 8),
        "alpha": float(alpha),
        "calibration_status": calibration_status,
        "null_semantics": (
            "Subject-level phenotype permutation tests H0_pheno: Y _|_ X. Because X retains "
            "intrinsic cell/donor co-expression manifold structure when Y is shuffled, nominal "
            "in-sample TrajDE p<=alpha proportions reflect manifold co-expression, NOT iid gene Type-I error."
        ),
        "mean_nominal_p_le_alpha_under_phenotype_shuffle": float(np.mean(nom_p_rates)) if nom_p_rates else None,
        "max_nominal_p_le_alpha_under_phenotype_shuffle": float(np.max(nom_p_rates)) if nom_p_rates else None,
        "mean_nominal_bh_le_alpha_under_phenotype_shuffle": float(np.mean(nom_bh_counts)) if nom_bh_counts else None,
        "draw_provenance": draw_records,
        "calibrated_de": cal_df,
    }


def run_confirmatory_oof_traj_de(
    adata: Any,
    *,
    sample_col: str,
    response_col: str,
    de_celltype: str,
    celltype_col: str,
    subject_col: str | None = None,
    subject_id_sep: str | None = None,
    primary_draw_only: bool = True,
    count_split_prop: float | None = 0.5,
    n_splits: int = 5,
    ncells_per_group: int = 400,
    ncomp_pls: int = 10,
    pls_function: str = "cppls",
    de_covariates: Sequence[str] = ("Age", "Sex", "log_count"),
    n_phenotype_permutations: int = 49,
    gene_wise_perm_null: bool = False,
    n_jobs: int = 1,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Subject-grouped K-fold out-of-fold (OOF) trajectory DE with explicit calibration scope.

    - When ``gene_wise_perm_null=False`` (default fast mode): computes subject-grouped OOF
      trajectory on ``X_traj`` and parametric NB-QLF ``traj_de_test`` on ``X_de``. Because
      binomial thinning leaves positive marginal covariance ``Cov(X^(1), X^(2)) = p(1-p)Var(Lambda)``
      whenever shared cross-gene donor latent factors exist, parametric ``p_Traj``/``adj_pval``
      showed limited calibration in the archived simulations, and is tagged
      ``'exploratory_subject_oof_countsplit_nb'`` (or ``'exploratory_subject_oof_nb'``) on
      general donor-structured scRNA-seq.
    - When ``gene_wise_perm_null=True``:
      additionally refits the OOF trajectory + DE across ``B`` subject-level phenotype shuffles
      to produce gene-wise aligned empirical p-values (``emp_p_oof_perm``, ``emp_q_bh_oof_perm``)
      and max-F adjusted empirical values (legacy column ``emp_p_maxf_fwer``).
      These are exploratory phenotype-null diagnostics. Folds were stratified on
      observed labels and are held fixed during global label shuffles; this is
      not a fully recomputed label-dependent procedure. Neither finite-sample
      FWER nor gene-wise FDR control is established. The legacy function and
      column names are retained for compatibility, not as inferential guarantees.
    """
    from .neighbors import resolve_effective_workers
    from .supervised import fit_pls

    _, w_info = resolve_effective_workers(n_jobs)
    if gene_wise_perm_null and int(n_phenotype_permutations) < 1:
        raise ValueError("gene_wise_perm_null=True requires n_phenotype_permutations >= 1")
    validate_open_backed_anndata(adata, caller="run_confirmatory_oof_traj_de")
    s_map = resolve_subject_mapping(
        adata.obs,
        sample_col=sample_col,
        subject_col=subject_col,
        subject_id_sep=subject_id_sep,
        strict_invariant_check=False,
    )
    use_samples = s_map.primary_samples if primary_draw_only else list(s_map.table[sample_col].astype(str))
    keep_c = adata.obs[sample_col].astype(str).isin(use_samples).to_numpy()
    sub_ad = adata[keep_c].copy()

    if count_split_prop is not None:
        ad_traj, ad_de = binomial_thin_counts(sub_ad, prop=float(count_split_prop), layer="counts", seed=seed)
    else:
        ad_traj, ad_de = sub_ad, sub_ad

    c_tr_csr = ad_traj.layers["counts"].tocsr().astype(np.float64)
    rs = np.asarray(c_tr_csr.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 10_000.0 / np.maximum(rs, 1.0), 0.0)
    norm_csr = (sp.diags(inv) @ c_tr_csr).tocsr()
    norm_csr.data = np.log1p(norm_csr.data)

    s_table = s_map.table.set_index(sample_col, drop=False).loc[use_samples].copy()
    y_raw = s_table[response_col]
    if pd.api.types.is_numeric_dtype(y_raw) and not isinstance(y_raw.dtype, pd.CategoricalDtype):
        y_num = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=np.float64)
    else:
        cats = list(y_raw.cat.categories) if isinstance(y_raw.dtype, pd.CategoricalDtype) else sorted(pd.unique(y_raw.astype(str)).tolist())
        cmap = {str(c): float(i) for i, c in enumerate(cats)}
        y_num = np.array([cmap[str(v)] for v in y_raw], dtype=np.float64)
    s_table["_y_num"] = y_num
    within_subject_labels = s_table.groupby("subject_id")["_y_num"].nunique(dropna=False)
    if bool((within_subject_labels > 1).any()):
        bad_subjs = sorted(within_subject_labels.index[within_subject_labels > 1].astype(str).tolist())
        raise ValueError(
            f"Subject-grouped OOF / phenotype permutation requires one response value per subject "
            f"(conflicting subjects: {bad_subjs[:5]}); use primary_draw_only=True or define a justified "
            "repeated-visit transformation."
        )

    folds = stratified_subject_kfold(
        s_table, y_col="_y_num", subject_col="subject_id", sample_col=sample_col, n_splits=min(int(n_splits), s_map.n_subjects)
    )
    donor_arr = ad_traj.obs[sample_col].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    sketch_by_s = {
        s: rng.choice(np.flatnonzero(donor_arr == s), size=min(int(ncells_per_group), int(np.sum(donor_arr == s))), replace=False)
        for s in use_samples
    }

    fold_hvg_cols: list[np.ndarray] = []
    for tr_s, _ in folds:
        tr_m = np.isin(donor_arr, tr_s)
        c_f = c_tr_csr[tr_m]
        mc = np.asarray(c_f.mean(axis=0)).ravel()
        sq = c_f.copy()
        sq.data = sq.data ** 2
        vc = np.maximum(0.0, (np.asarray(sq.sum(axis=0)).ravel() - c_f.shape[0] * (mc ** 2)) / max(c_f.shape[0] - 1, 1))
        pos = (mc > 0) & (vc > 0)
        poly = np.poly1d(np.polyfit(np.log10(mc[pos]), np.log10(vc[pos]), 2))
        rat = np.zeros_like(mc)
        rat[pos] = vc[pos] / np.maximum(10.0 ** poly(np.log10(mc[pos])), 1e-12)
        fold_hvg_cols.append(np.argsort(-rat)[: min(2000, int(np.sum(pos)))])

    s_to_pos = {s: i for i, s in enumerate(use_samples)}

    def _oof_for_y(y_vec: np.ndarray) -> np.ndarray:
        s_to_y = {use_samples[i]: float(y_vec[i]) for i in range(len(use_samples))}
        oof = np.zeros(len(use_samples), dtype=np.float64)
        for f_idx, (tr_s, te_s) in enumerate(folds):
            h_cols = fold_hvg_cols[f_idx]
            h_names = [str(ad_traj.var_names[c]) for c in h_cols]
            tr_p = np.concatenate([sketch_by_s[s] for s in tr_s])
            x_tr = norm_csr[tr_p][:, h_cols].toarray().astype(np.float64)
            y_tr_c = np.array([s_to_y[donor_arr[p]] for p in tr_p], dtype=np.float64)
            mdl = fit_pls(
                x_tr, y_tr_c, n_components=min(int(ncomp_pls), x_tr.shape[1] - 1, len(tr_p) - 2),
                scale=True, scale_max=10.0, method=pls_function, feature_names=h_names,
            )
            for s_te in te_s:
                preds = mdl.predict(norm_csr[donor_arr == s_te][:, h_cols], apply_training_scale=True).ravel()
                oof[s_to_pos[s_te]] = float(np.mean(preds))
        return oof

    oof_traj = _oof_for_y(y_num)
    oof_rho = float(spearmanr(oof_traj, y_num).statistic)

    pb_de = aggregate_pseudobulk(ad_de, group_by=(sample_col, celltype_col), layer="counts")
    sub_pb = pb_de[pb_de.obs[celltype_col].astype(str) == str(de_celltype)].copy()
    s_table["oof_trajectory"] = oof_traj
    keep_rows = [i for i, d in enumerate(sub_pb.obs[sample_col].astype(str)) if d in s_table.index]
    sub_pb = sub_pb[keep_rows].copy()
    for col in s_table.columns:
        if col not in sub_pb.obs.columns and not col.startswith("_"):
            sub_pb.obs[col] = s_table.loc[sub_pb.obs[sample_col].astype(str), col].to_numpy()
    sub_pb.obs["oof_trajectory"] = s_table.loc[sub_pb.obs[sample_col].astype(str), "oof_trajectory"].to_numpy()
    avail_cov = [str(c) for c in de_covariates if str(c) in sub_pb.obs.columns]
    valid_s = [
        str(nm) for i, nm in enumerate(sub_pb.obs_names)
        if not sub_pb.obs.iloc[i][["oof_trajectory"] + avail_cov].isna().any()
    ]
    de_df = traj_de_test(
        sub_pb, traj_var="oof_trajectory", latent_vars=avail_cov if avail_cov else None,
        samples=valid_s, complete_results=True,
    )

    gene_universe = list(de_df.index.astype(str))
    obs_f = de_df["f_statistic"].to_numpy(dtype=np.float64)

    # Pre-slice fixed pseudobulk count matrix & covariates on valid_s when gene_wise_perm_null=True
    counts_fixed_df: pd.DataFrame | None = None
    lat_valid_df: pd.DataFrame | None = None
    valid_donors: list[str] = []
    if gene_wise_perm_null:
        s_lookup = {str(nm): i for i, nm in enumerate(sub_pb.obs_names)}
        r_idx = np.array([s_lookup[nm] for nm in valid_s], dtype=np.int64)
        g_lookup = {str(g): i for i, g in enumerate(sub_pb.var_names)}
        c_idx = np.array([g_lookup[g] for g in gene_universe], dtype=np.int64)
        raw_pb_c = sub_pb.layers["counts"] if "counts" in sub_pb.layers else sub_pb.X
        sub_mat = raw_pb_c[r_idx, :][:, c_idx]
        counts_fixed_df = pd.DataFrame(
            (sub_mat.toarray() if sp.issparse(sub_mat) else np.asarray(sub_mat, dtype=np.float64)).T,
            index=gene_universe,
            columns=valid_s,
        )
        lat_valid_df = sub_pb.obs.loc[valid_s, avail_cov].copy() if avail_cov else None
        valid_donors = sub_pb.obs.loc[valid_s, sample_col].astype(str).tolist()

    perm_rhos: list[float] = []
    b_tot = int(n_phenotype_permutations)
    exceed_gw = np.zeros(len(gene_universe), dtype=np.int64)
    valid_b_gw = np.zeros(len(gene_universe), dtype=np.int64)
    exceed_maxf = np.zeros(len(gene_universe), dtype=np.int64)
    valid_b_maxf = 0

    if b_tot > 0:
        uniq_s = s_table[["subject_id", "_y_num"]].drop_duplicates("subject_id")
        u_ids = uniq_s["subject_id"].astype(str).to_numpy()
        u_vals = uniq_s["_y_num"].to_numpy(dtype=np.float64)
        for _ in range(b_tot):
            sh_u = dict(zip(u_ids, rng.permutation(u_vals)))
            y_perm = np.array([sh_u[sid] for sid in s_table["subject_id"].astype(str)], dtype=np.float64)
            t_perm = _oof_for_y(y_perm)
            perm_rhos.append(float(spearmanr(t_perm, y_perm).statistic))
            if gene_wise_perm_null and counts_fixed_df is not None:
                d_to_tp = {use_samples[i]: float(t_perm[i]) for i in range(len(use_samples))}
                tp_series = pd.Series([d_to_tp[d] for d in valid_donors], index=valid_s, dtype=np.float64)
                if float(np.std(tp_series.to_numpy(), ddof=1)) > 0:
                    pde = traj_de_test(
                        counts_fixed_df,
                        tp_series,
                        latent_vars=lat_valid_df,
                        apply_assay_filters=False,
                    ).reindex(gene_universe)
                    pf = pde["f_statistic"].to_numpy(dtype=np.float64)
                    ok = np.isfinite(pf) & np.isfinite(obs_f)
                    valid_b_gw[ok] += 1
                    exceed_gw[ok] += (pf[ok] >= obs_f[ok] - 1e-15).astype(np.int64)
                    fin_pf = pf[np.isfinite(pf)]
                    if fin_pf.size > 0:
                        valid_b_maxf += 1
                        max_f_b = float(np.max(fin_pf))
                        fin_obs = np.isfinite(obs_f)
                        exceed_maxf[fin_obs] += (max_f_b >= obs_f[fin_obs] - 1e-15).astype(np.int64)

    oof_perm_p = (
        (1.0 + sum(abs(r) >= abs(oof_rho) - 1e-15 for r in perm_rhos)) / (len(perm_rhos) + 1.0)
        if perm_rhos
        else None
    )

    if gene_wise_perm_null:
        if 1.0 / (b_tot + 1.0) > 0.05:
            warnings.warn(
                f"n_phenotype_permutations={b_tot} yields minimum attainable empirical p = "
                f"{1.0 / (b_tot + 1.0):.4f} > 0.05; use n_phenotype_permutations >= 199 for alpha=0.05.",
                UserWarning,
                stacklevel=2,
            )
        emp_p_gw = np.where(valid_b_gw > 0, (1.0 + exceed_gw) / (valid_b_gw + 1.0), np.nan)
        mc_se_gw = np.where(valid_b_gw > 0, np.sqrt(emp_p_gw * (1.0 - emp_p_gw) / (valid_b_gw + 1.0)), np.nan)
        emp_p_fwer = (
            np.where(np.isfinite(obs_f), (1.0 + exceed_maxf) / (valid_b_maxf + 1.0), np.nan)
            if valid_b_maxf > 0
            else np.full(len(gene_universe), np.nan)
        )
        de_df = de_df.copy()
        de_df["emp_p_oof_perm"] = emp_p_gw
        de_df["emp_p_oof_mc_se"] = mc_se_gw
        de_df["emp_n_valid_perms"] = valid_b_gw
        de_df["emp_q_bh_oof_perm"] = benjamini_hochberg(emp_p_gw)
        de_df["emp_q_by_oof_perm"] = benjamini_yekutieli(emp_p_gw)
        de_df["emp_p_maxf_fwer"] = emp_p_fwer
        de_df = de_df.sort_values(["emp_p_oof_perm", "p_Traj"], kind="mergesort")
        de_df.attrs["inference_tier"] = "exploratory_subject_oof_phenotype_permutation"
        de_df.attrs["shared_donor_latent_fdr_calibrated"] = False
        de_df.attrs["fold_policy"] = "observed_label_stratified_folds_fixed_during_permutation"
        de_df.attrs["gene_wise_error_control_established"] = False
        de_df.attrs["null_hypothesis"] = (
            "H0_pheno: subject phenotype Y is independent of counts X under subject-level exchangeability"
        )
        scope_note = (
            "Exploratory H0_pheno diagnostic: observed-label stratified folds are fixed during "
            "global subject phenotype shuffles. Exchangeability of the complete procedure and "
            "gene-wise FDR/strong FWER control are not established. Legacy emp_q_bh_oof_perm and "
            "emp_p_maxf_fwer columns must not be interpreted as confirmed gene discoveries. "
            f"Nominal permutation resolution is {1.0 / (b_tot + 1.0):.6f}."
        )
        warnings.warn(scope_note, UserWarning, stacklevel=2)
    else:
        de_df.attrs["inference_tier"] = (
            "exploratory_subject_oof_countsplit_nb" if count_split_prop is not None else "exploratory_subject_oof_nb"
        )
        de_df.attrs["parametric_calibration_scope"] = (
            "conditional_poisson_or_gene_independent_nb_without_shared_donor_latent_factors"
        )
        de_df.attrs["shared_donor_latent_fdr_calibrated"] = False
        scope_note = (
            "Parametric NB-QLF p_Traj/adj_pval on OOF/count-split trajectories showed limited "
            "calibration in archived simulations, NOT under shared cross-gene donor latent "
            "covariance (Cov(X^(1), X^(2)) = p(1-p)Var(Lambda) > 0). Pass gene_wise_perm_null=True "
            "for an additional exploratory phenotype-null diagnostic, not confirmatory gene-level calibration."
        )
        warnings.warn(scope_note, UserWarning, stacklevel=2)

    return {
        "inference_tier": de_df.attrs["inference_tier"],
        "confirmatory_calibration_established": False,
        "shared_donor_latent_fdr_calibrated": bool(de_df.attrs["shared_donor_latent_fdr_calibrated"]),
        "scope_note": scope_note,
        "subject_mapping_sha256": s_map.sha256,
        "n_samples_used": len(use_samples),
        "n_unique_subjects_used": int(s_table["subject_id"].nunique()),
        "primary_draw_only": bool(primary_draw_only),
        "count_split_prop": count_split_prop,
        "gene_wise_perm_null": bool(gene_wise_perm_null),
        "n_phenotype_permutations": b_tot,
        "min_attainable_perm_p": round(1.0 / (b_tot + 1.0), 8) if b_tot > 0 else None,
        "n_splits": len(folds),
        "workers_requested": w_info["workers_requested"],
        "workers_effective": w_info["workers_effective"],
        "available_cpus": w_info["available_cpus"],
        "exceeds_available_cpus": w_info["exceeds_available_cpus"],
        "oof_trajectory": pd.Series(oof_traj, index=use_samples, name="oof_trajectory"),
        "oof_phenotype_spearman_rho": oof_rho,
        "oof_phenotype_perm_p": oof_perm_p,
        "de_results": de_df,
    }


def calibrate_traj_de_donor_permutation(
    pipeline_fn: Callable[[np.random.Generator], pd.DataFrame],
    observed_de: pd.DataFrame,
    *,
    n_permutations: int = 20,
    alpha: float = 0.05,
    seed: int = 20261113,
    null_aggregation: str = "gene_wise",
    missing_gene_policy: str = "nan",
) -> dict[str, Any]:
    """Donor/subject-level phenotype permutation calibration diagnostic."""
    if null_aggregation not in ("gene_wise", "pooled_diagnostic"):
        raise ValueError("null_aggregation must be 'gene_wise' or 'pooled_diagnostic'")
    if missing_gene_policy not in ("nan", "conservative_one", "error"):
        raise ValueError("missing_gene_policy must be 'nan', 'conservative_one', or 'error'")

    rng = np.random.default_rng(seed)
    obs_out = observed_de.copy()
    if "gene_ID" in obs_out.columns:
        obs_out.index = pd.Index(obs_out["gene_ID"].astype(str))
    gene_universe = list(obs_out.index.astype(str))
    obs_f = obs_out["f_statistic"].to_numpy(dtype=np.float64) if "f_statistic" in obs_out.columns else None

    null_nom_rates: list[float] = []
    null_bh_discoveries: list[int] = []
    b_tot = int(n_permutations)
    perm_f_mat = np.full((b_tot, len(gene_universe)), np.nan, dtype=np.float64)
    finite_perm_mat = np.zeros((b_tot, len(gene_universe)), dtype=bool)

    for b in range(b_tot):
        perm_df = pipeline_fn(rng).copy()
        if "gene_ID" in perm_df.columns:
            perm_df.index = pd.Index(perm_df["gene_ID"].astype(str))
        pv = perm_df["p_Traj"].to_numpy(dtype=np.float64)
        qv = perm_df["adj_pval"].to_numpy(dtype=np.float64)
        valid = np.isfinite(pv)
        if np.any(valid):
            null_nom_rates.append(float(np.mean(pv[valid] <= alpha)))
            null_bh_discoveries.append(int(np.sum(qv[valid] <= alpha)))

        if obs_f is not None and "f_statistic" in perm_df.columns:
            aligned = perm_df.reindex(gene_universe)
            raw_f = aligned["f_statistic"].to_numpy(dtype=np.float64)
            fin_r = np.isfinite(raw_f)
            n_miss = int(np.sum(~fin_r))
            if n_miss > 0 and missing_gene_policy == "error":
                raise ValueError(f"Permutation {b} omitted {n_miss} genes from observed_de universe")
            finite_perm_mat[b] = fin_r
            frow = np.where(fin_r, raw_f, np.inf) if missing_gene_policy == "conservative_one" else raw_f
            perm_f_mat[b] = frow

    if obs_f is not None and b_tot > 0:
        obs_fin = np.isfinite(obs_f)
        valid_null = ~np.isnan(perm_f_mat) if missing_gene_policy == "conservative_one" else np.isfinite(perm_f_mat)
        vmask = valid_null & obs_fin[None, :]
        vb = np.sum(vmask, axis=0)
        fin_b = np.sum(finite_perm_mat & obs_fin[None, :], axis=0)
        miss_b = np.sum(~finite_perm_mat, axis=0)
        obs_out["emp_n_valid_perms"] = vb
        obs_out["emp_n_finite_perms"] = fin_b
        obs_out["emp_n_missing_perms"] = miss_b
        obs_out["emp_missing_perm_rate"] = miss_b / float(b_tot)
        if null_aggregation == "gene_wise":
            exc = np.sum(vmask & (perm_f_mat >= obs_f[None, :] - 1e-15), axis=0)
            emp_p = np.where(vb > 0, (1.0 + exc) / (vb + 1.0), np.nan)
            obs_out["emp_p_donor_perm"] = emp_p
            obs_out["emp_p_mc_se"] = np.where(vb > 0, np.sqrt(emp_p * (1.0 - emp_p) / (vb + 1.0)), np.nan)
            obs_out["emp_q_donor_perm"] = benjamini_hochberg(emp_p)
        else:
            warnings.warn(
                "null_aggregation='pooled_diagnostic' pools F-statistics across heterogeneous genes "
                "and is an uncalibrated diagnostic; use null_aggregation='gene_wise' for gene-aligned "
                "exploratory phenotype-null p-values (not general confirmatory FDR control).",
                UserWarning,
                stacklevel=2,
            )
            pooled_f = np.sort(perm_f_mat[finite_perm_mat])
            if pooled_f.size > 0:
                exceed = len(pooled_f) - np.searchsorted(pooled_f, obs_f, side="left")
                emp_p = (exceed + 1.0) / (len(pooled_f) + 1.0)
                obs_out["emp_p_donor_perm"] = emp_p
                obs_out["emp_q_donor_perm"] = benjamini_hochberg(emp_p)

    obs_out.attrs["inference_tier"] = "exploratory_subject_phenotype_permutation"
    obs_out.attrs["confirmatory_calibration_established"] = False
    obs_out.attrs["missing_gene_policy"] = missing_gene_policy

    mean_nom = float(np.mean(null_nom_rates)) if null_nom_rates else None
    max_nom = float(np.max(null_nom_rates)) if null_nom_rates else None
    return {
        "inference_tier": "exploratory_subject_phenotype_permutation",
        "confirmatory_calibration_established": False,
        "missing_gene_policy": missing_gene_policy,
        "n_permutations": b_tot,
        "alpha": float(alpha),
        "null_aggregation": null_aggregation,
        "calibration_status": (
            "GENE_WISE_ALIGNED_PHENOTYPE_NULL"
            if null_aggregation == "gene_wise"
            else "UNVERIFIED_DIAGNOSTIC_POOLED_NULL"
        ),
        "min_attainable_gene_wise_p": round(1.0 / (b_tot + 1.0), 8) if b_tot > 0 else None,
        "null_semantics_note": (
            "Permuting phenotype Y preserves intrinsic X co-expression structure; nominal p<=alpha "
            "fractions under Y shuffle are NOT iid gene-level Type-I error."
        ),
        "mean_nominal_p_le_alpha_under_phenotype_shuffle": mean_nom,
        "max_nominal_p_le_alpha_under_phenotype_shuffle": max_nom,
        "mean_bh_discoveries_under_phenotype_shuffle": float(np.mean(null_bh_discoveries)) if null_bh_discoveries else None,
        "mean_nominal_type1_error": mean_nom,
        "max_nominal_type1_error": max_nom,
        "mean_bh_false_discoveries": float(np.mean(null_bh_discoveries)) if null_bh_discoveries else None,
        "null_type1_rates": null_nom_rates,
        "null_bh_counts": null_bh_discoveries,
        "calibrated_de": obs_out,
    }
