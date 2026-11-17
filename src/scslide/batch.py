"""Control-only sample-level ComBat and CellANOVA batch-effect correction (F02, F17).

Exact native Python translations of ``CorrectSampleComBat`` and ``cellanova_calc_BE``
from ``LCGaoZzz/scSLIDE@968dbde299458100346c792aaf60f620c69f41fd`` (``R/batch_correction.R``),
with explicit safety policies for missing-control batches, rank-deficient / underdetermined
designs (B14), and memory-bounded lazy low-rank residual operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from scipy.special import gammaln

from ._validation import validate_bijective_ids, validate_numeric_matrix_values
from .sample import SampleLevelObject


# ---------------------------------------------------------------------------
# F02: CorrectSampleComBat (control-only Empirical Bayes + regression/subtraction)
# ---------------------------------------------------------------------------

def _combat_aprior(delta_hat: np.ndarray) -> float:
    m = float(np.mean(delta_hat))
    s2 = float(np.var(delta_hat, ddof=1)) if len(delta_hat) >= 2 else 0.0
    if s2 < np.finfo(np.float64).eps:
        return 1002.0
    return (2.0 * s2 + m * m) / s2


def _combat_bprior(delta_hat: np.ndarray) -> float:
    m = float(np.mean(delta_hat))
    s2 = float(np.var(delta_hat, ddof=1)) if len(delta_hat) >= 2 else 0.0
    if s2 < np.finfo(np.float64).eps:
        return 1001.0 * m
    return (m * s2 + (m ** 3)) / s2


def _combat_it_sol(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    *,
    conv: float = 1e-4,
    max_iter: int = 5000,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = float(sdat.shape[1])
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    change = 1.0
    count = 0
    while change > conv and count < max_iter:
        denom = t2 * n + d_old
        g_new = np.where(denom > 0, (t2 * n * g_hat + d_old * g_bar) / np.maximum(denom, 1e-300), g_hat)
        sum2 = np.sum((sdat - g_new[:, None]) ** 2, axis=1)
        d_new = (0.5 * sum2 + b) / (0.5 * n + a - 1.0)
        rel_g = np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-10))
        rel_d = np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-10))
        change = float(max(rel_g, rel_d))
        g_old = g_new
        d_old = d_new
        count += 1
    return g_old, d_old, count


def _combat_int_eprior(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized non-parametric empirical prior integration matching ``.combat_int_eprior``."""
    g_len, n_int = sdat.shape
    n = float(n_int)
    g_star = np.empty(g_len, dtype=np.float64)
    d_star = np.empty(g_len, dtype=np.float64)

    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    shape = 0.5 * n

    for i in range(g_len):
        sd_i = np.sqrt(max(d_hat[i] / n, 1e-300))
        z = (g_hat[i] - g_hat) / sd_i
        g_dens = (inv_sqrt_2pi / sd_i) * np.exp(-0.5 * z * z)
        s_gd = float(g_dens.sum())
        g_star[i] = float(np.dot(g_hat, g_dens) / s_gd) if s_gd > 0 else float(g_hat[i])

        diff = sdat[i, None, :] - g_hat[:, None]
        sum2_i = np.sum(diff * diff, axis=1)
        rate = 0.5 * sum2_i
        di = max(float(d_hat[i]), 1e-300)
        inv_di = 1.0 / di
        # dgamma(1 / d_hat[i], shape = n/2, rate = sum2_i/2) / d_hat[i]^2
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_dens = np.where(
                rate > 0,
                shape * np.log(np.maximum(rate, 1e-300))
                - gammaln(shape)
                + (shape - 1.0) * np.log(inv_di)
                - rate * inv_di
                - 2.0 * np.log(di),
                -np.inf,
            )
            max_ld = float(np.max(log_dens))
            if np.isfinite(max_ld):
                w = np.exp(log_dens - max_ld)
                sw = float(w.sum())
                d_star[i] = float(np.dot(d_hat, w) / sw) if sw > 0 else di
            else:
                d_star[i] = di
    return g_star, d_star


@dataclass
class SampleComBatResult:
    """Output and full diagnostic intermediates of :func:`correct_sample_combat`."""

    corrected: pd.DataFrame | SampleLevelObject | Any
    z_batch: pd.DataFrame
    z_global: pd.Series
    z_star: pd.DataFrame
    sigma2: pd.Series
    gamma_star: pd.DataFrame
    delta_star: pd.DataFrame
    beta: pd.Series
    eb_summary: pd.DataFrame
    attenuation_summary: pd.DataFrame | None = None
    uncontrolled_batches: list[str] = field(default_factory=list)


def correct_sample_combat(
    object_or_matrix: Any,
    *,
    batch_key: str | Sequence[Any],
    condition_key: str | Sequence[Any],
    control_label: str | Sequence[Any],
    metadata: pd.DataFrame | None = None,
    layer: str = "residuals",
    save_layer: str | None = None,
    par_prior: bool = True,
    method: str = "regression",
    correct_attenuation: bool = False,
    floor_beta: float | None = None,
    missing_control_policy: str = "global_mean_warn",
    return_diagnostics: bool = False,
) -> Any | SampleComBatResult:
    """Control-only sample-level ComBat batch correction matching ``CorrectSampleComBat`` (F02).

    Parameters
    ----------
    object_or_matrix
        A ``SampleLevelObject``, sample-level ``AnnData`` (``obs=sample, var=landmark``),
        or ``features x samples`` ``pd.DataFrame`` / ``np.ndarray``.
    missing_control_policy
        ``'global_mean_warn'`` (upstream legacy behavior: falls back to ``Z_global`` for
        batches with 0 controls and emits ``UserWarning``) or ``'error'`` (safety policy:
        raises ``ValueError`` when any batch lacks control samples).
    """
    if method not in ("regression", "subtraction"):
        raise ValueError("method must be 'regression' or 'subtraction'")
    if missing_control_policy not in ("global_mean_warn", "error"):
        raise ValueError("missing_control_policy must be 'global_mean_warn' or 'error'")
    if floor_beta is not None and (not isinstance(floor_beta, (int, float, np.number)) or not np.isfinite(floor_beta)):
        raise ValueError("floor_beta must be None or a single finite numeric value")

    if method == "subtraction":
        if correct_attenuation:
            warnings.warn("'correct_attenuation' is ignored when method='subtraction'", UserWarning, stacklevel=2)
        if floor_beta is not None and float(floor_beta) != 0.0:
            warnings.warn("'floor_beta' is ignored when method='subtraction'", UserWarning, stacklevel=2)

    is_sample_obj = isinstance(object_or_matrix, SampleLevelObject)
    is_adata = hasattr(object_or_matrix, "obs") and hasattr(object_or_matrix, "var_names") and hasattr(object_or_matrix, "layers")

    if is_sample_obj:
        s_obj: SampleLevelObject = object_or_matrix
        if layer == "residuals" and s_obj.residuals is not None:
            mat_df = s_obj.residuals.copy()
        elif layer in ("normalized", "data") and s_obj.normalized is not None:
            mat_df = s_obj.normalized.copy()
        elif layer == "counts":
            mat_df = s_obj.counts.astype(np.float64)
        elif s_obj.residuals is not None:
            mat_df = s_obj.residuals.copy()
        elif s_obj.normalized is not None:
            mat_df = s_obj.normalized.copy()
        else:
            raise ValueError(f"Requested layer {layer!r} is not populated on SampleLevelObject")
        meta_df = metadata if metadata is not None else s_obj.metadata
    elif is_adata:
        adata = object_or_matrix
        src = adata.layers[layer] if layer in adata.layers else adata.X
        arr_s_by_g = src.toarray() if sp.issparse(src) else np.asarray(src[:], dtype=np.float64)
        mat_df = pd.DataFrame(arr_s_by_g.T, index=adata.var_names.astype(str), columns=adata.obs_names.astype(str))
        meta_df = metadata if metadata is not None else adata.obs.copy()
    elif isinstance(object_or_matrix, pd.DataFrame):
        mat_df = object_or_matrix.astype(np.float64).copy()
        meta_df = metadata
    else:
        arr = np.asarray(object_or_matrix, dtype=np.float64)
        mat_df = pd.DataFrame(
            arr,
            index=[f"feature_{i}" for i in range(arr.shape[0])],
            columns=[f"sample_{j}" for j in range(arr.shape[1])],
        )
        meta_df = metadata

    validate_bijective_ids(mat_df.index, axis_name="feature axis")
    validate_bijective_ids(mat_df.columns, axis_name="sample axis")
    validate_numeric_matrix_values(mat_df.to_numpy(), name="data_mat")

    all_samples = list(mat_df.columns)
    data_mat = mat_df.to_numpy(dtype=np.float64).copy()
    n_genes, n_samples = data_mat.shape
    if n_genes < 2:
        raise ValueError("CorrectSampleComBat requires at least 2 features/landmarks for across-feature EB priors")

    if isinstance(batch_key, str):
        if meta_df is None or batch_key not in meta_df.columns:
            raise ValueError(f"Column {batch_key!r} not found in sample metadata")
        batches = meta_df.loc[all_samples, batch_key].astype(str).to_numpy()
    else:
        batches = np.asarray([str(b) for b in batch_key], dtype=object)
        if len(batches) != n_samples:
            raise ValueError("batch_key sequence length must match number of samples")

    if isinstance(condition_key, str):
        if meta_df is None or condition_key not in meta_df.columns:
            raise ValueError(f"Column {condition_key!r} not found in sample metadata")
        conditions = meta_df.loc[all_samples, condition_key].to_numpy()
    else:
        conditions = np.asarray(list(condition_key), dtype=object)
        if len(conditions) != n_samples:
            raise ValueError("condition_key sequence length must match number of samples")

    ctrl_labels = {control_label} if isinstance(control_label, (str, int, float)) else set(control_label)
    ctrl_str = {str(x) for x in ctrl_labels}
    is_control = np.array([c in ctrl_labels or str(c) in ctrl_str for c in conditions], dtype=bool)
    global_ctrl_pos = np.flatnonzero(is_control)
    if global_ctrl_pos.size == 0:
        raise ValueError(f"No control samples found matching control_label={control_label!r}")

    unique_batches: list[str] = []
    seen_b: set[str] = set()
    for b in batches:
        bs = str(b)
        if bs not in seen_b:
            seen_b.add(bs)
            unique_batches.append(bs)

    ctrl_pos_by_batch = {b: np.flatnonzero((batches == b) & is_control) for b in unique_batches}
    n_ctrl = {b: int(len(ctrl_pos_by_batch[b])) for b in unique_batches}
    uncontrolled = [b for b in unique_batches if n_ctrl[b] == 0]
    if uncontrolled and missing_control_policy == "error":
        raise ValueError(f"Batches without control samples are disallowed under missing_control_policy='error': {uncontrolled}")

    z_batch = np.full((n_genes, len(unique_batches)), np.nan, dtype=np.float64)
    for j, b in enumerate(unique_batches):
        pos = ctrl_pos_by_batch[b]
        if len(pos) >= 1:
            z_batch[:, j] = data_mat[:, pos].mean(axis=1)

    z_global = data_mat[:, global_ctrl_pos].mean(axis=1)

    batches_sigma = [b for b in unique_batches if n_ctrl[b] >= 2]
    df_within = sum(n_ctrl[b] - 1 for b in batches_sigma)
    if df_within >= 1:
        ss_within = np.zeros(n_genes, dtype=np.float64)
        for b in batches_sigma:
            j = unique_batches.index(b)
            resid = data_mat[:, ctrl_pos_by_batch[b]] - z_batch[:, j : j + 1]
            ss_within += np.sum(resid * resid, axis=1)
        sigma2 = ss_within / float(df_within)
    else:
        warnings.warn(
            "df_within < 1: cannot estimate pooled within-batch variance; falling back to row variance across all controls",
            UserWarning,
            stacklevel=2,
        )
        sigma2 = np.var(data_mat[:, global_ctrl_pos], axis=1, ddof=1) if global_ctrl_pos.size >= 2 else np.ones(n_genes)

    sigma2_safe = sigma2.copy()
    pos_s2 = sigma2_safe > 0
    if not np.any(pos_s2):
        sigma2_safe[:] = 1.0
    elif not np.all(pos_s2):
        sigma2_safe[~pos_s2] = float(np.median(sigma2_safe[pos_s2]))

    batches_obs = [b for b in unique_batches if n_ctrl[b] >= 1]
    if len(batches_obs) < 2:
        raise ValueError(
            f"Need at least 2 batches with control samples for Empirical Bayes shrinkage, found {len(batches_obs)}"
        )

    b_obs = len(batches_obs)
    gamma_hat = np.empty((b_obs, n_genes), dtype=np.float64)
    delta_hat = np.ones((b_obs, n_genes), dtype=np.float64)
    for i, b in enumerate(batches_obs):
        j = unique_batches.index(b)
        nb = n_ctrl[b]
        gamma_hat[i, :] = (z_batch[:, j] - z_global) / np.sqrt(sigma2_safe / float(nb))
        if nb >= 2:
            s_ctrl = (data_mat[:, ctrl_pos_by_batch[b]] - z_global[:, None]) / np.sqrt(sigma2_safe)[:, None]
            delta_hat[i, :] = np.var(s_ctrl, axis=1, ddof=1)

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    a_prior = np.array([_combat_aprior(delta_hat[i]) for i in range(b_obs)], dtype=np.float64)
    b_prior = np.array([_combat_bprior(delta_hat[i]) for i in range(b_obs)], dtype=np.float64)

    gamma_star = np.empty((b_obs, n_genes), dtype=np.float64)
    delta_star = np.empty((b_obs, n_genes), dtype=np.float64)
    for i, b in enumerate(batches_obs):
        s_ctrl = (data_mat[:, ctrl_pos_by_batch[b]] - z_global[:, None]) / np.sqrt(sigma2_safe)[:, None]
        if par_prior:
            gs, ds, _ = _combat_it_sol(
                s_ctrl, gamma_hat[i], delta_hat[i], float(gamma_bar[i]), float(t2[i]), float(a_prior[i]), float(b_prior[i])
            )
        else:
            gs, ds = _combat_int_eprior(s_ctrl, gamma_hat[i], delta_hat[i])
        gamma_star[i, :] = gs
        delta_star[i, :] = ds

    z_star = np.empty((n_genes, len(unique_batches)), dtype=np.float64)
    obs_lookup = {b: i for i, b in enumerate(batches_obs)}
    for j, b in enumerate(unique_batches):
        if b in obs_lookup:
            i = obs_lookup[b]
            z_star[:, j] = z_global + gamma_star[i, :] * np.sqrt(sigma2_safe / float(n_ctrl[b]))
        else:
            z_star[:, j] = z_global
            warnings.warn(
                f"Batch {b!r} has no control samples; falling back to global control mean",
                UserWarning,
                stacklevel=2,
            )

    beta_vec = np.ones(n_samples, dtype=np.float64)
    atten_rows: list[dict[str, Any]] = []

    if method == "regression":
        beta_vec[:] = np.nan
        for j, b in enumerate(unique_batches):
            s_pos = np.flatnonzero(batches == b)
            if s_pos.size == 0:
                continue
            zb = z_star[:, j]
            z_cent = zb - float(zb.mean())
            raw_denom = float(np.dot(z_cent, z_cent))
            if correct_attenuation and np.any(sigma2 > 0):
                me_var = float(np.sum(sigma2_safe / max(n_ctrl[b], 1)))
                corr_denom = raw_denom - me_var
                clamped = bool(corr_denom < 0.5 * raw_denom)
                denom = max(corr_denom, 0.5 * raw_denom)
            else:
                me_var = 0.0
                clamped = False
                denom = raw_denom

            atten_rows.append(
                {
                    "batch": b,
                    "total_var": raw_denom,
                    "noise_var": me_var,
                    "signal_var": denom,
                    "clamped": clamped,
                }
            )
            if denom < np.finfo(np.float64).eps:
                beta_vec[s_pos] = 0.0
            else:
                x_sub = data_mat[:, s_pos]
                x_cent = x_sub - x_sub.mean(axis=0, keepdims=True)
                beta_vec[s_pos] = (z_cent @ x_cent) / denom

        if floor_beta is not None:
            fb = float(floor_beta)
            beta_vec = np.maximum(beta_vec, fb)

        for j, b in enumerate(unique_batches):
            s_pos = np.flatnonzero(batches == b)
            if s_pos.size > 0:
                data_mat[:, s_pos] = data_mat[:, s_pos] - z_star[:, j : j + 1] * beta_vec[None, s_pos]
    else:
        for j, b in enumerate(unique_batches):
            s_pos = np.flatnonzero(batches == b)
            if s_pos.size > 0:
                data_mat[:, s_pos] = data_mat[:, s_pos] - z_star[:, j : j + 1]

    corrected_df = pd.DataFrame(data_mat, index=mat_df.index, columns=mat_df.columns)
    eb_df = pd.DataFrame(
        {
            "batch": batches_obs,
            "n_ctrl": [n_ctrl[b] for b in batches_obs],
            "gamma_bar": gamma_bar,
            "t2": t2,
            "a_prior": a_prior,
            "b_prior": b_prior,
        }
    )
    atten_df = pd.DataFrame(atten_rows) if atten_rows else None

    target_layer = save_layer or layer
    if is_sample_obj:
        new_attrs = dict(s_obj.attrs)
        new_attrs["combat_corrected_layer"] = target_layer
        new_attrs["combat_method"] = method
        out_obj = SampleLevelObject(
            counts=s_obj.counts.copy(),
            normalized=corrected_df if target_layer in ("normalized", "data") else (
                s_obj.normalized.copy() if s_obj.normalized is not None else None
            ),
            residuals=corrected_df if target_layer == "residuals" else (
                s_obj.residuals.copy() if s_obj.residuals is not None else None
            ),
            metadata=s_obj.metadata.copy() if s_obj.metadata is not None else None,
            embeddings={k: v.copy() for k, v in s_obj.embeddings.items()},
            attrs=new_attrs,
        )
        primary = out_obj
    elif is_adata:
        adata.layers[target_layer] = corrected_df.T.to_numpy(dtype=np.float64)
        sc_uns = dict(adata.uns.get("scslide", {}))
        sc_uns["combat"] = {
            "source_layer": layer,
            "target_layer": target_layer,
            "method": method,
            "par_prior": bool(par_prior),
            "correct_attenuation": bool(correct_attenuation),
            "floor_beta": floor_beta,
            "uncontrolled_batches": uncontrolled,
        }
        adata.uns["scslide"] = sc_uns
        primary = adata
    else:
        primary = corrected_df

    if not return_diagnostics:
        return primary

    return SampleComBatResult(
        corrected=primary,
        z_batch=pd.DataFrame(z_batch, index=mat_df.index, columns=unique_batches),
        z_global=pd.Series(z_global, index=mat_df.index),
        z_star=pd.DataFrame(z_star, index=mat_df.index, columns=unique_batches),
        sigma2=pd.Series(sigma2, index=mat_df.index),
        gamma_star=pd.DataFrame(gamma_star, index=batches_obs, columns=mat_df.index),
        delta_star=pd.DataFrame(delta_star, index=batches_obs, columns=mat_df.index),
        beta=pd.Series(beta_vec, index=mat_df.columns),
        eb_summary=eb_df,
        attenuation_summary=atten_df,
        uncontrolled_batches=uncontrolled,
    )


# ---------------------------------------------------------------------------
# F17: cellanova_calc_BE (control-pool batch subspace + lazy low-rank operator)
# ---------------------------------------------------------------------------

class LazyBatchCorrectedOperator(LinearOperator):
    """Memory-bounded lazy representation of ``X - U @ V^T`` (cells x genes orientation).

    Corresponds to BPCells ``TransformLinearResidual`` in ``cellanova_calc_BE``:
    ``X_corr = X - C_cells @ V_genes^T`` where ``C_cells = (X @ V_genes) - L @ (M_overall^T @ V_genes)``
    has shape ``(n_cells, k)`` and ``V_genes`` (``VV1T``) has shape ``(n_genes, k)``.
    """

    def __init__(self, base_cells_by_genes: Any, cell_factors: np.ndarray, gene_basis: np.ndarray) -> None:
        self.base = base_cells_by_genes
        self.cell_factors = np.asarray(cell_factors, dtype=np.float64)   # (n_cells, k)
        self.gene_basis = np.asarray(gene_basis, dtype=np.float64)       # (n_genes, k)
        super().__init__(dtype=np.float64, shape=base_cells_by_genes.shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).ravel()
        base_prod = np.asarray(self.base @ v, dtype=np.float64).ravel()
        corr = self.cell_factors @ (self.gene_basis.T @ v)
        return base_prod - corr

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        u = np.asarray(y, dtype=np.float64).ravel()
        base_prod = np.asarray(self.base.T @ u, dtype=np.float64).ravel()
        corr = self.gene_basis @ (self.cell_factors.T @ u)
        return base_prod - corr

    def to_dense(self) -> np.ndarray:
        b = self.base.toarray() if sp.issparse(self.base) else np.asarray(self.base[:], dtype=np.float64)
        return b - (self.cell_factors @ self.gene_basis.T)


@dataclass
class CellAnovaResult:
    """Output and SVD subspace diagnostics of :func:`cellanova_calc_be`."""

    corrected: np.ndarray | pd.DataFrame | LazyBatchCorrectedOperator | Any
    m_overall: np.ndarray          # (n_genes, n_pcs)
    singular_values: np.ndarray    # DD1 (k,)
    batch_basis: np.ndarray        # VV1T (n_genes, k)
    k_selected: int
    explained_variance_ratio: float


def cellanova_calc_be(
    adata_or_matrix: Any,
    *,
    integrate_key: str | Sequence[Any],
    control_dict: Sequence[str] | Mapping[str, Sequence[str]],
    reduction: str | np.ndarray,
    features: Sequence[str] | None = None,
    layer: str | None = "scale.data",
    var_cutoff: float = 0.9,
    k_max: int = 1500,
    k_select: int | None = None,
    new_layer_name: str = "cellanova_corrected",
    lazy_ondisk: bool = False,
    store_diagnostics: bool = True,
) -> Any | CellAnovaResult:
    """Native Python implementation of ``cellanova_calc_BE`` (F17, ``R/batch_correction.R``).

    Operates on ``cells x genes`` (AnnData/DataFrame/ndarray/CSR) or returns a
    :class:`LazyBatchCorrectedOperator` when ``lazy_ondisk=True`` so large sparse
    or backed matrices never materialize a dense ``N x G`` matrix.
    """
    if not (0.0 < float(var_cutoff) <= 1.0):
        raise ValueError("var_cutoff must satisfy 0 < var_cutoff <= 1")
    if int(k_max) <= 0:
        raise ValueError("k_max must be a positive integer")

    is_adata = hasattr(adata_or_matrix, "obs") and hasattr(adata_or_matrix, "var_names")
    if is_adata:
        adata = adata_or_matrix
        validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
        validate_bijective_ids(adata.var_names, axis_name="adata.var_names")
        var_lookup = {str(v): i for i, v in enumerate(adata.var_names)}
        if features is not None:
            feat_names = [str(f) for f in features if str(f) in var_lookup]
        elif "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
            feat_names = [str(v) for v, hv in zip(adata.var_names, adata.var["highly_variable"]) if hv]
        else:
            feat_names = [str(v) for v in adata.var_names]
        if len(feat_names) == 0:
            raise ValueError("No valid features found for cellanova_calc_be")
        col_idx = np.array([var_lookup[f] for f in feat_names], dtype=np.int64)
        src = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
        x_mat = src[:, col_idx]

        if isinstance(integrate_key, str):
            if integrate_key not in adata.obs.columns:
                raise ValueError(f"Column {integrate_key!r} not found in adata.obs")
            batches = adata.obs[integrate_key].astype(str).to_numpy()
        else:
            batches = np.asarray([str(b) for b in integrate_key], dtype=object)

        if isinstance(reduction, str):
            obsm_key = reduction if reduction in adata.obsm else f"X_{reduction}"
            if obsm_key not in adata.obsm:
                raise ValueError(f"Reduction {reduction!r} not found in adata.obsm")
            ll = np.asarray(adata.obsm[obsm_key][:], dtype=np.float64)
        else:
            ll = np.asarray(reduction, dtype=np.float64)
    else:
        if isinstance(adata_or_matrix, pd.DataFrame):
            feat_names = [str(c) for c in adata_or_matrix.columns]
            x_mat = adata_or_matrix.to_numpy(dtype=np.float64)
        elif sp.issparse(adata_or_matrix):
            feat_names = [f"gene_{j}" for j in range(adata_or_matrix.shape[1])]
            x_mat = adata_or_matrix.tocsr()
        else:
            x_mat = np.asarray(adata_or_matrix, dtype=np.float64)
            feat_names = [f"gene_{j}" for j in range(x_mat.shape[1])]
        batches = np.asarray([str(b) for b in integrate_key], dtype=object)
        ll = np.asarray(reduction, dtype=np.float64)

    n_cells, n_genes = x_mat.shape
    if len(batches) != n_cells:
        raise ValueError("integrate_key length must match number of cells")
    if ll.ndim != 2 or ll.shape[0] != n_cells:
        raise ValueError("reduction embedding rows must match number of cells")
    validate_numeric_matrix_values(ll, name="reduction embedding")

    # Drop all-zero embedding columns matching R colSums(abs(LL) > eps) == 0
    eps = np.finfo(np.float64).eps
    keep_ll = np.sum(np.abs(ll) > eps, axis=0) > 0
    ll = ll[:, keep_ll]
    n_pcs = ll.shape[1]
    if n_pcs == 0:
        raise ValueError("All columns in reduction embedding are zero")

    if isinstance(control_dict, Mapping):
        ctrl_groups = {str(k): [str(x) for x in v] for k, v in control_dict.items()}
    else:
        ctrl_groups = {"g1": [str(x) for x in control_dict]}

    batch_list: list[str] = []
    seen_b: set[str] = set()
    for b in batches:
        bs = str(b)
        if bs not in seen_b:
            seen_b.add(bs)
            batch_list.append(bs)

    missing_ctrl = sorted({b for grp in ctrl_groups.values() for b in grp if b not in seen_b})
    if missing_ctrl:
        raise ValueError(f"Control batches not found in integrate_key: {missing_ctrl}")

    cells_by_batch = {b: np.flatnonzero(batches == b) for b in batch_list}
    # Check underdetermined or rank-deficient batch embeddings (B14)
    small_batches = [f"{b} ({len(cells_by_batch[b])} cells)" for b in batch_list if len(cells_by_batch[b]) < n_pcs]
    if small_batches:
        raise ValueError(
            f"Batches in integrate_key have fewer cells than embedding dimensions ({n_pcs}), "
            f"producing underdetermined OLS: {small_batches}"
        )

    # Regression 1: per-batch OLS of GEX on LL -> coef shape (n_genes, n_pcs)
    overall_coef: dict[str, np.ndarray] = {}
    for b in batch_list:
        pos = cells_by_batch[b]
        ll_b = ll[pos, :]
        # Check numerical rank matching R qr() check against NA coefficients
        q_mat, r_mat = la.qr(ll_b, mode="economic")
        diag_r = np.abs(np.diag(r_mat))
        tol_r = 1e-7 * (diag_r[0] if diag_r.size > 0 else 1.0)
        if diag_r.size < n_pcs or np.any(diag_r <= tol_r):
            raise ValueError(
                f"Batch {b!r} embedding matrix is rank-deficient (rank < {n_pcs}); "
                "cannot identify batch regression coefficients without NAs"
            )
        sub_x = x_mat[pos, :]
        if sp.issparse(sub_x):
            qt_x = np.asarray((q_mat.T @ sub_x), dtype=np.float64)
        else:
            qt_x = q_mat.T @ np.asarray(sub_x, dtype=np.float64)
        coef_pcs_by_genes = la.solve_triangular(r_mat, qt_x, lower=False)
        overall_coef[b] = coef_pcs_by_genes.T  # (n_genes, n_pcs)

    m_overall = sum(overall_coef.values()) / float(len(overall_coef))

    # Regression 2: center batch coefficients within each control pool
    res_blocks: list[np.ndarray] = []
    for _grp_name, ctrl_batches in ctrl_groups.items():
        if len(ctrl_batches) == 0:
            continue
        m_grp = sum(overall_coef[d] for d in ctrl_batches) / float(len(ctrl_batches))
        centered_cols = [overall_coef[d] - m_grp for d in ctrl_batches]
        res_blocks.append(np.hstack(centered_cols))  # (n_genes, n_pcs * len(ctrl_batches))

    if not res_blocks:
        raise ValueError("control_dict contains no control batches")

    # res_combined in R is t(do.call(cbind, res)) -> shape (total_ctrl_cols, n_genes)
    res_combined = np.hstack(res_blocks).T
    total_var = float(np.sum(res_combined * res_combined))
    if total_var <= 0:
        raise ValueError("Control-pool batch residual matrix is identically zero; batch subspace has zero variance")

    max_possible = min(res_combined.shape) - 1
    if max_possible < 1:
        raise ValueError(
            f"Control residual matrix shape {res_combined.shape} is too small for truncated SVD (need min(dim) >= 2)"
        )

    if k_select is not None:
        k_use = int(k_select)
        if k_use < 1:
            raise ValueError("k_select must be >= 1")
        if k_use > max_possible:
            warnings.warn(f"k_select ({k_use}) exceeds max rank {max_possible}; reducing to {max_possible}", UserWarning, stacklevel=2)
            k_use = max_possible
        _, s_all, vt_all = la.svd(res_combined, full_matrices=False)
        dd1 = s_all[:k_use]
        vv1t = vt_all[:k_use, :].T
        exp_ratio = float(np.sum(dd1 ** 2) / total_var)
    else:
        k_cap = min(int(k_max), max_possible)
        # Exact thin SVD when min(res_combined.shape) <= 2000 (fast & avoids ARPACK convergence issues)
        if min(res_combined.shape) <= 2000:
            _, s_all, vt_all = la.svd(res_combined, full_matrices=False)
            s_pos = s_all[:k_cap]
            s_pos = s_pos[s_pos > 0]
            cum_var = np.cumsum(s_pos ** 2) / total_var
            hits = np.flatnonzero(cum_var >= float(var_cutoff))
            k_use = int(hits[0] + 1) if hits.size > 0 else len(s_pos)
            dd1 = s_all[:k_use]
            vv1t = vt_all[:k_use, :].T
            exp_ratio = float(cum_var[k_use - 1])
        else:
            k_pilot = min(k_cap, 100)
            _, s_p, vt_p = svds(res_combined, k=k_pilot)
            ord_p = np.argsort(s_p)[::-1]
            s_p, vt_p = s_p[ord_p], vt_p[ord_p]
            cum_p = np.cumsum(s_p ** 2) / total_var
            hits = np.flatnonzero(cum_p >= float(var_cutoff))
            if hits.size > 0:
                k_use = int(hits[0] + 1)
                dd1 = s_p[:k_use]
                vv1t = vt_p[:k_use, :].T
                exp_ratio = float(cum_p[k_use - 1])
            else:
                _, s_f, vt_f = svds(res_combined, k=k_cap)
                ord_f = np.argsort(s_f)[::-1]
                s_f, vt_f = s_f[ord_f], vt_f[ord_f]
                cum_f = np.cumsum(s_f ** 2) / total_var
                hits_f = np.flatnonzero(cum_f >= float(var_cutoff))
                k_use = int(hits_f[0] + 1) if hits_f.size > 0 else len(s_f)
                dd1 = s_f[:k_use]
                vv1t = vt_f[:k_use, :].T
                exp_ratio = float(cum_f[k_use - 1])

    # Compute compact cell factor C_cells = (X @ VV1T) - LL @ (M_overall^T @ VV1T) of shape (n_cells, k_use)
    proj_x = np.asarray(x_mat @ vv1t, dtype=np.float64)
    small_term = ll @ (m_overall.T @ vv1t)
    cell_factors = proj_x - small_term  # (n_cells, k_use)

    if lazy_ondisk:
        lazy_op = LazyBatchCorrectedOperator(x_mat, cell_factors=cell_factors, gene_basis=vv1t)
        return CellAnovaResult(
            corrected=lazy_op,
            m_overall=m_overall,
            singular_values=dd1,
            batch_basis=vv1t,
            k_selected=k_use,
            explained_variance_ratio=exp_ratio,
        )

    dense_x = x_mat.toarray().astype(np.float64) if sp.issparse(x_mat) else np.asarray(x_mat, dtype=np.float64)
    corrected = dense_x - (cell_factors @ vv1t.T)

    if is_adata:
        if len(feat_names) == adata.n_vars and feat_names == [str(v) for v in adata.var_names]:
            adata.layers[new_layer_name] = corrected
        else:
            adata.obsm[f"X_{new_layer_name}"] = corrected
        if store_diagnostics:
            sc_uns = dict(adata.uns.get("scslide", {}))
            sc_uns["cellanova"] = {
                "k_selected": int(k_use),
                "explained_variance_ratio": float(exp_ratio),
                "singular_values": dd1.tolist(),
                "feature_names": feat_names,
            }
            adata.uns["scslide"] = sc_uns
        return CellAnovaResult(
            corrected=adata,
            m_overall=m_overall,
            singular_values=dd1,
            batch_basis=vv1t,
            k_selected=k_use,
            explained_variance_ratio=exp_ratio,
        )

    out_mat = (
        pd.DataFrame(corrected, index=adata_or_matrix.index, columns=feat_names)
        if isinstance(adata_or_matrix, pd.DataFrame)
        else corrected
    )
    return CellAnovaResult(
        corrected=out_mat,
        m_overall=m_overall,
        singular_values=dd1,
        batch_basis=vv1t,
        k_selected=k_use,
        explained_variance_ratio=exp_ratio,
    )
