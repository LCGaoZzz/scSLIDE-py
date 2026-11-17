"""Supervised feature selection, exact multi-family PLS, streaming on-disk PLS, and projection.

Capabilities covered:
- F08: ``PredictPLS`` (:meth:`PLSModel.predict`, :func:`predict_pls`) and Seurat
  ``ProjectCellEmbeddings`` (:meth:`PLSModel.project_embeddings`, :func:`project_pls_embeddings`)
- F10: ``QuickCorTest`` (:func:`quick_cor_test`, :func:`quick_cor_select`)
- F13: ``RunPLS`` (:func:`run_pls`, :func:`fit_pls` — ``plsr``, ``cppls``, ``spls``)
- F18: ``cppls_ondisk`` (:func:`cppls_ondisk`)
- F19: ``kernelpls`` (:func:`kernelpls`, :func:`kernelpls_ondisk`)

Preserves v0.1.1 DataFrame named-feature alignment on :class:`PLSModel` while fixing the
single-response ``n_components`` cap (B05) and adding safe non-pickle persistence (B09).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.optimize import brentq

from ._validation import (
    StreamingMatrix,
    encode_response_design,
    validate_bijective_ids,
    validate_numeric_matrix_values,
)


# ---------------------------------------------------------------------------
# F10: QuickCorTest (both v0.1.1 table baseline and full grouped scSLIDE API)
# ---------------------------------------------------------------------------

def quick_cor_test(
    x: np.ndarray | pd.DataFrame | sp.spmatrix | sp.sparray,
    y: np.ndarray | pd.Series | pd.DataFrame,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Rank features by Pearson correlation with a response (preserves v0.1.1 table contract)."""
    from scipy.special import betainc

    sm = StreamingMatrix(x)
    n_obs, n_vars = sm.shape
    resp_mat, _, _ = encode_response_design(y, expected_nrow=n_obs, label="y", check_missing=True)
    if resp_mat.shape[1] != 1:
        raise ValueError("quick_cor_test expects a 1-D numeric response; use quick_cor_select for categorical/multi-column responses")
    response = resp_mat[:, 0]

    names = list(x.columns.astype(str)) if isinstance(x, pd.DataFrame) else [f"feature_{i}" for i in range(n_vars)]
    if isinstance(x, pd.DataFrame):
        validate_bijective_ids(x.columns, axis_name="feature names")

    if n_obs < 3 or np.std(response, ddof=1) == 0:
        return pd.DataFrame(
            [{"feature": nm, "correlation": 0.0, "p_value": 1.0, "selected": False} for nm in names]
        ).sort_values(["p_value", "feature"], ignore_index=True)

    x_mean, x_var = sm.mean_and_var(ddof=1)
    x_sd = np.sqrt(x_var)
    y_mean = float(response.mean())
    y_sd = float(np.std(response, ddof=1))
    xy = sm.xtm(response - y_mean)
    denom = (n_obs - 1.0) * x_sd * y_sd
    r_vec = np.zeros(n_vars, dtype=np.float64)
    valid = denom > 0
    r_vec[valid] = np.clip(xy[valid] / denom[valid], -1.0, 1.0)

    df = max(n_obs - 2, 1)
    p_vec = np.ones(n_vars, dtype=np.float64)
    perfect = valid & (np.abs(r_vec) >= 1.0 - 1e-15)
    p_vec[perfect] = 0.0
    interior = valid & ~perfect
    if np.any(interior):
        r2 = r_vec[interior] ** 2
        t2 = df * r2 / np.maximum(1.0 - r2, 1e-300)
        p_vec[interior] = betainc(0.5 * df, 0.5, df / (df + t2))

    rows = [
        {
            "feature": names[j],
            "correlation": float(r_vec[j]),
            "p_value": float(p_vec[j]),
            "selected": bool(p_vec[j] <= threshold),
        }
        for j in range(n_vars)
    ]
    return pd.DataFrame(rows).sort_values(["p_value", "feature"], ignore_index=True)


def quick_cor_select(
    adata_or_matrix: Any,
    y: str | Sequence[Any] | pd.Series | pd.DataFrame,
    *,
    group_by: str | Sequence[Any] | None = None,
    layer: str | None = "data",
    cor_cutoff: float | None = None,
    top_prop: float | None = 0.01,
    min_pct: float = 0.1,
    min_cells: int = 100,
    unlist: bool = True,
    reference_profile: str = "v6",
    return_details: bool = False,
) -> list[str] | dict[str, list[str] | None] | tuple[Any, pd.DataFrame]:
    """Native implementation of upstream ``QuickCorTest`` (F10, legacy & Seurat v6).

    Implements ``model.matrix(~ . + 0)`` categorical/numeric encoding, constant
    response removal per group, ``min_cells`` group skipping, ``min_pct`` gene
    detection filtering, closed-form cross-product Pearson correlation
    (``CorFromCrossprod``), and ``top_prop`` / ``cor_cutoff`` gene selection.
    """
    if cor_cutoff is None and top_prop is None:
        raise ValueError("Must provide at least one of 'top_prop' and 'cor_cutoff'")
    if top_prop is not None and not (0.0 <= float(top_prop) <= 1.0):
        raise ValueError("top_prop must be in [0, 1]")
    if reference_profile not in ("v6", "legacy"):
        raise ValueError("reference_profile must be 'v6' or 'legacy'")

    is_adata = hasattr(adata_or_matrix, "obs") and hasattr(adata_or_matrix, "var_names")
    if is_adata:
        adata = adata_or_matrix
        validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
        validate_bijective_ids(adata.var_names, axis_name="adata.var_names")
        feature_names = [str(v) for v in adata.var_names]
        raw_mat = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
        if isinstance(y, str):
            if y not in adata.obs.columns:
                raise ValueError(f"Response column {y!r} not found in adata.obs")
            y_df = adata.obs[[y]].copy()
        elif isinstance(y, pd.DataFrame):
            y_df = y.loc[adata.obs_names].copy()
        else:
            y_df = pd.DataFrame({"Y": list(y)}, index=adata.obs_names)

        if isinstance(group_by, str):
            if group_by not in adata.obs.columns:
                raise ValueError(f"Group column {group_by!r} not found in adata.obs")
            grp_series = adata.obs[group_by]
        elif group_by is not None:
            grp_series = pd.Series(list(group_by), index=adata.obs_names)
        else:
            grp_series = pd.Series(["all"] * adata.n_obs, index=adata.obs_names)
    else:
        raw_mat = adata_or_matrix
        n_obs, n_vars = raw_mat.shape
        feature_names = (
            [str(c) for c in raw_mat.columns]
            if isinstance(raw_mat, pd.DataFrame)
            else [f"feature_{i}" for i in range(n_vars)]
        )
        validate_bijective_ids(feature_names, axis_name="feature_names")
        idx_obs = raw_mat.index if isinstance(raw_mat, pd.DataFrame) else pd.RangeIndex(n_obs)
        if isinstance(y, pd.DataFrame):
            y_df = y.copy()
        elif isinstance(y, pd.Series):
            y_df = y.to_frame(name=str(y.name or "Y"))
        else:
            y_df = pd.DataFrame({"Y": list(y)}, index=idx_obs)
        grp_series = (
            pd.Series(list(group_by), index=idx_obs)
            if group_by is not None
            else pd.Series(["all"] * n_obs, index=idx_obs)
        )

    if y_df.shape[1] != 1:
        raise ValueError("When setting 'Y', one and only one variable is allowed")

    check_na = reference_profile == "v6"
    if check_na and y_df.iloc[:, 0].isna().any():
        n_miss = int(y_df.iloc[:, 0].isna().sum())
        raise ValueError(f"Y ({y_df.columns[0]!r}) has NA for {n_miss} cells. Subset to valid responses first.")

    # Preserve group appearance order matching Seurat CellsByIdentities
    grp_valid = grp_series.notna().to_numpy()
    group_order: list[str] = []
    seen_g: set[str] = set()
    for val in grp_series.to_numpy()[grp_valid]:
        gs = str(val)
        if gs not in seen_g:
            seen_g.add(gs)
            group_order.append(gs)

    grp_str = np.array([str(v) if ok else "" for v, ok in zip(grp_series.to_numpy(), grp_valid)], dtype=object)

    selected_by_group: dict[str, list[str] | None] = {}
    detail_rows: list[dict[str, Any]] = []

    for grp in group_order:
        row_pos = np.flatnonzero(grp_valid & (grp_str == grp))
        # In legacy profile, drop NA response rows within group
        if not check_na:
            not_na_y = y_df.iloc[row_pos, 0].notna().to_numpy()
            row_pos = row_pos[not_na_y]

        if len(row_pos) < min_cells:
            selected_by_group[grp] = None
            continue

        y_sub = y_df.iloc[row_pos]
        y_enc, y_cols, _ = encode_response_design(y_sub, expected_nrow=len(row_pos), label="Y", check_missing=True)
        # Remove constant response columns within this group
        keep_y = [j for j in range(y_enc.shape[1]) if len(np.unique(y_enc[:, j])) > 1]
        if len(keep_y) == 0:
            selected_by_group[grp] = None
            continue
        y_enc = y_enc[:, keep_y]
        y_cols = [y_cols[j] for j in keep_y]

        sm_grp = StreamingMatrix(raw_mat, row_indices=row_pos)
        det = sm_grp.detection_counts(strict_positive=True)
        keep_feat = np.flatnonzero(det >= min_pct * len(row_pos))
        if len(keep_feat) == 0:
            selected_by_group[grp] = []
            continue

        sm_filt = StreamingMatrix(raw_mat, row_indices=row_pos, col_indices=keep_feat)
        x_mean, x_var = sm_filt.mean_and_var(ddof=1)
        x_sd = np.sqrt(x_var)
        y_mean = y_enc.mean(axis=0)
        y_sd = np.std(y_enc, axis=0, ddof=1)
        xy = sm_filt.xtm(y_enc)  # (n_kept_genes, q)

        n_g = float(len(row_pos))
        cov = (xy - n_g * np.outer(x_mean, y_mean)) / (n_g - 1.0)
        denom = np.outer(x_sd, y_sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            cor_mat = cov / denom  # (n_kept_genes, q)
        cor_mat[~np.isfinite(cor_mat)] = np.nan

        abs_cor = np.abs(cor_mat)
        n_genes_grp = len(keep_feat)

        if y_enc.shape[1] == 1:
            vec = abs_cor[:, 0]
            # R order(..., decreasing=TRUE, na.last=TRUE): finite values sorted desc, ties by index asc, NAs last
            finite_idx = np.flatnonzero(np.isfinite(vec))
            na_idx = np.flatnonzero(~np.isfinite(vec))
            ord_fin = finite_idx[np.argsort(-vec[finite_idx], kind="mergesort")]
            ordered = np.concatenate([ord_fin, na_idx])

            if top_prop is not None:
                n_top = int(math.ceil(float(top_prop) * n_genes_grp))
                cand = ordered[:n_top].tolist() if n_top > 0 else []
            else:
                cand = ordered.tolist()

            if cor_cutoff is not None:
                pass_set = set(np.flatnonzero(np.isfinite(vec) & (vec >= float(cor_cutoff))).tolist())
                chosen_local = [idx for idx in cand if idx in pass_set]
            else:
                chosen_local = cand
        else:
            per_col_sets: list[int] = []
            for col_j in range(y_enc.shape[1]):
                vec = abs_cor[:, col_j]
                finite_idx = np.flatnonzero(np.isfinite(vec))
                na_idx = np.flatnonzero(~np.isfinite(vec))
                ord_fin = finite_idx[np.argsort(-vec[finite_idx], kind="mergesort")]
                ordered = np.concatenate([ord_fin, na_idx])
                if top_prop is not None:
                    n_top = int(math.ceil(float(top_prop) * n_genes_grp))
                    cand = ordered[:n_top].tolist() if n_top > 0 else []
                else:
                    cand = ordered.tolist()
                if cor_cutoff is not None:
                    pass_set = set(np.flatnonzero(np.isfinite(vec) & (vec >= float(cor_cutoff))).tolist())
                    cand = [idx for idx in cand if idx in pass_set]
                per_col_sets.extend(cand)
            chosen_local = sorted(set(per_col_sets))

        genes_grp = [feature_names[int(keep_feat[i])] for i in chosen_local]
        selected_by_group[grp] = genes_grp

        if return_details:
            for loc_i in chosen_local:
                g_name = feature_names[int(keep_feat[loc_i])]
                max_abs = float(np.nanmax(abs_cor[loc_i])) if np.any(np.isfinite(abs_cor[loc_i])) else 0.0
                detail_rows.append({"group": grp, "feature": g_name, "max_abs_cor": max_abs, "n_cells": int(len(row_pos))})

    if unlist:
        union_genes: list[str] = []
        seen_feat: set[str] = set()
        for grp in group_order:
            g_list = selected_by_group.get(grp)
            if g_list:
                for g in g_list:
                    if g not in seen_feat:
                        seen_feat.add(g)
                        union_genes.append(g)
        return (union_genes, pd.DataFrame(detail_rows)) if return_details else union_genes

    return (selected_by_group, pd.DataFrame(detail_rows)) if return_details else selected_by_group


# ---------------------------------------------------------------------------
# F19: Exact Dayal & MacGregor (1997) Kernel PLS (in-memory & streaming)
# ---------------------------------------------------------------------------

def kernelpls(
    x: Any,
    y: Any,
    ncomp: int,
    *,
    center: bool = True,
    stripped: bool = False,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Exact Dayal & MacGregor (1997) kernel-PLS matching ``pls::kernelpls.fit`` & ``KernelPLS.cpp``.

    Accepts dense arrays, DataFrames, scipy CSR/CSC sparse matrices, or backed
    HDF5/Zarr matrices (``n_obs x n_features``) without densifying ``x``.
    """
    ncomp = int(ncomp)
    if ncomp <= 0:
        raise ValueError("ncomp must be a positive integer")

    sm = x if isinstance(x, StreamingMatrix) else StreamingMatrix(x, chunk_size=chunk_size)
    n, p = sm.shape
    y_mat, y_names, _ = encode_response_design(y, expected_nrow=n, label="Y", check_missing=True)
    q = y_mat.shape[1]

    if ncomp > min(n - 1 if center else n, p):
        raise ValueError(f"ncomp ({ncomp}) exceeds valid rank bound min(n-1, p) = {min(n - 1 if center else n, p)}")

    if center:
        x_means, x_var = sm.mean_and_var(ddof=1)
        y_means = y_mat.mean(axis=0)
        y_centered = y_mat - y_means[None, :]
        xtotvar = float(x_var.sum() * max(n - 1, 1))
    else:
        x_means = np.zeros(p, dtype=np.float64)
        _, x_var = sm.mean_and_var(ddof=1)
        m0, _ = sm.mean_and_var(ddof=1)
        y_means = np.zeros(q, dtype=np.float64)
        y_centered = y_mat.copy()
        xtotvar = float(x_var.sum() * max(n - 1, 1) + n * np.sum(m0 ** 2))

    xty = sm.xtm(y_centered)  # (p, q); when y_centered has zero col sums, (X - mu)^T Y_c == X^T Y_c

    r_mat = np.zeros((p, ncomp), dtype=np.float64)
    p_mat = np.zeros((p, ncomp), dtype=np.float64)
    tq_mat = np.zeros((ncomp, q), dtype=np.float64)
    w_mat = np.zeros((p, ncomp), dtype=np.float64)
    scores = np.zeros((n, ncomp), dtype=np.float64)
    y_scores = np.zeros((n, ncomp), dtype=np.float64)
    xvar = np.zeros(ncomp, dtype=np.float64)
    tsqs = np.zeros(ncomp, dtype=np.float64)

    for a in range(ncomp):
        if q == 1:
            w_a = xty[:, 0].copy()
            norm = float(np.linalg.norm(w_a))
            if norm > 0:
                w_a /= norm
        elif q < p:
            gram = xty.T @ xty
            vals, vecs = np.linalg.eigh(gram)
            q_vec = vecs[:, -1]
            w_a = xty @ q_vec
            norm = float(np.linalg.norm(w_a))
            if norm > 0:
                w_a /= norm
        else:
            gram = xty @ xty.T
            vals, vecs = np.linalg.eigh(gram)
            w_a = vecs[:, -1]

        # Deterministic sign convention: first largest-magnitude entry positive
        max_idx = int(np.argmax(np.abs(w_a)))
        if w_a[max_idx] < 0:
            w_a = -w_a

        r_a = w_a.copy()
        if a > 0:
            r_a = r_a - r_mat[:, :a] @ (p_mat[:, :a].T @ w_a)

        t_a = sm.xm(r_a)
        if center:
            t_a = t_a - float(np.dot(x_means, r_a))
        tsq = float(np.dot(t_a, t_a))
        if tsq <= 0:
            raise ValueError(f"Degenerate zero-variance score encountered at component {a + 1}")

        p_a = sm.xtm(t_a) / tsq
        if center:
            p_a = p_a - x_means * (float(t_a.sum()) / tsq)

        q_a = (xty.T @ r_a) / tsq
        xty = xty - np.outer(tsq * p_a, q_a)

        r_mat[:, a] = r_a
        p_mat[:, a] = p_a
        tq_mat[a, :] = q_a
        tsqs[a] = tsq

        if not stripped:
            scores[:, a] = t_a
            w_mat[:, a] = w_a
            q_sq = float(np.dot(q_a, q_a))
            u_a = (y_centered @ q_a) / q_sq if q_sq > 0 else np.zeros(n, dtype=np.float64)
            if a > 0:
                proj = (scores[:, :a].T @ u_a) / tsqs[:a]
                u_a = u_a - scores[:, :a] @ proj
            y_scores[:, a] = u_a
            xvar[a] = float(np.dot(p_a, p_a)) * tsq

    coeffs = np.zeros((p, q, ncomp), dtype=np.float64)
    for a in range(ncomp):
        coeffs[:, :, a] = r_mat[:, : a + 1] @ tq_mat[: a + 1, :]

    out: dict[str, Any] = {
        "coefficients": coeffs,
        "Xmeans": x_means,
        "Ymeans": y_means,
        "response_names": y_names,
        "memory_mode": sm.memory_mode,
    }
    if not stripped:
        fitted = np.zeros((n, q, ncomp), dtype=np.float64)
        residuals = np.zeros((n, q, ncomp), dtype=np.float64)
        for a in range(ncomp):
            fit_c = scores[:, : a + 1] @ tq_mat[: a + 1, :]
            residuals[:, :, a] = y_centered - fit_c
            fitted[:, :, a] = fit_c + y_means[None, :] if center else fit_c

        r2, rmsep = compute_pls_r2_rmsep(y_mat, fitted)
        out.update(
            scores=scores,
            loadings=p_mat,
            loading_weights=w_mat,
            Yscores=y_scores,
            Yloadings=tq_mat.T,
            projection=r_mat,
            fitted_values=fitted,
            residuals=residuals,
            Xvar=xvar,
            Xtotvar=xtotvar,
            R2=r2,
            RMSEP=rmsep,
        )
    return out


kernelpls_ondisk = kernelpls


# ---------------------------------------------------------------------------
# F18: Exact Canonical Powered PLS (cppls / cppls_ondisk, v6 & legacy profiles)
# ---------------------------------------------------------------------------

def _cancorr_r_compatible(z: np.ndarray, y_prim: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact translation of ``BPCells::cancorr`` / ``pls:::cancorr`` with LAPACK pivoted QR."""
    n, ncx = z.shape
    ncy = y_prim.shape[1]

    qz, rz, piv_z = la.qr(z, mode="economic", pivoting=True)
    qy, ry, _ = la.qr(y_prim, mode="economic", pivoting=True)

    eps = np.finfo(np.float64).eps
    rz11 = abs(float(rz[0, 0])) if rz.size > 0 else 0.0
    if rz11 == 0.0:
        return np.zeros((ncx, 1), dtype=np.float64), 0.0
    thresh_z = eps * (2.0 ** math.floor(math.log2(rz11))) * max(n, ncx)
    dx = 0
    for i in range(min(n, ncx)):
        if abs(float(rz[i, i])) > thresh_z:
            dx += 1
        else:
            break
    if dx == 0:
        return np.zeros((ncx, 1), dtype=np.float64), 0.0

    ry11 = abs(float(ry[0, 0])) if ry.size > 0 else 0.0
    if ry11 == 0.0:
        return np.zeros((ncx, 1), dtype=np.float64), 0.0
    thresh_y = eps * (2.0 ** math.floor(math.log2(ry11))) * max(n, ncy)
    dy = 0
    for i in range(min(n, ncy)):
        if abs(float(ry[i, i])) > thresh_y:
            dy += 1
        else:
            break
    if dy == 0:
        return np.zeros((ncx, 1), dtype=np.float64), 0.0

    dxy = min(dx, dy)
    cross = qz[:, :dx].T @ qy[:, :dy]  # (dx, dy)
    u_svd, s_svd, _ = la.svd(cross, full_matrices=False)
    r_val = float(np.clip(s_svd[0], 0.0, 1.0))

    rz_sq = rz[:dx, :dx]
    a_trunc = la.solve_triangular(rz_sq, u_svd[:, :dxy], lower=False) * math.sqrt(max(n - 1.0, 1.0))
    a_out = np.zeros((ncx, dxy), dtype=np.float64)
    a_out[piv_z[:dx], :] = a_trunc
    return a_out, r_val


def cppls_ondisk(
    x: Any,
    y: Any,
    ncomp: int,
    *,
    y_add: Any = None,
    center: bool = True,
    stripped: bool = False,
    w_tol: float = 0.0,
    x_tol: float = 1e-12,
    reference_profile: str = "v6",
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Canonical Powered PLS (``gamma=0.5``) for dense, sparse CSR/CSC, or backed HDF5/Zarr inputs (F18).

    Explicitly supports both reference profiles discovered during upstream audit:
    - ``reference_profile='v6'`` (default, matching ``Seurat v6.0.0`` & ``pls >= 2.9-0``):
      centers ``Y`` and ``Y_add`` before ``cancorr(Z, Yprim)`` and ``q_a``.
    - ``reference_profile='legacy_ondisk'`` (matching ``scSLIDE@968dbde`` ``CPPLS.cpp`` & ``pls < 2.9-0``):
      passes uncentered ``Y`` and ``Y_add`` into ``cancorr(Z, Y)`` and ``q_a``.
    """
    if reference_profile not in ("v6", "legacy", "legacy_ondisk"):
        raise ValueError("reference_profile must be 'v6', 'legacy', or 'legacy_ondisk'")
    ncomp = int(ncomp)
    if ncomp <= 0:
        raise ValueError("ncomp must be a positive integer")

    sm = x if isinstance(x, StreamingMatrix) else StreamingMatrix(x, chunk_size=chunk_size)
    n, p = sm.shape
    if ncomp >= min(n, p):
        raise ValueError(f"ncomp ({ncomp}) must be less than min(n, p) = {min(n, p)}")

    y_raw, y_names, _ = encode_response_design(y, expected_nrow=n, label="Y", check_missing=True)
    nresp = y_raw.shape[1]

    if y_add is not None:
        y_add_raw, _, _ = encode_response_design(y_add, expected_nrow=n, label="Y.add", check_missing=True)
        q_add = y_add_raw.shape[1]
    else:
        y_add_raw = np.zeros((n, 0), dtype=np.float64)
        q_add = 0
    q_full = nresp + q_add

    center_responses = center and (reference_profile != "legacy_ondisk")

    if center:
        x_means, x_var = sm.mean_and_var(ddof=1)
        y_means = y_raw.mean(axis=0)
        xtotvar = float(x_var.sum() * max(n - 1, 1))
    else:
        x_means = np.zeros(p, dtype=np.float64)
        m0, x_var = sm.mean_and_var(ddof=1)
        y_means = np.zeros(nresp, dtype=np.float64)
        xtotvar = float(x_var.sum() * max(n - 1, 1) + n * np.sum(m0 ** 2))

    y_prim = y_raw - y_means[None, :] if center_responses else y_raw.copy()
    y_full = np.zeros((n, q_full), dtype=np.float64)
    y_full[:, :nresp] = y_prim
    if q_add > 0:
        y_full[:, nresp:] = y_add_raw - y_add_raw.mean(axis=0, keepdims=True) if center_responses else y_add_raw

    xty_full = sm.xtm(y_full)
    if center:
        xty_full = xty_full - np.outer(x_means, y_full.sum(axis=0))

    w_mat = np.zeros((p, ncomp), dtype=np.float64)
    p_mat = np.zeros((p, ncomp), dtype=np.float64)
    q_mat = np.zeros((nresp, ncomp), dtype=np.float64)
    tt_mat = np.zeros((n, ncomp), dtype=np.float64)
    tsqs = np.zeros(ncomp, dtype=np.float64)
    gammas = np.full(ncomp, 0.5, dtype=np.float64)
    canon_corr = np.zeros(ncomp, dtype=np.float64)
    a_weights = np.zeros((q_full, ncomp), dtype=np.float64)
    y_scores = np.zeros((n, ncomp), dtype=np.float64)

    # Detect exact zero-variance features upfront for smallNorm
    small_norm: list[int] = np.flatnonzero(np.sqrt(np.maximum(x_var * max(n - 1, 1), 0.0)) <= x_tol).tolist()

    for a in range(ncomp):
        w0 = xty_full.copy()
        if small_norm:
            w0[small_norm, :] = 0.0

        z = sm.xm(w0)
        if center:
            z = z - (x_means @ w0)[None, :]
        if a > 0:
            z = z - tt_mat[:, :a] @ (p_mat[:, :a].T @ w0)

        cca_a, cca_r = _cancorr_r_compatible(z, y_prim)
        canon_corr[a] = cca_r * cca_r
        if cca_a.shape[1] > 0:
            a_col = cca_a[:, 0]
            a_weights[: len(a_col), a] = a_col
            w_a = w0 @ a_col
        else:
            w_a = w0[:, 0].copy()

        if w_tol > 0:
            w_a[np.abs(w_a) < w_tol] = 0.0

        w_norm = float(np.linalg.norm(w_a))
        if w_norm > 0:
            w_a /= w_norm

        # Deterministic sign anchor (leaves canonical subspace & predictions invariant)
        max_idx = int(np.argmax(np.abs(w_a)))
        if w_a[max_idx] < 0:
            w_a = -w_a
            a_weights[:, a] = -a_weights[:, a]

        t_a = sm.xm(w_a)
        if center:
            t_a = t_a - float(np.dot(x_means, w_a))
        if a > 0:
            t_a = t_a - tt_mat[:, :a] @ (p_mat[:, :a].T @ w_a)

        tsq = float(np.dot(t_a, t_a))
        if tsq <= 0:
            raise ValueError(f"Degenerate zero-variance score encountered in CPPLS at component {a + 1}")

        p_a = sm.xtm(t_a)
        if center:
            p_a = p_a - x_means * float(t_a.sum())
        p_a = p_a / tsq
        if a > 0:
            p_a = p_a - p_mat[:, :a] @ ((tt_mat[:, :a].T @ t_a) / tsq)

        q_a = (y_prim.T @ t_a) / tsq
        xty_full = xty_full - np.outer(p_a, t_a @ y_full)

        w_mat[:, a] = w_a
        tt_mat[:, a] = t_a
        p_mat[:, a] = p_a
        q_mat[:, a] = q_a
        tsqs[a] = tsq

        if not stripped:
            q_sq = float(np.dot(q_a, q_a))
            u_a = (y_prim @ q_a) / q_sq if q_sq > 0 else np.zeros(n, dtype=np.float64)
            if a > 0:
                proj = (tt_mat[:, :a].T @ u_a) / tsqs[:a]
                u_a = u_a - tt_mat[:, :a] @ proj
            y_scores[:, a] = u_a

    coeffs = np.zeros((p, nresp, ncomp), dtype=np.float64)
    for a in range(ncomp):
        w_a = w_mat[:, : a + 1]
        p_a = p_mat[:, : a + 1]
        q_a = q_mat[:, : a + 1]
        ptw = p_a.T @ w_a
        coeffs[:, :, a] = w_a @ la.solve(ptw, q_a.T)

    out: dict[str, Any] = {
        "coefficients": coeffs,
        "Xmeans": x_means,
        "Ymeans": y_means,
        "gammas": gammas,
        "canonical_correlations": canon_corr,
        "A": a_weights,
        "smallNorm": [int(i + 1) for i in small_norm],
        "response_names": y_names,
        "reference_profile": reference_profile,
        "memory_mode": sm.memory_mode,
    }
    if not stripped:
        ptw_full = p_mat.T @ w_mat
        projection = la.solve(ptw_full.T, w_mat.T).T

        l_mat = np.zeros((ncomp, ncomp), dtype=np.float64)
        for a in range(1, ncomp):
            for j in range(a):
                l_mat[j, a] = float(np.dot(p_mat[:, j], w_mat[:, a]))
        ipl = np.eye(ncomp, dtype=np.float64) + l_mat

        fitted = np.zeros((n, nresp, ncomp), dtype=np.float64)
        residuals = np.zeros((n, nresp, ncomp), dtype=np.float64)
        for a in range(ncomp):
            w_a = w_mat[:, : a + 1]
            p_a = p_mat[:, : a + 1]
            q_a = q_mat[:, : a + 1]
            tipl = tt_mat[:, : a + 1] @ ipl[: a + 1, : a + 1]
            fit_a = tipl @ la.solve(p_a.T @ w_a, q_a.T)
            if center:
                fit_a = fit_a + y_means[None, :]
            fitted[:, :, a] = fit_a
            residuals[:, :, a] = y_raw - fit_a

        xvar = np.sum(p_mat ** 2, axis=0) * tsqs
        r2, rmsep = compute_pls_r2_rmsep(y_raw, fitted)
        out.update(
            scores=tt_mat,
            loadings=p_mat,
            loading_weights=w_mat,
            Yscores=y_scores,
            Yloadings=q_mat,
            projection=projection,
            fitted_values=fitted,
            residuals=residuals,
            Xvar=xvar,
            Xtotvar=xtotvar,
            R2=r2,
            RMSEP=rmsep,
        )
    return out


# ---------------------------------------------------------------------------
# Sparse PLS (Chun & Keles 2010, matching R spls::spls)
# ---------------------------------------------------------------------------

def _spls_ust(b: np.ndarray, eta: float) -> np.ndarray:
    out = np.zeros_like(b, dtype=np.float64)
    if eta < 1.0:
        max_abs = float(np.max(np.abs(b)))
        valb = np.abs(b) - eta * max_abs
        mask = valb >= 0.0
        out[mask] = valb[mask] * np.sign(b[mask])
    return out


def _spls_dv(z: np.ndarray, eta: float, kappa: float = 0.5, eps: float = 1e-4, maxstep: int = 100) -> np.ndarray:
    p, q = z.shape
    znorm1 = float(np.median(np.abs(z)))
    if znorm1 <= 0:
        return np.zeros((p, 1), dtype=np.float64)
    z_scaled = z / znorm1
    if q == 1:
        return _spls_ust(z_scaled, eta)

    m = z_scaled @ z_scaled.T
    dis = 10.0
    step = 1
    c = np.full((p, 1), 10.0, dtype=np.float64)
    c_old = c.copy()

    if abs(kappa - 0.5) < 1e-12:
        while dis > eps and step <= maxstep:
            u, _, vt = la.svd(m @ c, full_matrices=False)
            a = u @ vt
            c = _spls_ust(m @ a, eta)
            dis = float(np.max(np.abs(c - c_old)))
            c_old = c.copy()
            step += 1
        return c

    kappa2 = (1.0 - kappa) / (1.0 - 2.0 * kappa)
    eye_p = np.eye(p, dtype=np.float64)

    def h_fn(lam: float, m_cur: np.ndarray, c_cur: np.ndarray) -> float:
        alpha = la.solve(m_cur + lam * eye_p, m_cur @ c_cur, assume_a="pos")
        return float((alpha.T @ alpha).item() - 1.0 / (kappa2 ** 2))

    while dis > eps and step <= maxstep:
        while h_fn(eps, m, c) * h_fn(1e30, m, c) > 0 and h_fn(eps, m, c) <= 1e5:
            m = 2.0 * m
            c = 2.0 * c
        lam_star = brentq(h_fn, eps, 1e30, args=(m, c))
        a = kappa2 * la.solve(m + lam_star * eye_p, m @ c, assume_a="pos")
        c = _spls_ust(m @ a, eta)
        dis = float(np.max(np.abs(c - c_old)))
        c_old = c.copy()
        step += 1
    return c


def spls_fit(
    x: np.ndarray | pd.DataFrame,
    y: Any,
    ncomp: int,
    *,
    eta: float = 0.5,
    kappa: float = 0.5,
    select: str = "pls2",
    scale_x: bool = True,
    scale_y: bool = False,
    eps: float = 1e-4,
    maxstep: int = 100,
) -> dict[str, Any]:
    """Exact Chun & Keles (2010) sPLS matching ``spls::spls(..., fit='kernelpls')``."""
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must satisfy 0 <= eta < 1")
    arr_orig = x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float64)
    validate_numeric_matrix_values(arr_orig, name="x")
    n, p = arr_orig.shape
    k_comp = int(ncomp)
    if k_comp < 1 or k_comp >= n or k_comp > p:
        raise ValueError(f"ncomp ({k_comp}) must satisfy 1 <= ncomp <= min(n-1, p)")

    y_mat, y_names, _ = encode_response_design(y, expected_nrow=n, label="Y", check_missing=True)
    q = y_mat.shape[1]

    mu_y = y_mat.mean(axis=0)
    yc = y_mat - mu_y[None, :]
    mean_x = arr_orig.mean(axis=0)
    xc = arr_orig - mean_x[None, :]

    if scale_x:
        norm_x = np.sqrt(np.sum(xc ** 2, axis=0) / max(n - 1, 1))
        if np.any(norm_x < np.finfo(np.float64).eps):
            raise ValueError("Some columns of predictor matrix have zero variance for spls(scale_x=True)")
        xc = xc / norm_x[None, :]
    else:
        norm_x = np.ones(p, dtype=np.float64)

    if scale_y:
        norm_y = np.sqrt(np.sum(yc ** 2, axis=0) / max(n - 1, 1))
        if np.any(norm_y < np.finfo(np.float64).eps):
            raise ValueError("Some columns of response matrix have zero variance for spls(scale_y=True)")
        yc = yc / norm_y[None, :]
    else:
        norm_y = np.ones(q, dtype=np.float64)

    betahat = np.zeros((p, q), dtype=np.float64)
    betamat: list[np.ndarray] = []
    x1 = xc.copy()
    y1 = yc.copy()
    active_idx = np.array([], dtype=np.int64)
    pj = np.zeros((0, 0), dtype=np.float64)

    for k in range(1, k_comp + 1):
        z = x1.T @ y1
        what = _spls_dv(z, eta=eta, kappa=kappa, eps=eps, maxstep=maxstep).ravel()
        active_idx = np.flatnonzero((what != 0.0) | (betahat[:, 0] != 0.0))
        if active_idx.size == 0:
            # Fallback to single strongest correlation if eta thresholds everything
            active_idx = np.array([int(np.argmax(np.max(np.abs(z), axis=1)))], dtype=np.int64)

        x_active = xc[:, active_idx]
        sub_ncomp = min(k, len(active_idx), n - 1)
        pls_sub = kernelpls(x_active, yc, ncomp=sub_ncomp, center=True, stripped=False)
        betahat = np.zeros((p, q), dtype=np.float64)
        betahat[active_idx, :] = pls_sub["coefficients"][:, :, -1]
        betamat.append(betahat.copy())
        pj = pls_sub["projection"]

        if select == "pls2":
            y1 = yc - xc @ betahat
        elif select == "simpls":
            pw = pj @ la.solve(pj.T @ pj, pj.T)
            x1 = xc.copy()
            x1[:, active_idx] = xc[:, active_idx] - xc[:, active_idx] @ pw

    # Match RunPLS.default: embeddings = X_orig[:, A] @ projection
    cell_embeddings = arr_orig[:, active_idx] @ pj
    return {
        "scores": cell_embeddings,
        "projection": pj,
        "active_indices_0based": active_idx,
        "active_indices_1based": (active_idx + 1).tolist(),
        "betahat": betahat,
        "betamat": betamat,
        "Xmeans": mean_x,
        "Ymeans": mu_y,
        "normx": norm_x,
        "normy": norm_y,
        "response_names": y_names,
    }


def compute_pls_r2_rmsep(y: np.ndarray, fitted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative ``R2`` and ``RMSEP`` matching Seurat v6 ``(q, ncomp + 1)`` layout."""
    y_arr = np.asarray(y, dtype=np.float64)
    n, q = y_arr.shape
    ncomp = fitted_values.shape[2]
    sst = np.sum((y_arr - y_arr.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.zeros((q, ncomp + 1), dtype=np.float64)
    rmsep = np.zeros((q, ncomp + 1), dtype=np.float64)
    rmsep[:, 0] = np.sqrt(sst / max(n, 1))
    for a in range(ncomp):
        sse = np.sum((y_arr - fitted_values[:, :, a]) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2[:, a + 1] = np.where(sst > 0, 1.0 - sse / sst, 0.0)
        rmsep[:, a + 1] = np.sqrt(sse / max(n, 1))
    return r2, rmsep


# ---------------------------------------------------------------------------
# PLSModel (F08, F13, B04, B05, B09: alignment + prediction + projection + save/load)
# ---------------------------------------------------------------------------

@dataclass
class PLSModel:
    """Persisted, feature-aligned native PLS model supporting ``transform``, ``predict``, and Seurat projection.

    Preserves the v0.1.1 ``PLSModel`` interface (``feature_mean``, ``feature_scale``,
    ``feature_names``, ``transform``, ``predict``) while removing ``pickle`` reliance and
    supporting ``kernelpls``, ``cppls``, and ``spls``.
    """

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: list[str]
    method: str = "plsr"
    n_components: int = 2
    coefficients: np.ndarray | None = None        # (p, q, ncomp) in model-input space
    projection: np.ndarray | None = None          # (p, ncomp) or (|A|, ncomp) for spls
    loadings: np.ndarray | None = None            # (p, ncomp)
    loading_weights: np.ndarray | None = None     # (p, ncomp)
    x_means_internal: np.ndarray | None = None    # (p,) centering inside PLS solver
    y_means: np.ndarray | None = None             # (q,)
    response_names: list[str] = field(default_factory=list)
    x_var: np.ndarray | None = None
    x_totvar: float | None = None
    r2: np.ndarray | None = None
    rmsep: np.ndarray | None = None
    active_indices: np.ndarray | None = None      # 0-based active subset for spls
    reference_profile: str = "v6"
    scores_: np.ndarray | None = None
    model: Any = None                             # Optional legacy compatibility slot

    def _aligned_array(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Validate and align prediction features to the training schema (preserves v0.1.1 B04)."""
        if isinstance(x, pd.DataFrame):
            if x.columns.has_duplicates:
                raise ValueError("prediction features must have unique names")
            validate_bijective_ids(x.columns, axis_name="prediction features")
            expected = list(self.feature_names)
            missing = [name for name in expected if name not in x.columns]
            extra = [name for name in x.columns if name not in expected]
            if missing or extra:
                detail = []
                if missing:
                    detail.append(f"missing={missing}")
                if extra:
                    detail.append(f"unexpected={extra}")
                raise ValueError("prediction feature names do not match training schema (" + ", ".join(detail) + ")")
            arr = x.loc[:, expected].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
            raise ValueError(f"prediction must have {len(self.feature_names)} feature columns")
        validate_numeric_matrix_values(arr, name="prediction input")
        return arr

    def _aligned_streaming(self, x: Any, feature_names: Sequence[str] | None = None) -> StreamingMatrix:
        """Align a sparse/backed/AnnData/DataFrame input by feature names without densifying."""
        if isinstance(x, StreamingMatrix):
            if feature_names is None and x.n_vars == len(self.feature_names):
                return x
            if feature_names is not None:
                validate_bijective_ids(feature_names, axis_name="newdata features")
                lookup = {str(nm): idx for idx, nm in enumerate(feature_names)}
                missing = [nm for nm in self.feature_names if nm not in lookup]
                if missing:
                    raise ValueError(f"newdata is missing {len(missing)} required features: {missing[:5]}")
                sub_pos = np.array([lookup[nm] for nm in self.feature_names], dtype=np.int64)
                new_cols = sub_pos if x.col_indices is None else x.col_indices[sub_pos]
                return StreamingMatrix(x.source, row_indices=x.row_indices, col_indices=new_cols, chunk_size=x.chunk_size)
            raise ValueError(f"newdata has {x.n_vars} columns but model expects {len(self.feature_names)} features")
        if hasattr(x, "var_names") and hasattr(x, "X"):
            names = [str(v) for v in x.var_names]
            mat = x.X
        elif isinstance(x, pd.DataFrame):
            return StreamingMatrix(self._aligned_array(x))
        else:
            mat = x
            names = [str(v) for v in feature_names] if feature_names is not None else None

        if names is not None:
            validate_bijective_ids(names, axis_name="newdata features")
            lookup = {nm: idx for idx, nm in enumerate(names)}
            missing = [nm for nm in self.feature_names if nm not in lookup]
            if missing:
                raise ValueError(f"newdata is missing {len(missing)} required features: {missing[:5]}")
            col_idx = np.array([lookup[nm] for nm in self.feature_names], dtype=np.int64)
            return StreamingMatrix(mat, col_indices=col_idx)

        if mat.shape[1] != len(self.feature_names):
            raise ValueError(f"newdata has {mat.shape[1]} columns but model expects {len(self.feature_names)} features")
        return StreamingMatrix(mat)

    def transform(self, x: Any, *, ncomp: int | None = None, feature_names: Sequence[str] | None = None) -> np.ndarray:
        """Project observations into the PLS latent score space ``(n, k)``."""
        k = self.n_components if ncomp is None else int(ncomp)
        if not 1 <= k <= self.n_components:
            raise ValueError(f"ncomp must be between 1 and {self.n_components}")

        if self.model is not None and self.projection is None:
            arr = self._aligned_array(x)
            scaled = (arr - self.feature_mean) / self.feature_scale
            return self.model.transform(scaled)[:, :k]

        if isinstance(x, (np.ndarray, pd.DataFrame)):
            arr = self._aligned_array(x)
            scaled = (arr - self.feature_mean) / self.feature_scale
            if self.method == "spls" and self.active_indices is not None:
                return scaled[:, self.active_indices] @ self.projection[:, :k]
            xm_int = self.x_means_internal if self.x_means_internal is not None else np.zeros(scaled.shape[1])
            return (scaled - xm_int[None, :]) @ self.projection[:, :k]

        sm = self._aligned_streaming(x, feature_names=feature_names)
        inv_scale = 1.0 / self.feature_scale
        if self.method == "spls" and self.active_indices is not None:
            w_eff = np.zeros((len(self.feature_names), k), dtype=np.float64)
            w_eff[self.active_indices, :] = self.projection[:, :k]
            w_scaled = inv_scale[:, None] * w_eff
            offset = (self.feature_mean * inv_scale) @ w_eff
            return sm.xm(w_scaled) - offset[None, :]

        xm_int = self.x_means_internal if self.x_means_internal is not None else np.zeros(len(self.feature_names))
        proj_k = self.projection[:, :k]
        w_scaled = inv_scale[:, None] * proj_k
        offset = ((self.feature_mean * inv_scale) + xm_int) @ proj_k
        return sm.xm(w_scaled) - offset[None, :]

    def project_embeddings(
        self,
        query: Any,
        *,
        ncomp: int | None = None,
        feature_names: Sequence[str] | None = None,
        scale_max: float | None = 10.0,
        chunk_size: int = 8192,
    ) -> np.ndarray:
        """Project query ``data`` layer into PLS space matching Seurat ``ProjectCellEmbeddings(scale=TRUE)``.

        Applies ``min(scale_max, (X_query - feature_mean) / feature_scale) @ projection[:, :ncomp]``
        in streaming row chunks without materializing dense ``N_query x G``.
        """
        if self.projection is None:
            raise ValueError("PLSModel has no projection matrix")
        k = self.n_components if ncomp is None else int(ncomp)
        if not 1 <= k <= self.n_components:
            raise ValueError(f"ncomp must be between 1 and {self.n_components}")

        if self.method == "spls" and self.active_indices is not None:
            proj_full = np.zeros((len(self.feature_names), k), dtype=np.float64)
            proj_full[self.active_indices, :] = self.projection[:, :k]
        else:
            proj_full = self.projection[:, :k]

        sm = self._aligned_streaming(query, feature_names=feature_names)
        sm.chunk_size = chunk_size

        # If no upper clipping is requested, exact sparse matrix-matrix product avoids any dense chunk!
        if scale_max is None:
            inv_scale = 1.0 / self.feature_scale
            w_scaled = inv_scale[:, None] * proj_full
            offset = (self.feature_mean * inv_scale) @ proj_full
            return sm.xm(w_scaled) - offset[None, :]

        # Seurat FastSparseRowScaleWithKnownStats / BPCells::min_by_row:
        # Notice (min(X, mu + 10*sd) - mu)/sd == min(10, (X - mu)/sd), and since X >= 0 in log-normalized data
        # and mu + 10*sd > 0, ONLY non-zero entries in sparse X can ever exceed mu + 10*sd!
        thresh = self.feature_mean + float(scale_max) * self.feature_scale
        inv_scale = 1.0 / self.feature_scale
        w_scaled = inv_scale[:, None] * proj_full
        offset = (self.feature_mean * inv_scale) @ proj_full

        out = np.empty((sm.n_obs, k), dtype=np.float64)
        for start, stop, blk in sm.iter_row_chunks(chunk_size):
            if sp.issparse(blk) and np.all(thresh >= 0.0):
                b_csr = blk.tocsr().copy()
                b_csr.data = np.minimum(b_csr.data, thresh[b_csr.indices])
                out[start:stop] = np.asarray(b_csr @ w_scaled, dtype=np.float64) - offset[None, :]
            else:
                dense_blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk, dtype=np.float64)
                z_blk = (dense_blk - self.feature_mean[None, :]) * inv_scale[None, :]
                np.minimum(z_blk, float(scale_max), out=z_blk)
                out[start:stop] = z_blk @ proj_full
        return out

    def predict(
        self,
        x: Any,
        *,
        ncomp: int | Sequence[int] | None = None,
        apply_training_scale: bool = True,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Predict responses ``Y_hat`` matching ``PredictPLS`` (F08) and v0.1.1 ``PLSModel.predict``."""
        if self.model is not None and self.coefficients is None:
            arr = self._aligned_array(x)
            scaled = (arr - self.feature_mean) / self.feature_scale if apply_training_scale else arr
            return self.model.predict(scaled)

        if self.coefficients is None or self.y_means is None or self.x_means_internal is None:
            raise ValueError("PLSModel does not store regression coefficients (note: spls does not save PredictPLS model in R)")

        max_k = self.coefficients.shape[2]
        single_k = True
        if ncomp is None:
            k_list = [max_k]
        elif isinstance(ncomp, (int, np.integer)):
            k_list = [int(ncomp)]
        else:
            k_list = [int(v) for v in ncomp]
            single_k = False

        if any(k < 1 or k > max_k for k in k_list):
            raise ValueError(f"ncomp values must be between 1 and {max_k}")

        if isinstance(x, (np.ndarray, pd.DataFrame)):
            arr = self._aligned_array(x)
            x_in = (arr - self.feature_mean) / self.feature_scale if apply_training_scale else arr
            preds = []
            for k in k_list:
                b_mat = self.coefficients[:, :, k - 1]
                b0 = self.y_means - self.x_means_internal @ b_mat
                preds.append(x_in @ b_mat + b0[None, :])
            return preds[0] if single_k else np.stack(preds, axis=2)

        sm = self._aligned_streaming(x, feature_names=feature_names)
        inv_scale = (1.0 / self.feature_scale) if apply_training_scale else np.ones_like(self.feature_scale)
        mu_shift = (self.feature_mean * inv_scale) if apply_training_scale else np.zeros_like(self.feature_mean)
        preds = []
        for k in k_list:
            b_mat = self.coefficients[:, :, k - 1]
            b_eff = inv_scale[:, None] * b_mat
            b0_eff = self.y_means - (self.x_means_internal + mu_shift) @ b_mat
            preds.append(sm.xm(b_eff) + b0_eff[None, :])
        return preds[0] if single_k else np.stack(preds, axis=2)

    def save(self, path: str | Path) -> Path:
        """Persist model parameters safely without ``pickle`` (JSON metadata + compressed NPZ arrays)."""
        out_path = Path(path)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "schema_version": "scslide-pls-model-2",
            "method": self.method,
            "n_components": int(self.n_components),
            "feature_names": list(self.feature_names),
            "response_names": list(self.response_names),
            "x_totvar": float(self.x_totvar) if self.x_totvar is not None else None,
            "reference_profile": self.reference_profile,
        }
        arrays: dict[str, np.ndarray] = {
            "meta_json": np.array(json.dumps(meta, ensure_ascii=False)),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float64),
            "feature_scale": np.asarray(self.feature_scale, dtype=np.float64),
        }
        for key, attr in [
            ("coefficients", self.coefficients),
            ("projection", self.projection),
            ("loadings", self.loadings),
            ("loading_weights", self.loading_weights),
            ("x_means_internal", self.x_means_internal),
            ("y_means", self.y_means),
            ("x_var", self.x_var),
            ("r2", self.r2),
            ("rmsep", self.rmsep),
            ("active_indices", self.active_indices),
            ("scores_", self.scores_),
        ]:
            if attr is not None:
                arrays[key] = np.asarray(attr)
        np.savez_compressed(out_path, **arrays)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "PLSModel":
        """Load a :class:`PLSModel` safely with ``allow_pickle=False``."""
        in_path = Path(path)
        if not in_path.exists() and in_path.with_suffix(".npz").exists():
            in_path = in_path.with_suffix(".npz")
        with np.load(in_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"].item()))
            if meta.get("schema_version") != "scslide-pls-model-2":
                raise ValueError(f"Unsupported PLSModel schema: {meta.get('schema_version')!r}")
            opt = lambda k: z[k].copy() if k in z.files else None
            return cls(
                feature_mean=z["feature_mean"].copy(),
                feature_scale=z["feature_scale"].copy(),
                feature_names=[str(f) for f in meta["feature_names"]],
                method=str(meta["method"]),
                n_components=int(meta["n_components"]),
                coefficients=opt("coefficients"),
                projection=opt("projection"),
                loadings=opt("loadings"),
                loading_weights=opt("loading_weights"),
                x_means_internal=opt("x_means_internal"),
                y_means=opt("y_means"),
                response_names=[str(r) for r in meta.get("response_names", [])],
                x_var=opt("x_var"),
                x_totvar=meta.get("x_totvar"),
                r2=opt("r2"),
                rmsep=opt("rmsep"),
                active_indices=opt("active_indices"),
                reference_profile=str(meta.get("reference_profile", "v6")),
                scores_=opt("scores_"),
            )


def fit_pls(
    x: Any,
    y: Any,
    *,
    n_components: int = 2,
    scale: bool = True,
    scale_max: float | None = None,
    method: str = "plsr",
    y_add: Any = None,
    eta: float = 0.5,
    reference_profile: str = "v6",
    clamp_rank: bool = False,
    feature_names: Sequence[str] | None = None,
) -> PLSModel:
    """Fit a native deterministic PLS model (``plsr``/``kernelpls``, ``cppls``, or ``spls``).

    Fixes the v0.1.1 component-count bug (B05): ``n_components`` is bounded by the
    valid predictor/sample rank ``min(n - 1, p)``, NEVER by the number of response
    columns ``q``. Requests exceeding ``min(n - 1, p)`` raise ``ValueError`` unless
    ``clamp_rank=True`` (which emits an explicit ``UserWarning``).
    """
    method_norm = str(method).lower()
    if method_norm in ("plsr", "kernelpls"):
        method_norm = "plsr"
    elif method_norm not in ("cppls", "spls"):
        raise ValueError(f"Unsupported PLS method {method!r}; choose 'plsr', 'cppls', or 'spls'")

    if isinstance(x, pd.DataFrame):
        if x.columns.has_duplicates:
            raise ValueError("training features must have unique names")
        validate_bijective_ids(x.columns, axis_name="training features")
        names = [str(c) for c in x.columns]
    elif feature_names is not None:
        validate_bijective_ids(feature_names, axis_name="feature_names")
        names = [str(f) for f in feature_names]
    else:
        names = [f"feature_{i}" for i in range(x.shape[1])]

    n_req = int(n_components)
    if n_req < 1:
        raise ValueError("n_components must be >= 1")

    sm_raw = StreamingMatrix(x)
    n, p = sm_raw.shape
    if len(names) != p:
        raise ValueError("feature_names length must equal number of columns in x")
    if n < 2 or p < 1:
        raise ValueError("x must have at least 2 observations and 1 feature")

    max_comp = min(n - 1, p) if method_norm != "cppls" else min(n - 1, p - 1 if p > 1 else 1)
    if method_norm == "cppls" and n_req >= min(n, p) and not clamp_rank:
        raise ValueError(f"cppls requires n_components ({n_req}) < min(n, p) = {min(n, p)}")
    if n_req > max_comp:
        if not clamp_rank:
            raise ValueError(f"Requested n_components={n_req} exceeds valid rank bound {max_comp}")
        warnings.warn(
            f"Clamping requested n_components={n_req} to valid rank bound {max_comp}",
            UserWarning,
            stacklevel=2,
        )
        n_comp = max(1, max_comp)
    else:
        n_comp = n_req

    # Streaming path when x is sparse or backed and scale=False (e.g. BPCells IterableMatrix on data layer)
    if not scale and sm_raw._kind in ("scipy_sparse", "backed") and method_norm in ("plsr", "cppls"):
        mean_vec = np.zeros(p, dtype=np.float64)
        scale_vec = np.ones(p, dtype=np.float64)
        if method_norm == "plsr":
            res = kernelpls_ondisk(sm_raw, y, ncomp=n_comp, center=True, stripped=False)
        else:
            res = cppls_ondisk(
                sm_raw, y, ncomp=n_comp, y_add=y_add, center=True, stripped=False, reference_profile=reference_profile
            )
        return PLSModel(
            feature_mean=mean_vec,
            feature_scale=scale_vec,
            feature_names=names,
            method=method_norm,
            n_components=n_comp,
            coefficients=res["coefficients"],
            projection=res["projection"],
            loadings=res["loadings"],
            loading_weights=res["loading_weights"],
            x_means_internal=res["Xmeans"],
            y_means=res["Ymeans"],
            response_names=res["response_names"],
            x_var=res["Xvar"],
            x_totvar=res["Xtotvar"],
            r2=res["R2"],
            rmsep=res["RMSEP"],
            reference_profile=reference_profile,
            scores_=res["scores"],
        )

    # Materialize training matrix (note: in scSLIDE, TRAINING is the small donor-balanced sketch!)
    arr = x.toarray().astype(np.float64) if sp.issparse(x) else (
        x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float64)
    )
    validate_numeric_matrix_values(arr, name="training matrix x")

    if scale:
        mean_vec = arr.mean(axis=0)
        raw_sd = arr.std(axis=0, ddof=1)
        zero_sd = (raw_sd == 0) | ~np.isfinite(raw_sd)
        scale_vec = raw_sd.copy()
        scale_vec[zero_sd] = 1.0
        scaled = (arr - mean_vec[None, :]) / scale_vec[None, :]
        # Match Seurat ScaleData: zero-variance features become all-zero in scale.data, clipped at [-10, 10] if requested
        scaled[:, zero_sd] = 0.0
        if scale_max is not None:
            np.clip(scaled, -float(scale_max), float(scale_max), out=scaled)
    else:
        mean_vec = np.zeros(p, dtype=np.float64)
        scale_vec = np.ones(p, dtype=np.float64)
        scaled = arr

    if method_norm == "plsr":
        res = kernelpls(scaled, y, ncomp=n_comp, center=True, stripped=False)
        return PLSModel(
            feature_mean=mean_vec,
            feature_scale=scale_vec,
            feature_names=names,
            method="plsr",
            n_components=n_comp,
            coefficients=res["coefficients"],
            projection=res["projection"],
            loadings=res["loadings"],
            loading_weights=res["loading_weights"],
            x_means_internal=res["Xmeans"],
            y_means=res["Ymeans"],
            response_names=res["response_names"],
            x_var=res["Xvar"],
            x_totvar=res["Xtotvar"],
            r2=res["R2"],
            rmsep=res["RMSEP"],
            reference_profile=reference_profile,
            scores_=res["scores"],
        )

    if method_norm == "cppls":
        res = cppls_ondisk(
            scaled, y, ncomp=n_comp, y_add=y_add, center=True, stripped=False, reference_profile=reference_profile
        )
        return PLSModel(
            feature_mean=mean_vec,
            feature_scale=scale_vec,
            feature_names=names,
            method="cppls",
            n_components=n_comp,
            coefficients=res["coefficients"],
            projection=res["projection"],
            loadings=res["loadings"],
            loading_weights=res["loading_weights"],
            x_means_internal=res["Xmeans"],
            y_means=res["Ymeans"],
            response_names=res["response_names"],
            x_var=res["Xvar"],
            x_totvar=res["Xtotvar"],
            r2=res["R2"],
            rmsep=res["RMSEP"],
            reference_profile=reference_profile,
            scores_=res["scores"],
        )

    # spls
    res_s = spls_fit(scaled, y, ncomp=n_comp, eta=eta)
    return PLSModel(
        feature_mean=mean_vec,
        feature_scale=scale_vec,
        feature_names=names,
        method="spls",
        n_components=res_s["projection"].shape[1],
        coefficients=None,
        projection=res_s["projection"],
        loadings=res_s["projection"],
        x_means_internal=res_s["Xmeans"],
        y_means=res_s["Ymeans"],
        response_names=res_s["response_names"],
        active_indices=res_s["active_indices_0based"],
        reference_profile=reference_profile,
        scores_=res_s["scores"],
    )


def run_pls(
    adata_or_x: Any,
    y: str | Sequence[str] | Any,
    *,
    features: Sequence[str] | None = None,
    ncomp: int = 20,
    pls_function: str = "cppls",
    y_add: str | Sequence[str] | Any = None,
    eta: float = 0.5,
    layer: str | None = "data",
    scale_data: bool = True,
    scale_max: float | None = 10.0,
    reduction_name: str = "pls",
    save_model: bool = True,
    streaming: bool = False,
    reference_profile: str = "v6",
) -> PLSModel | tuple[Any, PLSModel]:
    """High-level ``RunPLS`` wrapper over ``AnnData`` or matrices (F13)."""
    if hasattr(adata_or_x, "obs") and hasattr(adata_or_x, "var_names"):
        adata = adata_or_x
        validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
        validate_bijective_ids(adata.var_names, axis_name="adata.var_names")
        var_lookup = {str(v): i for i, v in enumerate(adata.var_names)}
        if features is None:
            if "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
                feat_list = [str(v) for v, hv in zip(adata.var_names, adata.var["highly_variable"]) if hv]
            else:
                feat_list = [str(v) for v in adata.var_names]
        else:
            feat_list = [str(f) for f in features if str(f) in var_lookup]
        if len(feat_list) == 0:
            raise ValueError("No valid features found for run_pls")
        col_idx = np.array([var_lookup[f] for f in feat_list], dtype=np.int64)

        raw_mat = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
        if isinstance(y, str):
            y_in = adata.obs[[y]]
        elif isinstance(y, (list, tuple)) and all(isinstance(c, str) and c in adata.obs.columns for c in y):
            y_in = adata.obs[list(y)]
        else:
            y_in = y

        y_add_in = None
        if y_add is not None:
            if isinstance(y_add, str):
                y_add_in = adata.obs[[y_add]]
            elif isinstance(y_add, (list, tuple)) and all(isinstance(c, str) and c in adata.obs.columns for c in y_add):
                y_add_in = adata.obs[list(y_add)]
            else:
                y_add_in = y_add

        if streaming:
            sm_sub = StreamingMatrix(raw_mat, col_indices=col_idx)
            model = fit_pls(
                sm_sub,
                y_in,
                n_components=ncomp,
                scale=False,
                method=pls_function,
                y_add=y_add_in,
                eta=eta,
                reference_profile=reference_profile,
                clamp_rank=True,
                feature_names=feat_list,
            )
        else:
            sub = raw_mat[:, col_idx]
            if sp.issparse(sub):
                sub = sub.toarray()
            elif not isinstance(sub, np.ndarray):
                sub = np.asarray(sub[:])
                if sp.issparse(sub):
                    sub = sub.toarray()
            model = fit_pls(
                sub,
                y_in,
                n_components=ncomp,
                scale=scale_data,
                scale_max=scale_max if scale_data else None,
                method=pls_function,
                y_add=y_add_in,
                eta=eta,
                reference_profile=reference_profile,
                clamp_rank=True,
                feature_names=feat_list,
            )

        adata.obsm[f"X_{reduction_name}"] = model.scores_
        sc_uns = dict(adata.uns.get("scslide", {}))
        pls_reg = dict(sc_uns.get("pls", {}))
        pls_reg[reduction_name] = {
            "method": model.method,
            "n_components": int(model.n_components),
            "feature_names": list(model.feature_names),
            "response_names": list(model.response_names),
            "reference_profile": reference_profile,
            "saved_model": bool(save_model and model.coefficients is not None),
        }
        sc_uns["pls"] = pls_reg
        adata.uns["scslide"] = sc_uns
        return adata, model

    return fit_pls(
        adata_or_x,
        y,
        n_components=ncomp,
        scale=scale_data,
        scale_max=scale_max if scale_data else None,
        method=pls_function,
        y_add=y_add,
        eta=eta,
        reference_profile=reference_profile,
        clamp_rank=True,
        feature_names=features,
    )


def predict_pls(
    model: PLSModel,
    newdata: Any,
    *,
    ncomp: int | Sequence[int] | None = None,
    apply_training_scale: bool = True,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Functional ``PredictPLS`` entry point (F08)."""
    return model.predict(
        newdata,
        ncomp=ncomp,
        apply_training_scale=apply_training_scale,
        feature_names=feature_names,
    )


def project_pls_embeddings(
    model: PLSModel,
    query: Any,
    *,
    ncomp: int | None = None,
    feature_names: Sequence[str] | None = None,
    scale_max: float | None = 10.0,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Functional Seurat ``ProjectCellEmbeddings`` entry point for PLS reductions (F08/F09)."""
    return model.project_embeddings(
        query,
        ncomp=ncomp,
        feature_names=feature_names,
        scale_max=scale_max,
        chunk_size=chunk_size,
    )
