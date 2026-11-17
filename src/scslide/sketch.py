"""Reproducible ungrouped and grouped sketching utilities (F15 ``SketchDataByGroup``).

Supports both simple in-memory array sketching (preserving v0.1.1 APIs) and
Seurat-compatible grouped / leverage-score sketching over sparse CSR/CSC and
backed HDF5/Zarr matrices without dense ``N x G`` materialization.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ._validation import StreamingMatrix, validate_bijective_ids


def uniform_sketch(n: int, size: int, *, seed: int = 123) -> np.ndarray:
    """Sample cell indices without replacement using a fixed RNG seed."""
    if n < 1 or size < 1:
        raise ValueError("n and size must be positive")
    if size > n:
        raise ValueError("size cannot exceed n")
    return np.random.default_rng(seed).choice(n, size=size, replace=False)


def leverage_scores(matrix: Any, *, rank: int | None = None, center: bool = True) -> np.ndarray:
    """Compute normalized row leverage scores for a 2-D cells x features matrix.

    For dense inputs with ``center=True`` (default), preserves the exact v0.1.1
    centered thin-SVD definition. For sparse or backed inputs when ``center=False``,
    uses sparse/streaming SVD without densifying ``matrix``.
    """
    if sp.issparse(matrix) and not center:
        n_obs, n_vars = matrix.shape
        if n_obs == 0 or n_vars == 0:
            raise ValueError("matrix must be non-empty")
        max_k = min(n_obs, n_vars) - 1
        if max_k < 1:
            return np.full(n_obs, 1.0 / n_obs, dtype=np.float64)
        k = max_k if rank is None else max(1, min(int(rank), max_k))
        u, _, _ = svds(matrix.astype(np.float64), k=k)
        scores = np.sum(u * u, axis=1)
        tot = float(scores.sum())
        return scores / tot if tot > 0 else np.full(n_obs, 1.0 / n_obs, dtype=np.float64)

    x = matrix.toarray().astype(np.float64) if sp.issparse(matrix) else np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("matrix must be a non-empty 2-D array")
    rank = min(x.shape) if rank is None else max(1, min(int(rank), min(x.shape)))
    x_use = x - x.mean(axis=0, keepdims=True) if center else x
    u = np.linalg.svd(x_use, full_matrices=False)[0][:, :rank]
    scores = np.sum(u * u, axis=1)
    tot = float(scores.sum())
    if tot <= 0:
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
    return scores / tot


def leverage_sketch(
    matrix: Any,
    size: int,
    *,
    rank: int | None = None,
    seed: int = 123,
    center: bool | None = None,
) -> np.ndarray:
    """Sample row indices without replacement proportional to leverage score."""
    if center is None:
        center = not sp.issparse(matrix)
    scores = leverage_scores(matrix, rank=rank, center=center)
    if size < 1 or size > len(scores):
        raise ValueError("size must be between 1 and number of rows")
    return np.random.default_rng(seed).choice(len(scores), size=size, replace=False, p=scores)


def seurat_leverage_scores(
    matrix: Any,
    *,
    nsketch: int = 5000,
    ndims: int | None = None,
    eps: float = 0.5,
    seed: int = 123,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Compute Seurat v5/v6 ``LeverageScore`` on a ``cells x features`` matrix (uncentered ``data`` layer).

    Matches ``Seurat:::LeverageScore.default`` & ``LeverageScore.StdAssay``:
    - When ``n_cells < 1.5 * nsketch``: computes top ``nv = min(50, p - 1, n - 1)``
      left singular vectors of uncentered ``cells x features`` via sparse ``svds``
      (equivalent to right singular vectors of ``features x cells`` in ``irlba``),
      returning ``rowSums(U^2)``.
    - When ``n_cells >= 1.5 * nsketch``: builds a sparse ``CountSketch`` ``S``
      (``nsketch x n_cells``), computes ``S @ A`` (``nsketch x p``), extracts upper
      triangular ``R`` from QR decomposition (automatically dropping zero-variance
      / rank-deficient feature columns just like ``LeverageScore.StdAssay``), forms
      ``R^{-1} @ JL`` with Johnson-Lindenstrauss sparse projection ``JL``, and
      accumulates row squared norms in chunks without materializing dense ``N x G``.
    """
    sm = matrix if isinstance(matrix, StreamingMatrix) else StreamingMatrix(matrix, chunk_size=chunk_size)
    n_cells, n_features = sm.shape
    if n_cells < 2 or n_features < 2:
        raise ValueError(
            f"Cannot compute leverage scores from {n_features} features and {n_cells} cells: at least 2 of each needed"
        )

    # Filter zero-variance / all-zero features proactively (matching LeverageScore.StdAssay retry)
    _, feat_var = sm.mean_and_var(ddof=1)
    keep_feat = np.flatnonzero(feat_var > 0)
    if len(keep_feat) < 2:
        return np.ones(n_cells, dtype=np.float64)
    if len(keep_feat) < n_features:
        col_idx = keep_feat if sm.col_indices is None else sm.col_indices[keep_feat]
        sm = StreamingMatrix(sm.source, row_indices=sm.row_indices, col_indices=col_idx, chunk_size=chunk_size)
        n_cells, n_features = sm.shape

    if n_cells < int(1.5 * nsketch):
        nv = min(50, n_features - 1, n_cells - 1)
        if nv < 1:
            return np.ones(n_cells, dtype=np.float64)
        if sm._kind == "scipy_sparse":
            rng = np.random.default_rng(seed)
            v0 = rng.normal(size=min(n_cells, n_features))
            u, _, _ = svds(sm._csr.astype(np.float64), k=nv, v0=v0)
            scores = np.sum(u * u, axis=1)
        elif sm._kind in ("ndarray", "dataframe"):
            blocks = [blk for _, _, blk in sm.iter_row_chunks()]
            dense = np.vstack(blocks)
            u, _, _ = np.linalg.svd(dense, full_matrices=False)
            scores = np.sum(u[:, :nv] ** 2, axis=1)
        else:
            op = sp.linalg.LinearOperator(
                shape=(n_cells, n_features),
                matvec=sm.xm,
                rmatvec=sm.xtm,
                dtype=np.float64,
            )
            rng = np.random.default_rng(seed)
            v0 = rng.normal(size=min(n_cells, n_features))
            u, _, _ = svds(op, k=nv, v0=v0)
            scores = np.sum(u * u, axis=1)
        scores = np.maximum(scores, 1e-15)
        return scores

    # CountSketch + QR + JLEmbed streaming path for large N
    ndims_eff = n_cells if ndims is None else int(ndims)
    nsketch_eff = int(nsketch)
    if nsketch_eff < int(math.ceil(1.1 * n_features)):
        nsketch_eff = min(int(math.ceil(1.1 * n_features)), ndims_eff)
        warnings.warn(
            f"'nsketch' is close to feature count ({n_features}); adjusted to {nsketch_eff}",
            UserWarning,
            stacklevel=2,
        )
    nsketch_eff = min(nsketch_eff, ndims_eff)

    rng = np.random.default_rng(seed)
    sketch_rows = rng.integers(0, nsketch_eff, size=n_cells)
    sketch_signs = rng.choice(np.array([-1.0, 1.0]), size=n_cells)

    # Accumulate SA = S @ A of shape (nsketch_eff, n_features) in streaming chunks
    sa = np.zeros((nsketch_eff, n_features), dtype=np.float64)
    for start, stop, blk in sm.iter_row_chunks():
        s_sub = sp.csr_matrix(
            (sketch_signs[start:stop], (sketch_rows[start:stop], np.arange(stop - start))),
            shape=(nsketch_eff, stop - start),
        )
        prod = s_sub @ blk
        sa += prod.toarray() if sp.issparse(prod) else np.asarray(prod, dtype=np.float64)

    # Pivoted QR handles any residual collinearity safely
    _, r_mat, piv = la.qr(sa, mode="economic", pivoting=True)
    diag_abs = np.abs(np.diag(r_mat))
    tol = np.finfo(np.float64).eps * max(sa.shape) * (diag_abs[0] if diag_abs.size > 0 else 1.0)
    eff_rank = int(np.sum(diag_abs > tol))
    if eff_rank < 1:
        return np.ones(n_cells, dtype=np.float64)

    r_valid = r_mat[:eff_rank, :eff_rank]
    r_inv_trunc = la.solve_triangular(r_valid, np.eye(eff_rank, dtype=np.float64), lower=False)
    r_inv = np.zeros((n_features, eff_rank), dtype=np.float64)
    r_inv[piv[:eff_rank], :] = r_inv_trunc

    denom = (eps ** 2) / 2.0 - (eps ** 3) / 3.0
    d_jl = max(16, int(math.floor(4.0 * math.log(max(ndims_eff, 2)) / denom)))
    s_sparsity = max(1, int(math.ceil(math.sqrt(d_jl))))
    probs = [1.0 / (2.0 * s_sparsity), 1.0 - 1.0 / s_sparsity, 1.0 / (2.0 * s_sparsity)]
    jl = rng.choice(np.array([-1.0, 0.0, 1.0]), size=(eff_rank, d_jl), p=probs)
    rhs = r_inv @ jl  # (n_features, d_jl) dense small matrix

    z_scores = np.zeros(n_cells, dtype=np.float64)
    for start, stop, blk in sm.iter_row_chunks():
        proj = blk @ rhs
        proj_arr = np.asarray(proj, dtype=np.float64)
        z_scores[start:stop] = np.einsum("ij,ij->i", proj_arr, proj_arr)

    return np.maximum(z_scores, 1e-15)


def sketch_data_by_group(
    adata_or_matrix: Any,
    *,
    group_by: str | Sequence[Any] | None = None,
    ncells: int | Sequence[int] | Mapping[str, int] = 1000,
    method: str = "Uniform",
    features: Sequence[str] | None = None,
    layer: str | None = "data",
    cells: Sequence[str] | None = None,
    min_cells_per_group: int = 10,
    var_name: str = "leverage.score",
    sketched_name: str = "sketch",
    seed: int = 123,
    reference_profile: str = "v6",
    return_anndata: bool = False,
) -> dict[str, Any] | Any:
    """Native Python implementation of ``SketchDataByGroup`` (F15) and ``SketchData``.

    Returns a dictionary ``{"cell_ids": list[str], "indices": np.ndarray, "leverage_scores": pd.Series, "group_counts": dict}``
    (and registers ``adata.uns['scslide']['sketches'][sketched_name]`` + ``adata.obs[var_name]`` when given an ``AnnData``),
    or returns a sketched ``AnnData`` subset when ``return_anndata=True``.
    """
    method_norm = str(method)
    if method_norm not in ("Uniform", "LeverageScore"):
        raise ValueError("method must be 'Uniform' or 'LeverageScore'")
    if reference_profile not in ("v6", "legacy"):
        raise ValueError("reference_profile must be 'v6' or 'legacy'")

    is_adata = hasattr(adata_or_matrix, "obs") and hasattr(adata_or_matrix, "var_names")
    if is_adata:
        adata = adata_or_matrix
        cell_ids_arr, _ = validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
        cell_ids = [str(c) for c in cell_ids_arr]
        if layer is not None and layer in adata.layers:
            raw_mat = adata.layers[layer]
        elif layer in (None, "X", "data"):
            raw_mat = adata.X
        else:
            raise ValueError(f"Layer {layer!r} not found in AnnData")

        col_idx = None
        if features is not None:
            var_lookup = {str(v): i for i, v in enumerate(adata.var_names)}
            present = [var_lookup[str(f)] for f in features if str(f) in var_lookup]
            if len(present) == 0:
                raise ValueError("None of the requested sketch features are present in AnnData.var_names")
            col_idx = np.asarray(present, dtype=np.int64)
        sm = StreamingMatrix(raw_mat, col_indices=col_idx)

        if isinstance(group_by, str):
            if group_by not in adata.obs.columns:
                raise ValueError(f"Column {group_by!r} not found in adata.obs")
            group_series = adata.obs[group_by]
        elif group_by is not None:
            group_series = pd.Series(list(group_by), index=cell_ids)
        else:
            group_series = None
    else:
        sm = StreamingMatrix(adata_or_matrix)
        cell_ids = (
            [str(x) for x in adata_or_matrix.index]
            if isinstance(adata_or_matrix, pd.DataFrame)
            else [f"cell_{i}" for i in range(sm.n_obs)]
        )
        validate_bijective_ids(cell_ids, axis_name="matrix row IDs")
        group_series = pd.Series(list(group_by), index=cell_ids) if group_by is not None else None

    id_to_pos = {cid: i for i, cid in enumerate(cell_ids)}

    # Pre-defined cell list override (supported in legacy SketchDataByGroup and SketchData)
    if cells is not None:
        if reference_profile == "v6" and group_series is not None:
            warnings.warn(
                "Seurat v6 removed 'cells' override on SketchDataByGroup; honoring explicit cell list under native contract.",
                UserWarning,
                stacklevel=2,
            )
        chosen_ids = [str(c) for c in cells if str(c) in id_to_pos]
        if len(chosen_ids) == 0:
            raise ValueError("None of the user-supplied 'cells' were found in the dataset")
        chosen_idx = np.array([id_to_pos[c] for c in chosen_ids], dtype=np.int64)
        lev_series = pd.Series(1.0, index=cell_ids, name=var_name)
        if is_adata:
            _register_sketch_on_adata(adata_or_matrix, sketched_name, chosen_ids, var_name, lev_series)
            if return_anndata:
                return adata_or_matrix[chosen_idx].copy()
        return {
            "cell_ids": chosen_ids,
            "indices": chosen_idx,
            "leverage_scores": lev_series,
            "group_counts": {"predefined": len(chosen_ids)},
        }

    if method_norm == "LeverageScore":
        lev_vals = seurat_leverage_scores(sm, seed=seed)
    else:
        lev_vals = np.ones(sm.n_obs, dtype=np.float64)
    lev_series = pd.Series(lev_vals, index=cell_ids, name=var_name)

    # Ungrouped SketchData fallback
    if group_series is None:
        if not isinstance(ncells, (int, np.integer)) or int(ncells) < 1:
            raise ValueError("Ungrouped sketching requires a positive integer ncells")
        target_n = min(int(ncells), sm.n_obs)
        if sm.n_obs <= target_n:
            chosen_idx = np.arange(sm.n_obs, dtype=np.int64)
        else:
            probs = lev_vals / float(lev_vals.sum())
            chosen_idx = np.random.default_rng(seed).choice(sm.n_obs, size=target_n, replace=False, p=probs)
        chosen_ids = [cell_ids[i] for i in chosen_idx]
        if is_adata:
            _register_sketch_on_adata(adata_or_matrix, sketched_name, chosen_ids, var_name, lev_series)
            if return_anndata:
                return adata_or_matrix[chosen_idx].copy()
        return {
            "cell_ids": chosen_ids,
            "indices": chosen_idx,
            "leverage_scores": lev_series,
            "group_counts": {"all": len(chosen_ids)},
        }

    # Grouped SketchDataByGroup
    valid_mask = group_series.notna().to_numpy()
    groups: list[str] = []
    seen_g: set[str] = set()
    for val in group_series.to_numpy()[valid_mask]:
        g_str = str(val)
        if g_str not in seen_g:
            seen_g.add(g_str)
            groups.append(g_str)

    if len(groups) == 0:
        raise ValueError("No non-NA groups found in group_by")

    if isinstance(ncells, (int, np.integer)):
        if int(ncells) < 1:
            raise ValueError("ncells must be a positive integer")
        ncells_map = {g: int(ncells) for g in groups}
    elif isinstance(ncells, Mapping):
        raw_map = {str(k): int(v) for k, v in ncells.items()}
        if len(raw_map) == 0 or any(v < 1 for v in raw_map.values()):
            raise ValueError("ncells mapping must contain positive integers")
        default_n = next(iter(raw_map.values()))
        ncells_map = {g: raw_map.get(g, default_n) for g in groups}
    else:
        seq = [int(x) for x in ncells]
        if len(seq) != len(groups) or any(v < 1 for v in seq):
            raise ValueError("Unnamed ncells sequence must match number of groups and contain positive integers")
        ncells_map = dict(zip(groups, seq))

    g_str_all = np.array([str(v) if m else "" for v, m in zip(group_series.to_numpy(), valid_mask)], dtype=object)
    sampled_indices: list[np.ndarray] = []
    group_counts: dict[str, int] = {}

    for g_idx, grp in enumerate(groups, start=1):
        grp_pos = np.flatnonzero(valid_mask & (g_str_all == grp))
        if grp_pos.size == 0:
            continue
        if grp_pos.size < min_cells_per_group or grp_pos.size <= ncells_map[grp]:
            keep_pos = grp_pos
        else:
            q = ncells_map[grp]
            w = lev_vals[grp_pos]
            probs = w / float(w.sum())
            # Match R deterministic per-group seed offset (seed + which(groups == group) + 1)
            rng_grp = np.random.default_rng(int(seed) + g_idx + 1)
            keep_pos = rng_grp.choice(grp_pos, size=q, replace=False, p=probs)
        sampled_indices.append(keep_pos)
        group_counts[grp] = int(len(keep_pos))

    if not sampled_indices:
        raise ValueError("No cells were sampled; check group_by and ncells parameters")

    all_idx = np.concatenate(sampled_indices).astype(np.int64)
    chosen_ids = [cell_ids[i] for i in all_idx]
    if is_adata:
        _register_sketch_on_adata(adata_or_matrix, sketched_name, chosen_ids, var_name, lev_series)
        if return_anndata:
            return adata_or_matrix[all_idx].copy()

    return {
        "cell_ids": chosen_ids,
        "indices": all_idx,
        "leverage_scores": lev_series,
        "group_counts": group_counts,
    }


def _register_sketch_on_adata(
    adata: Any,
    sketched_name: str,
    chosen_ids: list[str],
    var_name: str,
    lev_series: pd.Series,
) -> None:
    if not getattr(adata, "isbacked", False):
        adata.obs[var_name] = lev_series.reindex(adata.obs_names.astype(str)).to_numpy(dtype=np.float64)
    sc_uns = dict(adata.uns.get("scslide", {}))
    sketches = dict(sc_uns.get("sketches", {}))
    sketches[str(sketched_name)] = {
        "cell_ids": list(chosen_ids),
        "n_cells": int(len(chosen_ids)),
    }
    sc_uns["sketches"] = sketches
    adata.uns["scslide"] = sc_uns
