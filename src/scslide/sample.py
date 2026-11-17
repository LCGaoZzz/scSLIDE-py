"""Sample-level scSLIDE operations, structured Pearson operators, and sample-object builders.

Mirrors both the pinned legacy ``scSLIDE`` package (``968dbde2``) and ``Seurat`` v6.0.0:
landmark-by-cell neighbour counts are collapsed by sample, all landmark query cells are
excluded before aggregation, zero-query samples remain explicit zero columns, and
Chi-squared normalization uses Pearson residuals from the row/column independence model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable, Sequence
import re
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ._validation import validate_bijective_ids, validate_numeric_matrix_values, validate_open_backed_anndata


def landmark_matrix(
    neighbor_index: np.ndarray,
    sample_for_cell: Iterable[Any],
    landmark_ids: Iterable[Any],
    *,
    k: int = 5,
    cell_ids: Iterable[Any] | None = None,
    cells_use: Iterable[Any] | None = None,
    sample_order: Sequence[Any] | None = None,
    allow_unexcluded_queries: bool = False,
) -> pd.DataFrame:
    """Aggregate cell-to-landmark neighbours into a landmark x sample count matrix.

    Parameters
    ----------
    neighbor_index
        One-based ``(n_query, stored_k)`` integer array matching Seurat ``Neighbor@nn.idx``.
    sample_for_cell
        Sample/donor label for each row of ``neighbor_index``.
    landmark_ids
        Ordered landmark cell identifiers corresponding to ``1..n_landmarks``.
    k
        Number of nearest landmarks per query cell to aggregate. Repeated landmark
        indices in a row are counted with multiplicity, matching ``sparseMatrix(..., x=1)``.
    cell_ids
        Explicit query cell identifiers. Required whenever ``landmark_ids`` are strings
        so that landmark query self-exclusion is unambiguous (unless
        ``allow_unexcluded_queries=True`` is explicitly passed).
    cells_use
        Optional subset of query cell IDs to retain before aggregation.
    sample_order
        Optional explicit sample ordering (e.g. factor levels), preserving zero-query
        samples as explicit zero columns.
    allow_unexcluded_queries
        Only when ``True`` may ``cell_ids`` be omitted alongside non-integer ``landmark_ids``.
    """
    nn = np.asarray(neighbor_index)
    if nn.ndim != 2 or not np.issubdtype(nn.dtype, np.integer) or np.issubdtype(nn.dtype, np.bool_):
        raise ValueError("neighbor_index must be a 2-D integer array")
    n_cells, stored_k = nn.shape
    if n_cells == 0:
        raise ValueError("neighbor_index cannot have zero rows")
    if not 1 <= k <= stored_k:
        raise ValueError(f"k ({k}) must be between 1 and stored neighbor count ({stored_k})")

    raw_landmarks = list(landmark_ids)
    if len(raw_landmarks) == 0:
        raise ValueError("landmark_ids cannot be empty")
    lm_str, _ = validate_bijective_ids(raw_landmarks, axis_name="landmark_ids")
    n_landmarks = len(lm_str)

    nn_k = np.asarray(nn[:, :k], dtype=np.int64)
    if np.any(nn_k < 1) or np.any(nn_k > n_landmarks):
        raise ValueError(f"neighbor indices must be one-based landmark IDs in 1..{n_landmarks}")

    raw_samples = list(sample_for_cell)
    if len(raw_samples) != n_cells:
        raise ValueError("sample_for_cell length must equal neighbor_index rows")

    # Handle NA sample entries identically to GenerateSampleObject (excluded from grouping)
    valid_sample_mask = np.array(
        [not (s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s)) for s in raw_samples],
        dtype=bool,
    )
    if not np.any(valid_sample_mask):
        raise ValueError("sample_for_cell contains no non-missing sample labels")

    # Validate no stringification collisions among distinct sample values
    unique_raw_samples: list[Any] = []
    seen_typed: set[tuple[type, Any]] = set()
    for s in np.asarray(raw_samples, dtype=object)[valid_sample_mask]:
        py_s = s.item() if isinstance(s, np.generic) else s
        tk = (type(py_s), py_s)
        if tk not in seen_typed:
            seen_typed.add(tk)
            unique_raw_samples.append(py_s)
    validate_bijective_ids(unique_raw_samples, axis_name="sample_for_cell")

    if sample_order is not None:
        sample_ids, _ = validate_bijective_ids(sample_order, axis_name="sample_order")
    else:
        sample_ids = np.array([str(s) for s in unique_raw_samples], dtype=object)

    if cell_ids is None:
        all_int_landmarks = all(
            isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
            for x in raw_landmarks
        )
        if not all_int_landmarks and not allow_unexcluded_queries:
            raise ValueError(
                "cell_ids must be provided when landmark_ids are non-integer strings; "
                "omitting query cell_ids would silently disable landmark self-exclusion. "
                "Pass cell_ids explicitly (or allow_unexcluded_queries=True if queries are disjoint)."
            )
        cells_str = np.array([str(i) for i in range(n_cells)], dtype=object)
    else:
        raw_cells = list(cell_ids)
        if len(raw_cells) != n_cells:
            raise ValueError("cell_ids length must equal neighbor_index rows")
        cells_str, _ = validate_bijective_ids(raw_cells, axis_name="cell_ids")

    keep = valid_sample_mask.copy()
    if cells_use is not None:
        use_list = [str(u.item() if isinstance(u, np.generic) else u) for u in cells_use]
        in_use = np.isin(cells_str, np.asarray(use_list, dtype=object))
        if not np.any(in_use):
            raise ValueError("None of the cells in 'cells_use' were found in query cell_ids")
        keep &= in_use

    # Exclude ALL query cells whose ID belongs to landmark_ids (not merely diagonal self-edges)
    keep &= ~np.isin(cells_str, lm_str)

    sample_to_col = {str(s): idx for idx, s in enumerate(sample_ids)}
    sample_str_arr = np.array([str(s) if m else "" for s, m in zip(raw_samples, valid_sample_mask)], dtype=object)
    in_sample_order = np.array([s in sample_to_col for s in sample_str_arr], dtype=bool)
    keep &= in_sample_order

    out = np.zeros((n_landmarks, len(sample_ids)), dtype=np.int64)
    kept_queries = np.flatnonzero(keep)
    if kept_queries.size > 0:
        col_idx = np.fromiter((sample_to_col[sample_str_arr[q]] for q in kept_queries), dtype=np.int64, count=kept_queries.size)
        lm_zero = (nn_k[kept_queries, :] - 1).reshape(-1)
        sm_rep = np.repeat(col_idx, k)
        flat_idx = lm_zero * len(sample_ids) + sm_rep
        binc = np.bincount(flat_idx, minlength=n_landmarks * len(sample_ids))
        out = binc.reshape((n_landmarks, len(sample_ids)))

    return pd.DataFrame(out, index=pd.Index(lm_str.astype(str)), columns=pd.Index(sample_ids.astype(str)))


def expected_counts(counts: np.ndarray | sp.spmatrix | sp.sparray | pd.DataFrame) -> pd.DataFrame | np.ndarray:
    """Expected counts ``E = outer(row_sums, col_sums) / total`` under row/column independence."""
    is_frame = isinstance(counts, pd.DataFrame)
    mat = counts.to_numpy(dtype=np.float64) if is_frame else counts
    validate_numeric_matrix_values(mat, name="counts", non_negative=True, require_integer=False)
    if sp.issparse(mat):
        r = np.asarray(mat.sum(axis=1), dtype=np.float64).ravel()
        c = np.asarray(mat.sum(axis=0), dtype=np.float64).ravel()
    else:
        arr = np.asarray(mat, dtype=np.float64)
        r = arr.sum(axis=1)
        c = arr.sum(axis=0)
    total = float(r.sum())
    if total <= 0:
        raise ValueError("counts must have a positive total")
    exp = np.outer(r, c) / total
    if is_frame:
        return pd.DataFrame(exp, index=counts.index, columns=counts.columns)
    return exp


class PearsonResidualOperator(LinearOperator):
    """Exact structured sparse + rank-one linear operator for Pearson residuals.

    On positive margins ``(r_l > 0, c_s > 0)``,
    ``R = sqrt(T) * D_r^{-1/2} C D_c^{-1/2} - (sqrt(r) @ sqrt(c).T) / sqrt(T)``,
    and zero-margin rows/columns evaluate identically to ``0``.
    Supports ``O(nnz(C) + L + S)`` matrix-vector products without materializing dense ``L x S``.
    """

    def __init__(
        self,
        counts: np.ndarray | sp.spmatrix | sp.sparray | pd.DataFrame,
        *,
        center_rows: bool = False,
        require_integer_counts: bool = False,
    ) -> None:
        if isinstance(counts, pd.DataFrame):
            self.index = counts.index
            self.columns = counts.columns
            mat = sp.csr_matrix(counts.to_numpy(dtype=np.float64))
        elif sp.issparse(counts):
            self.index = None
            self.columns = None
            mat = counts.tocsr().astype(np.float64)
        else:
            self.index = None
            self.columns = None
            mat = sp.csr_matrix(np.asarray(counts, dtype=np.float64))

        validate_numeric_matrix_values(
            mat, name="counts", non_negative=True, require_integer=require_integer_counts
        )
        self._csr = mat
        self.row_sums = np.asarray(mat.sum(axis=1), dtype=np.float64).ravel()
        self.col_sums = np.asarray(mat.sum(axis=0), dtype=np.float64).ravel()
        self.total = float(self.row_sums.sum())
        if self.total <= 0:
            raise ValueError("counts must have a positive total")

        self.pos_row = self.row_sums > 0
        self.pos_col = self.col_sums > 0
        self.inv_sqrt_r = np.zeros_like(self.row_sums)
        self.inv_sqrt_r[self.pos_row] = 1.0 / np.sqrt(self.row_sums[self.pos_row])
        self.inv_sqrt_c = np.zeros_like(self.col_sums)
        self.inv_sqrt_c[self.pos_col] = 1.0 / np.sqrt(self.col_sums[self.pos_col])
        self.sqrt_r = np.sqrt(np.maximum(self.row_sums, 0.0))
        self.sqrt_c = np.sqrt(np.maximum(self.col_sums, 0.0))
        self.sqrt_t = np.sqrt(self.total)
        self.center_rows = bool(center_rows)
        super().__init__(dtype=np.float64, shape=mat.shape)

        self._row_means: np.ndarray | None = None
        if self.center_rows:
            ones = np.ones(self.shape[1], dtype=np.float64)
            self._row_means = self._raw_matvec(ones) / float(self.shape[1])

    def _raw_matvec(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).ravel()
        v_masked = np.where(self.pos_col, v, 0.0)
        term1 = self.sqrt_t * self.inv_sqrt_r * (self._csr @ (self.inv_sqrt_c * v_masked))
        term2 = (self.sqrt_r / self.sqrt_t) * float(np.dot(self.sqrt_c, v_masked))
        out = term1 - term2
        out[~self.pos_row] = 0.0
        return out

    def _raw_rmatvec(self, y: np.ndarray) -> np.ndarray:
        u = np.asarray(y, dtype=np.float64).ravel()
        u_masked = np.where(self.pos_row, u, 0.0)
        term1 = self.sqrt_t * self.inv_sqrt_c * (self._csr.T @ (self.inv_sqrt_r * u_masked))
        term2 = (self.sqrt_c / self.sqrt_t) * float(np.dot(self.sqrt_r, u_masked))
        out = term1 - term2
        out[~self.pos_col] = 0.0
        return out

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).ravel()
        out = self._raw_matvec(v)
        if self.center_rows and self._row_means is not None:
            out = out - self._row_means * float(v.sum())
        return out

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        u = np.asarray(y, dtype=np.float64).ravel()
        out = self._raw_rmatvec(u)
        if self.center_rows and self._row_means is not None:
            out = out - float(np.dot(self._row_means, u))
        return out

    def to_dense(self) -> pd.DataFrame | np.ndarray:
        c_dense = self._csr.toarray()
        exp = np.outer(self.row_sums, self.col_sums) / self.total
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (c_dense - exp) / np.sqrt(exp)
        res[~np.isfinite(res)] = 0.0
        if self.center_rows:
            res = res - res.mean(axis=1, keepdims=True)
        if self.index is not None and self.columns is not None:
            return pd.DataFrame(res, index=self.index, columns=self.columns)
        return res


def pearson_residuals(
    counts: np.ndarray | sp.spmatrix | sp.sparray | pd.DataFrame,
    *,
    require_integer_counts: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Return Pearson residuals ``(C - E) / sqrt(E)`` for a landmark x sample count matrix.

    Undefined cells (zero expected count ``E == 0``) are set to ``0.0``, matching R
    ``NormalizeChiSquared``. Negative, NaN, or infinite inputs raise ``ValueError``.
    """
    is_frame = isinstance(counts, pd.DataFrame)
    mat = counts.to_numpy(dtype=np.float64) if is_frame else counts
    validate_numeric_matrix_values(
        mat, name="counts", non_negative=True, require_integer=require_integer_counts
    )
    arr = mat.toarray().astype(np.float64) if sp.issparse(mat) else np.asarray(mat, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("counts must have a positive total")
    expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = (arr - expected) / np.sqrt(expected)
    residual[~np.isfinite(residual)] = 0.0
    if is_frame:
        return pd.DataFrame(residual, index=counts.index, columns=counts.columns)
    return residual


def normalize_chi_squared(
    object_or_counts: Any,
    *,
    layer: str = "counts",
    save_layer: str = "residuals",
    reference_profile: str = "v6",
    return_operator: bool = False,
) -> Any:
    """Apply ``NormalizeChiSquared`` to a matrix, ``SampleLevelObject``, or sample-level ``AnnData``.

    Under ``reference_profile='v6'``, non-integer counts are rejected just like
    ``Seurat::NormalizeChiSquared.default``; under ``reference_profile='legacy'``,
    any finite non-negative matrix with positive sum is accepted.
    """
    if reference_profile not in ("v6", "legacy"):
        raise ValueError("reference_profile must be 'v6' or 'legacy'")
    req_int = reference_profile == "v6"

    if isinstance(object_or_counts, SampleLevelObject):
        if save_layer.startswith("counts"):
            raise ValueError("save_layer cannot overwrite a counts layer")
        res = pearson_residuals(object_or_counts.counts, require_integer_counts=req_int)
        attrs = dict(object_or_counts.attrs)
        attrs["normalization"] = "chi_squared"
        attrs["reference_profile"] = reference_profile
        return SampleLevelObject(
            counts=object_or_counts.counts.copy(),
            normalized=object_or_counts.normalized.copy() if object_or_counts.normalized is not None else None,
            residuals=res,
            metadata=object_or_counts.metadata.copy() if object_or_counts.metadata is not None else None,
            embeddings={k: v.copy() for k, v in object_or_counts.embeddings.items()},
            attrs=attrs,
        )

    # Sample-level AnnData (obs = sample, var = landmark)
    if hasattr(object_or_counts, "obs_names") and hasattr(object_or_counts, "var_names") and hasattr(object_or_counts, "layers"):
        adata = object_or_counts
        validate_open_backed_anndata(adata, caller="normalize_chi_squared")
        if save_layer.startswith("counts"):
            raise ValueError("save_layer cannot start with 'counts'")
        src = adata.layers[layer] if layer in adata.layers else (adata.X if layer in (None, "X") else None)
        if src is None:
            raise ValueError(f"Layer {layer!r} not found in AnnData")
        # AnnData is sample x landmark; transpose to landmark x sample, compute, transpose back
        c_lm_by_s = src.T
        if return_operator:
            return PearsonResidualOperator(c_lm_by_s, require_integer_counts=req_int)
        res_lm_by_s = pearson_residuals(c_lm_by_s, require_integer_counts=req_int)
        adata.layers[save_layer] = np.asarray(res_lm_by_s, dtype=np.float64).T
        meta = dict(adata.uns.get("scslide", {}))
        meta["normalization"] = "chi_squared"
        meta["residuals_layer"] = save_layer
        meta["reference_profile"] = reference_profile
        adata.uns["scslide"] = meta
        return adata

    if return_operator:
        return PearsonResidualOperator(object_or_counts, require_integer_counts=req_int)
    return pearson_residuals(object_or_counts, require_integer_counts=req_int)


def log_normalize(
    counts: np.ndarray | sp.spmatrix | sp.sparray | pd.DataFrame,
    scale_factor: float = 10_000.0,
) -> pd.DataFrame | np.ndarray | sp.csr_matrix:
    """Column-wise library-size normalization followed by ``log1p`` (landmark x sample orientation)."""
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    is_frame = isinstance(counts, pd.DataFrame)
    mat = counts.to_numpy(dtype=np.float64) if is_frame else counts
    validate_numeric_matrix_values(mat, name="counts", non_negative=True, require_integer=False)

    if sp.issparse(mat):
        csc = mat.tocsc().astype(np.float64)
        totals = np.asarray(csc.sum(axis=0), dtype=np.float64).ravel()
        inv = np.where(totals > 0, scale_factor / np.maximum(totals, 1e-300), 0.0)
        scaled = csc @ sp.diags(inv)
        scaled = scaled.tocsr()
        scaled.data = np.log1p(scaled.data)
        return scaled

    arr = np.asarray(mat, dtype=np.float64)
    totals = arr.sum(axis=0)
    normalized = np.divide(arr, totals, out=np.zeros_like(arr), where=totals > 0) * scale_factor
    out = np.log1p(normalized)
    return pd.DataFrame(out, index=counts.index, columns=counts.columns) if is_frame else out


@dataclass
class SampleLevelObject:
    """Explicit sample-level representation with landmark rows and sample/donor columns."""

    counts: pd.DataFrame
    normalized: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    embeddings: dict[str, pd.DataFrame] = field(default_factory=dict)
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.counts, pd.DataFrame):
            self.counts = pd.DataFrame(self.counts)
        if self.counts.shape[0] == 0 or self.counts.shape[1] == 0:
            raise ValueError("SampleLevelObject counts must have at least one landmark and one sample")
        lm_ids, _ = validate_bijective_ids(self.counts.index, axis_name="landmark axis")
        sm_ids, _ = validate_bijective_ids(self.counts.columns, axis_name="sample axis")
        validate_numeric_matrix_values(
            self.counts.to_numpy(), name="counts", non_negative=True, require_integer=False
        )
        self.counts = self.counts.copy()
        self.counts.index = pd.Index(lm_ids.astype(str))
        self.counts.columns = pd.Index(sm_ids.astype(str))

        if self.normalized is not None:
            if not isinstance(self.normalized, pd.DataFrame):
                raise ValueError("normalized must be a pandas DataFrame with matching axes")
            validate_bijective_ids(self.normalized.index, axis_name="normalized landmark axis")
            validate_bijective_ids(self.normalized.columns, axis_name="normalized sample axis")
            validate_numeric_matrix_values(self.normalized.to_numpy(), name="normalized", non_negative=False)
            norm_idx = pd.Index(self.normalized.index.astype(str))
            norm_cols = pd.Index(self.normalized.columns.astype(str))
            if set(norm_idx) != set(self.counts.index) or set(norm_cols) != set(self.counts.columns):
                raise ValueError("normalized rows and columns must match counts axes")
            self.normalized = self.normalized.copy()
            self.normalized.index = norm_idx
            self.normalized.columns = norm_cols
            self.normalized = self.normalized.loc[self.counts.index, self.counts.columns]

        if self.residuals is not None:
            if not isinstance(self.residuals, pd.DataFrame):
                raise ValueError("residuals must be a pandas DataFrame with matching axes")
            validate_bijective_ids(self.residuals.index, axis_name="residuals landmark axis")
            validate_bijective_ids(self.residuals.columns, axis_name="residuals sample axis")
            validate_numeric_matrix_values(self.residuals.to_numpy(), name="residuals", non_negative=False)
            res_idx = pd.Index(self.residuals.index.astype(str))
            res_cols = pd.Index(self.residuals.columns.astype(str))
            if set(res_idx) != set(self.counts.index) or set(res_cols) != set(self.counts.columns):
                raise ValueError("residual rows and columns must match counts axes")
            self.residuals = self.residuals.copy()
            self.residuals.index = res_idx
            self.residuals.columns = res_cols
            self.residuals = self.residuals.loc[self.counts.index, self.counts.columns]

        if self.metadata is not None:
            if not isinstance(self.metadata, pd.DataFrame):
                raise ValueError("metadata must be a pandas DataFrame indexed by sample ID")
            meta_ids, _ = validate_bijective_ids(self.metadata.index, axis_name="sample metadata index")
            meta = self.metadata.copy()
            meta.index = pd.Index(meta_ids.astype(str))
            missing = self.counts.columns.difference(meta.index)
            if len(missing) > 0:
                raise ValueError(f"sample metadata is missing sample IDs: {list(missing)}")
            self.metadata = meta.loc[self.counts.columns]

        cleaned_emb: dict[str, pd.DataFrame] = {}
        for key, emb in self.embeddings.items():
            if not isinstance(emb, pd.DataFrame):
                emb = pd.DataFrame(emb, index=self.counts.columns)
            emb_ids, _ = validate_bijective_ids(emb.index, axis_name=f"embedding '{key}' index")
            ef = emb.copy()
            ef.index = pd.Index(emb_ids.astype(str))
            missing = self.counts.columns.difference(ef.index)
            if len(missing) > 0:
                raise ValueError(f"embedding '{key}' is missing sample IDs: {list(missing)}")
            validate_numeric_matrix_values(ef.loc[self.counts.columns].to_numpy(), name=f"embedding '{key}'")
            cleaned_emb[str(key)] = ef.loc[self.counts.columns]
        self.embeddings = cleaned_emb

    @property
    def n_landmarks(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[1])

    def centered_residuals(self, *, scale_max: float | None = 10.0) -> pd.DataFrame:
        """Return feature-centered Pearson residuals matching ``Seurat::ScaleData(..., do.scale=FALSE, do.center=TRUE, scale.max=10)``.

        Notice that ``Seurat:::FastDenseRowScale`` clips the upper bound at ``+scale.max``
        (``np.minimum(centered, scale_max)``) while leaving negative deviations unclipped.
        Pass ``scale_max=None`` to disable upper-bound clipping.
        """
        if self.residuals is None:
            raise ValueError("SampleLevelObject has no residuals; run chi_squared normalization first")
        arr = self.residuals.to_numpy(dtype=np.float64)
        centered = arr - arr.mean(axis=1, keepdims=True)
        if scale_max is not None:
            if not np.isfinite(scale_max) or float(scale_max) <= 0:
                raise ValueError("scale_max must be positive and finite when provided")
            centered = np.minimum(centered, float(scale_max))
        return pd.DataFrame(centered, index=self.residuals.index, columns=self.residuals.columns)

    def subset_samples(
        self,
        samples: Sequence[str],
        *,
        recompute_normalization: bool = False,
    ) -> "SampleLevelObject":
        """Subset columns (samples/donors) of a :class:`SampleLevelObject`.

        By default (``recompute_normalization=False``), retains the pre-filter
        ``residuals`` and ``normalized`` layers sliced to ``samples`` — matching
        ``Seurat::subset(sample_obj, subset = nCount_LMC >= ...)``. Set
        ``recompute_normalization=True`` to recompute Chi-squared Pearson residuals
        or LogNormalize values on the subsetted count matrix.
        """
        req_arr, _ = validate_bijective_ids(samples, axis_name="subset samples")
        req_cols = pd.Index(req_arr.astype(str))
        missing = req_cols.difference(self.counts.columns)
        if len(missing) > 0:
            raise ValueError(f"Requested samples not found in SampleLevelObject: {list(missing[:5])}")

        sub_counts = self.counts.loc[:, req_cols].copy()
        sub_meta = self.metadata.loc[req_cols].copy() if self.metadata is not None else None
        sub_emb = {k: v.loc[req_cols].copy() for k, v in self.embeddings.items()}
        new_attrs = dict(self.attrs)
        new_attrs["subset_from_n_samples"] = int(self.n_samples)
        new_attrs["subset_normalization_policy"] = "recomputed" if recompute_normalization else "retained_prefilter"

        if recompute_normalization:
            norm_mode = str(self.attrs.get("normalization", "chi_squared"))
            ref_prof = str(self.attrs.get("reference_profile", "v6"))
            sf = float(self.attrs.get("scale_factor", 10_000.0))
            rebuilt = SampleLevelObject.from_counts(
                sub_counts,
                normalization=norm_mode,
                scale_factor=sf,
                metadata=sub_meta,
                reference_profile=ref_prof,
                **{k: v for k, v in new_attrs.items() if k not in ("normalization", "scale_factor", "reference_profile")},
            )
            rebuilt.embeddings = sub_emb
            return rebuilt

        sub_norm = self.normalized.loc[:, req_cols].copy() if self.normalized is not None else None
        sub_res = self.residuals.loc[:, req_cols].copy() if self.residuals is not None else None
        return SampleLevelObject(
            counts=sub_counts,
            normalized=sub_norm,
            residuals=sub_res,
            metadata=sub_meta,
            embeddings=sub_emb,
            attrs=new_attrs,
        )

    @classmethod
    def from_counts(
        cls,
        counts: pd.DataFrame,
        *,
        normalization: str = "chi_squared",
        scale_factor: float = 10_000.0,
        metadata: pd.DataFrame | None = None,
        reference_profile: str = "v6",
        **attrs: object,
    ) -> "SampleLevelObject":
        norm_lower = normalization.lower()
        if norm_lower in ("chi_squared", "chisquared"):
            residuals = pearson_residuals(counts, require_integer_counts=(reference_profile == "v6"))
            return cls(
                counts=counts,
                residuals=residuals,
                metadata=metadata,
                attrs={"normalization": "chi_squared", "reference_profile": reference_profile, **attrs},
            )
        if norm_lower in ("log_normalize", "lognormalize"):
            normalized = log_normalize(counts, scale_factor=scale_factor)
            return cls(
                counts=counts,
                normalized=normalized,
                metadata=metadata,
                attrs={
                    "normalization": "log_normalize",
                    "scale_factor": float(scale_factor),
                    "reference_profile": reference_profile,
                    **attrs,
                },
            )
        if norm_lower == "none":
            return cls(
                counts=counts,
                metadata=metadata,
                attrs={"normalization": "none", "reference_profile": reference_profile, **attrs},
            )
        raise ValueError("normalization must be 'chi_squared', 'log_normalize', or 'none'")


def extract_sample_metadata(
    cell_metadata: pd.DataFrame,
    group_by: str,
    *,
    sample_ids: Sequence[str] | None = None,
    valid_cells: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract metadata columns that are constant within each sample/donor (matching ``GenerateSampleObject``)."""
    if group_by not in cell_metadata.columns:
        raise ValueError(f"Grouping column {group_by!r} not found in cell_metadata")

    df = cell_metadata
    if valid_cells is not None:
        df = df.loc[list(valid_cells)]
    df = df.loc[df[group_by].notna()].copy()
    df[group_by] = df[group_by].astype(str)

    consistent_cols: list[str] = [group_by]
    grouped = df.groupby(group_by, sort=False, observed=True)
    for col in df.columns:
        if col == group_by:
            continue
        # Match R length(unique(x)) == 1 within each donor (NA counts as a distinct value in R unique())
        ok = True
        for _, sub in grouped[col]:
            n_val = sub.nunique(dropna=False)
            if n_val > 1:
                ok = False
                break
        if ok:
            consistent_cols.append(col)

    sample_meta = df[consistent_cols].drop_duplicates(subset=[group_by]).set_index(group_by, drop=False)
    sample_meta.index.name = None
    if sample_ids is not None:
        req = pd.Index([str(s) for s in sample_ids])
        # Support R Seurat CreateCategoryMatrix underscore-to-hyphen normalization if needed
        if not req.isin(sample_meta.index).all():
            hyphen_map = {idx.replace("_", "-"): idx for idx in sample_meta.index}
            if all(s in hyphen_map for s in req):
                sample_meta.index = pd.Index([idx.replace("_", "-") for idx in sample_meta.index])
        sample_meta = sample_meta.reindex(req)
    return sample_meta


def generate_sample_object(
    object_or_nn: Any,
    *,
    group_by: str = "Donor",
    nn_name: str | None = None,
    k_nn: int = 5,
    sketch_assay: str = "LANDMARK",
    normalization_method: str = "ChiSquared",
    scale_factor: float = 10_000.0,
    rename_group_by: str | None = None,
    add_meta_data: bool = True,
    cells_use: Iterable[str] | None = None,
    return_seurat_like: bool = True,
    return_anndata: bool = False,
    center_residuals: bool = False,
    scale_max: float | None = 10.0,
    seurat_hyphenate_samples: bool = False,
    seurat_hyphenate_landmarks: bool = False,
    reference_profile: str = "v6",
    sample_for_cell: Iterable[Any] | None = None,
    landmark_ids: Iterable[Any] | None = None,
    cell_ids: Iterable[Any] | None = None,
    cell_metadata: pd.DataFrame | None = None,
) -> SampleLevelObject | pd.DataFrame | Any:
    """Native Python implementation of ``GenerateSampleObject`` (F05).

    Accepts either:
    1. A prepared cell-level ``AnnData`` produced by :func:`prepare_sample_object`
       (or carrying rectangular neighbor assignments in ``adata.obsm`` / ``adata.uns['scslide']``), or
    2. Explicit ``neighbor_index`` together with ``sample_for_cell``, ``landmark_ids``,
       ``cell_ids``, and optional ``cell_metadata``.
    """
    if hasattr(object_or_nn, "obs") and hasattr(object_or_nn, "obsm") and hasattr(object_or_nn, "uns"):
        adata = object_or_nn
        validate_open_backed_anndata(adata, caller="generate_sample_object")
        from .anndata import load_rectangular_neighbors

        if nn_name is None:
            sc_uns = adata.uns.get("scslide", {})
            reg_nns = list(sc_uns.get("neighbors", {}).keys())
            defaults = [n for n in ("weighted.nn", "single.nn") if n in reg_nns]
            if len(defaults) == 0:
                raise ValueError(
                    "No default NN object ('weighted.nn' or 'single.nn') found in AnnData; specify nn_name"
                )
            if len(defaults) > 1:
                raise ValueError(
                    f"Multiple default NN objects found ({defaults}); specify nn_name to disambiguate"
                )
            nn_name = defaults[0]

        nn_idx, _, q_ids, lm_ids_reg = load_rectangular_neighbors(adata, nn_name)
        if landmark_ids is None:
            sketches = adata.uns.get("scslide", {}).get("sketches", {})
            if sketch_assay in sketches and "cell_ids" in sketches[sketch_assay]:
                landmark_ids = sketches[sketch_assay]["cell_ids"]
            else:
                landmark_ids = lm_ids_reg
        cell_ids = q_ids
        cell_metadata = adata.obs.loc[list(q_ids)].copy()
        if group_by not in cell_metadata.columns:
            raise ValueError(f"Metadata column {group_by!r} not found in AnnData.obs")
        sample_for_cell = cell_metadata[group_by].tolist()
        nn_arr = nn_idx
    else:
        nn_arr = np.asarray(object_or_nn)
        if sample_for_cell is None or landmark_ids is None:
            raise ValueError("When passing a neighbor_index array, sample_for_cell and landmark_ids are required")

    samples_raw = list(sample_for_cell)
    valid_mask = np.array(
        [not (s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s)) for s in samples_raw],
        dtype=bool,
    )
    valid_samples_series = pd.Series(samples_raw)[valid_mask]
    if isinstance(cell_metadata, pd.DataFrame) and group_by in cell_metadata.columns and isinstance(
        cell_metadata[group_by].dtype, pd.CategoricalDtype
    ):
        cat_levels = [str(x) for x in cell_metadata[group_by].cat.categories if str(x) in set(valid_samples_series.astype(str))]
        ordered_samples = cat_levels if len(cat_levels) >= 2 else sorted(valid_samples_series.astype(str).unique().tolist())
    else:
        # R as.factor() sorts character levels lexicographically!
        ordered_samples = sorted(valid_samples_series.astype(str).unique().tolist())

    if len(ordered_samples) < 2:
        only_val = ordered_samples[0] if ordered_samples else "NONE"
        raise ValueError(
            f"The grouping variable {group_by!r} has fewer than two levels ({only_val!r}); "
            "at least two samples/groups are required."
        )

    counts_df = landmark_matrix(
        nn_arr,
        samples_raw,
        landmark_ids,
        k=k_nn,
        cell_ids=cell_ids,
        cells_use=cells_use,
        sample_order=ordered_samples,
    )

    canonical_landmarks = list(counts_df.index)
    display_to_landmark: dict[str, str] = {}
    if rename_group_by is not None and cell_metadata is not None:
        if rename_group_by not in cell_metadata.columns:
            warnings.warn(
                f"Cannot find metadata column {rename_group_by!r} to rename landmarks; falling back to landmark IDs.",
                UserWarning,
                stacklevel=2,
            )
            base_labels = canonical_landmarks
        else:
            missing_lm = [lm for lm in canonical_landmarks if lm not in cell_metadata.index]
            if missing_lm:
                warnings.warn(
                    "Some landmark IDs were not found in cell_metadata index; falling back to landmark IDs.",
                    UserWarning,
                    stacklevel=2,
                )
                base_labels = canonical_landmarks
            else:
                base_labels = [str(cell_metadata.loc[lm, rename_group_by]) for lm in canonical_landmarks]
    else:
        base_labels = canonical_landmarks

    new_rows = [re.sub(r"\s", "_", f"{lbl}_LM{i + 1}") for i, lbl in enumerate(base_labels)]
    if seurat_hyphenate_landmarks:
        new_rows = [r.replace("_", "-") for r in new_rows]
        validate_bijective_ids(new_rows, axis_name="hyphenated landmark display IDs")
    for disp, orig in zip(new_rows, canonical_landmarks):
        display_to_landmark[disp] = orig
    counts_df.index = pd.Index(new_rows)

    sample_id_map: dict[str, str] | None = None
    if seurat_hyphenate_samples:
        hyphen_cols = [c.replace("_", "-") for c in counts_df.columns]
        validate_bijective_ids(hyphen_cols, axis_name="hyphenated sample IDs")
        sample_id_map = {h: orig for h, orig in zip(hyphen_cols, counts_df.columns)}
        counts_df.columns = pd.Index(hyphen_cols)

    if not return_seurat_like and not return_anndata:
        return counts_df

    meta_out: pd.DataFrame | None = None
    if add_meta_data and cell_metadata is not None and group_by in cell_metadata.columns:
        valid_cell_ids = None
        if reference_profile == "v6" and cell_ids is not None:
            c_arr = np.asarray([str(c) for c in cell_ids], dtype=object)
            valid_cell_ids = [c for c, ok in zip(c_arr, valid_mask) if ok and c in cell_metadata.index]
        meta_out = extract_sample_metadata(
            cell_metadata,
            group_by=group_by,
            sample_ids=list(counts_df.columns),
            valid_cells=valid_cell_ids,
        )
    else:
        meta_out = pd.DataFrame({group_by: list(counts_df.columns)}, index=counts_df.columns)

    # Record nCount_LMC and nFeature_LMC just like CreateSeuratObject(counts, assay = new_assay_name)
    meta_out = meta_out.copy()
    meta_out["nCount_LMC"] = counts_df.sum(axis=0).to_numpy(dtype=np.int64)
    meta_out["nFeature_LMC"] = (counts_df > 0).sum(axis=0).to_numpy(dtype=np.int64)

    norm_choice = normalization_method.lower()
    if norm_choice == "chisquared":
        norm_arg = "chi_squared"
    elif norm_choice == "lognormalize":
        norm_arg = "log_normalize"
    elif norm_choice == "none":
        norm_arg = "none"
    else:
        raise ValueError("normalization_method must be 'ChiSquared', 'LogNormalize', or 'none'")

    # Guard all-zero counts when ChiSquared is requested
    if norm_arg == "chi_squared" and float(counts_df.to_numpy().sum()) <= 0:
        raise ValueError("LMC count matrix has zero total assignments; cannot compute ChiSquared residuals")

    sample_obj = SampleLevelObject.from_counts(
        counts_df,
        normalization=norm_arg,
        scale_factor=scale_factor,
        metadata=meta_out,
        reference_profile=reference_profile,
        group_by=group_by,
        k_nn=int(k_nn),
        nn_name=nn_name,
        sketch_assay=sketch_assay,
        landmark_id_map=display_to_landmark,
        canonical_landmark_ids=canonical_landmarks,
        sample_id_map=sample_id_map,
    )

    if center_residuals and sample_obj.residuals is not None:
        sample_obj.attrs["centered_residuals"] = True
        sample_obj.attrs["scale_max"] = scale_max

    if return_anndata:
        from .anndata import sample_object_to_anndata

        s_adata = sample_object_to_anndata(
            sample_obj,
            include_centered_residuals=center_residuals,
            scale_max=scale_max,
        )
        return s_adata

    return sample_obj
