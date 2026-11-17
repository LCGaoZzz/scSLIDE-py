"""Internal validation and storage-aware streaming matrix operators for scSLIDE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


def validate_open_backed_anndata(adata: Any, *, caller: str = "scSLIDE") -> None:
    """Ensure a backed AnnData object still holds an open backing store handle."""
    if not getattr(adata, "isbacked", False):
        return
    fobj = getattr(adata, "file", None)
    if fobj is None:
        raise RuntimeError(
            f"{caller}: backed AnnData has no backing file manager; reopen the HDF5/Zarr store first."
        )
    # AnnData BackingFileManager exposes is_open (bool) and _file (h5py.File)
    is_open = getattr(fobj, "is_open", None)
    if is_open is False:
        raise RuntimeError(
            f"{caller}: backed AnnData file handle is closed; reopen the HDF5/Zarr store before calling {caller}."
        )
    raw_h5 = getattr(fobj, "_file", None)
    if raw_h5 is not None and not bool(raw_h5):
        raise RuntimeError(
            f"{caller}: backed HDF5 file handle is closed; reopen the HDF5 store before calling {caller}."
        )


def validate_bijective_ids(
    ids: Iterable[Any],
    *,
    axis_name: str = "axis",
    allow_reversible_map: bool = False,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Validate that axis identifiers are non-empty, unique, and collision-free under stringification.

    Rejects None/NA, empty strings, duplicate raw values, and collisions where
    distinct raw values (such as integer ``1`` and string ``'1'``) stringify to
    the same identifier unless ``allow_reversible_map=True``.
    """
    raw_list = list(ids)
    if len(raw_list) == 0:
        raise ValueError(f"{axis_name} identifiers cannot be empty")

    str_list: list[str] = []
    seen_raw: list[tuple[type, Any]] = []
    raw_set: set[tuple[type, Any]] = set()
    str_to_raw: dict[str, tuple[type, Any]] = {}

    for pos, item in enumerate(raw_list):
        if item is None or (isinstance(item, float) and np.isnan(item)) or pd.isna(item):
            raise ValueError(f"{axis_name} contains missing/NA identifier at position {pos}")
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{axis_name} cannot use boolean values as identifiers at position {pos}")
        # Normalize numpy scalars to Python scalars for deterministic type tagging
        py_val = item.item() if isinstance(item, np.generic) else item
        typed_key = (type(py_val), py_val)
        if typed_key in raw_set:
            raise ValueError(f"{axis_name} contains duplicate identifier {py_val!r}")
        raw_set.add(typed_key)
        seen_raw.append(typed_key)

        s = str(py_val)
        if s == "" or s.strip() == "":
            raise ValueError(f"{axis_name} contains empty/whitespace identifier at position {pos}")
        if s in str_to_raw and str_to_raw[s] != typed_key:
            if not allow_reversible_map:
                prev_type, prev_val = str_to_raw[s]
                raise ValueError(
                    f"{axis_name} has stringification collision between {prev_val!r} ({prev_type.__name__}) "
                    f"and {py_val!r} ({type(py_val).__name__}); provide unique string IDs or an explicit mapping"
                )
            s = f"{type(py_val).__name__}:{s}"
            if s in str_to_raw:
                raise ValueError(f"{axis_name} reversible string mapping still collided on {s!r}")
        str_to_raw[s] = typed_key
        str_list.append(s)

    arr = np.asarray(str_list, dtype=object)
    rev_map = {s: val for s, (_, val) in zip(str_list, seen_raw)} if allow_reversible_map else None
    return arr, rev_map


def validate_numeric_matrix_values(
    arr: np.ndarray | sp.spmatrix | sp.sparray,
    *,
    name: str = "matrix",
    non_negative: bool = False,
    require_integer: bool = False,
) -> None:
    """Validate finite, optionally non-negative and integer-valued numeric entries."""
    if sp.issparse(arr):
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2-D matrix")
        data = np.asarray(arr.data)
        if data.size == 0:
            return
        if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
            raise ValueError(f"{name} must contain numeric values, got dtype {data.dtype}")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contains NaN or infinite values; silent coercion is forbidden")
        if non_negative and np.any(data < 0):
            raise ValueError(f"{name} must be non-negative")
        if require_integer and not np.all(data == np.round(data)):
            raise ValueError(f"{name} must contain integer count values")
    else:
        a = np.asarray(arr)
        if a.ndim != 2 or a.size == 0:
            raise ValueError(f"{name} must be a non-empty 2-D matrix")
        if not np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_):
            raise ValueError(f"{name} must contain numeric values, got dtype {a.dtype}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains NaN or infinite values; silent coercion is forbidden")
        if non_negative and np.any(a < 0):
            raise ValueError(f"{name} must be non-negative")
        if require_integer and not np.all(a == np.round(a)):
            raise ValueError(f"{name} must contain integer count values")


def encode_response_design(
    y: Any,
    *,
    expected_nrow: int | None = None,
    label: str = "Y",
    check_missing: bool = True,
) -> tuple[np.ndarray, list[str], bool]:
    """Encode numeric or categorical response(s) matching R ``model.matrix(~ . + 0, data = Y)``.

    Returns ``(y_matrix_float64, column_names, is_categorical)``.
    """
    if y is None:
        raise ValueError(f"'{label}' cannot be None")

    if isinstance(y, pd.Series):
        df = y.to_frame(name=str(y.name) if y.name is not None else label)
        return _encode_dataframe_response(df, expected_nrow=expected_nrow, label=label, check_missing=check_missing)

    if isinstance(y, pd.DataFrame):
        return _encode_dataframe_response(y, expected_nrow=expected_nrow, label=label, check_missing=check_missing)

    arr = np.asarray(y)
    if arr.ndim == 1:
        if expected_nrow is not None and len(arr) != expected_nrow:
            raise ValueError(f"nrow({label}) is {len(arr)} but expected {expected_nrow}")
        if np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            out = arr.astype(np.float64).reshape(-1, 1)
            if check_missing and not np.all(np.isfinite(out)):
                n_miss = int(np.sum(~np.isfinite(out[:, 0])))
                raise ValueError(f"{label} has NA/non-finite values for {n_miss} observations")
            return out, [f"{label}1"], False
        # String/object categorical vector -> R factor levels (sorted unique strings unless Categorical)
        s = pd.Series(y, name=label)
        return _encode_dataframe_response(
            s.to_frame(), expected_nrow=expected_nrow, label=label, check_missing=check_missing
        )

    if arr.ndim == 2:
        if expected_nrow is not None and arr.shape[0] != expected_nrow:
            raise ValueError(f"nrow({label}) is {arr.shape[0]} but expected {expected_nrow}")
        if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            raise ValueError(f"2-D array '{label}' must be numeric; pass a DataFrame for categorical columns")
        out = arr.astype(np.float64, copy=False)
        if check_missing and not np.all(np.isfinite(out)):
            n_miss = int(np.sum(np.any(~np.isfinite(out), axis=1)))
            raise ValueError(f"{label} has NA/non-finite values for {n_miss} observations")
        cols = [f"{label}{i + 1}" for i in range(out.shape[1])]
        return out, cols, False

    raise ValueError(f"'{label}' must be a 1-D vector, 2-D numeric matrix, Series, or DataFrame")


def _encode_dataframe_response(
    df: pd.DataFrame,
    *,
    expected_nrow: int | None,
    label: str,
    check_missing: bool,
) -> tuple[np.ndarray, list[str], bool]:
    if expected_nrow is not None and len(df) != expected_nrow:
        raise ValueError(f"nrow({label}) is {len(df)} but expected {expected_nrow}")
    if df.shape[1] == 0:
        raise ValueError(f"'{label}' DataFrame has zero columns")

    if check_missing:
        na_rows = df.isna().any(axis=1)
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                na_rows = na_rows | ~np.isfinite(s.to_numpy(dtype=np.float64, na_value=np.nan))
        n_miss = int(na_rows.sum())
        if n_miss > 0:
            cols_str = ", ".join(repr(str(c)) for c in df.columns)
            raise ValueError(
                f"{label} ({cols_str}) has NA for {n_miss} cells/samples. "
                "Please subset to observations with a valid response first."
            )

    blocks: list[np.ndarray] = []
    col_names: list[str] = []
    any_cat = False
    first_cat_encoded = False

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            vals = s.to_numpy(dtype=np.float64).reshape(-1, 1)
            blocks.append(vals)
            col_names.append(str(col))
        else:
            any_cat = True
            if isinstance(s.dtype, pd.CategoricalDtype):
                levels = [lvl for lvl in s.cat.categories]
                codes = s.cat.codes.to_numpy()
            else:
                str_vals = s.astype(str).to_numpy()
                levels = sorted(pd.unique(str_vals).tolist())
                level_to_idx = {lvl: idx for idx, lvl in enumerate(levels)}
                codes = np.array([level_to_idx[v] for v in str_vals], dtype=int)
            n = len(s)
            k = len(levels)
            if not first_cat_encoded:
                # R model.matrix(~ . + 0): first categorical variable keeps ALL k dummy columns!
                mat = np.zeros((n, k), dtype=np.float64)
                valid = codes >= 0
                mat[np.nonzero(valid)[0], codes[valid]] = 1.0
                blocks.append(mat)
                col_names.extend([f"{col}{lvl}" for lvl in levels])
                first_cat_encoded = True
            else:
                # Subsequent categorical variables in ~ . + 0 drop reference level 0 (treatment contrasts)
                if k > 1:
                    mat = np.zeros((n, k - 1), dtype=np.float64)
                    for j in range(1, k):
                        mat[codes == j, j - 1] = 1.0
                    blocks.append(mat)
                    col_names.extend([f"{col}{lvl}" for lvl in levels[1:]])

    out = np.hstack(blocks) if blocks else np.zeros((len(df), 0), dtype=np.float64)
    return out, col_names, any_cat


@dataclass
class StreamingMatrix:
    """Unified cells-by-features (n x p) matrix operator over dense, sparse CSR/CSC, or backed HDF5/Zarr."""

    source: Any
    row_indices: np.ndarray | None = None
    col_indices: np.ndarray | None = None
    chunk_size: int = 8192

    def __post_init__(self) -> None:
        raw_shape = self.source.shape
        if len(raw_shape) != 2:
            raise ValueError("StreamingMatrix source must be 2-D")
        self._n_raw, self._p_raw = int(raw_shape[0]), int(raw_shape[1])
        self.n_obs = len(self.row_indices) if self.row_indices is not None else self._n_raw
        self.n_vars = len(self.col_indices) if self.col_indices is not None else self._p_raw

        if isinstance(self.source, pd.DataFrame):
            self._kind = "dataframe"
            self.memory_mode = "dense_in_memory"
        elif isinstance(self.source, np.ndarray):
            self._kind = "ndarray"
            self.memory_mode = "dense_in_memory"
        elif sp.issparse(self.source):
            self._kind = "scipy_sparse"
            self.memory_mode = "sparse_native"
            # Materialize lightweight CSR submatrix if row/col sliced (preserves sparsity)
            mat = self.source
            if not sp.isspmatrix_csr(mat) and not isinstance(mat, sp.csr_array):
                mat = mat.tocsr()
            if self.row_indices is not None:
                mat = mat[self.row_indices, :]
            if self.col_indices is not None:
                mat = mat[:, self.col_indices]
            self._csr = mat
        else:
            self._kind = "backed"
            self.memory_mode = "backed_chunked"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    def iter_row_chunks(self, chunk_size: int | None = None):
        """Yield ``(start, stop, block)`` where ``block`` is dense or CSR of shape ``(stop-start, p)``."""
        step = chunk_size or self.chunk_size
        if self._kind == "scipy_sparse":
            for s in range(0, self.n_obs, step):
                e = min(self.n_obs, s + step)
                yield s, e, self._csr[s:e]
            return

        if self._kind in ("ndarray", "dataframe"):
            base = self.source.to_numpy(dtype=np.float64, copy=False) if self._kind == "dataframe" else self.source
            for s in range(0, self.n_obs, step):
                e = min(self.n_obs, s + step)
                rows = slice(s, e) if self.row_indices is None else self.row_indices[s:e]
                blk = base[rows, :]
                if self.col_indices is not None:
                    blk = blk[:, self.col_indices]
                yield s, e, np.asarray(blk, dtype=np.float64)
            return

        # Backed HDF5 / Zarr / AnnData SparseDataset or h5py Dataset
        for s in range(0, self.n_obs, step):
            e = min(self.n_obs, s + step)
            if self.row_indices is None:
                blk = self.source[s:e]
            else:
                ridx = np.asarray(self.row_indices[s:e], dtype=np.int64)
                # HDF5 / SparseDataset requires sorted unique indices; reorder if needed
                order = np.argsort(ridx)
                sorted_ridx = ridx[order]
                uniq, inv = np.unique(sorted_ridx, return_inverse=True)
                blk_u = self.source[uniq]
                inv_back = np.empty_like(order)
                inv_back[order] = np.arange(len(order))
                remap = inv[inv_back]
                blk = blk_u[remap]
            if self.col_indices is not None:
                blk = blk[:, self.col_indices]
            yield s, e, blk

    def mean_and_var(self, *, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact column means and sample variances without densifying full X."""
        if self._kind == "scipy_sparse":
            m = self._csr
            n = self.n_obs
            sums = np.asarray(m.sum(axis=0), dtype=np.float64).ravel()
            mean = sums / max(n, 1)
            sq = m.copy()
            sq.data = np.asarray(sq.data, dtype=np.float64) ** 2
            sum_sq = np.asarray(sq.sum(axis=0), dtype=np.float64).ravel()
            ss = np.maximum(0.0, sum_sq - n * (mean ** 2))
            denom = max(n - ddof, 1)
            return mean, ss / denom

        n = self.n_obs
        sums = np.zeros(self.n_vars, dtype=np.float64)
        sum_sq = np.zeros(self.n_vars, dtype=np.float64)
        for _, _, blk in self.iter_row_chunks():
            if sp.issparse(blk):
                sums += np.asarray(blk.sum(axis=0), dtype=np.float64).ravel()
                b2 = blk.copy()
                b2.data = np.asarray(b2.data, dtype=np.float64) ** 2
                sum_sq += np.asarray(b2.sum(axis=0), dtype=np.float64).ravel()
            else:
                b = np.asarray(blk, dtype=np.float64)
                sums += b.sum(axis=0)
                sum_sq += np.einsum("ij,ij->j", b, b)
        mean = sums / max(n, 1)
        ss = np.maximum(0.0, sum_sq - n * (mean ** 2))
        denom = max(n - ddof, 1)
        return mean, ss / denom

    def xm(self, m: np.ndarray) -> np.ndarray:
        """Compute ``X @ M`` where ``M`` is ``(p, k)`` or ``(p,)``."""
        m_arr = np.asarray(m, dtype=np.float64)
        is_1d = m_arr.ndim == 1
        if is_1d:
            m_arr = m_arr.reshape(-1, 1)
        if self._kind == "scipy_sparse":
            out = self._csr @ m_arr
            out = np.asarray(out, dtype=np.float64)
            return out.ravel() if is_1d else out

        out = np.empty((self.n_obs, m_arr.shape[1]), dtype=np.float64)
        for s, e, blk in self.iter_row_chunks():
            res = blk @ m_arr
            out[s:e] = np.asarray(res, dtype=np.float64)
        return out.ravel() if is_1d else out

    def xtm(self, m: np.ndarray) -> np.ndarray:
        """Compute ``X.T @ M`` where ``M`` is ``(n, q)`` or ``(n,)``."""
        m_arr = np.asarray(m, dtype=np.float64)
        is_1d = m_arr.ndim == 1
        if is_1d:
            m_arr = m_arr.reshape(-1, 1)
        if self._kind == "scipy_sparse":
            out = self._csr.T @ m_arr
            out = np.asarray(out, dtype=np.float64)
            return out.ravel() if is_1d else out

        out = np.zeros((self.n_vars, m_arr.shape[1]), dtype=np.float64)
        for s, e, blk in self.iter_row_chunks():
            res = blk.T @ m_arr[s:e]
            out += np.asarray(res, dtype=np.float64)
        return out.ravel() if is_1d else out

    def detection_counts(self, *, strict_positive: bool = True) -> np.ndarray:
        """Count observations per feature with ``x > 0`` (or stored nonzero)."""
        if self._kind == "scipy_sparse":
            m = self._csr
            if strict_positive:
                pos = m.data > 0
                if np.all(pos):
                    return np.bincount(m.indices, minlength=self.n_vars).astype(np.int64)
                return np.bincount(m.indices[pos], minlength=self.n_vars).astype(np.int64)
            nz = m.data != 0
            return np.bincount(m.indices[nz], minlength=self.n_vars).astype(np.int64)

        counts = np.zeros(self.n_vars, dtype=np.int64)
        for _, _, blk in self.iter_row_chunks():
            if sp.issparse(blk):
                mask = blk.data > 0 if strict_positive else blk.data != 0
                counts += np.bincount(blk.indices[mask], minlength=self.n_vars)
            else:
                b = np.asarray(blk)
                counts += np.sum(b > 0 if strict_positive else b != 0, axis=0)
        return counts
