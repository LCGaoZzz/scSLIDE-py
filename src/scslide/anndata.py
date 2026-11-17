"""Production AnnData / Scanpy adapters with strict statistical-unit and storage contracts.

Sample-level AnnData contract:
- ``obs`` = samples / donors (``observation_unit == 'sample'``)
- ``var`` = landmarks (``variable_unit == 'landmark'``)
- ``layers['counts']`` = landmark assignment counts (sample x landmark)
- ``layers['normalized']`` = column library-size log1p values (optional)
- ``layers['residuals']`` = Pearson residuals from independence model (optional)
- ``layers['scale.data']`` = feature-centered Pearson residuals for DiffusionMap (optional)

Cell-level rectangular query-by-landmark neighbor assignments are stored in
``adata.obsm['scslide_<nn_name>_idx']`` and ``adata.obsm['scslide_<nn_name>_dist']``
paired with landmark ID provenance in ``adata.uns['scslide']['neighbors'][nn_name]`` —
never in ``.obsp`` (which AnnData reserves for square obs x obs graphs).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._validation import validate_bijective_ids, validate_numeric_matrix_values, validate_open_backed_anndata
from .sample import SampleLevelObject


SAMPLE_SCHEMA_VERSION = "sample-level-2"
LEGACY_SAMPLE_SCHEMA_VERSIONS = {"sample-level-1", "sample-level-2"}


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AnnData support requires the optional dependency: pip install scslide-py[anndata]") from exc
    return ad


def _json_safe_value(value: Any) -> Any:
    """Convert nested metadata values into HDF5/Zarr/AnnData-safe serializable scalars/containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim <= 2 and value.size <= 10_000:
            return value.tolist()
        return {"__ndarray_shape__": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return str(value)


def _extract_2d_layer(
    matrix: Any,
    *,
    index: pd.Index,
    columns: pd.Index,
    layer_name: str,
    non_negative: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Read dense, CSR/CSC, or backed HDF5/Zarr layer into a validated sample x landmark DataFrame."""
    materialization = "dense_in_memory"
    if sp.issparse(matrix):
        materialization = f"sparse_{matrix.format}_materialized_for_sample_frame"
        arr = matrix.toarray()
    elif hasattr(matrix, "toarray"):
        materialization = f"{type(matrix).__name__}_materialized_for_sample_frame"
        arr = matrix.toarray()
    else:
        # Backed HDF5 Dataset or Zarr array: slice [:] explicitly
        if not isinstance(matrix, np.ndarray):
            materialization = f"backed_{type(matrix).__name__}_sliced"
            arr = np.asarray(matrix[:])
            if sp.issparse(arr):
                arr = arr.toarray()
        else:
            arr = np.asarray(matrix)

    if arr.ndim != 2 or arr.shape != (len(index), len(columns)):
        raise ValueError(
            f"AnnData layer {layer_name!r} shape {arr.shape} does not match axes ({len(index)}, {len(columns)})"
        )
    validate_numeric_matrix_values(arr, name=f"AnnData layer {layer_name!r}", non_negative=non_negative)
    return pd.DataFrame(arr, index=index.astype(str), columns=columns.astype(str)), materialization


def sample_object_to_anndata(
    obj: SampleLevelObject,
    *,
    metadata: pd.DataFrame | None = None,
    include_normalized: bool = True,
    include_residuals: bool = True,
    include_centered_residuals: bool = False,
    scale_max: float | None = 10.0,
    sparse_counts: bool = False,
) -> Any:
    """Convert a :class:`SampleLevelObject` to a schema-tagged sample-level ``AnnData``."""
    ad = _require_anndata()
    counts = obj.counts
    sm_arr, _ = validate_bijective_ids(counts.columns, axis_name="sample_ids")
    lm_arr, _ = validate_bijective_ids(counts.index, axis_name="landmark_ids")
    sample_ids = pd.Index(sm_arr.astype(str))
    landmark_ids = pd.Index(lm_arr.astype(str))

    obs = pd.DataFrame(index=sample_ids)
    source_metadata = metadata if metadata is not None else obj.metadata
    if source_metadata is not None:
        meta_ids, _ = validate_bijective_ids(source_metadata.index, axis_name="sample metadata index")
        meta_df = source_metadata.copy()
        meta_df.index = pd.Index(meta_ids.astype(str))
        missing = sample_ids.difference(meta_df.index)
        if len(missing):
            raise ValueError(f"sample metadata is missing IDs: {list(missing)}")
        obs = meta_df.loc[sample_ids].copy()

    var = pd.DataFrame(index=landmark_ids)
    lm_map = obj.attrs.get("landmark_id_map")
    if isinstance(lm_map, dict) and all(k in lm_map for k in landmark_ids):
        var["canonical_landmark_id"] = [str(lm_map[k]) for k in landmark_ids]

    x_dense = counts.T.to_numpy()
    validate_numeric_matrix_values(x_dense, name="counts", non_negative=True)
    x_mat = sp.csr_matrix(x_dense) if sparse_counts else x_dense.copy()

    out = ad.AnnData(X=x_mat, obs=obs, var=var)
    out.layers["counts"] = sp.csr_matrix(x_dense) if sparse_counts else x_dense.copy()

    if include_normalized and obj.normalized is not None:
        if not obj.normalized.index.equals(counts.index) or not obj.normalized.columns.equals(counts.columns):
            raise ValueError("normalized matrix axes must match counts")
        out.layers["normalized"] = obj.normalized.T.to_numpy(dtype=np.float64)

    if include_residuals and obj.residuals is not None:
        if not obj.residuals.index.equals(counts.index) or not obj.residuals.columns.equals(counts.columns):
            raise ValueError("residual matrix axes must match counts")
        res_t = obj.residuals.T.to_numpy(dtype=np.float64)
        out.layers["residuals"] = res_t
        if include_centered_residuals or bool(obj.attrs.get("centered_residuals", False)):
            eff_smax = obj.attrs.get("scale_max", scale_max) if "scale_max" in obj.attrs else scale_max
            cent_t = res_t - res_t.mean(axis=0, keepdims=True)
            if eff_smax is not None:
                cent_t = np.minimum(cent_t, float(eff_smax))
            out.layers["scale.data"] = cent_t

    for emb_name, emb_df in obj.embeddings.items():
        key = emb_name if emb_name.startswith("X_") else f"X_{emb_name.lower()}"
        aligned = emb_df.loc[sample_ids]
        out.obsm[key] = aligned.to_numpy(dtype=np.float64)

    provenance_attrs = {
        str(k): _json_safe_value(v)
        for k, v in obj.attrs.items()
        if k not in ("anndata_materialized", "anndata_materialization_mode")
    }
    out.uns["scslide"] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "observation_unit": "sample",
        "variable_unit": "landmark",
        "counts_layer": "counts",
        "normalization": obj.attrs.get("normalization"),
        "sparse_counts": bool(sparse_counts),
        "provenance_json": json.dumps(provenance_attrs, ensure_ascii=False, sort_keys=True),
    }
    return out


def sample_object_from_anndata(
    adata: Any,
    *,
    layer: str = "counts",
    normalized_layer: str | None = "normalized",
    residuals_layer: str | None = "residuals",
    strict_schema: bool = True,
    confirm_sample_landmark_units: bool = False,
) -> SampleLevelObject:
    """Read a sample-level ``AnnData`` object into a validated :class:`SampleLevelObject`.

    Statistical-unit safety contract (B06):
    - If ``adata.uns['scslide']`` declares ``observation_unit != 'sample'`` or
      ``variable_unit != 'landmark'`` (for example, a cell x gene AnnData), this
      function always raises ``ValueError``.
    - If ``adata.uns['scslide']`` is absent and ``strict_schema=True`` (default),
      callers must explicitly pass ``confirm_sample_landmark_units=True`` to import
      an external sample-by-landmark AnnData; matrix shape is never used to guess units.
    """
    _require_anndata()
    validate_open_backed_anndata(adata, caller="sample_object_from_anndata")
    if getattr(adata, "n_obs", 0) == 0 or getattr(adata, "n_vars", 0) == 0:
        raise ValueError("AnnData must contain at least one sample and one landmark")

    validate_bijective_ids(adata.obs_names, axis_name="AnnData obs_names")
    validate_bijective_ids(adata.var_names, axis_name="AnnData var_names")

    raw_uns = dict(adata.uns.get("scslide", {})) if hasattr(adata, "uns") and "scslide" in adata.uns else {}
    if raw_uns:
        obs_unit = raw_uns.get("observation_unit")
        var_unit = raw_uns.get("variable_unit")
        schema_ver = raw_uns.get("schema_version")
        if obs_unit != "sample" or var_unit != "landmark":
            raise ValueError(
                f"Expected sample x landmark AnnData, but scSLIDE schema declares "
                f"observation_unit={obs_unit!r}, variable_unit={var_unit!r}"
            )
        if strict_schema and schema_ver not in LEGACY_SAMPLE_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported scSLIDE sample AnnData schema_version: {schema_ver!r}")
    elif strict_schema and not confirm_sample_landmark_units:
        raise ValueError(
            "AnnData lacks scSLIDE sample-level schema metadata (uns['scslide']). "
            "Ordinary cell x gene AnnData objects cannot be imported as sample x landmark objects. "
            "If this is an external sample x landmark AnnData, pass confirm_sample_landmark_units=True."
        )

    if layer:
        if layer not in adata.layers:
            raise ValueError(f"Requested counts layer {layer!r} not found in AnnData.layers")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X

    counts_s_by_l, mat_mode = _extract_2d_layer(
        matrix, index=adata.obs_names, columns=adata.var_names, layer_name=layer or "X", non_negative=True
    )
    counts = counts_s_by_l.T

    normalized = None
    if normalized_layer is not None and normalized_layer in adata.layers:
        norm_s_by_l, _ = _extract_2d_layer(
            adata.layers[normalized_layer],
            index=adata.obs_names,
            columns=adata.var_names,
            layer_name=normalized_layer,
            non_negative=False,
        )
        normalized = norm_s_by_l.T

    residuals = None
    if residuals_layer is not None and residuals_layer in adata.layers:
        res_s_by_l, _ = _extract_2d_layer(
            adata.layers[residuals_layer],
            index=adata.obs_names,
            columns=adata.var_names,
            layer_name=residuals_layer,
            non_negative=False,
        )
        residuals = res_s_by_l.T

    metadata = adata.obs[:].copy() if getattr(adata, "isbacked", False) else adata.obs.copy()
    metadata.index = pd.Index(metadata.index.astype(str))

    embeddings: dict[str, pd.DataFrame] = {}
    if hasattr(adata, "obsm"):
        for key in adata.obsm.keys():
            val = np.asarray(adata.obsm[key][:])
            if val.ndim == 2 and val.shape[0] == adata.n_obs:
                emb_name = key[2:] if key.startswith("X_") else str(key)
                cols = [f"{emb_name}_{i + 1}" for i in range(val.shape[1])]
                embeddings[emb_name] = pd.DataFrame(val, index=metadata.index, columns=cols)

    attrs: dict[str, Any] = {}
    if "provenance_json" in raw_uns and isinstance(raw_uns["provenance_json"], str):
        try:
            attrs.update(json.loads(raw_uns["provenance_json"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    for k, v in raw_uns.items():
        if k != "provenance_json":
            attrs.setdefault(k, v)
    attrs["observation_unit"] = "sample"
    attrs["variable_unit"] = "landmark"
    attrs["anndata_materialized"] = True
    attrs["anndata_materialization_mode"] = mat_mode

    return SampleLevelObject(
        counts=counts,
        normalized=normalized,
        residuals=residuals,
        metadata=metadata,
        embeddings=embeddings,
        attrs=attrs,
    )


def store_rectangular_neighbors(
    adata: Any,
    nn_name: str,
    *,
    nn_idx: np.ndarray,
    nn_dist: np.ndarray | None,
    landmark_ids: Sequence[str],
    query_ids: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Persist query-by-landmark rectangular neighbors safely in ``adata.obsm`` + ``adata.uns['scslide']``."""
    validate_open_backed_anndata(adata, caller="store_rectangular_neighbors")
    validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
    lm_arr, _ = validate_bijective_ids(landmark_ids, axis_name="landmark_ids")
    idx_arr = np.asarray(nn_idx, dtype=np.int64)
    if idx_arr.ndim != 2 or idx_arr.shape[0] != adata.n_obs:
        raise ValueError(f"nn_idx shape {idx_arr.shape} must match (adata.n_obs={adata.n_obs}, k)")
    if np.any(idx_arr < 1) or np.any(idx_arr > len(lm_arr)):
        raise ValueError("nn_idx must contain one-based indices into landmark_ids")

    if query_ids is not None:
        q_arr, _ = validate_bijective_ids(query_ids, axis_name="query_ids")
        if not np.array_equal(q_arr.astype(str), np.asarray(adata.obs_names.astype(str))):
            raise ValueError("query_ids must match adata.obs_names in order")

    idx_key = f"scslide_{nn_name}_idx"
    dist_key = f"scslide_{nn_name}_dist"
    adata.obsm[idx_key] = idx_arr
    if nn_dist is not None:
        d_arr = np.asarray(nn_dist, dtype=np.float64)
        if d_arr.shape != idx_arr.shape:
            raise ValueError("nn_dist shape must match nn_idx")
        validate_numeric_matrix_values(d_arr, name="nn_dist", non_negative=True)
        adata.obsm[dist_key] = d_arr

    sc_uns = dict(adata.uns.get("scslide", {}))
    sc_uns.setdefault("observation_unit", "cell")
    sc_uns.setdefault("variable_unit", "feature")
    neighbors_reg = dict(sc_uns.get("neighbors", {}))
    neighbors_reg[str(nn_name)] = {
        "index_base": 1,
        "obsm_idx_key": idx_key,
        "obsm_dist_key": dist_key if nn_dist is not None else None,
        "landmark_ids": [str(x) for x in lm_arr],
        "n_landmarks": int(len(lm_arr)),
        "k_stored": int(idx_arr.shape[1]),
        "params": _json_safe_value(params or {}),
    }
    sc_uns["neighbors"] = neighbors_reg
    adata.uns["scslide"] = sc_uns


def load_rectangular_neighbors(
    adata: Any,
    nn_name: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    """Load and validate rectangular query x landmark neighbors from ``adata`` (safe across row slicing)."""
    validate_open_backed_anndata(adata, caller="load_rectangular_neighbors")
    sc_uns = adata.uns.get("scslide", {})
    neighbors_reg = sc_uns.get("neighbors", {})
    if nn_name not in neighbors_reg:
        raise ValueError(f"Neighbor registry {nn_name!r} not found in adata.uns['scslide']['neighbors']")
    reg = neighbors_reg[nn_name]
    idx_key = reg["obsm_idx_key"]
    dist_key = reg.get("obsm_dist_key")
    landmark_ids = [str(x) for x in reg["landmark_ids"]]
    if idx_key not in adata.obsm:
        raise ValueError(f"Expected neighbor index array adata.obsm[{idx_key!r}] is missing")
    nn_idx = np.asarray(adata.obsm[idx_key][:], dtype=np.int64)
    if nn_idx.ndim != 2 or nn_idx.shape[0] != adata.n_obs:
        raise ValueError("Stored neighbor index shape does not match current adata.n_obs")
    if np.any(nn_idx < 1) or np.any(nn_idx > len(landmark_ids)):
        raise ValueError("Stored neighbor indices out of range for registered landmark_ids")
    nn_dist = np.asarray(adata.obsm[dist_key][:], dtype=np.float64) if dist_key and dist_key in adata.obsm else None
    query_ids = np.asarray(adata.obs_names.astype(str), dtype=object)
    return nn_idx, nn_dist, query_ids, landmark_ids


def write_sample_zarr(adata: Any, path: str | Path) -> Path:
    """Write a sample-level AnnData to Zarr store cleanly."""
    _require_anndata()
    validate_open_backed_anndata(adata, caller="write_sample_zarr")
    out_path = Path(path)
    try:
        adata.write_zarr(out_path)
    except KeyError as exc:
        # Zarr 2 wraps a filesystem permission failure as KeyError('.zgroup').
        # Preserve unrelated serialization errors, but expose the actionable
        # filesystem cause when the destination cannot be written.
        permission_error = exc.__cause__ or exc.__context__
        if isinstance(permission_error, PermissionError):
            raise PermissionError(
                f"write_sample_zarr: cannot write {out_path}; check destination permissions."
            ) from permission_error
        raise
    return out_path


def landmark_object_from_anndata(adata: Any) -> Any:
    """Restore a :class:`~scslide.landmark.LandmarkObject` from a schema-tagged landmark-level ``AnnData``."""
    from .landmark import LandmarkObject

    _require_anndata()
    validate_open_backed_anndata(adata, caller="landmark_object_from_anndata")
    raw_uns = dict(adata.uns.get("scslide", {})) if hasattr(adata, "uns") and "scslide" in adata.uns else {}
    if raw_uns.get("observation_unit") != "landmark" or raw_uns.get("variable_unit") != "sample":
        raise ValueError(
            "AnnData does not declare scSLIDE landmark-level units "
            "(observation_unit='landmark', variable_unit='sample')."
        )
    dens_df, _ = _extract_2d_layer(
        adata.X, index=adata.obs_names, columns=adata.var_names, layer_name="X", non_negative=True
    )
    if "X_wnn_umap" not in adata.obsm:
        raise ValueError("Landmark AnnData is missing adata.obsm['X_wnn_umap']")
    umap_arr = np.asarray(adata.obsm["X_wnn_umap"][:], dtype=np.float64)
    umap_df = pd.DataFrame(umap_arr, index=adata.obs_names.astype(str), columns=["UMAP_1", "UMAP_2"])
    nn_idx = np.asarray(adata.obsm["scslide_landmark_nn_idx"][:], dtype=np.int64)
    nn_dist = np.asarray(adata.obsm["scslide_landmark_nn_dist"][:], dtype=np.float64)
    meta = adata.obs[:].copy() if getattr(adata, "isbacked", False) else adata.obs.copy()
    meta.index = pd.Index(meta.index.astype(str))
    attrs: dict[str, Any] = {}
    if "attrs_json" in raw_uns and isinstance(raw_uns["attrs_json"], str):
        attrs = json.loads(raw_uns["attrs_json"])
    elif isinstance(raw_uns.get("attrs"), dict):
        attrs = dict(raw_uns["attrs"])
    return LandmarkObject(
        density_by_sample=dens_df,
        umap=umap_df,
        metadata=meta,
        nn_idx_landmark=nn_idx,
        nn_dist_landmark=nn_dist,
        attrs=attrs,
    )
