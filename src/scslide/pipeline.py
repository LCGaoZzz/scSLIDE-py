"""End-to-end scSLIDE single-cell preparation and sample-level orchestration (F09 ``PrepareSampleObject``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._validation import StreamingMatrix, validate_bijective_ids, validate_open_backed_anndata
from .anndata import sample_object_to_anndata, store_rectangular_neighbors
from .de import (
    SubjectMappingResult,
    aggregate_pseudobulk,
    resolve_subject_mapping,
    run_confirmatory_oof_traj_de,
    traj_de_permutation_test,
    traj_de_test,
)
from .neighbors import NeighborResult, find_mm_nn, find_sm_nn
from .sample import SampleLevelObject, generate_sample_object
from .sketch import sketch_data_by_group
from .supervised import PLSModel, fit_pls, project_pls_embeddings, quick_cor_select
from .trajectory import DiffusionMapResult, PrincipalCurveResult, fit_principal_curve, run_diffusion_map


def normalize_and_hvg(
    adata: Any,
    *,
    counts_layer: str | None = "counts",
    data_layer: str = "data",
    scale_factor: float = 10_000.0,
    n_top_genes: int = 2000,
    chunk_size: int = 8192,
) -> Any:
    """Sparse/chunked Seurat-equivalent ``NormalizeData`` + ``FindVariableFeatures(vst)`` without densifying ``X``."""
    validate_open_backed_anndata(adata, caller="normalize_and_hvg")
    raw_src = adata.layers[counts_layer] if (counts_layer is not None and counts_layer in adata.layers) else adata.X
    if sp.issparse(raw_src):
        csr = raw_src.tocsr().astype(np.float64)
        row_sums = np.asarray(csr.sum(axis=1), dtype=np.float64).ravel()
        inv_s = np.where(row_sums > 0, float(scale_factor) / np.maximum(row_sums, 1e-300), 0.0)
        norm_csr = sp.diags(inv_s) @ csr
        norm_csr = norm_csr.tocsr()
        norm_csr.data = np.log1p(norm_csr.data)
        adata.layers[data_layer] = norm_csr
    else:
        arr = np.asarray(raw_src[:], dtype=np.float64)
        rs = arr.sum(axis=1, keepdims=True)
        norm_arr = np.log1p(np.divide(arr, rs, out=np.zeros_like(arr), where=rs > 0) * float(scale_factor))
        adata.layers[data_layer] = norm_arr

    # Streaming VST-style dispersion ranking on raw counts (Seurat vst approximation via loess log-var vs log-mean)
    sm_c = StreamingMatrix(raw_src, chunk_size=chunk_size)
    mean_c, var_c = sm_c.mean_and_var(ddof=1)
    pos = (mean_c > 0) & (var_c > 0)
    hvg_mask = np.zeros(sm_c.n_vars, dtype=bool)
    std_var = np.zeros(sm_c.n_vars, dtype=np.float64)
    if np.any(pos):
        lx = np.log10(mean_c[pos])
        ly = np.log10(var_c[pos])
        deg = min(2, max(1, len(lx) - 1))
        poly = np.poly1d(np.polyfit(lx, ly, deg))
        expected_var = 10.0 ** poly(lx)
        # Standardized variance ratio
        ratio = var_c[pos] / np.maximum(expected_var, 1e-12)
        std_var[pos] = ratio
        top_k = min(int(n_top_genes), int(np.sum(pos)))
        top_pos = np.flatnonzero(pos)[np.argsort(-ratio, kind="mergesort")[:top_k]]
        hvg_mask[top_pos] = True

    adata.var["means"] = mean_c
    adata.var["variances"] = var_c
    adata.var["variances_norm"] = std_var
    adata.var["highly_variable"] = hvg_mask
    return adata


@dataclass
class PreparedSampleState:
    """Structured state returned by :func:`prepare_sample_object` (F09)."""

    adata: Any
    pls_model: PLSModel
    neighbors: NeighborResult
    training_cell_ids: list[str]
    landmark_cell_ids: list[str]
    selected_hvgs: list[str]
    cor_selected_genes: list[str]
    nn_name: str
    pls_reduction_use: str


def prepare_sample_object(
    adata: Any,
    *,
    y: str,
    add_hvg: bool = True,
    group_by_cortest: str | None = None,
    cor_cutoff: float | None = None,
    top_prop: float | None = 0.01,
    sketch_training: bool = True,
    group_by_sketch: str | None = None,
    ncells_per_group: int | Mapping[str, int] = 1000,
    training_assay_name: str = "TRAINING",
    training_sketch_method: str = "Uniform",
    training_cells: Sequence[str] | None = None,
    ncells_landmark: int = 2000,
    landmark_assay_name: str = "LANDMARK",
    landmark_sketch_method: str = "LeverageScore",
    landmark_cells: Sequence[str] | None = None,
    ncomp: int = 10,
    pls_function: str = "cppls",
    pls_reduction_name: str = "pls",
    modality: str = "multi",
    k_nn: int = 5,
    knn_range: int = 200,
    l2_norm: bool = True,
    name_reduction_1: str = "pca",
    dims_reduction_1: Sequence[int] | None = None,
    nn_name: str | None = None,
    fix_wnn_weights: Sequence[float] | None = (0.5, 0.5),
    data_layer: str = "data",
    variable_features: Sequence[str] | None = None,
    nn_method: str = "exact",
    annoy_n_trees: int = 50,
    n_jobs: int = 1,
    seed: int = 123,
    reference_profile: str = "v6",
    return_state: bool = False,
) -> Any | PreparedSampleState:
    """Native Python implementation of ``PrepareSampleObject`` (F09, legacy & Seurat v6).

    Executes the complete cell-level preparation chain on ``adata`` (dense, sparse CSR/CSC,
    or backed HDF5/Zarr) without materializing dense ``N_all x G`` expression matrices:
    1. Validates ``y`` (rejecting missing responses upfront under ``reference_profile='v6'``).
    2. Optional grouped training sketch (``SketchDataByGroup``, default ``Uniform``).
    3. Optional response-correlated gene augmentation (``QuickCorTest`` on training subset).
    4. Landmark cell selection from the training subset (``SketchData``, default ``LeverageScore``
       computed over original variable features).
    5. Supervised PLS learning (``plsr`` / ``cppls`` / ``spls``) on scaled training HVGs.
    6. Streaming projection of full dataset into PLS space (``ProjectCellEmbeddings``).
    7. Multi-modal WNN (``FindmmNN``) or single-modal kNN (``FindsmNN``) from all cells to landmarks,
       stored cleanly via :func:`store_rectangular_neighbors`.
    """
    if modality not in ("multi", "single"):
        raise ValueError("modality must be 'multi' or 'single'")
    if pls_function not in ("plsr", "spls", "cppls"):
        raise ValueError("pls_function must be 'plsr', 'spls', or 'cppls'")
    if reference_profile not in ("v6", "legacy", "legacy_v5_bug"):
        raise ValueError("reference_profile must be 'v6', 'legacy', or 'legacy_v5_bug'")

    validate_open_backed_anndata(adata, caller="prepare_sample_object")
    validate_bijective_ids(adata.obs_names, axis_name="adata.obs_names")
    validate_bijective_ids(adata.var_names, axis_name="adata.var_names")

    if y not in adata.obs.columns:
        raise ValueError(f"Response column {y!r} not found in adata.obs")
    if reference_profile == "v6" and adata.obs[y].isna().any():
        n_miss = int(adata.obs[y].isna().sum())
        raise ValueError(f"Y ({y!r}) has NA for {n_miss} cells. Subset to cells with a valid response first.")

    if nn_name is None:
        nn_name_eff = "weighted.nn" if modality == "multi" else "single.nn"
    else:
        nn_name_eff = str(nn_name)

    # Resolve reduction 1 upfront when modality == 'multi'
    red1_key: str | None = None
    if modality == "multi":
        for cand_k in (name_reduction_1, f"X_{name_reduction_1}"):
            if cand_k in adata.obsm:
                red1_key = cand_k
                break
        if red1_key is None:
            raise ValueError(
                f"Complementary embedding {name_reduction_1!r} (or 'X_{name_reduction_1}') not found in adata.obsm"
            )

    # Resolve base VariableFeatures
    var_lookup = {str(v): i for i, v in enumerate(adata.var_names)}
    if variable_features is not None:
        var_feat = [str(f) for f in variable_features if str(f) in var_lookup]
    elif "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
        var_feat = [str(v) for v, hv in zip(adata.var_names, adata.var["highly_variable"]) if hv]
    else:
        raise ValueError("Variable features are not set; run normalize_and_hvg / sc.pp.highly_variable_genes or pass variable_features")
    if len(var_feat) == 0:
        raise ValueError("Zero variable features found in adata.var_names")

    raw_data_mat = adata.layers[data_layer] if data_layer in adata.layers else adata.X
    all_cell_ids = [str(c) for c in adata.obs_names]
    cell_to_idx = {cid: i for i, cid in enumerate(all_cell_ids)}

    # 1. Training sketch
    if sketch_training:
        if training_cells is not None:
            train_ids = [str(c) for c in training_cells if str(c) in cell_to_idx]
            train_idx = np.array([cell_to_idx[c] for c in train_ids], dtype=np.int64)
            sc_uns = dict(adata.uns.get("scslide", {}))
            sk_reg = dict(sc_uns.get("sketches", {}))
            sk_reg[training_assay_name] = {"cell_ids": train_ids, "n_cells": len(train_ids)}
            sc_uns["sketches"] = sk_reg
            adata.uns["scslide"] = sc_uns
        else:
            sk_train = sketch_data_by_group(
                adata,
                group_by=group_by_sketch,
                ncells=ncells_per_group,
                method=training_sketch_method,
                features=var_feat,
                layer=data_layer,
                sketched_name=training_assay_name,
                seed=seed,
                reference_profile="v6" if reference_profile == "v6" else "legacy",
            )
            train_ids = sk_train["cell_ids"]
            train_idx = sk_train["indices"]
    else:
        train_ids = all_cell_ids
        train_idx = np.arange(adata.n_obs, dtype=np.int64)

    # 2. QuickCorTest on training subset (layer = data)
    cor_genes: list[str] = []
    if add_hvg:
        # Materialize lightweight CSR view or pass sliced AnnData view
        if sp.issparse(raw_data_mat):
            train_mat = raw_data_mat.tocsr()[train_idx]
            train_df_y = adata.obs.iloc[train_idx][[y]]
            grp_ct = adata.obs.iloc[train_idx][group_by_cortest] if group_by_cortest else None
            cor_genes = quick_cor_select(
                train_mat,
                train_df_y,
                group_by=grp_ct,
                cor_cutoff=cor_cutoff,
                top_prop=top_prop,
                reference_profile="v6" if reference_profile == "v6" else "legacy",
            )
            # Map feature_i back to adata.var_names
            cor_genes = [str(adata.var_names[int(g.split("_", 1)[1])]) for g in cor_genes]
        else:
            sub_ad = adata[train_idx]
            cor_genes = quick_cor_select(
                sub_ad,
                y=y,
                group_by=group_by_cortest,
                layer=data_layer,
                cor_cutoff=cor_cutoff,
                top_prop=top_prop,
                reference_profile="v6" if reference_profile == "v6" else "legacy",
            )
        seen_hvg = set(var_feat)
        all_hvg = list(var_feat) + [g for g in cor_genes if g not in seen_hvg]
    else:
        all_hvg = list(var_feat)

    # 3. Landmark sketch FROM training subset using original var_feat (matching PrepareSampleObject!)
    if landmark_cells is not None:
        lm_ids = [str(c) for c in landmark_cells if str(c) in cell_to_idx]
        lm_idx_global = np.array([cell_to_idx[c] for c in lm_ids], dtype=np.int64)
        sc_uns = dict(adata.uns.get("scslide", {}))
        sk_reg = dict(sc_uns.get("sketches", {}))
        sk_reg[landmark_assay_name] = {"cell_ids": lm_ids, "n_cells": len(lm_ids)}
        sc_uns["sketches"] = sk_reg
        adata.uns["scslide"] = sc_uns
    else:
        vf_cols = np.array([var_lookup[f] for f in var_feat], dtype=np.int64)
        sm_train_vf = StreamingMatrix(raw_data_mat, row_indices=train_idx, col_indices=vf_cols)
        target_lm = min(int(ncells_landmark), len(train_idx))
        if len(train_idx) <= target_lm:
            lm_local = np.arange(len(train_idx), dtype=np.int64)
        elif landmark_sketch_method == "LeverageScore":
            from .sketch import seurat_leverage_scores

            lev = seurat_leverage_scores(sm_train_vf, seed=seed)
            probs = lev / float(lev.sum())
            lm_local = np.random.default_rng(seed).choice(len(train_idx), size=target_lm, replace=False, p=probs)
        else:
            lm_local = np.random.default_rng(seed).choice(len(train_idx), size=target_lm, replace=False)

        lm_idx_global = train_idx[lm_local]
        lm_ids = [all_cell_ids[int(i)] for i in lm_idx_global]
        sc_uns = dict(adata.uns.get("scslide", {}))
        sk_reg = dict(sc_uns.get("sketches", {}))
        sk_reg[landmark_assay_name] = {"cell_ids": lm_ids, "n_cells": len(lm_ids)}
        sc_uns["sketches"] = sk_reg
        adata.uns["scslide"] = sc_uns

    # 4. Fit PLS on training subset (ScaleData with scale.max=10 + RunPLS)
    hvg_cols = np.array([var_lookup[f] for f in all_hvg], dtype=np.int64)
    sm_train_hvg = StreamingMatrix(raw_data_mat, row_indices=train_idx, col_indices=hvg_cols)
    blocks = [blk.toarray() if sp.issparse(blk) else np.asarray(blk, dtype=np.float64) for _, _, blk in sm_train_hvg.iter_row_chunks()]
    train_x_dense = np.vstack(blocks)
    train_y_df = adata.obs.iloc[train_idx][[y]].copy()

    pls_prof = "v6" if reference_profile == "v6" else "legacy_ondisk"
    pls_model = fit_pls(
        train_x_dense,
        train_y_df,
        n_components=ncomp,
        scale=True,
        scale_max=10.0,
        method=pls_function,
        reference_profile=pls_prof,
        clamp_rank=True,
        feature_names=all_hvg,
    )

    # 5. Project PLS to full dataset (ProjectCellEmbeddings) when sketch_training == True
    if sketch_training:
        sm_full_hvg = StreamingMatrix(raw_data_mat, col_indices=hvg_cols)
        full_pls = project_pls_embeddings(pls_model, sm_full_hvg, ncomp=pls_model.n_components, scale_max=10.0)
        pls_use_name = f"proj.{pls_reduction_name}"
        if not getattr(adata, "isbacked", False):
            adata.obsm[f"X_{pls_use_name}"] = full_pls
    else:
        full_pls = np.asarray(pls_model.scores_, dtype=np.float64)
        pls_use_name = pls_reduction_name
        if not getattr(adata, "isbacked", False):
            adata.obsm[f"X_{pls_use_name}"] = full_pls

    # 6. Build cell-to-landmark WNN or single-modal NN graph
    if modality == "multi":
        assert red1_key is not None
        emb1 = np.asarray(adata.obsm[red1_key][:], dtype=np.float64)
        if dims_reduction_1 is None:
            d1 = list(range(min(30, emb1.shape[1])))
        else:
            d1 = [int(d) for d in dims_reduction_1 if 0 <= int(d) < emb1.shape[1]]
            if len(d1) < len(list(dims_reduction_1)):
                warnings.warn(
                    f"dims_reduction_1 exceeded columns of {red1_key} ({emb1.shape[1]}); truncated to valid dimensions",
                    UserWarning,
                    stacklevel=2,
                )
        emb1 = emb1[:, d1]
        nn_res = find_mm_nn(
            [emb1, full_pls],
            lm_idx_global,
            k_nn=k_nn,
            knn_range=knn_range,
            l2_norm=l2_norm,
            fix_wnn_weights=fix_wnn_weights,
            query_ids=all_cell_ids,
            landmark_ids=lm_ids,
            nn_method=nn_method,
            annoy_n_trees=annoy_n_trees,
            reference_profile="legacy_v5_bug" if reference_profile == "legacy_v5_bug" else "v6",
            n_jobs=n_jobs,
            seed=seed,
        )
    else:
        nn_res = find_sm_nn(
            full_pls,
            lm_idx_global,
            k_nn=k_nn,
            l2_norm=l2_norm,
            query_ids=all_cell_ids,
            landmark_ids=lm_ids,
            nn_method=nn_method,
            annoy_n_trees=annoy_n_trees,
            n_jobs=n_jobs,
            seed=seed,
        )

    if not getattr(adata, "isbacked", False):
        store_rectangular_neighbors(
            adata,
            nn_name_eff,
            nn_idx=nn_res.nn_idx,
            nn_dist=nn_res.nn_dist,
            landmark_ids=lm_ids,
            query_ids=all_cell_ids,
            params=nn_res.params,
        )

    state = PreparedSampleState(
        adata=adata,
        pls_model=pls_model,
        neighbors=nn_res,
        training_cell_ids=train_ids,
        landmark_cell_ids=lm_ids,
        selected_hvgs=all_hvg,
        cor_selected_genes=cor_genes,
        nn_name=nn_name_eff,
        pls_reduction_use=pls_use_name,
    )
    return state if return_state else adata


@dataclass
class WorkflowResult:
    """Complete end-to-end scSLIDE native workflow output."""

    prepared: PreparedSampleState
    sample_object: SampleLevelObject
    sample_adata: Any
    diffusion: DiffusionMapResult
    principal_curve: PrincipalCurveResult | None = None
    pseudobulk_adata: Any | None = None
    de_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    subject_mapping: SubjectMappingResult | None = None
    inference_summary: dict[str, Any] = field(default_factory=dict)


def run_scslide_workflow(
    adata: Any,
    *,
    sample_col: str,
    response_col: str,
    subject_col: str | None = None,
    subject_id_sep: str | None = None,
    celltype_col: str | None = None,
    embedding_key: str | None = None,
    dims_embedding: Sequence[int] | None = None,
    ncells_per_group: int = 1000,
    ncells_landmark: int = 2000,
    ncomp_pls: int = 20,
    pls_function: str = "cppls",
    k_nn: int = 5,
    knn_range: int = 200,
    nn_method: str = "exact",
    fix_wnn_weights: Sequence[float] | None = (0.5, 0.5),
    min_lmc_count: int | None = None,
    recompute_residuals_after_filter: bool = False,
    scale_max_sample: float | None = 10.0,
    ncomp_diffusion: int = 10,
    fit_curve: bool = True,
    de_celltype: str | None = None,
    de_covariates: Sequence[str] = ("Age", "Sex", "log_count"),
    de_inference_mode: str = "descriptive_r_compat",
    n_permutations: int = 199,
    n_jobs: int = 1,
    seed: int = 123,
    reference_profile: str = "v6",
) -> WorkflowResult:
    """High-level user pipeline from cell-level ``AnnData`` to sample trajectory and donor/subject DE."""
    validate_open_backed_anndata(adata, caller="run_scslide_workflow")
    valid_modes = (
        "descriptive_r_compat",
        "exploratory_insample_residual_perm",
        "confirmatory_residual_perm",
        "exploratory_oof_countsplit",
        "confirmatory_oof_countsplit",
        "confirmatory_oof_phenotype_perm",
        "exploratory_oof_phenotype_perm",
    )
    if de_inference_mode not in valid_modes:
        raise ValueError(f"de_inference_mode must be one of {valid_modes}")

    legacy_mode_map = {
        "confirmatory_residual_perm": "exploratory_insample_residual_perm",
        "confirmatory_oof_countsplit": "exploratory_oof_countsplit",
        "confirmatory_oof_phenotype_perm": "exploratory_oof_phenotype_perm",
    }
    effective_de_mode = legacy_mode_map.get(de_inference_mode, de_inference_mode)
    if de_inference_mode in legacy_mode_map:
        warnings.warn(
            f"de_inference_mode={de_inference_mode!r} is a legacy compatibility alias and does not establish "
            f"confirmatory FDR/FWER guarantees; executing exploratory mode {effective_de_mode!r}. "
            f"Migrate callers to de_inference_mode={effective_de_mode!r} (or 'descriptive_r_compat' for standard R-compatible workflows).",
            UserWarning,
            stacklevel=2,
        )

    subj_map = resolve_subject_mapping(
        adata.obs,
        sample_col=sample_col,
        subject_col=subject_col,
        subject_id_sep=subject_id_sep,
        strict_invariant_check=False,
    )

    modality = "multi" if embedding_key is not None else "single"
    prep = prepare_sample_object(
        adata,
        y=response_col,
        add_hvg=True,
        group_by_cortest=celltype_col,
        sketch_training=True,
        group_by_sketch=sample_col,
        ncells_per_group=ncells_per_group,
        ncells_landmark=ncells_landmark,
        ncomp=ncomp_pls,
        pls_function=pls_function,
        modality=modality,
        k_nn=k_nn,
        knn_range=knn_range,
        nn_method=nn_method,
        name_reduction_1=embedding_key or "pca",
        dims_reduction_1=dims_embedding,
        fix_wnn_weights=fix_wnn_weights,
        n_jobs=n_jobs,
        seed=seed,
        reference_profile=reference_profile,
        return_state=True,
    )
    assert isinstance(prep, PreparedSampleState)

    s_obj = generate_sample_object(
        prep.neighbors.nn_idx,
        group_by=sample_col,
        k_nn=k_nn,
        normalization_method="ChiSquared",
        rename_group_by=celltype_col,
        add_meta_data=True,
        center_residuals=True,
        scale_max=scale_max_sample,
        reference_profile="v6" if reference_profile == "v6" else "legacy",
        sample_for_cell=adata.obs[sample_col].tolist(),
        landmark_ids=prep.landmark_cell_ids,
        cell_ids=[str(c) for c in adata.obs_names],
        cell_metadata=adata.obs,
    )
    assert isinstance(s_obj, SampleLevelObject)

    if min_lmc_count is not None and s_obj.metadata is not None:
        keep_s = list(s_obj.metadata.index[s_obj.metadata["nCount_LMC"] >= int(min_lmc_count)].astype(str))
        if len(keep_s) < 3:
            raise ValueError(f"Fewer than 3 samples remain after min_lmc_count={min_lmc_count} filter")
        s_obj = s_obj.subset_samples(keep_s, recompute_normalization=recompute_residuals_after_filter)

    dm = run_diffusion_map(s_obj, ncomp=ncomp_diffusion, layer="scale.data", scale_max=scale_max_sample)
    assert isinstance(dm, DiffusionMapResult)
    if s_obj.metadata is not None:
        for j in range(min(3, dm.embeddings.shape[1])):
            s_obj.metadata[f"traj_DC{j + 1}"] = dm.embeddings.iloc[:, j].to_numpy()

    pcurve: PrincipalCurveResult | None = None
    if fit_curve and dm.embeddings.shape[1] >= 2:
        pcurve = fit_principal_curve(dm.embeddings.iloc[:, :2], maxit=20)
        if s_obj.metadata is not None:
            s_obj.metadata["traj_overall"] = pcurve.lambda_

    s_adata = sample_object_to_anndata(
        s_obj, include_centered_residuals=True, scale_max=scale_max_sample
    )

    pb_adata = None
    de_dict: dict[str, pd.DataFrame] = {}
    inf_summary: dict[str, Any] = {
        "de_inference_mode": de_inference_mode,
        "de_inference_mode_effective": effective_de_mode,
        "legacy_mode_alias_used": bool(de_inference_mode in legacy_mode_map),
        "confirmatory_calibration_established": False,
        "n_samples": subj_map.n_samples,
        "n_unique_subjects": subj_map.n_subjects,
        "has_repeated_subject_draws": subj_map.n_subjects < subj_map.n_samples,
        "subject_mapping_sha256": subj_map.sha256,
        "nn_backend_requested": prep.neighbors.params.get("backend_requested"),
        "nn_backend_effective": prep.neighbors.params.get("backend_effective"),
        "nn_backend_version": prep.neighbors.params.get("backend_version"),
        "workers_requested": prep.neighbors.params.get("workers_requested", 1),
        "workers_effective": prep.neighbors.params.get("workers_effective", 1),
        "build_workers_effective": prep.neighbors.params.get("build_workers_effective", 1),
        "query_workers_effective": prep.neighbors.params.get("query_workers_effective", 1),
        "available_cpus": prep.neighbors.params.get("available_cpus"),
        "exceeds_available_cpus": prep.neighbors.params.get("exceeds_available_cpus", False),
    }
    if de_celltype is not None and celltype_col is not None:
        if de_inference_mode in (
            "exploratory_oof_countsplit", "confirmatory_oof_countsplit",
            "confirmatory_oof_phenotype_perm", "exploratory_oof_phenotype_perm",
        ):
            use_gw_perm = de_inference_mode in (
                "confirmatory_oof_phenotype_perm", "exploratory_oof_phenotype_perm",
            )
            oof_res = run_confirmatory_oof_traj_de(
                adata,
                sample_col=sample_col,
                response_col=response_col,
                de_celltype=de_celltype,
                celltype_col=celltype_col,
                subject_col=subject_col,
                subject_id_sep=subject_id_sep,
                ncells_per_group=min(ncells_per_group, 500),
                ncomp_pls=min(ncomp_pls, 10),
                pls_function=pls_function,
                de_covariates=de_covariates,
                n_phenotype_permutations=n_permutations if use_gw_perm else 49,
                gene_wise_perm_null=use_gw_perm,
                n_jobs=n_jobs,
                seed=seed,
            )
            de_dict[de_celltype] = oof_res["de_results"]
            inf_summary.update(
                {k: v for k, v in oof_res.items() if k not in ("de_results", "oof_trajectory")}
            )
        else:
            pb_adata = aggregate_pseudobulk(adata, group_by=(sample_col, celltype_col))
            sub_mask = pb_adata.obs[celltype_col].astype(str) == str(de_celltype)
            if np.any(sub_mask) and s_obj.metadata is not None:
                sub_pb = pb_adata[sub_mask].copy()
                s_meta = s_obj.metadata.set_index(sample_col, drop=False)
                keep_pb_rows = [i for i, d in enumerate(sub_pb.obs[sample_col].astype(str)) if d in s_meta.index]
                sub_pb = sub_pb[keep_pb_rows].copy()
                for col in s_meta.columns:
                    if col not in sub_pb.obs.columns:
                        sub_pb.obs[col] = s_meta.loc[sub_pb.obs[sample_col].astype(str), col].to_numpy()
                avail_cov = [c for c in de_covariates if c in sub_pb.obs.columns]
                traj_col = "traj_overall" if "traj_overall" in sub_pb.obs.columns else "traj_DC1"
                if de_inference_mode in ("exploratory_insample_residual_perm", "confirmatory_residual_perm"):
                    de_dict[de_celltype] = traj_de_permutation_test(
                        sub_pb,
                        traj_var=traj_col,
                        latent_vars=avail_cov if avail_cov else None,
                        n_permutations=n_permutations,
                        seed=seed,
                        trajectory_provenance="in_sample_learned",
                    )
                else:
                    de_dict[de_celltype] = traj_de_test(
                        sub_pb,
                        traj_var=traj_col,
                        latent_vars=avail_cov if avail_cov else None,
                    )
                inf_summary["inference_tier"] = de_dict[de_celltype].attrs.get("inference_tier")

    return WorkflowResult(
        prepared=prep,
        sample_object=s_obj,
        sample_adata=s_adata,
        diffusion=dm,
        principal_curve=pcurve,
        pseudobulk_adata=pb_adata,
        de_results=de_dict,
        subject_mapping=subj_map,
        inference_summary=inf_summary,
    )
