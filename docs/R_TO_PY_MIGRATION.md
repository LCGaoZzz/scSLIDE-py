# R (`Seurat v6` / `scSLIDE`) to Python (`scslide-py`) Migration Guide (`v0.3.3`)

`scslide-py` (`v0.3.3`) provides a native Python replacement for the 19 capabilities (`F01`–`F19`) of upstream R `scSLIDE` (`v0.1.1`) and `Seurat` (`v6.0.0`) sample-level workflows. It does **not** directly execute `.R` scripts or read `.rds` files implicitly via `rpy2`. Users migrating from R perform a one-time, ID-verified export from Seurat (or start directly from `AnnData` `.h5ad` / `.zarr`).

## 1. Object, Axis, Layer & Convention Translation

| Concept | Upstream R (`Seurat` / `scSLIDE`) | Python (`scslide-py` `v0.3.3`) | Migration & Safety Contract |
|---|---|---|---|
| Cell-level expression orientation | `genes × cells` (`dgCMatrix` in `object[["RNA"]]`) | `cells × genes` (`adata` with CSR/CSC/backed `X` & `layers`) | Transpose on one-time export (`Matrix::t()` or `anndata` reader); verify `adata.obs_names` (cells) and `adata.var_names` (genes) before running. |
| Raw UMI counts vs log-normalized data | `LayerData(obj[["RNA"]], "counts")` / `"data"` | `adata.layers["counts"]` / `adata.layers["data"]` | Keep integer counts in `layers["counts"]` for `aggregate_pseudobulk` & `traj_de_test`; `normalize_and_hvg(adata)` populates `layers["data"]` (`log1p(CP10k)`) without densifying `X`. |
| Cell-level embeddings | `Embeddings(obj, "pca")` / `"azimuth_embed"` | `adata.obsm["X_pca"]` / `adata.obsm["X_scVI"]` | Pass `embedding_key="X_pca"` (or `name_reduction_1="X_pca"`, `dims_reduction_1=range(k)`). Zero-based column slicing in Python (`range(10)` corresponds to R `1:10L`). |
| Sample vs biological subject | Often conflated as `group.by = "Donor"` | `sample_col` + `resolve_subject_mapping(..., subject_col=..., subject_id_sep=..., time_col=...)` | Audit longitudinal repeat visits/specimens upfront; restrict subject-level inference or primary trajectories with `s_obj.subset_samples(subj_map.primary_samples, recompute_normalization=False)`. |
| Rectangular query-to-landmark neighbors | `obj[["weighted.nn"]]` (`Neighbor@nn.idx`, 1-based `1..L`) | `NeighborResult.nn_idx` & `adata.obsm["scslide_<name>_idx"]` | **1-based landmark indices (`1..L`) are preserved** inside `NeighborResult.nn_idx` and `landmark_matrix` to match R `Neighbor@nn.idx`. Stored in `.obsm` + `.uns["scslide"]["neighbors"]` (never `.obsp`). |
| Sample-level `LMC` object | `Seurat` object (`landmarks × samples`, `counts` + `data` = ChiSq residuals, `scale.data` = centered clipped) | `SampleLevelObject` (`counts`, `residuals`, `normalized` as `landmarks × samples` DataFrames) OR sample-level `AnnData` (`samples × landmarks`) | `SampleLevelObject` keeps R's `landmarks × samples` DataFrame orientation; `sample_object_to_anndata(s_obj, include_centered_residuals=True)` writes schema-tagged `samples × landmarks` `.h5ad`/`.zarr` (`uns["scslide"]["observation_unit"] == "sample"`). |
| Sample filtering (`nCount_LMC >= cutoff`) | `subset(sample_all, subset = nCount_LMC >= 1000)` | `s_all.subset_samples(keep_ids, recompute_normalization=False)` | Default `recompute_normalization=False` preserves pre-filter Pearson residuals identically to `Seurat::subset`. |
| Sample residual centering & clipping | `ScaleData(sample_obj, do.scale=FALSE, do.center=TRUE)` (`scale.max=10` upper clip) | `s_obj.centered_residuals(scale_max=10.0)` | Clips positive outliers at `+10.0` while leaving negative values unclipped (`FastDenseRowScale` semantics). |
| Reference PLS model & query projection | `RunPLS(..., save.model=TRUE)` + `ProjectCellEmbeddings` / `PredictPLS` | `PLSModel.save("model.npz")` / `PLSModel.load("model.npz")` + `project_pls_embeddings` / `predict_pls` | Non-pickle `.npz` (`allow_pickle=False`). `DataFrame` inputs enforce an exact bijective match to `model.feature_names` (missing or extra columns raise `ValueError`); `AnnData` or sparse/dense matrices with `feature_names=...` select `model.feature_names` in order from a superset and raise `ValueError` if any training feature is missing or duplicated (never zero-fills silently). |
| Trajectory DE (`TrajDETest`) | `TrajDETest(bulk_obj, traj.var=..., latent.vars=c("Age","Sex","log_count"))` | `traj_de_test(sub_pb, traj_var=..., latent_vars=["Age","Sex","log_count"])` | Matches `glmGamPoi::glm_gp(..., size_factors=FALSE)` + quasi-likelihood F-test (`library_size_mode="covariate"`, `test_mode="ql_f"`). Nominal `p_Traj`/`adj_pval` on a learned trajectory are **descriptive** (`inference_tier="descriptive_post_selection_r_compatible"`). |

## 2. One-Time Export from an Existing R `Seurat` `.rds` Object

If your cohort currently lives in an R `Seurat` `.rds` file, run [`../examples/export_seurat_for_scslide_py.R`](../examples/export_seurat_for_scslide_py.R) once in R to export a portable directory (`counts_cells_by_genes.mtx`, `genes.txt`, `cells.txt`, `cell_metadata.csv`, optional `variable_features.txt`, optional `embedding_<reduction>.csv`, `export_summary.txt`, and `export_manifest.json`), then load it into Python `AnnData` (via `load_exported_r_bundle` in [`../examples/r_to_python_migration_quickstart.py`](../examples/r_to_python_migration_quickstart.py)) and verify cell/gene/sample counts and IDs before running `scslide-py`. Core `scslide-py` never invokes R afterwards.

> **Small-cohort DiffusionMap note (`N <= 20` demo cohorts)**: Upstream `RunDiffusionMap` selects largest-magnitude eigenvalues (`eigen_which="LM"`) and weights eigenvectors by `sqrt(max(lambda, 0))`. On very small synthetic cohorts (`N = 10..12`), negative oscillatory eigenvalues can exceed the second positive eigenvalue in magnitude and get truncated to `0.0`; pass `eigen_which="LA"` (as `r_to_python_migration_quickstart.py` does when `n_samples <= 20`) to retain strictly positive 2D diffusion coordinates for visualization.

## 3. Complete Portable Python Workflow (No Developer Paths Required)

Run [`../examples/r_to_python_migration_quickstart.py`](../examples/r_to_python_migration_quickstart.py) in any clean environment with `scslide-py[anndata]` installed:

```bash
python -m pip install "scslide-py[anndata]"
python examples/r_to_python_migration_quickstart.py --out-dir ./scslide_demo_out
```

That script executes end-to-end offline using `sl.build_synthetic_cohort_fixture()` (clearly tagged `synthetic_demo_only=True`), demonstrating:
1. Cohort subject-vs-draw auditing (`resolve_subject_mapping`);
2. Sketched CPPLS + cell-to-landmark WNN (`prepare_sample_object`, `nn_method="exact"`, `n_jobs=1`);
3. Sample-level LMC contingency & Chi-squared Pearson residuals (`generate_sample_object` + `subset_samples`);
4. Diffusion Map & Principal Curve trajectory inference (`run_diffusion_map`, `fit_principal_curve`);
5. R-compatible descriptive trajectory DE (`aggregate_pseudobulk` + `traj_de_test`);
6. Safe persistence & reload (`sample_object_to_anndata` -> `.h5ad` & `.zarr`, `PLSModel.save` -> `.npz`) and named-feature query projection (`project_pls_embeddings`);
7. Publication-ready Diffusion, Clinical Scatter, Landmark Correlation UMAP (`umap_backend="native"`), and Annotated Heatmap figures with automatic `.csv` source-data sidecars.

## 4. Actionable Guardrails for Common Migration Pitfalls

`scslide-py` fails fast with explicit errors rather than silently coercing mismatched inputs:

- **Passing a `cell × gene` `AnnData` into `sample_object_from_anndata`**: raises `ValueError` (`observation_unit != 'sample'`); call `prepare_sample_object` + `generate_sample_object` first.
- **Missing or duplicate query features in `PLSModel.predict` / `project_pls_embeddings`**: raises `ValueError` listing missing/duplicate feature names; align gene symbols explicitly rather than zero-filling.
- **Stringification ID collisions (`1` vs `'1'`) or duplicate cell/sample IDs**: raises `ValueError` (`validate_bijective_ids`).
- **Closed backed HDF5/Zarr handle (`adata.file.close()`)**: raises `RuntimeError` instructing the caller to reopen the store.
- **Changing within-subject response across longitudinal visits in OOF/permutation modes**: raises `ValueError` instructing the caller to pass `primary_draw_only=True` or define a justified subject-level response.
- **Requesting `umap_backend="auto"` without `umap-learn` installed (`allow_fallback=False` default)**: raises `ImportError` instructing the caller to install `scslide-py[ann]` or pass `umap_backend="native"`.
