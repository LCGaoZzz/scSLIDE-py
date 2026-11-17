# Release Scope: Stable Core, Experimental Extensions, and Explicit Boundaries (`v0.3.3`)

This document defines what `scslide-py` (`v0.3.3`) supports as its stable core workflow, what remains explicitly experimental, and what is outside scope.

## 1. Stable Core (Supported Research Workflow Replacement)

`scslide-py` provides a native Python semantic and scientific-workflow counterpart to the 19 capabilities (`F01`–`F19`, detailed in [`algorithm.md`](algorithm.md)) of the upstream R `scSLIDE` (`v0.1.1`) and `Seurat` (`v6.0.0`) sample-level single-cell pipeline, without calling `R` or `rpy2` at runtime:

1. **Cohort & Statistical-Unit Preparation**:
   - Sparse/chunked log-normalization and VST dispersion ranking (`normalize_and_hvg`);
   - Sample-to-biological-subject auditing for repeated draws/specimens (`resolve_subject_mapping`, `stratified_subject_kfold`);
   - Grouped uniform and CountSketch+QR+JL leverage-score sketching (`sketch_data_by_group`, `seurat_leverage_scores`, `F15`);
   - Grouped response-correlation gene selection (`quick_cor_select`, `quick_cor_test`, `F10`).
2. **Supervised Cell-Level PLS & Query Projection**:
   - Native `cppls` (`v6` & `legacy_ondisk`), `plsr` (`kernelpls`), and `spls` solvers across dense, sparse CSR/CSC, and streaming inputs (`fit_pls`, `run_pls`, `cppls_ondisk`, `kernelpls`, `kernelpls_ondisk`, `spls_fit`, `F13`, `F18`, `F19`);
   - Safe non-pickle `.npz` model serialization (`PLSModel.save` / `PLSModel.load`, schema `scslide-pls-model-2`), strict named-feature alignment, response prediction (`predict_pls`, `F08`), and streaming query projection (`project_pls_embeddings`, `F08`).
3. **Cell-to-Landmark Neighborhoods & Sample-Level Contingency (`LMC`)**:
   - Single-modality kNN (`find_sm_nn`, `landmark_neighbors`, `F04`) and multi-modal Weighted Nearest Neighbors (`find_mm_nn`, `compute_modality_weights`, `F03`) with deterministic `nn_method="exact"` (`scipy.spatial.cKDTree`, default stable path);
   - 1-based landmark neighbor aggregation with complete landmark query self-exclusion (`landmark_matrix`, `generate_sample_object`, `SampleLevelObject`, `F05`), Chi-squared Pearson residuals (`normalize_chi_squared`, `pearson_residuals`, `PearsonResidualOperator`, `F06`), `Seurat::subset` pre-filter residual retention (`subset_samples`), and `Seurat::ScaleData` upper-clipped centering (`centered_residuals(scale_max=10.0)`).
4. **Sample Geometry, Trajectories, Batch Adjustment & Descriptive Effects**:
   - Locally scaled anisotropic Diffusion Maps (`run_diffusion_map`, `compare_spectral_embeddings`, `F12`) and Hastie-Stuetzle / Reinsch smoothing-spline Principal Curves (`fit_principal_curve`, `project_to_curve`, `F12`);
   - Control-referenced batch adjustment (`correct_sample_combat`, `F02`; `cellanova_calc_be` + `LazyBatchCorrectedOperator`, `F17`);
   - Donor-by-cell-type pseudobulk count aggregation (`aggregate_pseudobulk`), `filter_by_expr`, `trajectory_fold_change`, and R-compatible `glmGamPoi` quasi-likelihood F-test descriptive trajectory DE (`traj_de_test`, `de_inference_mode="descriptive_r_compat"`, `F16`).
5. **Persistence, Landmark UMAP & Publication Graphics**:
   - Schema-validated `AnnData` H5AD/Zarr round-trips (`sample_object_to_anndata`, `sample_object_from_anndata`, `write_sample_zarr`, `store_rectangular_neighbors`, `load_rectangular_neighbors`, `landmark_object_from_anndata`);
   - Landmark-level density/correlation objects and projected UMAP (`build_landmark_object`, `run_and_project_umap`, `F01`, `F11`) with explicit `umap_backend` provenance (`"umap-learn"` vs `"native"`, no silent runtime exception fallback);
   - Context-isolated publication plots (`plot_landmark_object`, `sample_level_dim_plot`, `plot_sample_diffusion`, `plot_clinical_scatter`, `plot_severity_ridges`, `plot_trajectory_bins`, `plot_annotated_heatmap`, `plot_calibration`, `F07`, `F14`) that never mutate global `matplotlib.rcParams` and emit `.csv` source-data sidecars.

## 2. Experimental Extensions (Explicit Opt-In Only; No Confirmatory Guarantee)

The following post-selection / resampling procedures are retained as **explicitly exploratory** diagnostics (`confirmatory_calibration_established = False`, `shared_donor_latent_fdr_calibrated = False`):

- `run_confirmatory_oof_traj_de` (and workflow modes `exploratory_oof_countsplit`, `exploratory_oof_phenotype_perm`):
  - Parametric NB-QLF p/q values on binomial count-split OOF trajectories do **not** control FDR when donors share cross-gene latent covariance (`Cov(X^(1), X^(2)) = p(1-p)Var(Lambda) > 0`).
  - Phenotype-shuffle p-values (`gene_wise_perm_null=True`) hold observed-label stratified folds fixed during global subject label shuffles and test a global phenotype-independence null (`H0_pheno: Y _|_ X`) rather than individual conditional gene nulls; neither finite-sample strong FWER nor gene-wise FDR control is established.
- `traj_de_permutation_test` (and workflow mode `exploratory_insample_residual_perm`):
  - Permutes covariate-residualized trajectory values (`t_resid`), not expression residuals; external trajectory provenance alone does not establish exchangeability or multiplicity control.
- `run_unified_permutation_inference` and `calibrate_traj_de_donor_permutation`:
  - Subject-level phenotype-null diagnostics conditional on subject exchangeability and user callback fidelity; `missing_gene_policy` (`"conservative_one"`, `"nan"`, `"error"`) explicitly records finite vs missing permutation counts (`emp_n_finite_perms`, `emp_n_missing_perms`, `emp_missing_perm_rate`).
- Legacy compatibility names (`run_confirmatory_oof_traj_de`, `emp_p_maxf_fwer`, and workflow aliases `confirmatory_residual_perm`, `confirmatory_oof_countsplit`, `confirmatory_oof_phenotype_perm`) are retained solely to avoid breaking existing scripts; workflow aliases emit a `UserWarning` directing callers to `exploratory_*` or `descriptive_r_compat`.

## 3. Neighbor & UMAP Backend Scope

| Backend | Extra / Package | Status | Behavior & Boundaries |
|---|---|---|---|
| `nn_method="exact"` | Core (`scipy.spatial.cKDTree`) | **Stable default** | Exact float64 Euclidean kNN with deterministic tie-breaking; invariant across `n_jobs` (`1`, `4`, `8`, `-1`). |
| `nn_method="annoy"` | `scslide-py[annoy]` (`annoy>=1.17`) | **Supported optional approximate** | Matches Seurat `RcppAnnoy` (`n_trees=50` default). Forwards resolved `n_jobs` to `AnnoyIndex.build` (`build_workers_effective`) and runs serial queries (`query_workers_effective=1`). Multithreaded forest construction can alter approximate neighbor ties slightly relative to `n_jobs=1` or `exact`; downstream scientific preservation is verified on COVID-19 and SEA-AD cases, not bit-identical neighbor indices. |
| `nn_method="pynndescent"` | `scslide-py[ann]` (`pynndescent>=0.5`) | **Experimental optional approximate** | Approximate kNN graph search for exploratory scaling; carries `numba` JIT warmup overhead on medium cohorts and stochastic variation across seeds/threads. Use `exact` or `annoy` for R-aligned workflows. |
| `umap_backend="native"` | Core (`scipy`) | **Stable built-in** | Force-directed cross-entropy UMAP layout on precomputed landmark kNN without requiring `umap-learn`. |
| `umap_backend="umap-learn"` | `scslide-py[ann]` (`umap-learn>=0.5`) | **Supported optional** | Calls `umap-learn` simplicial embedding. Under default `umap_backend="auto", allow_fallback=False`, missing `umap-learn` raises an actionable `ImportError` rather than silently switching algorithms. |

## 4. Explicit Out-of-Scope & Non-Claims

- **Not a drop-in R script or `.rds` reader**: Existing R scripts and Seurat `.rds` objects require one-time explicit export of counts, metadata, and embeddings (see [`R_TO_PY_MIGRATION.md`](R_TO_PY_MIGRATION.md) and `examples/export_seurat_for_scslide_py.R`).
- **Not a replacement for the full Seurat ecosystem**: Covers the 19 `scSLIDE` sample-level and trajectory capabilities, not unrelated Seurat spatial, VDJ, or integration pipelines.
- **Retrospective scientific scope**: Compatibility and directional preservation on COVID-19 (40 draws / 36 subjects) and SEA-AD Microglia-PVM (89 total / 76 filtered / 75 DE / 74 QC donors) do not establish prospective clinical equivalence, absence of confounding (e.g., cell-count or QC/PMI sensitivity), universal superiority over pseudobulk/CLR/scLKME, or general post-selection FDR/FWER control.
- **Platform & version verification boundary**: Verified on Linux `x86_64` across Python `3.10`, `3.11`, and `3.12` (see [`SUPPORTED_ENVIRONMENTS.md`](SUPPORTED_ENVIRONMENTS.md)). macOS, Windows, and Python `>=3.13` are not claimed verified in `v0.3.3`.
