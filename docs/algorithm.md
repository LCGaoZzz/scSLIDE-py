# Algorithm Map, Support Matrix & Dual-Reference Compatibility (`v0.3.3`)

`scslide-py` (`v0.3.3`) implements all 19 upstream `scSLIDE` / Seurat v6 capabilities natively in Python without `Rscript` or `rpy2` runtime bridges, with explicit neighbor-worker resolution using affinity and visible conventional cgroup limits (`resolve_effective_workers`, `n_jobs`) and explicitly exploratory inference extensions.

The tables describe implemented paths; support is limited to the verified environments and fixtures documented in [`SUPPORTED_ENVIRONMENTS.md`](SUPPORTED_ENVIRONMENTS.md) and [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md). They are not a guarantee for every future dependency or storage combination. Explicit positive worker requests may exceed availability; separate thread pools need their own limits.

## 1. Upstream R-to-Python Parameter & Capability Mapping (`F01`–`F19`)

| ID | Upstream R Export (`Seurat v6` / `scSLIDE`) | `scslide-py` `v0.3.3` Public API | Key Parameter & Numerical Notes |
|---|---|---|---|
| `F01` | `BuildLandmarkObject` | `build_landmark_object`, `LandmarkObject`, `landmark_object_from_anndata` | Transposes log-normalized LMC densities (`obs = landmark`, `var = sample`), fits landmark UMAP with explicit `umap_backend`, computes `cor.coef_<feature>` (`na.omit`). |
| `F02` | `CorrectSampleComBat` | `correct_sample_combat`, `SampleComBatResult` | Empirical-Bayes / non-parametric control-referenced sample batch correction (`regression` & `subtraction`, `d_hat == 0` guard). |
| `F03` | `FindmmNN` | `find_mm_nn`, `compute_modality_weights`, `NeighborResult` | Clamps internal `k_nn` to `>= 20` matching Seurat; supports `nn_method='exact'` (`cKDTree` float64, checked against stored R `RANN::nn2` fixtures), `'annoy'` (`n_trees=50`, matching Seurat `RcppAnnoy`), and `'pynndescent'`. |
| `F04` | `FindsmNN` | `find_sm_nn`, `landmark_neighbors`, `NeighborResult` | Single-modality query-to-landmark Euclidean kNN with optional L2 row normalization and deterministic tie-breaking. |
| `F05` | `GenerateSampleObject` | `generate_sample_object`, `landmark_matrix`, `SampleLevelObject` | 1-based landmark neighbor aggregation with complete landmark query self-exclusion; `SampleLevelObject.subset_samples(..., recompute_normalization=False)` matches `Seurat::subset` by retaining pre-filter residuals. |
| `F06` | `NormalizeChiSquared` | `normalize_chi_squared`, `pearson_residuals`, `expected_counts`, `PearsonResidualOperator` | Exact `(C - E)/sqrt(E)` with zero-margin `0.0`; `SampleLevelObject.centered_residuals(scale_max=10.0)` clips upper bound `+10.0` matching `Seurat::ScaleData(do.scale=FALSE, do.center=TRUE, scale.max=10)` (`FastDenseRowScale`). |
| `F07` | `PlotLandmarkObject` | `plot_landmark_object` | Context-isolated diverging UMAP correlation facets and donor density views with `.csv` source-data sidecars. |
| `F08` | `PredictPLS` / `ProjectCellEmbeddings` | `predict_pls`, `project_pls_embeddings`, `PLSModel` | Named-feature alignment, safe `.npz` persistence (`PLSModel.save`/`load`), streaming chunked projection. |
| `F09` | `PrepareSampleObject` | `prepare_sample_object`, `run_scslide_workflow`, `normalize_and_hvg` | Full cell-to-landmark preparation across dense, CSR/CSC, and open backed HDF5/Zarr `AnnData`. |
| `F10` | `QuickCorTest` | `quick_cor_select`, `quick_cor_test` | Grouped `model.matrix(~ . + 0)` response encoding, `min_cells=100`, `min_pct=0.1`, exact streamed cross-product correlations. |
| `F11` | `RunAndProjectUMAP` | `run_and_project_umap` | Fits landmark subgraph UMAP and projects all query cells; records `umap_info` (`backend_requested`, `backend_effective`, `fallback_reason`). |
| `F12` | `RunDiffusionMap` | `run_diffusion_map`, `fit_principal_curve`, `compare_spectral_embeddings` | Exact `destiny::DiffusionMap` locally scaled anisotropic Markov operator (`cosine_mode='covertree_cor'`) + Green-Silverman `df=5` Reinsch smoothing spline `principal_curve`. |
| `F13` | `RunPLS` | `run_pls`, `fit_pls`, `spls_fit`, `PLSModel` | Exact `cppls` (`v6` & `legacy_ondisk`), `plsr` (`kernelpls`), and `spls` (`ust` soft-thresholding). |
| `F14` | `SampleLevelDimPlot` | `sample_level_dim_plot`, `plot_sample_diffusion`, `plot_annotated_heatmap` | Non-mutating `plt.rc_context`, `ax=` composition, `show=False`, editable PDF/SVG text, automatic `.csv` sidecars. |
| `F15` | `SketchDataByGroup` | `sketch_data_by_group`, `seurat_leverage_scores`, `uniform_sketch`, `leverage_sketch` | Grouped uniform and CountSketch+QR+JL leverage sketching without densifying sparse/backed matrices. |
| `F16` | `TrajDETest` | `traj_de_test`, `traj_de_permutation_test`, `run_unified_permutation_inference`, `run_confirmatory_oof_traj_de` | Exact `glmGamPoi` NB-QLF descriptive test plus unified subject-level and OOF+count-split exploratory diagnostics. |
| `F17` | `cellanova_calc_BE` | `cellanova_calc_be`, `CellAnovaResult`, `LazyBatchCorrectedOperator` | Control-pool SVD nuisance subspace estimation and out-of-core lazy batch removal. |
| `F18` | `cppls_ondisk` | `cppls_ondisk` | Two-pass chunked CPPLS matching `Seurat` BPCells on-disk solver. |
| `F19` | `kernelpls` | `kernelpls`, `kernelpls_ondisk` | Dayal & MacGregor (1997) KernelPLS across dense, sparse, and backed matrices. |

## 2. Storage, Dependency & Lifecycle Support Matrix

| Input / Storage Mode | Supported Operations | Contract & Error Behavior |
|---|---|---|
| Dense `ndarray` / `DataFrame` | All `F01`–`F19` | Rejects `NaN`/`Inf`, duplicate IDs, and `int`/`str` ID stringification collisions (`1` vs `'1'`). |
| `scipy.sparse` `csr_matrix` / `csc_matrix` | All cell-level, pseudobulk, thinning & sample-level adapters | Never densifies `N x G` expression during normalization, sketching, `QuickCorSelect`, `kernelpls`, `cppls_ondisk`, projection, or `binomial_thin_counts`. |
| Backed `AnnData` (`backed='r'` HDF5) & Zarr (`v2`/`v3`) | `normalize_and_hvg` (in-memory), `prepare_sample_object`, `aggregate_pseudobulk`, `sample_object_from_anndata`, `write_sample_zarr` | Validates open backing file handle upfront (`validate_open_backed_anndata`); closed handles raise immediate actionable `RuntimeError`. |
| Core install (`scslide-py`) | Matrix/DataFrame workflows, `SampleLevelObject`, `PLSModel`, `DiffusionMap`, `TrajDETest`, native UMAP, plots | Importing `AnnData` adapters without `[anndata]` raises actionable `ImportError`. |
| `[anndata]` extra (without `[ann]`) | Full `AnnData`/H5AD/Zarr pipeline + `nn_method='exact'` + `umap_backend='native'` | Default `umap_backend='auto', allow_fallback=False` raises actionable `ImportError` until `[ann]` is installed or `umap_backend='native'` / `allow_fallback=True` is explicitly chosen. |
| `[anndata,ann]` extras | Adds `umap-learn` (`umap_backend='umap-learn'`) and `pynndescent` (`nn_method='pynndescent'`, experimental approximate kNN) | Runtime/numerical exceptions inside `umap-learn` always propagate and are never masked by silent fallback. `[ann]` does **not** install `annoy`. |
| `[annoy]` extra (`pip install "scslide-py[annoy]"`) | Adds `annoy` (`nn_method='annoy'`, `n_trees=50` default matching Seurat `RcppAnnoy`) | Passes resolved `n_jobs` to `AnnoyIndex.build(..., n_jobs=eff_workers)` (`build_workers_effective`) and runs serial vector queries (`query_workers_effective=1`). |
