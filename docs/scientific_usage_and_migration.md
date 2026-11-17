# Scientific use and migration to v0.3.3

The completed scientific campaigns used retrospective COVID blood and SEA-AD Microglia-PVM subsets. Neither is an untouched external cohort. See [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md) and [`R_TO_PY_MIGRATION.md`](R_TO_PY_MIGRATION.md) for the release scope and R-to-Python migration guide; repository archive `../validation/README.md` records the full S1–S7 adjudications superseding overstrong claims in earlier campaign reports.

## Inference modes

| API | Output meaning | Supported interpretation |
|---|---|---|
| `traj_de_test` / `descriptive_r_compat` | R-compatible NB quasi-likelihood effects and nominal p/q values | Descriptive post-selection ranking; no de novo FDR guarantee for a learned trajectory. |
| `run_unified_permutation_inference` | User-supplied complete pipeline under subject phenotype shuffles | Phenotype-null diagnostic, conditional on exchangeability and the callback implementing the same procedure. It does not automatically test every conditional gene null. |
| `traj_de_permutation_test` | Permutes residualized trajectory values, holding counts fixed | Exploratory for both external and in-sample trajectories. External origin alone does not establish exchangeable residuals or independence of repeated specimens. This is not an implemented expression-residual Freedman–Lane procedure. |
| `run_confirmatory_oof_traj_de(..., gene_wise_perm_null=False)` | OOF/count-split parametric NB effects | Exploratory; shared donor latent effects can induce dependence between split counts. Limited simulations are not a general calibration proof. |
| `run_confirmatory_oof_traj_de(..., gene_wise_perm_null=True)` / `exploratory_oof_phenotype_perm` | Gene-aligned phenotype-null and maximum-statistic empirical values | Exploratory. Observed-label stratified folds remain fixed during global phenotype shuffles. Neither gene-wise FDR nor strong FWER is established; the global phenotype null differs from individual gene conditional nulls. |

v0.3.3 retains historical function, argument and column names for compatibility. Workflow aliases `confirmatory_residual_perm`, `confirmatory_oof_countsplit` and `confirmatory_oof_phenotype_perm` emit a `UserWarning` directing callers to `exploratory_*` (or `descriptive_r_compat`), record `legacy_mode_alias_used=True` and `confirmatory_calibration_established=False`, and execute the exploratory tier. `shared_donor_latent_fdr_calibrated` is false. `emp_q_bh_oof_perm`, `emp_q_by_oof_perm` and `emp_p_maxf_fwer` are numerical diagnostic columns, not validated guarantees. The historical 879 unsplit COVID hits are exploratory candidates. Increasing the permutation count improves resolution but does not repair exchangeability or establish the null being tested.

Subject-grouped OOF and phenotype shuffling reject multiple distinct response values within a retained subject (even when `n_phenotype_permutations=0`). Use an explicitly chosen primary visit (`primary_draw_only=True`), or a separately justified repeated-visit analysis. No repeated-visit transformation is silently invented.

## Cohorts, sensitivity and comparator boundaries

Specify sample, subject, counts and embedding explicitly. Audit repeated visits with `resolve_subject_mapping`; keep subjects together in CV. Train-fitted HVGs, PCA, scaling, landmarks and decoders are necessary for inductive use. Shared full-cohort PCA/scVI analyses remain transductive. Inspect multiple diffusion coordinates without selecting a favorable axis on the final evaluation endpoint.

Retain donor QC and covariate sensitivity results. SEA-AD CPS associations weaken after QC adjustment; non-significant PMI associations cannot exclude confounding. COVID cell counts are associated with disease groups in these subsets. Gene/program similarity across languages does not resolve shared confounding.

The tested pure-composition and conditional-mean simulations favor CLR and pseudobulk, respectively. The mean-matched shape simulation supports distributional representations, including scLKME; scSLIDE did not dominate. Rare-state and nuisance/batch scenarios are useful counterexamples. These finite results do not prove a universal null space or guarantee that adding CLR preserves every other task. R/transductive and Python/strict-inductive columns must not be compared as one matched leaderboard.

## Resource policy

Neighbor APIs default to `n_jobs=1`. Negative requests resolve against affinity and visible cgroup limits at conventional Linux mounts, including process ancestors; custom mounts or hidden container ancestors are not verified. Positive requests are explicit and may exceed detected availability (`exceeds_available_cpus=True`). `Annoy` forwards the resolved count to forest construction (`workers_effective` / `build_workers_effective`) while running serial queries (`query_workers_effective=1`). Exact `cKDTree` and `PyNNDescent` receive the resolved parameter. BLAS, numba and external processes have their own pools: use measured affinity/quota and explicit pool limits for a total CPU budget. Requested workers are not observed thread counts or CPU-time quotas.

| Path | Scope |
|---|---|
| Core | NumPy/pandas/SciPy/scikit-learn/matplotlib; no R runtime dependency. |
| `[anndata]` | AnnData/H5AD/Zarr adapters; explicit count/data layers and open backed handles required. |
| `[ann]` | Adds PyNNDescent (experimental approximate kNN) and umap-learn. Does **not** install Annoy. |
| `[annoy]` | Adds Annoy (`nn_method='annoy'`, matching Seurat `RcppAnnoy` with `n_trees=50`). |
| UMAP without `[ann]` | Choose `umap_backend='native'`; strict auto mode raises if umap-learn is absent. |
| Models and outputs | Non-pickle PLS `.npz`, sample/landmark object round trips, plotting and CSV exports; see executable tutorial. |

See [`SUPPORTED_ENVIRONMENTS.md`](SUPPORTED_ENVIRONMENTS.md) for the verified Linux Python `3.10`/`3.11`/`3.12` matrix and pinned `constraints/` profiles. Stored R golden comparison (`47 checks + 6 negative controls`) verifies numerical parity against archived R outputs; the optional GitHub Actions R job only runs `sessionInfo()` as an environment probe and does not regenerate live R oracles.

With `missing_gene_policy="conservative_one"`, absent/null-invalid gene statistics count as exceedances (`empirical p = 1.0` if all draws are missing), and `emp_n_valid_perms`, `emp_n_finite_perms`, `emp_n_missing_perms`, and `emp_missing_perm_rate` record the exact missingness breakdown. The `nan` policy conditions on finite draws and needs a justified missingness mechanism; no error-control claim follows automatically.
