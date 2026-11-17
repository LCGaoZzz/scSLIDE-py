# Supported Environments, Extras, Storage & Resource Matrix (`v0.3.3`)

`pyproject.toml` declares open lower bounds so `scslide-py` composes cleanly inside scientific Python environments without artificial upper-bound solver lockouts, while `constraints/` records the exact dependency sets verified during release-candidate acceptance. Lower-bound metadata does **not** claim that every future unverified combination is tested.

## 1. Operating System, Python & Build Backend Matrix

| Axis | Verified in `v0.3.3` | Unverified / Out of Claimed Scope |
|---|---|---|
| Operating system & architecture | Linux `x86_64` (local compute node + GitHub Actions `ubuntu-latest`) | macOS (`arm64`/`x86_64`), Windows (`win_amd64`), 32-bit platforms |
| Runtime Python (`Requires-Python: >=3.10`) | CPython `3.10` (`3.10.12`), `3.11` (`3.11.15`), `3.12` (`3.12.14`) | CPython `>=3.13`, PyPy, GraalPy |
| PEP 517 build backend (`pyproject.toml`) | `setuptools>=77.0.3` (`setuptools.build_meta`), tested at exact floor `77.0.3` and the original modern build environment `84.2.1` (`PEP 639` `license = "MIT"` + `license-files = ["LICENSE", "NOTICE"]`) | `setuptools < 77.0.0` (incompatible with PEP 639 string `license`) |

## 2. Installation Profiles & Optional Extras

| Install Command | Included Dependencies | Verified Capabilities | Expected Behavior for Omitted Extras |
|---|---|---|---|
| `pip install scslide-py` (`core`) | `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`, `scikit-learn>=1.3`, `matplotlib>=3.7` | `scslide-calibrate` CLI, dense/sparse matrix `PLSModel`, `find_sm_nn`/`find_mm_nn(nn_method='exact')`, `SampleLevelObject`, `run_diffusion_map`, `fit_principal_curve`, `traj_de_test`, `correct_sample_combat`, `cellanova_calc_be`, `umap_backend='native'`, all `scslide.pl` plots, `build_synthetic_cohort_fixture(as_anndata=False)` | Calling `AnnData` adapters raises `ImportError` pointing to `scslide-py[anndata]`. |
| `pip install "scslide-py[anndata]"` | Core + `anndata>=0.10`, `h5py>=3.8`, `zarr>=2.16` | Full `AnnData` / `.h5ad` / `.zarr` workflows (`run_scslide_workflow`, `prepare_sample_object`, `aggregate_pseudobulk`, `sample_object_to_anndata` / `from_anndata`, `write_sample_zarr`) with `nn_method='exact'` and `umap_backend='native'` | Default `umap_backend='auto', allow_fallback=False` raises `ImportError` until `[ann]` is installed or `umap_backend='native'` / `allow_fallback=True` is explicitly passed. |
| `pip install "scslide-py[anndata,ann]"` | Above + `pynndescent>=0.5`, `umap-learn>=0.5` | Enables `umap_backend='umap-learn'` (and `'auto'`) plus `nn_method='pynndescent'` (experimental approximate kNN). **Note:** `[ann]` does **not** install `annoy`. | Calling `nn_method='annoy'` raises `ImportError` pointing to `scslide-py[annoy]` or `pip install annoy`. |
| `pip install "scslide-py[anndata,ann,annoy]"` | Above + `annoy>=1.17` | Enables `nn_method='annoy'` (`RcppAnnoy`-compatible Euclidean forest with `n_jobs` build worker forwarding). Requires either a platform wheel or a working C++14 toolchain. | All optional backends enabled. |
| `pip install "scslide-py[anndata,ann,annoy,dev]"` | Above + `pytest>=7`, `ruff>=0.5` | Full regression test suite (`pytest -o pythonpath= --import-mode=importlib`) and static linting (`ruff check`). | Development/CI profile. |

## 3. Verified Representative Dependency Profiles (`constraints/`)

Three concrete, solvable dependency profiles were installed in clean virtual environments outside the repository and verified against the final `v0.3.3` wheel:

1. **Minimum-solvable floor (`constraints/verified-minimum-py310.txt`, Python `3.10.12`)**:
   - `numpy==1.24.4`, `pandas==2.0.3`, `scipy==1.10.1`, `scikit-learn==1.3.2`, `matplotlib==3.7.5`, `anndata==0.10.8`, `h5py==3.8.0`, `zarr==2.16.1`.
2. **Primary reference profile (`constraints/verified-primary-py311.txt`, Python `3.11.15`)**:
   - `numpy==2.2.6` / `2.4.x`, `pandas==2.3.3` / `3.x`, `scipy==1.16.3`+, `scikit-learn==1.7.2`+, `matplotlib==3.10.9`+, `anndata==0.11.4`+, `h5py==3.14.0`+, `zarr==2.18.7`+, `pynndescent>=0.6.0`, `umap-learn>=0.5.12`, `annoy==1.17.3`.
3. **Modern compatible profile (`constraints/verified-modern-py312.txt`, Python `3.12.14`)**:
   - `numpy==2.5.5`, `pandas==3.1.1`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `anndata==0.13.4`, `h5py==3.16.0`, `zarr==3.4.2`, `pynndescent==0.7.0`, `umap-learn==0.6.0`, `annoy==1.17.3`.

## 4. Storage, Sparsity & Memory Semantics

- **Cell-level sparse operations (`scipy.sparse.csr_matrix` / `csc_matrix`)**: `normalize_and_hvg`, `sketch_data_by_group`, `seurat_leverage_scores`, `quick_cor_select`, `kernelpls`, `kernelpls_ondisk`, `cppls_ondisk`, `project_pls_embeddings`, `aggregate_pseudobulk`, and `binomial_thin_counts` stream over sparse chunks without materializing a dense `N_cells × G_genes` array.
- **Training sketch materialization**: In `prepare_sample_object(..., sketch_training=True)`, only the sketched `N_train × G_hvg` submatrix (typically `10^3`–`10^4` cells × `2,000`–`3,500` genes) is materialized in float64 for centered/clipped PLS fitting, matching Seurat's sketched `TRAINING` assay.
- **Sample-level matrices (`SampleLevelObject` & sample `AnnData`)**: Landmark-by-sample (`L × S`, typically `≤2,500 × ≤500`) matrices are small and stored densely in memory (with optional `sparse_counts=True` on `sample_object_to_anndata`).
- **Backed `AnnData` (`ad.read_h5ad(..., backed='r')`) & Zarr (`v2` / `v3`)**: Supported for read-streaming in `prepare_sample_object`, `aggregate_pseudobulk`, `sample_object_from_anndata`, and `run_diffusion_map`. Backed handles must remain open; closed handles fail immediately with an actionable `RuntimeError` (`validate_open_backed_anndata`). In-place mutation helpers (`normalize_and_hvg`) require in-memory `AnnData` or writable layers.

## 5. Worker Resolution, Cgroup Quotas & External Thread Pools

- All neighbor APIs default to `n_jobs=1`.
- `resolve_effective_workers(n_jobs)` resolves negative requests (`n_jobs=-1`, `-2`, ...) against `min(len(os.sched_getaffinity(0)), visible_cgroup_quota)`, inspecting both root mounts (`/sys/fs/cgroup/cpu.max`, `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`) and visible process ancestors in `/proc/self/cgroup`. Custom non-standard cgroup mounts or hidden container ancestors outside the visible namespace are not claimed verified.
- Explicit positive `n_jobs > 0` requests are honored as requested and set `exceeds_available_cpus=True` in returned metadata when above detected availability (never silently clamped).
- `n_jobs` controls parallel queries for exact neighbors (`cKDTree.query(..., workers=...)`); `cKDTree` construction is serial (`build_workers_effective=1`). For Annoy it controls `AnnoyIndex.build(..., n_jobs=...)`, while queries are serial. For PyNNDescent it is passed to `NNDescent(..., n_jobs=...)`. Native BLAS/OpenMP (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) and `numba` (`NUMBA_NUM_THREADS`) maintain separate thread pools and must be configured via environment variables or `threadpoolctl` when enforcing a strict process-wide CPU budget. These fields describe the backend worker configuration, not observed active CPU or total process threads.
