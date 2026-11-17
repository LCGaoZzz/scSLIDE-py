# scSLIDE-py Tutorial (`v0.3.3`)

Every Python block in this tutorial is executable offline against an installed `scslide-py[anndata,ann]` (or `scslide-py[anndata]` because `umap_backend="native"` is used below) wheel without R or private developer paths. All datasets constructed below are synthetic demonstrations (`synthetic_demo_only=True`), not biological discoveries. CI executes every fenced `python` block in this file from `/tmp` with `PYTHONPATH=""`.

## 1. Subject-Aware Cohort Audit & Sample Contingency Matrix (`F05`, `F06`, `WP2`)

Before running sample-level trajectory discovery or cross-validation on clinical cohorts with longitudinal blood draws (such as COMBAT COVID-19 IDs `S00045-V1`, `S00045-V2`), audit the `sample -> subject` mapping with `resolve_subject_mapping`:

```python
import tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scslide as sl

rng = np.random.default_rng(20261114)
draws = [
    "S01-V1", "S01-V2",  # longitudinal repeat draw for subject S01
    "S02-V1", "S02-V2",  # longitudinal repeat draw for subject S02
    "S03-V1", "S04-V1", "S05-V1", "S06-V1",
    "H01-V1", "H02-V1", "H03-V1", "H04-V1",
]
diseases = ["SEV", "SEV", "CRIT", "CRIT", "MILD", "MILD", "SEV", "CRIT", "HV", "HV", "HV", "HV"]
ages = [62.0, 62.0, 58.0, 58.0, 49.0, 51.0, 66.0, 70.0, 44.0, 48.0, 53.0, 59.0]
sexes = ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F", "M", "F"]
visit_day = [3.0, 7.0, 10.0, 14.0, 5.0, 6.0, 9.0, 12.0, 0.0, 0.0, 0.0, 0.0]

cells_per_draw = 30
n_genes = 60
n_cells = len(draws) * cells_per_draw
cell_ids = [f"cell_{i:04d}" for i in range(n_cells)]
cell_draw = np.repeat(draws, cells_per_draw)
cell_dis = np.repeat(diseases, cells_per_draw)
cell_age = np.repeat(ages, cells_per_draw)
cell_sex = np.repeat(sexes, cells_per_draw)
cell_day = np.repeat(visit_day, cells_per_draw)
cell_type = np.tile(["CD14 monocyte", "CD4 T"], n_cells // 2)

sev_score = np.where(cell_dis == "HV", 0.0, np.where(cell_dis == "MILD", 1.0, 2.5))
rates = np.full((n_cells, n_genes), 1.2, dtype=np.float64)
mono_mask = cell_type == "CD14 monocyte"
rates[mono_mask, :8] += 2.0 + 1.8 * sev_score[mono_mask, None]
counts_csr = sp.csr_matrix(rng.poisson(rates).astype(np.float64))

adata = ad.AnnData(
    X=counts_csr.copy(),
    obs=pd.DataFrame(
        {
            "Donor": cell_draw,
            "Disease": cell_dis,
            "Age": cell_age,
            "Sex": cell_sex,
            "TimeSinceOnset": cell_day,
            "cell_type": cell_type,
        },
        index=cell_ids,
    ),
    var=pd.DataFrame(index=[f"GENE_{g:03d}" for g in range(n_genes)]),
)
adata.layers["counts"] = counts_csr.copy()
sl.normalize_and_hvg(adata, n_top_genes=40)
u_pca, sv_pca, _ = np.linalg.svd(adata.layers["data"].toarray(), full_matrices=False)
adata.obsm["X_pca"] = u_pca[:, :8] * sv_pca[:8]

subj_map = sl.resolve_subject_mapping(
    adata.obs,
    sample_col="Donor",
    subject_id_sep="-",
    invariant_cols=["Age", "Sex"],
    time_col="TimeSinceOnset",
)
assert subj_map.n_samples == 12 and subj_map.n_subjects == 10
assert len(subj_map.primary_samples) == 10
```

## 2. Core scSLIDE Chain, Subsetting, Persistence & Query Projection (`F03`–`F19`)

`prepare_sample_object` learns the supervised `PLSModel` and builds the cell-to-landmark WNN graph; `generate_sample_object` computes integer LMC counts and Chi-squared Pearson residuals; `subset_samples` preserves pre-filter Pearson residuals by default (`Seurat::subset` semantics); and `centered_residuals(scale_max=10.0)` matches `Seurat::ScaleData(do.scale=FALSE, do.center=TRUE, scale.max=10)`:

```python
work_tmp = Path(tempfile.mkdtemp(prefix="scslide_tutorial_"))

prep = sl.prepare_sample_object(
    adata,
    y="Disease",
    add_hvg=True,
    group_by_cortest="cell_type",
    sketch_training=True,
    group_by_sketch="Donor",
    ncells_per_group=20,
    ncells_landmark=36,
    ncomp=4,
    pls_function="cppls",
    modality="multi",
    k_nn=5,
    knn_range=36,
    name_reduction_1="X_pca",
    dims_reduction_1=range(8),
    fix_wnn_weights=(0.5, 0.5),
    nn_method="exact",
    seed=123,
    reference_profile="v6",
    return_state=True,
)

s_obj_all = sl.generate_sample_object(
    prep.neighbors.nn_idx,
    group_by="Donor",
    k_nn=5,
    normalization_method="ChiSquared",
    rename_group_by="cell_type",
    center_residuals=True,
    scale_max=10.0,
    sample_for_cell=adata.obs["Donor"],
    landmark_ids=prep.landmark_cell_ids,
    cell_ids=adata.obs_names,
    cell_metadata=adata.obs,
)

# Restrict to one primary baseline blood draw per unique human subject (10 subjects)
s_obj = s_obj_all.subset_samples(subj_map.primary_samples, recompute_normalization=False)
dm = sl.run_diffusion_map(s_obj, ncomp=3, metric="cosine", scale_max=10.0)
pcurve = sl.fit_principal_curve(dm.embeddings.iloc[:, :2], maxit=20)
s_obj.metadata["traj_DC1"] = dm.embeddings.iloc[:, 0].to_numpy()
s_obj.metadata["traj_overall"] = pcurve.lambda_

# Persist and reload sample-level AnnData (.h5ad) + PLSModel (.npz)
s_h5ad = work_tmp / "sample_level.h5ad"
sl.sample_object_to_anndata(s_obj, include_centered_residuals=True).write_h5ad(s_h5ad)
s_reloaded = sl.sample_object_from_anndata(ad.read_h5ad(s_h5ad))
assert s_reloaded.counts.equals(s_obj.counts)

pls_path = prep.pls_model.save(work_tmp / "cppls_model.npz")
pls_reloaded = sl.PLSModel.load(pls_path)
new_query_pls = sl.project_pls_embeddings(
    pls_reloaded, adata.layers["data"][:25], feature_names=list(adata.var_names), ncomp=4
)
assert new_query_pls.shape == (25, 4)
```

## 3. Explicit UMAP Backend, Plotting Sidecars & Three Inference Modes (`WP2`, `WP3`)

UMAP functions record `backend_requested`, `backend_effective`, `backend_version`, and `fallback_reason` without catch-all silent algorithm switching. For differential expression along sample trajectories, separate descriptive R-compatible testing from subject-aware exploratory diagnostics:

```python
import matplotlib.pyplot as plt

# 1. Landmark correlation UMAP with explicit backend and automatic .csv source-data sidecar
dim_res = sl.sample_level_dim_plot(
    adata,
    s_obj_all,
    features_to_test=["nCount_LMC"],
    k_nn=8,
    umap_backend="native",
    output=work_tmp / "landmark_dimplot",
)
plt.close("all")
assert dim_res["umap_info"]["backend_effective"] == "native"
assert (work_tmp / "landmark_dimplot.csv").exists()

# 2. Mode A — R-compatible descriptive TrajDETest (glmGamPoi QL-F on fixed/in-sample trajectory)
pb = sl.aggregate_pseudobulk(adata, group_by=("Donor", "cell_type"), layer="counts")
mono_pb = pb[
    (pb.obs["cell_type"] == "CD14 monocyte")
    & (pb.obs["Donor"].astype(str).isin(subj_map.primary_samples))
].copy()
mono_pb.obs["traj_overall"] = s_obj.metadata.loc[mono_pb.obs["Donor"].astype(str), "traj_overall"].to_numpy()
mono_pb.obs["Age"] = s_obj.metadata.loc[mono_pb.obs["Donor"].astype(str), "Age"].to_numpy()
mono_pb.obs["Sex"] = s_obj.metadata.loc[mono_pb.obs["Donor"].astype(str), "Sex"].to_numpy()

de_descriptive = sl.traj_de_test(mono_pb, traj_var="traj_overall", latent_vars=["Age", "Sex", "log_count"])
assert de_descriptive.attrs["inference_tier"] == "descriptive_post_selection_r_compatible"

# 3. Mode C1 — Exploratory trajectory-residual permutation
de_cond_perm = sl.traj_de_permutation_test(
    mono_pb, traj_var="traj_overall", latent_vars=["Age", "Sex", "log_count"], n_permutations=39, seed=42,
    trajectory_provenance="in_sample_learned"
)
assert "emp_p_gene" in de_cond_perm.columns and "emp_q_bh" in de_cond_perm.columns

# 4. Mode C2 — Exploratory OOF/count-splitting and phenotype-null diagnostics
oof_exp = sl.run_confirmatory_oof_traj_de(
    adata,
    sample_col="Donor",
    subject_id_sep="-",
    response_col="Disease",
    de_celltype="CD14 monocyte",
    celltype_col="cell_type",
    primary_draw_only=True,
    count_split_prop=0.5,
    n_splits=3,
    n_phenotype_permutations=19,
    gene_wise_perm_null=False,
    n_jobs=1,
    seed=42,
)
assert oof_exp["n_unique_subjects_used"] == 10
assert oof_exp["de_results"].attrs["inference_tier"] == "exploratory_subject_oof_countsplit_nb"

oof_perm = sl.run_confirmatory_oof_traj_de(
    adata,
    sample_col="Donor",
    subject_id_sep="-",
    response_col="Disease",
    de_celltype="CD14 monocyte",
    celltype_col="cell_type",
    primary_draw_only=True,
    count_split_prop=0.5,
    n_splits=3,
    n_phenotype_permutations=19,
    gene_wise_perm_null=True,
    n_jobs=1,
    seed=42,
)
assert oof_perm["de_results"].attrs["inference_tier"] == "exploratory_subject_oof_phenotype_permutation"
assert "emp_p_maxf_fwer" in oof_perm["de_results"].columns
```
