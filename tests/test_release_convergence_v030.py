"""Regression and negative-control test suite for v0.3.0 release convergence (WP1-WP3)."""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scslide as sl


def test_wp1_scaledata_upper_clip_and_subset_samples_residual_policy():
    # Construct LMC counts with an extreme positive outlier (> +10 residual) and negative deviation (< -10)
    counts = pd.DataFrame(
        [
            [250, 0, 0, 0],
            [0, 250, 250, 250],
            [40, 40, 40, 1],
            [10, 10, 10, 49],
        ],
        index=["LM1", "LM2", "LM3", "LM4"],
        columns=["S1", "S2", "S3", "S4"],
    )
    meta = pd.DataFrame({"Donor": ["S1", "S2", "S3", "S4"], "nCount_LMC": counts.sum(axis=0).to_numpy()}, index=counts.columns)
    obj = sl.SampleLevelObject.from_counts(counts, normalization="chi_squared", metadata=meta, reference_profile="v6")

    unclipped = obj.centered_residuals(scale_max=None)
    clipped = obj.centered_residuals(scale_max=10.0)
    assert float(unclipped.to_numpy().max()) > 10.0
    assert np.isclose(float(clipped.to_numpy().max()), 10.0)
    # Negative values below -10 are NOT clipped (matching Seurat:::FastDenseRowScale)
    neg_mask = unclipped.to_numpy() < -10.0
    if np.any(neg_mask):
        assert np.allclose(clipped.to_numpy()[neg_mask], unclipped.to_numpy()[neg_mask])

    # subset_samples default preserves pre-filter residuals identically (Seurat::subset semantics)
    sub_retained = obj.subset_samples(["S1", "S2", "S3"], recompute_normalization=False)
    pd.testing.assert_frame_equal(sub_retained.residuals, obj.residuals.loc[:, ["S1", "S2", "S3"]])
    assert sub_retained.attrs["subset_normalization_policy"] == "retained_prefilter"

    # recompute_normalization=True recomputes Pearson residuals on the 3-sample submatrix
    sub_recomp = obj.subset_samples(["S1", "S2", "S3"], recompute_normalization=True)
    expected_recomp = sl.pearson_residuals(counts.loc[:, ["S1", "S2", "S3"]], require_integer_counts=True)
    pd.testing.assert_frame_equal(sub_recomp.residuals, expected_recomp)
    assert not np.allclose(sub_retained.residuals.to_numpy(), sub_recomp.residuals.to_numpy())


def test_wp3_umap_explicit_backend_no_silent_fallback_and_landmark_roundtrip(tmp_path, monkeypatch):
    rng = np.random.default_rng(101)
    n_cells, n_lm = 30, 12
    c_ids = [f"c{i}" for i in range(n_cells)]
    lm_ids = c_ids[:n_lm]
    emb = rng.normal(size=(n_cells, 5))
    nn_res = sl.find_sm_nn(emb, np.arange(n_lm), k_nn=6, query_ids=c_ids, landmark_ids=lm_ids)
    donors = ["D1"] * 10 + ["D2"] * 10 + ["D3"] * 10
    meta = pd.DataFrame({"Donor": donors, "traj": [0.0] * 10 + [1.0] * 10 + [2.0] * 10}, index=c_ids)

    s_obj = sl.generate_sample_object(
        nn_res.nn_idx, group_by="Donor", k_nn=4, sample_for_cell=donors,
        landmark_ids=lm_ids, cell_ids=c_ids, cell_metadata=meta,
    )
    s_obj.metadata["traj_num"] = [0.1, 1.2, 2.5]

    # 1. Explicit native backend is deterministic for fixed seed and records provenance
    lm_nat1 = sl.build_landmark_object(
        nn_res.nn_idx, s_obj, features_to_test=["traj_num"], k_nn=5, seed=77,
        umap_backend="native", cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist, cell_metadata=meta,
    )
    lm_nat2 = sl.build_landmark_object(
        nn_res.nn_idx, s_obj, features_to_test=["traj_num"], k_nn=5, seed=77,
        umap_backend="native", cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist, cell_metadata=meta,
    )
    assert np.allclose(lm_nat1.umap.to_numpy(), lm_nat2.umap.to_numpy())
    assert lm_nat1.attrs["umap_info"]["backend_requested"] == "native"
    assert lm_nat1.attrs["umap_info"]["backend_effective"] == "native"
    assert lm_nat1.attrs["umap_info"]["fallback_reason"] is None

    # LandmarkObject -> AnnData -> HDF5/Zarr round-trip preserves umap_info
    lm_ad = lm_nat1.to_anndata()
    h5_p = tmp_path / "lm.h5ad"
    lm_ad.write_h5ad(h5_p)
    lm_reloaded = sl.landmark_object_from_anndata(ad.read_h5ad(h5_p))
    pd.testing.assert_frame_equal(lm_reloaded.umap, lm_nat1.umap)
    assert lm_reloaded.attrs["umap_info"]["backend_effective"] == "native"

    # 2. Simulate missing umap-learn package: strict modes raise ImportError, allow_fallback=True warns & records reason
    monkeypatch.setitem(sys.modules, "umap", None)
    monkeypatch.setitem(sys.modules, "umap.umap_", None)
    with pytest.raises(ImportError, match="umap-learn"):
        sl.run_and_project_umap(
            nn_res.nn_idx, nn_name="single.nn", k_nn=4, umap_backend="umap-learn",
            cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist,
        )
    with pytest.raises(ImportError, match="umap-learn"):
        sl.run_and_project_umap(
            nn_res.nn_idx, nn_name="single.nn", k_nn=4, umap_backend="auto", allow_fallback=False,
            cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist,
        )
    with pytest.warns(UserWarning, match="allow_fallback=True"):
        fb_out = sl.run_and_project_umap(
            nn_res.nn_idx, nn_name="single.nn", k_nn=4, umap_backend="auto", allow_fallback=True,
            cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist,
        )
    assert fb_out["umap_info"]["backend_requested"] == "auto"
    assert fb_out["umap_info"]["backend_effective"] == "native"
    assert "ImportError" in str(fb_out["umap_info"]["fallback_reason"]) or "ModuleNotFoundError" in str(fb_out["umap_info"]["fallback_reason"])
    monkeypatch.undo()

    # 3. Runtime / numerical exception inside umap-learn MUST NEVER silently fall back even if allow_fallback=True!
    try:
        import umap.umap_ as umap_internal
        has_umap = True
    except ImportError:
        has_umap = False

    if has_umap:
        ok_u = sl.run_and_project_umap(
            nn_res.nn_idx, nn_name="single.nn", k_nn=4, umap_backend="umap-learn",
            cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist,
        )
        assert ok_u["umap_info"]["backend_effective"] == "umap-learn"
        assert ok_u["umap_info"]["fallback_reason"] is None

        def _boom(*args, **kwargs):
            raise RuntimeError("Deliberate umap-learn internal numerical failure")

        monkeypatch.setattr(umap_internal, "simplicial_set_embedding", _boom)
        with pytest.raises(RuntimeError, match="Deliberate umap-learn internal numerical failure"):
            sl.run_and_project_umap(
                nn_res.nn_idx, nn_name="single.nn", k_nn=4, umap_backend="auto", allow_fallback=True,
                cell_ids=c_ids, landmark_cell_ids=lm_ids, nn_dist=nn_res.nn_dist,
            )


def test_wp3_closed_backed_anndata_lifecycle_and_missing_ann_errors(tmp_path, monkeypatch):
    counts = pd.DataFrame([[5, 2, 1], [1, 6, 3]], index=["L1", "L2"], columns=["S1", "S2", "S3"])
    s_obj = sl.SampleLevelObject.from_counts(counts, normalization="chi_squared")
    s_ad = sl.sample_object_to_anndata(s_obj, include_centered_residuals=True)
    p_h5 = tmp_path / "backed_sample.h5ad"
    s_ad.write_h5ad(p_h5)

    b_ad = ad.read_h5ad(p_h5, backed="r")
    b_ad.file.close()
    with pytest.raises(RuntimeError, match="closed"):
        sl.sample_object_from_anndata(b_ad)
    with pytest.raises(RuntimeError, match="closed"):
        sl.run_diffusion_map(b_ad, ncomp=2)

    # Missing annoy / pynndescent raises actionable ImportError
    emb = np.random.default_rng(3).normal(size=(20, 4))
    monkeypatch.setitem(sys.modules, "annoy", None)
    with pytest.raises(ImportError, match="annoy"):
        sl.find_sm_nn(emb, np.arange(8), k_nn=4, nn_method="annoy")
    monkeypatch.setitem(sys.modules, "pynndescent", None)
    with pytest.raises(ImportError, match="pynndescent"):
        sl.find_sm_nn(emb, np.arange(8), k_nn=4, nn_method="pynndescent")


def test_wp2_subject_mapping_grouped_cv_thinning_and_unified_genewise_null():
    # 1. Subject mapping & conflict detection on longitudinal draws
    obs = pd.DataFrame(
        {
            "Donor": [
                "S01-V1-PB", "S01-V2-PB",
                "S02-V1-PB", "S02-V2-PB",
                "S03-V1-PB", "S04-V1-PB",
                "S05-V1-PB", "S06-V1-PB",
            ],
            "Disease": ["SEV", "SEV", "SEV", "CRIT", "HV", "HV", "MILD", "MILD"],
            "Age": [60, 60, 52, 52, 41, 38, 49, 55],
            "Time": [3, 6, 10, 14, 0, 0, 5, 7],
        }
    )
    smap = sl.resolve_subject_mapping(
        obs, sample_col="Donor", subject_id_sep="-", invariant_cols=["Age"],
        check_discordant_cols=["Disease"], time_col="Time",
    )
    assert smap.n_samples == 8 and smap.n_subjects == 6
    assert "S01" in smap.multi_draw_subjects and "S02" in smap.multi_draw_subjects
    assert "S02" in smap.discordant_columns["Disease"]
    assert "S01-V1-PB" in smap.primary_samples and "S01-V2-PB" not in smap.primary_samples

    # Conflicting invariant column raises ValueError
    obs_bad = obs.copy()
    obs_bad.loc[1, "Age"] = 99
    with pytest.raises(ValueError, match="violated invariant"):
        sl.resolve_subject_mapping(obs_bad, sample_col="Donor", subject_id_sep="-", invariant_cols=["Age"])

    # Stratified subject K-fold keeps multi-draw subjects strictly disjoint across folds
    folds = sl.stratified_subject_kfold(smap.table, y_col="Disease", subject_col="subject_id", sample_col="Donor", n_splits=2)
    for tr_s, te_s in folds:
        tr_subjs = {smap.sample_to_subject[s] for s in tr_s}
        te_subjs = {smap.sample_to_subject[s] for s in te_s}
        assert tr_subjs.isdisjoint(te_subjs)

    # 2. Sparse O(nnz) Binomial count thinning preserves exact matrix sum and sparsity
    rng = np.random.default_rng(12)
    csr = sp.csr_matrix(rng.poisson(1.5, size=(40, 25)))
    c1, c2 = sl.binomial_thin_counts(csr, prop=0.5, seed=99)
    assert sp.isspmatrix_csr(c1) and sp.isspmatrix_csr(c2)
    assert np.array_equal((c1 + c2).toarray(), csr.toarray())

    # 3. Unified permutation inference + gene-wise vs pooled diagnostic + Freedman-Lane residual perm
    n_subj, n_g = 16, 30
    subj_names = [f"U{i:02d}" for i in range(n_subj)]
    y_map = {s: ("CASE" if i >= 8 else "HV") for i, s in enumerate(subj_names)}
    traj_v = np.array([0.0] * 8 + [2.0] * 8, dtype=np.float64) + rng.normal(scale=0.2, size=n_subj)
    age_v = np.linspace(-1.0, 1.0, n_subj)
    counts_g = rng.poisson(30.0, size=(n_g, n_subj)).astype(np.float64)
    # Plant true trajectory signal in G00, G01
    counts_g[0] = rng.poisson(np.exp(3.0 + 0.8 * traj_v))
    counts_g[1] = rng.poisson(np.exp(3.0 + 0.7 * traj_v))
    c_df = pd.DataFrame(counts_g, index=[f"G{i:02d}" for i in range(n_g)], columns=subj_names)
    lat_df = pd.DataFrame({"Age": age_v}, index=subj_names)

    perm_res = sl.traj_de_permutation_test(c_df, pd.Series(traj_v, index=subj_names), latent_vars=lat_df, n_permutations=39, seed=7)
    assert perm_res.attrs["inference_tier"] == "exploratory_external_traj_residual_permutation"
    assert set(perm_res.head(2).index) == {"G00", "G01"}
    assert np.allclose(perm_res["emp_min_attainable_p"].to_numpy(), 1.0 / 40.0)

    def _unified_pipe(lbl_map: dict[str, str], spec: sl.InferencePipelineSpec, p_rng: np.random.Generator) -> pd.DataFrame:
        tv = np.array([2.0 if lbl_map[s] == "CASE" else 0.0 for s in subj_names], dtype=np.float64) + p_rng.normal(scale=1e-3, size=n_subj)
        return sl.traj_de_test(
            c_df, pd.Series(tv, index=subj_names), latent_vars=lat_df,
            apply_assay_filters=False, test_mode=spec.test_mode, library_size_mode=spec.library_size_mode,
        )

    spec = sl.InferencePipelineSpec(ncomp_pls=4, knn_range=50)
    uni_out = sl.run_unified_permutation_inference(_unified_pipe, y_map, spec=spec, n_permutations=19, seed=55)
    assert uni_out["calibration_status"] == "UNIFIED_GENE_WISE_PHENOTYPE_NULL"
    assert len(uni_out["draw_provenance"]) == 19
    assert all(d["spec_sha256"] == spec.sha256() for d in uni_out["draw_provenance"])
    assert uni_out["calibrated_de"].loc["G00", "emp_p_gene_wise"] <= 0.10

    with pytest.warns(UserWarning, match="pooled_diagnostic"):
        uni_pool = sl.run_unified_permutation_inference(
            _unified_pipe, y_map, spec=spec, n_permutations=9, seed=55, null_aggregation="pooled_diagnostic"
        )
    assert uni_pool["calibration_status"] == "UNVERIFIED_DIAGNOSTIC_POOLED_NULL"


def test_wp2_and_wp3_workflow_confirmatory_modes_annoy_and_unit_contracts(tmp_path):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(303)
    # 6 subjects, with S01 and S02 each having 2 blood draws -> 8 total draws
    draws = ["S01-V1", "S01-V2", "S02-V1", "S02-V2", "S03-V1", "S04-V1", "S05-V1", "S06-V1"]
    dis = ["SEV", "SEV", "CRIT", "CRIT", "HV", "HV", "MILD", "HV"]
    ages = [61.0, 61.0, 58.0, 58.0, 42.0, 47.0, 53.0, 39.0]
    sexes = ["M", "M", "F", "F", "M", "F", "M", "F"]
    cells_per_draw = 30
    tot_c = len(draws) * cells_per_draw
    c_ids = [f"cell_{i:03d}" for i in range(tot_c)]
    c_draw = np.repeat(draws, cells_per_draw)
    c_dis = np.repeat(dis, cells_per_draw)
    c_age = np.repeat(ages, cells_per_draw)
    c_sex = np.repeat(sexes, cells_per_draw)
    c_type = np.tile(["Mono", "T"], tot_c // 2)

    sev_v = np.where(c_dis == "HV", 0.0, np.where(c_dis == "MILD", 1.0, 2.5))
    rates = np.full((tot_c, 36), 0.8, dtype=np.float64)
    mono_m = c_type == "Mono"
    rates[mono_m, :5] += 2.0 + 1.6 * sev_v[mono_m, None]
    counts = sp.csr_matrix(rng.poisson(rates).astype(np.float64))

    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame({"Donor": c_draw, "Disease": c_dis, "Age": c_age, "Sex": c_sex, "cell_type": c_type}, index=c_ids),
        var=pd.DataFrame(index=[f"G_{j:02d}" for j in range(36)]),
    )
    adata.layers["counts"] = counts.copy()
    sl.normalize_and_hvg(adata, n_top_genes=24)
    u, sv, _ = np.linalg.svd(adata.layers["data"].toarray(), full_matrices=False)
    adata.obsm["X_pca"] = u[:, :6] * sv[:6]

    # Confirmatory OOF + count-split mode in run_scslide_workflow
    wf_oof = sl.run_scslide_workflow(
        adata,
        sample_col="Donor",
        subject_id_sep="-",
        response_col="Disease",
        celltype_col="cell_type",
        embedding_key="X_pca",
        dims_embedding=range(6),
        ncells_per_group=20,
        ncells_landmark=30,
        ncomp_pls=3,
        k_nn=4,
        ncomp_diffusion=3,
        de_celltype="Mono",
        de_inference_mode="confirmatory_oof_countsplit",
        seed=42,
    )
    assert wf_oof.subject_mapping is not None
    assert wf_oof.subject_mapping.n_samples == 8
    assert wf_oof.subject_mapping.n_subjects == 6
    assert wf_oof.inference_summary["has_repeated_subject_draws"] is True
    assert wf_oof.de_results["Mono"].attrs["inference_tier"] == "exploratory_subject_oof_countsplit_nb"
    assert wf_oof.inference_summary["workers_effective"] == 1

    # Integer count contract check on aggregate_pseudobulk when require_integer_counts=True
    ad_nonint = adata.copy()
    ad_nonint.layers["counts"] = adata.layers["data"].copy()
    with pytest.raises(ValueError, match="integer"):
        sl.aggregate_pseudobulk(ad_nonint, group_by=("Donor", "cell_type"), require_integer_counts=True)

    # Optional annoy neighbor backend check when installed
    try:
        import annoy  # noqa: F401
        has_annoy = True
    except ImportError:
        has_annoy = False
    if has_annoy:
        nn_ann = sl.find_mm_nn(
            [adata.obsm["X_pca"], adata.obsm["X_pca"][:, :3]],
            np.arange(25),
            k_nn=5,
            knn_range=25,
            fix_wnn_weights=(0.5, 0.5),
            nn_method="annoy",
            annoy_n_trees=20,
        )
        assert nn_ann.params["backend_effective"] == "annoy"
        assert nn_ann.params["annoy_n_trees"] == 20

    # Plot helper ax= / source-data / non-mutating rcParams check
    orig_lw = plt.rcParams["axes.linewidth"]
    fig_ext, ax_ext = plt.subplots(figsize=(4, 3))
    fig_ret, ax_ret, src_df = sl.plot_sample_diffusion(
        wf_oof.sample_object, group_by="Disease", ax=ax_ext, output=tmp_path / "diff_panel"
    )
    assert fig_ret is fig_ext and ax_ret is ax_ext
    assert len(src_df) == 8
    assert (tmp_path / "diff_panel.csv").exists()
    assert plt.rcParams["axes.linewidth"] == orig_lw
    plt.close(fig_ext)

