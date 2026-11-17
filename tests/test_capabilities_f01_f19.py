
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scslide as sl


def _make_synthetic_cohort_adata(seed: int = 20261113, n_donors: int = 12, cells_per_donor: int = 60, n_genes: int = 80):
    rng = np.random.default_rng(seed)
    donor_ids = [f"D{i + 1:02d}" for i in range(n_donors)]
    diseases = ["HV"] * (n_donors // 3) + ["MILD"] * (n_donors // 3) + ["SEV"] * (n_donors - 2 * (n_donors // 3))
    batches = ["B1" if i % 2 == 0 else "B2" for i in range(n_donors)]
    ages = np.linspace(35.0, 78.0, n_donors)
    sexes = ["F" if i % 2 == 0 else "M" for i in range(n_donors)]

    total_cells = n_donors * cells_per_donor
    cell_ids = [f"cell_{i:04d}" for i in range(total_cells)]
    cell_donor = np.repeat(donor_ids, cells_per_donor)
    cell_dis = np.repeat(diseases, cells_per_donor)
    cell_batch = np.repeat(batches, cells_per_donor)
    cell_age = np.repeat(ages, cells_per_donor)
    cell_sex = np.repeat(sexes, cells_per_donor)
    cell_types = np.tile(["CD14 monocyte", "CD4 T", "NK"], total_cells // 3)

    severity_num = np.where(cell_dis == "HV", 0.0, np.where(cell_dis == "MILD", 1.0, 2.5))
    rates = np.full((total_cells, n_genes), 0.6, dtype=np.float64)
    # Cell-type markers + severity gradient on CD14 monocytes
    mono_mask = cell_types == "CD14 monocyte"
    rates[mono_mask, :8] += 2.5 + 1.8 * severity_num[mono_mask, None]
    rates[cell_types == "CD4 T", 8:16] += 3.0
    rates[cell_types == "NK", 16:24] += 3.0
    counts = rng.poisson(rates).astype(np.float32)

    obs = pd.DataFrame(
        {
            "Donor": cell_donor,
            "Disease": cell_dis,
            "Batch": cell_batch,
            "Age": cell_age,
            "Sex": cell_sex,
            "cell_type": cell_types,
        },
        index=cell_ids,
    )
    var = pd.DataFrame(index=[f"GENE_{j:03d}" for j in range(n_genes)])
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    adata.layers["counts"] = sp.csr_matrix(counts)
    sl.normalize_and_hvg(adata, n_top_genes=40)
    # Deterministic cell-type embedding (8-D SVD on log-normalized data)
    u, s, _ = np.linalg.svd(adata.layers["data"].toarray(), full_matrices=False)
    adata.obsm["X_pca"] = u[:, :8] * s[:8]
    return adata


def test_b11_neighbors_wnn_and_negative_controls():
    rng = np.random.default_rng(31)
    emb1 = rng.normal(size=(80, 6))
    emb2 = rng.normal(size=(80, 4))
    lm_idx = np.arange(0, 25)
    q_ids = [f"c{i}" for i in range(80)]

    sm_res = sl.find_sm_nn(emb1, lm_idx, k_nn=5, query_ids=q_ids)
    assert sm_res.nn_idx.shape == (80, 5)
    assert np.all(sm_res.nn_idx[:25, 0] == np.arange(1, 26))

    # Swapping query and reference changes output shape & assignments (B11 negative control)
    swapped = sl.landmark_neighbors(emb1, emb1[lm_idx], k=5)
    assert swapped.shape == (25, 5) != sm_res.nn_idx.shape

    # Multi-modal WNN: both fixed weights and learned weights
    mm_fixed = sl.find_mm_nn([emb1, emb2], lm_idx, k_nn=5, knn_range=25, fix_wnn_weights=(0.5, 0.5), query_ids=q_ids)
    mm_learned = sl.find_mm_nn([emb1, emb2], lm_idx, k_nn=5, knn_range=25, fix_wnn_weights=None, query_ids=q_ids)
    assert mm_fixed.nn_idx.shape == (80, 20)  # upstream clamps internal WNN k_nn to min 20
    assert mm_learned.modality_weights is not None
    w_sum = mm_learned.modality_weights[0] + mm_learned.modality_weights[1]
    assert np.allclose(w_sum, 1.0)


def test_b12_diffusion_principal_curve_and_spectral_identifiability():
    t = np.linspace(0.0, np.pi, 24)
    pts = np.column_stack([np.cos(t), np.sin(t), 0.2 * t])
    df = pd.DataFrame(pts, index=[f"S{i:02d}" for i in range(24)], columns=["f1", "f2", "f3"])
    dm = sl.run_diffusion_map(df, ncomp=4, metric="euclidean")
    assert dm.embeddings.shape == (24, 4)

    # Sign flip on isolated mode is legally equivalent
    flipped = dm.embeddings.copy()
    flipped.iloc[:, 0] *= -1.0
    cmp_ok = sl.compare_spectral_embeddings(dm.embeddings, flipped, dm.eigenvalues, dm.eigenvalues)
    assert cmp_ok["equivalent_under_legal_gauge"] is True

    # Mixing two non-degenerate modes via a 45-degree rotation is REJECTED (not masked by Procrustes!)
    rot = np.eye(4)
    c45 = np.sqrt(0.5)
    rot[:2, :2] = [[c45, -c45], [c45, c45]]
    mixed = pd.DataFrame(dm.embeddings.to_numpy() @ rot, index=dm.embeddings.index, columns=dm.embeddings.columns)
    cmp_bad = sl.compare_spectral_embeddings(dm.embeddings, mixed, dm.eigenvalues, dm.eigenvalues)
    assert cmp_bad["equivalent_under_legal_gauge"] is False

    # Unequal eigenvalues are REJECTED even if coordinates match
    cmp_bad_ev = sl.compare_spectral_embeddings(
        dm.embeddings, dm.embeddings, dm.eigenvalues, dm.eigenvalues + 0.05
    )
    assert cmp_bad_ev["spectra_match"] is False
    assert cmp_bad_ev["equivalent_under_legal_gauge"] is False

    pcurve = sl.fit_principal_curve(dm.embeddings.iloc[:, :2], maxit=15)
    assert len(pcurve.lambda_) == 24
    assert abs(np.corrcoef(pcurve.lambda_, t)[0, 1]) > 0.9


def test_b13_traj_de_detects_covariate_vs_offset_and_qlf_vs_wald():
    adata = _make_synthetic_cohort_adata()
    pb = sl.aggregate_pseudobulk(adata, group_by=("Donor", "cell_type"))
    mono_pb = pb[pb.obs["cell_type"] == "CD14 monocyte"].copy()
    donor_meta = adata.obs[["Donor", "Disease", "Age", "Sex"]].drop_duplicates("Donor").set_index("Donor")
    for col in ("Disease", "Age", "Sex"):
        mono_pb.obs[col] = donor_meta.loc[mono_pb.obs["Donor"].astype(str), col].to_numpy()
    mono_pb.obs["traj"] = np.where(mono_pb.obs["Disease"] == "HV", 0.0, np.where(mono_pb.obs["Disease"] == "MILD", 1.0, 2.5))

    de_faithful = sl.traj_de_test(
        mono_pb, traj_var="traj", latent_vars=["Age", "Sex", "log_count"], complete_results=True
    )
    de_offset = sl.traj_de_test(
        mono_pb,
        traj_var="traj",
        latent_vars=["Age", "Sex", "log_count"],
        library_size_mode="offset",
        complete_results=True,
    )
    de_wald = sl.traj_de_test(
        mono_pb,
        traj_var="traj",
        latent_vars=["Age", "Sex", "log_count"],
        test_mode="wald",
        complete_results=True,
    )
    assert "beta_log_count" in de_faithful.columns
    assert "beta_log_count" not in de_offset.columns
    assert not np.allclose(de_faithful["beta_Traj"].to_numpy(), de_offset.loc[de_faithful.index, "beta_Traj"].to_numpy())
    assert not np.allclose(de_faithful["p_Traj"].to_numpy(), de_wald.loc[de_faithful.index, "p_Traj"].to_numpy())
    # Planted monocyte severity markers rank at the top
    top10 = set(de_faithful.sort_values("p_Traj").head(10)["gene_ID"])
    assert len(top10.intersection({f"GENE_{j:03d}" for j in range(8)})) >= 4


def test_b14_combat_and_cellanova_batch_correction_and_edge_policies():
    rng = np.random.default_rng(55)
    genes = [f"L{i}" for i in range(25)]
    samples = [f"S{j}" for j in range(12)]
    meta = pd.DataFrame(
        {
            "Batch": ["B1"] * 4 + ["B2"] * 4 + ["B3"] * 4,
            "Condition": ["HV", "HV", "CASE", "CASE"] * 3,
        },
        index=samples,
    )
    base = rng.normal(size=(25, 12))
    base[:, :4] += 2.0   # B1 batch shift
    base[:, 4:8] -= 1.5  # B2 batch shift
    df = pd.DataFrame(base, index=genes, columns=samples)

    res_reg = sl.correct_sample_combat(
        df,
        batch_key="Batch",
        condition_key="Condition",
        control_label="HV",
        metadata=meta,
        method="regression",
        correct_attenuation=True,
        floor_beta=0.0,
        return_diagnostics=True,
    )
    res_sub = sl.correct_sample_combat(
        df,
        batch_key="Batch",
        condition_key="Condition",
        control_label="HV",
        metadata=meta,
        par_prior=False,
        method="subtraction",
        return_diagnostics=True,
    )
    assert res_reg.corrected.shape == (25, 12)
    assert res_sub.corrected.shape == (25, 12)

    # Batch with 0 controls errors under missing_control_policy='error' and warns under 'global_mean_warn'
    meta_no_ctrl = meta.copy()
    meta_no_ctrl.loc[meta_no_ctrl["Batch"] == "B3", "Condition"] = "CASE"
    with pytest.raises(ValueError, match="without control"):
        sl.correct_sample_combat(
            df, batch_key="Batch", condition_key="Condition", control_label="HV",
            metadata=meta_no_ctrl, missing_control_policy="error"
        )
    with pytest.warns(UserWarning, match="no control"):
        sl.correct_sample_combat(
            df, batch_key="Batch", condition_key="Condition", control_label="HV",
            metadata=meta_no_ctrl, missing_control_policy="global_mean_warn"
        )

    # Fewer than 2 batches with controls always raises ValueError
    meta_one_b = meta.copy()
    meta_one_b.loc[meta_one_b["Batch"] != "B1", "Condition"] = "CASE"
    with pytest.raises(ValueError, match="at least 2 batches"):
        sl.correct_sample_combat(
            df, batch_key="Batch", condition_key="Condition", control_label="HV", metadata=meta_one_b
        )

    # cellanova_calc_be dense == lazy_ondisk, and underdetermined batch raises ValueError
    adata = _make_synthetic_cohort_adata(n_donors=6, cells_per_donor=30, n_genes=30)
    ca_dense = sl.cellanova_calc_be(
        adata.layers["data"],
        integrate_key=adata.obs["Donor"].tolist(),
        control_dict={"ctrl": ["D01", "D02", "D03"]},
        reduction=adata.obsm["X_pca"][:, :4],
        k_select=2,
    )
    ca_lazy = sl.cellanova_calc_be(
        adata.layers["data"],
        integrate_key=adata.obs["Donor"].tolist(),
        control_dict={"ctrl": ["D01", "D02", "D03"]},
        reduction=adata.obsm["X_pca"][:, :4],
        k_select=2,
        lazy_ondisk=True,
    )
    assert np.allclose(ca_dense.corrected, ca_lazy.corrected.to_dense())

    with pytest.raises(ValueError, match="fewer cells than embedding dimensions"):
        sl.cellanova_calc_be(
            adata.layers["data"][:10],
            integrate_key=["D1"] * 3 + ["D2"] * 7,
            control_dict=["D1", "D2"],
            reduction=np.random.default_rng(1).normal(size=(10, 5)),
        )


def test_b15_end_to_end_native_pipeline_and_plots_without_r_in_path(tmp_path, monkeypatch):
    # Remove PATH completely so any hidden Rscript / R subprocess call would immediately fail
    monkeypatch.setenv("PATH", "")
    orig_font = plt.rcParams["font.size"]

    adata = _make_synthetic_cohort_adata()
    wf = sl.run_scslide_workflow(
        adata,
        sample_col="Donor",
        response_col="Disease",
        celltype_col="cell_type",
        embedding_key="X_pca",
        dims_embedding=range(6),
        ncells_per_group=25,
        ncells_landmark=40,
        ncomp_pls=4,
        pls_function="cppls",
        k_nn=5,
        ncomp_diffusion=4,
        fit_curve=True,
        de_celltype="CD14 monocyte",
        de_covariates=("Age", "Sex", "log_count"),
    )
    assert wf.sample_object.counts.shape == (40, 12)
    assert "CD14 monocyte" in wf.de_results

    # F01 + F07 + F14 + F11 + PlotAnnotatedHeatmap + CSV sidecars
    dim_out = sl.sample_level_dim_plot(
        adata,
        wf.sample_object,
        features_to_test=["traj_DC1", "traj_overall"],
        k_nn=8,
        umap_backend="native",
        output=tmp_path / "landmark_cor",
    )
    assert isinstance(dim_out["landmark_obj"], sl.LandmarkObject)
    assert dim_out["umap_info"]["backend_effective"] == "native"
    assert (tmp_path / "landmark_cor.png").exists()
    assert (tmp_path / "landmark_cor.csv").exists()
    plt.close("all")

    sl_umap = sl.run_and_project_umap(adata, nn_name="weighted.nn", k_nn=5, umap_backend="native")
    assert "X_proj.sketched.umap" in sl_umap.obsm
    assert sl_umap.uns["scslide"]["umap"]["sketched.umap"]["backend_effective"] == "native"

    de_df = wf.de_results["CD14 monocyte"]
    top_genes = list(de_df.head(10)["gene_ID"])
    sub_pb = wf.pseudobulk_adata[wf.pseudobulk_adata.obs["cell_type"] == "CD14 monocyte"].copy()
    sub_pb.obs["traj_overall"] = wf.sample_object.metadata.loc[sub_pb.obs["Donor"].astype(str), "traj_overall"].to_numpy()
    sub_pb.obs["Disease"] = wf.sample_object.metadata.loc[sub_pb.obs["Donor"].astype(str), "Disease"].to_numpy()
    fig_hm, _, hm_src = sl.plot_annotated_heatmap(
        sub_pb, top_genes, sort_by="traj_overall", anno_var="Disease", output=tmp_path / "traj_hm"
    )
    plt.close(fig_hm)
    assert (tmp_path / "traj_hm.png").exists() and (tmp_path / "traj_hm.csv").exists()
    assert len(hm_src) == len(top_genes)
    # Global rcParams remained untouched
    assert plt.rcParams["font.size"] == orig_font
