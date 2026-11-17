"""Regression tests for v0.3.1 worker/affinity quota control and scoped statistical inference (S6 & S7)."""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scslide as sl
from scslide.neighbors import _detect_cgroup_cpu_quota


def test_s6_resolve_effective_workers_affinity_and_cgroup_bounds(tmp_path):
    # Invalid inputs rejected explicitly
    with pytest.raises(ValueError, match="cannot be 0"):
        sl.resolve_effective_workers(0)
    with pytest.raises(ValueError, match="non-zero integer"):
        sl.resolve_effective_workers(True)  # type: ignore[arg-type]

    # Default None / 1 -> strict single worker
    eff1, info1 = sl.resolve_effective_workers(1)
    assert eff1 == 1 and info1["workers_effective"] == 1
    eff_none, info_none = sl.resolve_effective_workers(None)
    assert eff_none == 1 and info_none["workers_effective"] == 1

    # Negative n_jobs resolves against available quota and NEVER returns -1
    eff_all, info_all = sl.resolve_effective_workers(-1, available_cpus_override=4)
    assert eff_all == 4 and info_all["workers_effective"] == 4 and info_all["available_cpus"] == 4
    eff_m2, _ = sl.resolve_effective_workers(-2, available_cpus_override=4)
    assert eff_m2 == 3
    eff_clamp, _ = sl.resolve_effective_workers(-99, available_cpus_override=4)
    assert eff_clamp == 1

    # Linux cgroup v2 and v1 quota file parsing
    cg_v2 = tmp_path / "cpu.max"
    cg_v2.write_text("400000 100000\n", encoding="utf-8")
    assert _detect_cgroup_cpu_quota(cgroup_v2_max_path=cg_v2) == 4
    cg_v2_max = tmp_path / "cpu_unlimited.max"
    cg_v2_max.write_text("max 100000\n", encoding="utf-8")
    q_v1 = tmp_path / "cfs_quota"
    p_v1 = tmp_path / "cfs_period"
    q_v1.write_text("250000\n", encoding="utf-8")
    p_v1.write_text("100000\n", encoding="utf-8")
    assert _detect_cgroup_cpu_quota(cgroup_v2_max_path=cg_v2_max, cgroup_v1_quota_path=q_v1, cgroup_v1_period_path=p_v1) == 2

    # Real OS sched_setaffinity check on Linux: n_jobs=-1 inside a 1-CPU mask resolves to 1 (not os.cpu_count())
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        orig_mask = os.sched_getaffinity(0)
        if len(orig_mask) >= 2:
            first_cpu = min(orig_mask)
            try:
                os.sched_setaffinity(0, {first_cpu})
                eff_pinned, info_pinned = sl.resolve_effective_workers(-1)
                assert eff_pinned == 1
                assert info_pinned["available_cpus"] == 1
            finally:
                os.sched_setaffinity(0, orig_mask)


def test_s6_n_jobs_propagation_and_exact_numerical_invariance():
    rng = np.random.default_rng(20261115)
    draws = [f"D{i:02d}" for i in range(10)]
    dis = ["HV"] * 5 + ["CASE"] * 5
    ages = np.linspace(35.0, 75.0, 10)
    sexes = ["F", "M"] * 5
    cells_per_draw = 30
    n_cells = len(draws) * cells_per_draw
    counts = sp.csr_matrix(rng.poisson(1.8, size=(n_cells, 45)).astype(np.float64))

    def _make_adata() -> ad.AnnData:
        a = ad.AnnData(
            X=counts.copy(),
            obs=pd.DataFrame(
                {
                    "Donor": np.repeat(draws, cells_per_draw),
                    "Disease": np.repeat(dis, cells_per_draw),
                    "Age": np.repeat(ages, cells_per_draw),
                    "Sex": np.repeat(sexes, cells_per_draw),
                    "cell_type": np.tile(["Mono", "T", "NK"], n_cells // 3),
                },
                index=[f"c_{i:03d}" for i in range(n_cells)],
            ),
            var=pd.DataFrame(index=[f"G_{j:02d}" for j in range(45)]),
        )
        a.layers["counts"] = counts.copy()
        sl.normalize_and_hvg(a, n_top_genes=30)
        u, sv, _ = np.linalg.svd(a.layers["data"].toarray(), full_matrices=False)
        a.obsm["X_pca"] = u[:, :6] * sv[:6]
        return a

    wf1 = sl.run_scslide_workflow(
        _make_adata(),
        sample_col="Donor",
        response_col="Disease",
        celltype_col="cell_type",
        embedding_key="X_pca",
        dims_embedding=range(6),
        ncells_per_group=20,
        ncells_landmark=40,
        knn_range=40,
        ncomp_pls=4,
        de_celltype="Mono",
        n_jobs=1,
        seed=77,
    )
    wf4 = sl.run_scslide_workflow(
        _make_adata(),
        sample_col="Donor",
        response_col="Disease",
        celltype_col="cell_type",
        embedding_key="X_pca",
        dims_embedding=range(6),
        ncells_per_group=20,
        ncells_landmark=40,
        knn_range=40,
        ncomp_pls=4,
        de_celltype="Mono",
        n_jobs=4,
        seed=77,
    )
    assert wf1.prepared.neighbors.params["workers_effective"] == 1
    assert wf4.prepared.neighbors.params["workers_effective"] == 4
    assert wf4.inference_summary["workers_effective"] == 4
    assert np.array_equal(wf1.prepared.neighbors.nn_idx, wf4.prepared.neighbors.nn_idx)
    assert np.allclose(wf1.prepared.neighbors.nn_dist, wf4.prepared.neighbors.nn_dist, atol=1e-12)
    pd.testing.assert_frame_equal(wf1.sample_object.counts, wf4.sample_object.counts)
    assert np.allclose(
        wf1.de_results["Mono"]["beta_Traj"].to_numpy(),
        wf4.de_results["Mono"]["beta_Traj"].to_numpy(),
        atol=1e-12,
    )


def test_s7_scoped_inference_tiers_and_confirmatory_oof_phenotype_permutation():
    rng = np.random.default_rng(20261116)
    draws = [f"S{i:02d}-V1" for i in range(16)]
    dis = ["HV"] * 8 + ["CASE"] * 8
    ages = np.tile(np.linspace(40.0, 70.0, 8), 2)
    sexes = ["F", "M"] * 8
    cells_per_draw = 36
    n_cells = len(draws) * cells_per_draw
    cell_dis = np.repeat(dis, cells_per_draw)
    cell_ct = np.tile(["Mono", "T"], n_cells // 2)

    rates = np.full((n_cells, 40), 5.0, dtype=np.float64)
    sig_mask = (cell_ct == "Mono") & (cell_dis == "CASE")
    rates[sig_mask, :3] += 9.0
    rates[sig_mask, 3:9] -= 4.4
    counts = sp.csr_matrix(rng.poisson(rates).astype(np.float64))

    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(
            {
                "Donor": np.repeat(draws, cells_per_draw),
                "Disease": cell_dis,
                "Age": np.repeat(ages, cells_per_draw),
                "Sex": np.repeat(sexes, cells_per_draw),
                "cell_type": cell_ct,
            },
            index=[f"cell_{i:03d}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"G_{j:02d}" for j in range(40)]),
    )
    adata.layers["counts"] = counts.copy()
    sl.normalize_and_hvg(adata, n_top_genes=30)
    u, sv, _ = np.linalg.svd(adata.layers["data"].toarray(), full_matrices=False)
    adata.obsm["X_pca"] = u[:, :5] * sv[:5]

    # 1. In-sample residual permutation inside workflow emits UserWarning & tags exploratory
    with pytest.warns(UserWarning, match="in-sample learned trajectory"):
        wf_ins = sl.run_scslide_workflow(
            adata,
            sample_col="Donor",
            subject_id_sep="-",
            response_col="Disease",
            celltype_col="cell_type",
            embedding_key="X_pca",
            dims_embedding=range(5),
            ncells_per_group=20,
            ncells_landmark=30,
            knn_range=30,
            ncomp_pls=3,
            de_celltype="Mono",
            de_inference_mode="exploratory_insample_residual_perm",
            n_permutations=19,
            seed=11,
        )
    assert wf_ins.de_results["Mono"].attrs["inference_tier"] == "exploratory_insample_traj_residual_permutation"

    # 2. Fast parametric OOF + countsplit emits scope warning & marks shared_donor_latent_fdr_calibrated=False
    with pytest.warns(UserWarning, match="shared cross-gene donor latent"):
        oof_exp = sl.run_confirmatory_oof_traj_de(
            adata,
            sample_col="Donor",
            subject_id_sep="-",
            response_col="Disease",
            de_celltype="Mono",
            celltype_col="cell_type",
            n_splits=3,
            n_phenotype_permutations=9,
            gene_wise_perm_null=False,
            seed=21,
        )
    assert oof_exp["inference_tier"] == "exploratory_subject_oof_countsplit_nb"
    assert oof_exp["shared_donor_latent_fdr_calibrated"] is False

    # 3. Historical phenotype-null calculations are deterministic, but exploratory.
    with pytest.warns(UserWarning, match="minimum attainable empirical p"):
        oof_conf_a = sl.run_confirmatory_oof_traj_de(
            adata,
            sample_col="Donor",
            subject_id_sep="-",
            response_col="Disease",
            de_celltype="Mono",
            celltype_col="cell_type",
            n_splits=3,
            ncomp_pls=2,
            n_phenotype_permutations=14,
            gene_wise_perm_null=True,
            n_jobs=1,
            seed=21,
        )
    oof_conf_b = sl.run_confirmatory_oof_traj_de(
        adata,
        sample_col="Donor",
        subject_id_sep="-",
        response_col="Disease",
        de_celltype="Mono",
        celltype_col="cell_type",
        n_splits=3,
        ncomp_pls=2,
        n_phenotype_permutations=29,
        gene_wise_perm_null=True,
        n_jobs=1,
        seed=21,
    )
    oof_conf_c = sl.run_confirmatory_oof_traj_de(
        adata,
        sample_col="Donor",
        subject_id_sep="-",
        response_col="Disease",
        de_celltype="Mono",
        celltype_col="cell_type",
        n_splits=3,
        ncomp_pls=2,
        n_phenotype_permutations=29,
        gene_wise_perm_null=True,
        n_jobs=1,
        seed=21,
    )
    assert oof_conf_a["inference_tier"] == "exploratory_subject_oof_phenotype_permutation"
    assert oof_conf_b["shared_donor_latent_fdr_calibrated"] is False
    df_c = oof_conf_b["de_results"]
    assert {"emp_p_oof_perm", "emp_q_bh_oof_perm", "emp_p_maxf_fwer"}.issubset(df_c.columns)
    pd.testing.assert_frame_equal(df_c, oof_conf_c["de_results"])
    top3_by_maxf = set(df_c.sort_values(["emp_p_maxf_fwer", "p_Traj"], kind="mergesort").head(3).index)
    assert top3_by_maxf == {"G_00", "G_01", "G_02"}
    assert float(df_c.loc["G_00", "emp_p_maxf_fwer"]) <= 0.10
    assert float(df_c.loc["G_01", "emp_p_maxf_fwer"]) <= 0.10
    assert float(df_c.loc["G_02", "emp_p_maxf_fwer"]) <= 0.10
