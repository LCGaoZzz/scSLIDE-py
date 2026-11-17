"""Targeted release-candidate regressions for v0.3.3 (C02-C07, P3 portable fixture)."""
from __future__ import annotations

import sys
import os
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scslide as sl


def test_zarr_unwritable_destination_preserves_actionable_permission_error(tmp_path):
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("requires unprivileged POSIX permission enforcement")
    directory = tmp_path / "readonly"
    directory.mkdir()
    directory.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            sl.write_sample_zarr(ad.AnnData(np.eye(3)), directory / "sample.zarr")
    finally:
        directory.chmod(0o700)


def test_exact_worker_provenance_separates_serial_build_from_parallel_queries():
    rng = np.random.default_rng(42)
    e1, e2 = rng.normal(size=(36, 5)), rng.normal(size=(36, 4))
    for method, embeddings in [(sl.find_sm_nn, e1), (sl.find_mm_nn, [e1, e2])]:
        one = method(embeddings, np.arange(16), k_nn=5, nn_method="exact", n_jobs=1)
        four = method(embeddings, np.arange(16), k_nn=5, nn_method="exact", n_jobs=4)
        np.testing.assert_array_equal(one.nn_idx, four.nn_idx)
        np.testing.assert_array_equal(one.nn_dist, four.nn_dist)
        assert four.params["workers_effective"] == 4
        assert four.params["build_workers_effective"] == 1
        assert four.params["query_workers_effective"] == 4


def test_c02_annoy_public_api_records_resolved_build_and_serial_query_workers(monkeypatch):
    build_calls: list[tuple[int, int]] = []

    class FakeAnnoyIndex:
        def __init__(self, dim: int, metric: str):
            self.dim = dim
            self.metric = metric

        def set_seed(self, seed: int) -> None:
            pass

        def add_item(self, idx: int, vec: np.ndarray) -> None:
            pass

        def build(self, n_trees: int, n_jobs: int = -1) -> None:
            build_calls.append((int(n_trees), int(n_jobs)))

        def get_nns_by_vector(self, vec: np.ndarray, k: int, include_distances: bool = True):
            return list(range(k)), [float(0.05 * i) for i in range(k)]

    monkeypatch.setitem(sys.modules, "annoy", SimpleNamespace(AnnoyIndex=FakeAnnoyIndex))
    rng = np.random.default_rng(20261116)
    emb1 = rng.normal(size=(36, 5))
    emb2 = rng.normal(size=(36, 4))
    lm_idx = np.arange(16)

    sm = sl.find_sm_nn(emb1, lm_idx, k_nn=5, nn_method="annoy", annoy_n_trees=25, n_jobs=3)
    assert sm.params["backend_effective"] == "annoy"
    assert sm.params["annoy_n_trees"] == 25
    assert sm.params["workers_requested"] == 3
    assert sm.params["workers_effective"] == 3
    assert sm.params["build_workers_effective"] == 3
    assert sm.params["query_workers_effective"] == 1

    mm = sl.find_mm_nn(
        [emb1, emb2],
        lm_idx,
        k_nn=5,
        knn_range=16,
        fix_wnn_weights=(0.5, 0.5),
        nn_method="annoy",
        annoy_n_trees=30,
        n_jobs=4,
    )
    assert mm.params["workers_requested"] == 4
    assert mm.params["workers_effective"] == 4
    assert mm.params["build_workers_effective"] == 4
    assert mm.params["query_workers_effective"] == 1
    assert (25, 3) in build_calls
    assert (30, 4) in build_calls


def test_c03_explicit_positive_worker_request_above_available_cpus_is_flagged_not_clamped():
    eff, info = sl.resolve_effective_workers(6, available_cpus_override=2)
    assert eff == 6
    assert info["workers_requested"] == 6
    assert info["workers_effective"] == 6
    assert info["available_cpus"] == 2
    assert info["exceeds_available_cpus"] is True


@pytest.mark.parametrize("policy,expected_p", [("conservative_one", 1.0), ("nan", np.nan)])
def test_c04_missing_permutation_gene_accounting_and_provenance(policy: str, expected_p: float):
    observed = pd.DataFrame(
        {
            "gene_ID": ["g_present", "g_all_missing", "g_half_missing"],
            "f_statistic": [12.0, 12.0, 12.0],
            "p_Traj": [0.005, 0.005, 0.005],
            "adj_pval": [0.01, 0.01, 0.01],
        }
    )
    step = {"i": 0}

    def _pipe_uni(labels, spec, rng):
        step["i"] += 1
        if step["i"] == 1:
            return observed
        # Permutations 1..2 include g_half_missing (below observed); 3..4 omit it; g_all_missing always omitted
        if step["i"] in (2, 3):
            return pd.DataFrame(
                {"gene_ID": ["g_present", "g_half_missing"], "f_statistic": [1.0, 1.0], "p_Traj": [0.5, 0.5], "adj_pval": [0.8, 0.8]}
            )
        return pd.DataFrame({"gene_ID": ["g_present"], "f_statistic": [1.0], "p_Traj": [0.5], "adj_pval": [0.8]})

    out_u = sl.run_unified_permutation_inference(
        _pipe_uni,
        {f"S{i}": i % 2 for i in range(6)},
        n_permutations=4,
        missing_gene_policy=policy,
    )
    assert out_u["confirmatory_calibration_established"] is False
    assert out_u["missing_gene_policy"] == policy
    df_u = out_u["calibrated_de"]
    assert df_u.attrs["missing_gene_policy"] == policy
    assert df_u.loc["g_all_missing", "emp_n_finite_perms"] == 0
    assert df_u.loc["g_all_missing", "emp_n_missing_perms"] == 4
    assert np.isclose(df_u.loc["g_all_missing", "emp_missing_perm_rate"], 1.0)
    assert df_u.loc["g_half_missing", "emp_n_finite_perms"] == 2
    assert df_u.loc["g_half_missing", "emp_n_missing_perms"] == 2
    assert np.isclose(df_u.loc["g_half_missing", "emp_missing_perm_rate"], 0.5)
    if np.isnan(expected_p):
        assert np.isnan(df_u.loc["g_all_missing", "emp_p_gene_wise"])
        assert np.isclose(df_u.loc["g_half_missing", "emp_p_gene_wise"], 1.0 / 3.0)
    else:
        assert np.isclose(df_u.loc["g_all_missing", "emp_p_gene_wise"], 1.0)
        # 2 missing draws count as exceedances -> (1 + 2) / (4 + 1) = 0.6
        assert np.isclose(df_u.loc["g_half_missing", "emp_p_gene_wise"], 0.6)


def test_c05_changing_subject_response_rejected_even_when_permutations_zero():
    counts = sp.csr_matrix(np.ones((12, 6)))
    adata = ad.AnnData(
        counts.copy(),
        obs=pd.DataFrame(
            {
                "sample": np.repeat(["s1-v1", "s1-v2", "s2-v1", "s3-v1", "s4-v1", "s5-v1"], 2),
                "subject": np.repeat(["s1", "s1", "s2", "s3", "s4", "s5"], 2),
                "response": np.repeat(["HV", "CASE", "HV", "CASE", "HV", "CASE"], 2),
                "type": ["Mono"] * 12,
            },
            index=[f"c{i}" for i in range(12)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    adata.layers["counts"] = counts
    with pytest.raises(ValueError, match="one response value per subject"):
        sl.run_confirmatory_oof_traj_de(
            adata,
            sample_col="sample",
            subject_col="subject",
            response_col="response",
            celltype_col="type",
            de_celltype="Mono",
            primary_draw_only=False,
            n_splits=2,
            n_phenotype_permutations=0,
            gene_wise_perm_null=False,
        )


def test_c06_legacy_confirmatory_workflow_aliases_warn_and_expose_exploratory_provenance():
    adata = sl.build_synthetic_cohort_fixture(seed=17, n_subjects=8, n_repeat_subjects=1, cells_per_sample=24, n_genes=36)
    assert adata.uns["scslide_demo"]["synthetic_demo_only"] is True

    with pytest.warns(UserWarning, match="legacy compatibility alias"):
        wf = sl.run_scslide_workflow(
            adata,
            sample_col="Donor",
            subject_id_sep="-",
            response_col="Disease",
            celltype_col="cell_type",
            embedding_key="X_pca",
            dims_embedding=range(6),
            ncells_per_group=18,
            ncells_landmark=30,
            knn_range=30,
            ncomp_pls=3,
            de_celltype="CD14_Mono",
            de_inference_mode="confirmatory_oof_countsplit",
            seed=9,
        )
    assert wf.inference_summary["legacy_mode_alias_used"] is True
    assert wf.inference_summary["de_inference_mode"] == "confirmatory_oof_countsplit"
    assert wf.inference_summary["de_inference_mode_effective"] == "exploratory_oof_countsplit"
    assert wf.inference_summary["confirmatory_calibration_established"] is False
    assert wf.de_results["CD14_Mono"].attrs["inference_tier"] == "exploratory_subject_oof_countsplit_nb"

    core_dict = sl.build_synthetic_cohort_fixture(seed=17, n_subjects=6, n_repeat_subjects=0, cells_per_sample=12, n_genes=20, as_anndata=False)
    assert core_dict["scslide_demo"]["synthetic_demo_only"] is True
    assert core_dict["counts"].shape == (72, 20)
