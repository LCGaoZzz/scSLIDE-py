"""Regressions for defects located while verifying the completed campaign."""
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scslide as sl
from scslide.neighbors import _detect_cgroup_cpu_quota, _knn_query


def test_annoy_receives_resolved_worker_budget(monkeypatch):
    calls = []

    class FakeIndex:
        def __init__(self, *args):
            pass

        def set_seed(self, seed):
            pass

        def add_item(self, *args):
            pass

        def build(self, n_trees, n_jobs=-1):
            calls.append((n_trees, n_jobs))

        def get_nns_by_vector(self, vector, k, include_distances):
            return list(range(k)), [0.0] * k

    monkeypatch.setitem(sys.modules, "annoy", SimpleNamespace(AnnoyIndex=FakeIndex))
    x = np.eye(4)
    _knn_query(x, x, 2, method="annoy", n_jobs=1)
    _knn_query(x, x, 2, method="annoy", n_jobs=3)
    assert calls == [(50, 1), (50, 3)]


def test_nested_cgroup_uses_tightest_visible_ancestor(tmp_path):
    root = tmp_path / "cgroup"
    leaf = root / "user.slice" / "process.scope"
    leaf.mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n")
    (leaf.parent / "cpu.max").write_text("250000 100000\n")
    (leaf / "cpu.max").write_text("400000 100000\n")
    proc = tmp_path / "membership"
    proc.write_text("0::/user.slice/process.scope\n")
    assert _detect_cgroup_cpu_quota(
        cgroup_v2_max_path=root / "cpu.max",
        cgroup_v1_quota_path=tmp_path / "absent_quota",
        cgroup_v1_period_path=tmp_path / "absent_period",
        proc_cgroup_path=proc,
    ) == 2


def test_permutation_rejects_changing_response_within_subject():
    counts = sp.csr_matrix(np.ones((8, 5)))
    obj = ad.AnnData(
        counts.copy(),
        obs=pd.DataFrame({
            "sample": np.repeat(["s1_v1", "s1_v2", "s2_v1", "s2_v2"], 2),
            "subject": np.repeat(["s1", "s1", "s2", "s2"], 2),
            "response": np.repeat([0, 1, 0, 0], 2),
            "type": ["T"] * 8,
        }, index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    obj.layers["counts"] = counts
    with pytest.raises(ValueError, match="one response value per subject"):
        sl.run_confirmatory_oof_traj_de(
            obj, sample_col="sample", subject_col="subject", response_col="response",
            celltype_col="type", de_celltype="T", primary_draw_only=False,
            n_splits=2, n_phenotype_permutations=1, gene_wise_perm_null=True,
        )


@pytest.mark.parametrize("api", ["unified", "callback"])
def test_missing_gene_conservative_policy_counts_exceedances(api):
    observed = pd.DataFrame({"gene_ID": ["g1", "g2"], "f_statistic": [10., 10.],
                             "p_Traj": [.01, .01], "adj_pval": [.01, .01]})
    missing = pd.DataFrame({"gene_ID": ["g1"], "f_statistic": [0.],
                            "p_Traj": [1.], "adj_pval": [1.]})
    if api == "unified":
        calls = []

        def pipeline(labels, spec, rng):
            calls.append(1)
            return observed if len(calls) == 1 else missing

        result = sl.run_unified_permutation_inference(
            pipeline, {str(i): i % 2 for i in range(6)},
            n_permutations=4, missing_gene_policy="conservative_one",
        )["calibrated_de"]
        assert result.loc["g2", "emp_n_valid_perms"] == 4
        column = "emp_p_gene_wise"
    else:
        result = sl.calibrate_traj_de_donor_permutation(
            lambda rng: missing, observed, n_permutations=4,
            missing_gene_policy="conservative_one",
        )["calibrated_de"]
        column = "emp_p_donor_perm"
    assert result.loc["g2", column] == 1.
    assert result.loc["g1", column] == .2
