import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scslide import (
    cppls_ondisk,
    fit_pls,
    kernelpls,
    predict_pls,
    project_pls_embeddings,
    quick_cor_select,
    quick_cor_test,
)


def test_pls_round_trip_shapes():
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(20, 5)), columns=list("abcde"))
    y = x[["a", "b"]].to_numpy() + rng.normal(scale=0.1, size=(20, 2))
    model = fit_pls(x, y, n_components=2)
    assert model.transform(x).shape == (20, 2)
    assert model.predict(x).shape == (20, 2)


def test_b05_single_response_supports_multiple_components_across_solvers():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(36, 8)), columns=[f"g{i}" for i in range(8)])
    y_single = 2.0 * x["g0"] - 1.2 * x["g3"] + 0.7 * x["g5"] + rng.normal(scale=0.15, size=36)

    for method in ("plsr", "cppls", "spls"):
        model = fit_pls(x, y_single, n_components=5, method=method)
        scores = model.transform(x)
        assert scores.shape == (36, 5), f"{method} truncated single-response components"
        if method != "spls":
            pred_single = predict_pls(model, x, ncomp=5)
            pred_multi = predict_pls(model, x, ncomp=[1, 3, 5])
            assert pred_single.shape == (36, 1)
            assert pred_multi.shape == (36, 1, 3)

    # Exceeding valid rank without clamp_rank=True raises ValueError (no silent truncation)
    with pytest.raises(ValueError, match="exceeds|less than"):
        fit_pls(x, y_single, n_components=12, method="plsr", clamp_rank=False)


def test_streaming_sparse_kernelpls_and_cppls_match_dense():
    rng = np.random.default_rng(77)
    raw = rng.poisson(1.2, size=(50, 12)).astype(float)
    csr = sp.csr_matrix(raw)
    y_cat = pd.DataFrame({"Disease": ["HV"] * 15 + ["MILD"] * 20 + ["SEV"] * 15})

    d_kpls = kernelpls(raw, y_cat, ncomp=4)
    s_kpls = kernelpls(csr, y_cat, ncomp=4)
    assert np.allclose(d_kpls["scores"], s_kpls["scores"])
    assert np.allclose(d_kpls["coefficients"], s_kpls["coefficients"])

    d_cppls = cppls_ondisk(raw, y_cat, ncomp=4, reference_profile="v6")
    s_cppls = cppls_ondisk(csr, y_cat, ncomp=4, reference_profile="v6")
    assert np.allclose(d_cppls["scores"], s_cppls["scores"])
    assert np.allclose(d_cppls["projection"], s_cppls["projection"])


def test_quick_cor_has_expected_columns():
    x = pd.DataFrame({"signal": [1, 2, 3, 4], "constant": [1, 1, 1, 1]})
    out = quick_cor_test(x, [1, 2, 3, 4])
    assert list(out.columns) == ["feature", "correlation", "p_value", "selected"]


def test_b10_grouped_categorical_quick_cor_select():
    rng = np.random.default_rng(11)
    n = 260
    groups = np.array(["T_cell"] * 120 + ["Mono"] * 120 + ["Rare"] * 20)
    disease = np.array(["HV"] * 60 + ["SEV"] * 60 + ["HV"] * 60 + ["CRIT"] * 60 + ["HV"] * 20)
    mat = np.abs(rng.normal(loc=0.5, scale=0.4, size=(n, 20)))
    # Inject strong group-specific signals
    mat[:120, 0] += (disease[:120] == "SEV") * 4.0
    mat[120:240, 1] += (disease[120:240] == "CRIT") * 4.5
    df = pd.DataFrame(mat, columns=[f"G{i}" for i in range(20)])

    by_grp = quick_cor_select(
        df,
        pd.Series(disease, name="Disease"),
        group_by=groups,
        top_prop=0.15,
        cor_cutoff=0.3,
        min_cells=100,
        min_pct=0.1,
        unlist=False,
    )
    assert by_grp["Rare"] is None  # < min_cells=100 skipped
    assert "G0" in by_grp["T_cell"]
    assert "G1" in by_grp["Mono"]

    # top_prop=None with cor_cutoff works cleanly (Seurat v6 fix)
    only_cut = quick_cor_select(
        df,
        pd.Series(disease, name="Disease"),
        group_by=groups,
        top_prop=None,
        cor_cutoff=0.5,
        min_cells=100,
        unlist=True,
    )
    assert "G0" in only_cut and "G1" in only_cut


def test_b04_pls_prediction_transform_and_projection_align_named_features():
    rng = np.random.default_rng(8)
    x = pd.DataFrame(np.abs(rng.normal(size=(28, 5))), columns=["a", "b", "c", "d", "e"])
    model = fit_pls(x, x[["a", "b"]], n_components=3, method="cppls")
    reordered = x[["e", "d", "b", "a", "c"]]
    assert np.allclose(model.predict(x), model.predict(reordered))
    assert np.allclose(model.transform(x), model.transform(reordered))
    assert np.allclose(project_pls_embeddings(model, x), project_pls_embeddings(model, reordered))


def test_pls_rejects_missing_or_duplicate_features():
    rng = np.random.default_rng(9)
    x = pd.DataFrame(rng.normal(size=(18, 3)), columns=["a", "b", "c"])
    model = fit_pls(x, x[["a"]], n_components=2)
    with pytest.raises(ValueError):
        model.predict(x.drop(columns="b"))
    duplicate = x.rename(columns={"c": "b"})
    with pytest.raises(ValueError):
        model.predict(duplicate)
