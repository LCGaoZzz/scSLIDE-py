import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scslide import (
    PearsonResidualOperator,
    expected_counts,
    generate_sample_object,
    landmark_matrix,
    normalize_chi_squared,
    pearson_residuals,
)


def test_landmark_self_exclusion_and_totals():
    nn = np.array([[1, 2], [1, 2], [1, 2], [2, 1], [2, 1]])
    samples = ["A", "A", "B", "B", "B"]
    cells = ["L1", "L2", "c3", "c4", "c5"]
    counts = landmark_matrix(nn, samples, ["L1", "L2"], k=2, cell_ids=cells)
    assert counts.loc[:, "A"].sum() == 0
    assert counts.to_numpy().sum() == 6


def test_b01_omitting_cell_ids_with_string_landmarks_errors():
    nn = np.array([[1, 2], [2, 1], [1, 2]])
    with pytest.raises(ValueError, match="cell_ids must be provided"):
        landmark_matrix(nn, ["A", "B", "B"], ["L1", "L2"], k=2)


def test_lmc_contract_fixture_zero_sample_and_duplicate_neighbors():
    fixture_path = Path(__file__).resolve().parent / "fixtures/lmc_contract.json"
    spec = json.loads(fixture_path.read_text(encoding="utf-8"))
    counts = landmark_matrix(
        np.asarray(spec["neighbor_index"], dtype=int),
        spec["sample_for_query"],
        spec["landmark_ids"],
        k=spec["k"],
        cell_ids=spec["query_ids"],
        sample_order=spec["sample_order"],
    )
    expected_known = pd.DataFrame(
        spec["known_answer_counts_landmark_by_sample"],
        index=spec["landmark_ids"],
        columns=spec["sample_order"],
    )
    pd.testing.assert_frame_equal(counts, expected_known)
    assert int(counts.to_numpy().sum()) == spec["known_answer_assignment_total"]
    assert int(counts["C"].sum()) == 0


def test_b02_query_permutation_equivariance():
    rng = np.random.default_rng(2026)
    n_landmarks, n_queries = 6, 40
    landmarks = [f"LM{i + 1}" for i in range(n_landmarks)]
    cells = landmarks + [f"q{i + 1:03d}" for i in range(n_queries - n_landmarks)]
    samples = np.array(["S1"] * 10 + ["S2"] * 15 + ["S3"] * 15, dtype=object)
    nn = rng.integers(1, n_landmarks + 1, size=(n_queries, 5))
    meta = pd.DataFrame(
        {"Donor": samples, "Condition": np.where(samples == "S1", "HV", "CASE")},
        index=cells,
    )

    obj1 = generate_sample_object(
        nn,
        group_by="Donor",
        k_nn=4,
        sample_for_cell=samples,
        landmark_ids=landmarks,
        cell_ids=cells,
        cell_metadata=meta,
    )
    perm = rng.permutation(n_queries)
    obj2 = generate_sample_object(
        nn[perm],
        group_by="Donor",
        k_nn=4,
        sample_for_cell=samples[perm],
        landmark_ids=landmarks,
        cell_ids=np.asarray(cells, dtype=object)[perm],
        cell_metadata=meta.iloc[perm],
    )
    pd.testing.assert_frame_equal(obj1.counts, obj2.counts.loc[obj1.counts.index, obj1.counts.columns])
    pd.testing.assert_frame_equal(obj1.residuals, obj2.residuals.loc[obj1.residuals.index, obj1.residuals.columns])
    pd.testing.assert_frame_equal(obj1.metadata, obj2.metadata.loc[obj1.metadata.index])


def test_b03_pearson_independence_expectation_and_structured_operator():
    counts = pd.DataFrame([[10, 20, 0], [30, 40, 0], [0, 0, 0]], index=["L1", "L2", "L3"], columns=["A", "B", "C"])
    residual = pearson_residuals(counts)
    exp = expected_counts(counts)
    assert np.isfinite(residual.to_numpy()).all()
    assert np.allclose(exp.loc[["L1", "L2"], ["A", "B"]].to_numpy(), [[12.0, 18.0], [28.0, 42.0]])
    assert np.all(exp.loc["L3"] == 0.0) and np.all(exp["C"] == 0.0)

    # Structured sparse+rank-one operator matches dense Pearson residuals identically
    op = PearsonResidualOperator(counts)
    v = np.array([0.7, -1.3, 2.5])
    u = np.array([1.2, -0.4, 0.9])
    assert np.allclose(op @ v, residual.to_numpy() @ v)
    assert np.allclose(op.T @ u, residual.to_numpy().T @ u)
    pd.testing.assert_frame_equal(op.to_dense(), residual)

    # Scaling law R(a * C) = sqrt(a) * R(C) (NOT scale-invariant!)
    res_4x = pearson_residuals(counts * 4)
    assert np.allclose(res_4x.to_numpy(), 2.0 * residual.to_numpy())

    # Forbidden coercions rejected
    with pytest.raises(ValueError):
        pearson_residuals(np.array([[1.0, -1.0], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        pearson_residuals(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="integer"):
        normalize_chi_squared(np.array([[1.5, 2.0], [3.0, 4.0]]), reference_profile="v6")


def test_zero_expected_is_finite():
    counts = np.array([[2, 0], [0, 0]])
    assert np.isfinite(pearson_residuals(counts)).all()
