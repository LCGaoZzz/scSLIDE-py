import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from scslide import (
    PLSModel,
    SampleLevelObject,
    fit_pls,
    load_rectangular_neighbors,
    sample_object_from_anndata,
    sample_object_to_anndata,
    store_rectangular_neighbors,
    write_sample_zarr,
)


def _object():
    counts = pd.DataFrame([[1, 2, 0], [3, 1, 4]], index=["L1", "L2"], columns=["S1", "S2", "S3"])
    normalized = counts.astype(float) / np.maximum(counts.sum(axis=0), 1) * 100
    residuals = normalized - normalized.mean(axis=1).to_numpy()[:, None]
    metadata = pd.DataFrame(
        {"condition": pd.Categorical(["A", "B", "A"]), "age": [45.0, 62.0, 51.0]},
        index=counts.columns,
    )
    emb = pd.DataFrame([[0.1, -0.2], [0.4, 0.3], [-0.5, -0.1]], index=counts.columns, columns=["DC_1", "DC_2"])
    return SampleLevelObject(
        counts,
        normalized=normalized,
        residuals=residuals,
        metadata=metadata,
        embeddings={"DiffMap": emb},
        attrs={"normalization": "log_normalize", "k_nn": 5},
    )


def test_sample_level_anndata_round_trip_dense():
    obj = _object()
    restored = sample_object_from_anndata(sample_object_to_anndata(obj))
    pd.testing.assert_frame_equal(restored.counts, obj.counts)
    pd.testing.assert_frame_equal(restored.metadata[["age"]], obj.metadata[["age"]])
    assert list(restored.metadata["condition"].astype(str)) == ["A", "B", "A"]
    pd.testing.assert_frame_equal(restored.normalized, obj.normalized)
    assert "diffmap" in restored.embeddings


def test_b06_rejects_ordinary_cell_anndata_without_explicit_unit_declaration():
    cell_adata = ad.AnnData(
        X=sp.csr_matrix([[1, 0, 2], [0, 3, 1]]),
        obs=pd.DataFrame({"Donor": ["D1", "D2"]}, index=["cell_1", "cell_2"]),
        var=pd.DataFrame(index=["GENE1", "GENE2", "GENE3"]),
    )
    cell_adata.layers["counts"] = cell_adata.X.copy()
    with pytest.raises(ValueError, match="schema"):
        sample_object_from_anndata(cell_adata)

    # Explicit cell-unit tag is always rejected even if confirm_sample_landmark_units=True
    cell_adata.uns["scslide"] = {"schema_version": "cell-1", "observation_unit": "cell", "variable_unit": "gene"}
    with pytest.raises(ValueError, match="observation_unit"):
        sample_object_from_anndata(cell_adata, confirm_sample_landmark_units=True)


def test_b07_rejects_int_and_string_id_collision():
    bad_df = pd.DataFrame([[1, 2], [3, 4]], index=[1, "1"], columns=["S1", "S2"])
    with pytest.raises(ValueError, match="collision"):
        SampleLevelObject(bad_df)


def test_b08_and_b09_csr_csc_backed_h5ad_zarr_and_model_round_trip(tmp_path):
    obj = _object()
    for matrix in (sp.csr_matrix(obj.counts.T.to_numpy()), sp.csc_matrix(obj.counts.T.to_numpy())):
        adata = ad.AnnData(X=matrix, obs=pd.DataFrame(index=obj.counts.columns), var=pd.DataFrame(index=obj.counts.index))
        adata.layers["counts"] = matrix
        restored = sample_object_from_anndata(adata, confirm_sample_landmark_units=True)
        pd.testing.assert_frame_equal(restored.counts, obj.counts)
        assert "sparse_" in str(restored.attrs.get("anndata_materialization_mode"))

    # H5AD dense & backed='r' round-trip
    s_ad = sample_object_to_anndata(obj, include_centered_residuals=True)
    h5_path = tmp_path / "sample_level.h5ad"
    s_ad.write_h5ad(h5_path)

    reloaded_mem = ad.read_h5ad(h5_path)
    obj_mem = sample_object_from_anndata(reloaded_mem)
    pd.testing.assert_frame_equal(obj_mem.counts, obj.counts)
    pd.testing.assert_frame_equal(obj_mem.residuals, obj.residuals)
    assert obj_mem.attrs.get("k_nn") == 5

    reloaded_backed = ad.read_h5ad(h5_path, backed="r")
    obj_backed = sample_object_from_anndata(reloaded_backed)
    pd.testing.assert_frame_equal(obj_backed.counts, obj.counts)
    reloaded_backed.file.close()

    # Zarr round-trip
    zarr_path = write_sample_zarr(s_ad, tmp_path / "sample_level.zarr")
    reloaded_zarr = ad.read_zarr(zarr_path)
    obj_zarr = sample_object_from_anndata(reloaded_zarr)
    pd.testing.assert_frame_equal(obj_zarr.counts, obj.counts)

    # Rectangular query x landmark neighbor registry (never .obsp)
    cell_ad = ad.AnnData(
        X=sp.csr_matrix(np.ones((6, 4))),
        obs=pd.DataFrame({"Donor": ["D1", "D1", "D2", "D2", "D3", "D3"]}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    nn_idx = np.array([[1, 2], [2, 1], [1, 2], [2, 1], [1, 1], [2, 2]])
    nn_dist = np.array([[0.0, 0.4], [0.0, 0.4], [0.2, 0.5], [0.1, 0.6], [0.3, 0.3], [0.2, 0.2]])
    store_rectangular_neighbors(cell_ad, "weighted.nn", nn_idx=nn_idx, nn_dist=nn_dist, landmark_ids=["c0", "c1"])
    assert len(cell_ad.obsp.keys()) == 0
    sub_ad = cell_ad[[0, 2, 4]].copy()
    sub_idx, sub_dist, sub_q, lm_ids = load_rectangular_neighbors(sub_ad, "weighted.nn")
    assert list(sub_q) == ["c0", "c2", "c4"]
    assert lm_ids == ["c0", "c1"]
    assert np.array_equal(sub_idx, nn_idx[[0, 2, 4]])
    assert np.allclose(sub_dist, nn_dist[[0, 2, 4]])

    # Safe non-pickle PLSModel save/load round-trip
    rng = np.random.default_rng(19)
    x_df = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f"gene_{i}" for i in range(6)])
    y_vec = 1.5 * x_df["gene_0"] - 0.8 * x_df["gene_3"] + rng.normal(scale=0.1, size=30)
    m = fit_pls(x_df, y_vec, n_components=4, method="cppls")
    saved = m.save(tmp_path / "pls_model.npz")
    m_loaded = PLSModel.load(saved)
    assert np.allclose(m.predict(x_df), m_loaded.predict(x_df))
    assert np.allclose(m.transform(x_df), m_loaded.transform(x_df))
