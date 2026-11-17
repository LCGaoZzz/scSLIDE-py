#!/usr/bin/env python3
"""Portable newcomer & R-to-Python migration quickstart for scslide-py (v0.3.3).

Runs offline in any clean virtual environment with `scslide-py[anndata]` installed,
requiring zero private repository paths, zero external downloads, and zero R runtime.
Uses `scslide.build_synthetic_cohort_fixture()` (explicitly marked `synthetic_demo_only=True`)
or an exported directory produced by `examples/export_seurat_for_scslide_py.R`.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
import scipy.sparse as sp

import scslide as sl


def load_exported_r_bundle(bundle_dir: Path) -> ad.AnnData:
    """Ingest a one-time R Seurat bundle exported by `export_seurat_for_scslide_py.R`."""
    cells = (bundle_dir / "cells.txt").read_text(encoding="utf-8").splitlines()
    genes = (bundle_dir / "genes.txt").read_text(encoding="utf-8").splitlines()
    counts_csr = mmread(bundle_dir / "counts_cells_by_genes.mtx").tocsr().astype(np.float64)
    meta = pd.read_csv(bundle_dir / "cell_metadata.csv")
    if "cell_id" in meta.columns:
        meta = meta.set_index("cell_id", drop=True)
    meta = meta.loc[cells].copy()

    adata = ad.AnnData(X=counts_csr.copy(), obs=meta, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts_csr.copy()
    sl.normalize_and_hvg(adata, n_top_genes=min(2000, len(genes)))

    vf_path = bundle_dir / "variable_features.txt"
    if vf_path.exists():
        vf_set = set(vf_path.read_text(encoding="utf-8").splitlines())
        adata.var["highly_variable"] = [g in vf_set for g in adata.var_names]

    for emb_csv in sorted(bundle_dir.glob("embedding_*.csv")):
        name = emb_csv.stem.replace("embedding_", "")
        emb_df = pd.read_csv(emb_csv, index_col=0).loc[cells]
        adata.obsm[f"X_{name}"] = emb_df.to_numpy(dtype=np.float64)
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description="Portable scslide-py user & R-migration workflow")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for models, objects, plots & CSVs")
    parser.add_argument("--r-bundle", type=Path, default=None, help="Optional R bundle from export_seurat_for_scslide_py.R")
    parser.add_argument("--sample-col", type=str, default="Donor")
    parser.add_argument("--subject-sep", type=str, default="-")
    parser.add_argument("--response-col", type=str, default="Disease")
    parser.add_argument("--celltype-col", type=str, default="cell_type")
    parser.add_argument("--de-celltype", type=str, default="CD14_Mono")
    parser.add_argument("--embedding-key", type=str, default="X_pca")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    fig_dir = out_dir / "figures"
    obj_dir = out_dir / "objects"
    fig_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    if args.r_bundle is not None:
        adata = load_exported_r_bundle(args.r_bundle.resolve())
        input_mode = "exported_r_seurat_bundle"
    else:
        adata = sl.build_synthetic_cohort_fixture(seed=20261116, n_subjects=12, n_repeat_subjects=2, cells_per_sample=30, n_genes=60)
        input_mode = "synthetic_demo_fixture"

    # 1. Audit sample -> biological subject mapping
    subj_map = sl.resolve_subject_mapping(
        adata.obs,
        sample_col=args.sample_col,
        subject_id_sep=args.subject_sep if args.subject_sep else None,
        invariant_cols=[c for c in ("Age", "Sex") if c in adata.obs.columns],
        time_col="TimeSinceOnset" if "TimeSinceOnset" in adata.obs.columns else None,
    )

    # 2. Prepare cell-level supervised PLS + WNN landmarks (R PrepareSampleObject counterpart)
    n_emb = adata.obsm[args.embedding_key].shape[1]
    prep = sl.prepare_sample_object(
        adata,
        y=args.response_col,
        add_hvg=True,
        group_by_cortest=args.celltype_col,
        sketch_training=True,
        group_by_sketch=args.sample_col,
        ncells_per_group=25,
        ncells_landmark=min(48, adata.n_obs // 2),
        ncomp=min(4, n_emb),
        pls_function="cppls",
        modality="multi",
        k_nn=5,
        knn_range=min(48, adata.n_obs // 2),
        name_reduction_1=args.embedding_key,
        dims_reduction_1=range(min(8, n_emb)),
        fix_wnn_weights=(0.5, 0.5),
        nn_method="exact",
        n_jobs=1,
        seed=123,
        reference_profile="v6",
        return_state=True,
    )

    # 3. Generate sample-level LMC object & restrict to primary baseline visit per subject
    s_all = sl.generate_sample_object(
        prep.neighbors.nn_idx,
        group_by=args.sample_col,
        k_nn=5,
        normalization_method="ChiSquared",
        rename_group_by=args.celltype_col,
        center_residuals=True,
        scale_max=10.0,
        sample_for_cell=adata.obs[args.sample_col],
        landmark_ids=prep.landmark_cell_ids,
        cell_ids=adata.obs_names,
        cell_metadata=adata.obs,
    )
    s_primary = s_all.subset_samples(subj_map.primary_samples, recompute_normalization=False)

    # 4. Sample-level Diffusion Map & Principal Curve trajectory
    # Use eigen_which="LA" (largest algebraic positive modes) when sample count is small (<=20)
    # so DC_2 is a non-degenerate positive diffusion coordinate rather than a truncated negative mode.
    eig_mode = "LA" if s_primary.n_samples <= 20 else "LM"
    dm = sl.run_diffusion_map(s_primary, ncomp=3, metric="cosine", scale_max=10.0, eigen_which=eig_mode)
    pcurve = sl.fit_principal_curve(dm.embeddings.iloc[:, :2], maxit=20)
    s_primary.metadata["traj_DC1"] = dm.embeddings.iloc[:, 0].to_numpy()
    s_primary.metadata["traj_DC2"] = dm.embeddings.iloc[:, 1].to_numpy()
    s_primary.metadata["traj_overall"] = pcurve.lambda_

    # 5. R-compatible descriptive Trajectory DE (glmGamPoi QL-F)
    pb = sl.aggregate_pseudobulk(adata, group_by=(args.sample_col, args.celltype_col), layer="counts")
    sub_pb = pb[
        (pb.obs[args.celltype_col].astype(str) == args.de_celltype)
        & (pb.obs[args.sample_col].astype(str).isin(subj_map.primary_samples))
    ].copy()
    for col in ("Age", "Sex", "traj_overall", args.response_col):
        if col in s_primary.metadata.columns:
            sub_pb.obs[col] = s_primary.metadata.loc[sub_pb.obs[args.sample_col].astype(str), col].to_numpy()
    covs = [c for c in ("Age", "Sex", "log_count") if c in sub_pb.obs.columns]
    de_desc = sl.traj_de_test(sub_pb, traj_var="traj_overall", latent_vars=covs, complete_results=True)
    de_csv = out_dir / "descriptive_traj_de.csv"
    de_desc.to_csv(de_csv, index=False)

    # 6. Persist & reload sample AnnData (.h5ad + .zarr) and non-pickle PLSModel (.npz) + query projection
    s_h5ad = obj_dir / "sample_primary.h5ad"
    s_zarr = obj_dir / "sample_primary.zarr"
    s_ad = sl.sample_object_to_anndata(s_primary, include_centered_residuals=True)
    s_ad.write_h5ad(s_h5ad)
    sl.write_sample_zarr(s_ad, s_zarr)
    s_reloaded_h5 = sl.sample_object_from_anndata(ad.read_h5ad(s_h5ad))
    s_reloaded_zarr = sl.sample_object_from_anndata(ad.read_zarr(s_zarr))
    assert s_reloaded_h5.counts.equals(s_primary.counts)
    assert s_reloaded_zarr.counts.equals(s_primary.counts)

    pls_npz = prep.pls_model.save(obj_dir / "cppls_model.npz")
    pls_reloaded = sl.PLSModel.load(pls_npz)
    # Permute model feature columns deliberately to verify DataFrame named-feature alignment
    perm_cols = list(reversed(pls_reloaded.feature_names))
    idx_perm = [adata.var_names.get_loc(g) for g in perm_cols]
    q_sub = adata.layers["data"][:20, idx_perm]
    if sp.issparse(q_sub):
        q_df = pd.DataFrame(q_sub.toarray(), index=adata.obs_names[:20], columns=perm_cols)
    else:
        q_df = pd.DataFrame(np.asarray(q_sub), index=adata.obs_names[:20], columns=perm_cols)
    q_proj = sl.project_pls_embeddings(pls_reloaded, q_df)
    # Also verify sparse matrix + superset feature_names alignment across all cohort genes
    all_rev = list(reversed(adata.var_names))
    idx_all_rev = [adata.var_names.get_loc(g) for g in all_rev]
    ref_proj = sl.project_pls_embeddings(prep.pls_model, adata.layers["data"][:20, idx_all_rev], feature_names=all_rev)
    max_query_align_diff = float(np.max(np.abs(q_proj - ref_proj)))
    assert max_query_align_diff <= 1e-12

    # 7. Publication plots + CSV source-data sidecars without mutating global rcParams
    rc_before = dict(plt.rcParams)
    fig_dm, _, _ = sl.plot_sample_diffusion(
        s_primary, group_by=args.response_col, curve_coords=pcurve, output=fig_dir / "sample_diffusion"
    )
    plt.close(fig_dm)
    dim_out = sl.sample_level_dim_plot(
        adata, s_all, features_to_test=["nCount_LMC"], k_nn=8, umap_backend="native", output=fig_dir / "landmark_umap"
    )
    plt.close("all")
    top_genes = list(de_desc.head(min(12, len(de_desc)))["gene_ID"])
    fig_hm, _, _ = sl.plot_annotated_heatmap(
        sub_pb, top_genes, sort_by="traj_overall", anno_var=args.response_col, output=fig_dir / "trajectory_heatmap"
    )
    plt.close(fig_hm)
    assert plt.rcParams["font.size"] == rc_before["font.size"]

    receipt = {
        "scslide_py_version": md.version("scslide-py"),
        "scslide_module_file": str(Path(sl.__file__).resolve()),
        "input_mode": input_mode,
        "synthetic_demo_only": bool(adata.uns.get("scslide_demo", {}).get("synthetic_demo_only", False)),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_samples_all": int(subj_map.n_samples),
        "n_subjects_primary": int(subj_map.n_subjects),
        "nn_params": prep.neighbors.params,
        "umap_info": dim_out["umap_info"],
        "de_inference_tier": de_desc.attrs.get("inference_tier"),
        "n_genes_tested_de": int(len(de_desc)),
        "max_query_named_alignment_abs_diff": max_query_align_diff,
        "outputs": {
            "descriptive_de_csv": str(de_csv.name),
            "sample_h5ad": str(s_h5ad.relative_to(out_dir)),
            "sample_zarr": str(s_zarr.relative_to(out_dir)),
            "pls_model_npz": str(pls_npz.relative_to(out_dir)),
            "figures": sorted(str(p.relative_to(out_dir)) for p in fig_dir.glob("*")),
        },
    }
    (out_dir / "user_task_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
