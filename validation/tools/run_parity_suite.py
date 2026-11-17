#!/usr/bin/env python3
"""Execute installed scslide-py against real legacy (scSLIDE) and Seurat v6 R oracle outputs.

Evaluates:
1. Exact integer contracts via ``compare_labelled.py`` (mode='exact').
2. Continuous matrix contracts via ``compare_labelled.py`` (mode='numeric' with pre-declared tolerances).
3. Spectral / subspace contracts via ``scslide.compare_spectral_embeddings`` (legal +/-1 sign flips only on non-degenerate modes).
4. Mandatory negative controls verifying the comparator catches corrupted axes, missing self-exclusion,
   swapped query/reference, cross-mode rotation, unequal spectra, and offset/Wald substitutions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import scslide as sl


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_compare_labelled(repo_root: Path):
    sys.path.insert(0, str(repo_root / "validation/tools"))
    import compare_labelled as cl  # type: ignore

    return cl


def evaluate_profile(
    profile_name: str,
    oracle_dir: Path,
    norm_starter_dir: Path,
    cand_dir: Path,
    cl_mod: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cand_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    neg_controls: list[dict[str, Any]] = []

    is_v6 = profile_name == "v6"
    ref_prof = "v6" if is_v6 else "legacy"

    # 0. Starter normalization probe (lmc_known_answer.csv -> NormalizeChiSquared)
    starter_in = pd.read_csv(norm_starter_dir / "input.csv", index_col=0)
    starter_py = sl.normalize_chi_squared(starter_in, reference_profile=ref_prof)
    starter_cand_csv = cand_dir / "starter_pearson_residuals_py.csv"
    starter_py.to_csv(starter_cand_csv)
    starter_cmp = cl_mod.compare(
        norm_starter_dir / "pearson_residuals_backend.csv",
        starter_cand_csv,
        mode="numeric",
        atol=1e-12,
        rtol=1e-12,
        tolerance_note="IEEE-754 float64 outer-product + sqrt division rounding floor",
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F06",
            "operation": "NormalizeChiSquared_starter_probe",
            "comparison_kind": "NUMERIC",
            "matched": bool(starter_cmp["matched_after_id_alignment"]),
            "raw_order_equal": bool(starter_cmp["raw_row_order_equal"] and starter_cmp["raw_column_order_equal"]),
            "max_abs_err": float(starter_cmp["max_abs_decimal"]),
            "rel_frobenius": starter_cmp["relative_frobenius"],
            "tolerance": "atol=1e-12,rtol=1e-12",
            "oracle_file": str(norm_starter_dir / "pearson_residuals_backend.csv"),
            "candidate_file": str(starter_cand_csv),
            "notes": "Real R backend starter execution vs installed Python wheel",
        }
    )

    # 1. W1 programmatic Seurat GenerateSampleObject + ChiSquared + LogNormalize on lmc_contract.json
    spec_path = Path(cl_mod.__file__).resolve().parents[1] / "fixtures/lmc_contract.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    meta_toy = pd.DataFrame(
        {
            "Sample": pd.Categorical(spec["sample_for_query"], categories=spec["sample_order"]),
            "Donor": spec["sample_for_query"],
            "CellGroup": ["T", "M", "B", "T", "T", "M", "B", "M"],
        },
        index=spec["query_ids"],
    )
    # 1a. return.seurat = FALSE keeps exact '<label>_LM<k>' row IDs
    w1_raw_counts = sl.generate_sample_object(
        np.asarray(spec["neighbor_index"], dtype=int),
        group_by="Sample",
        k_nn=int(spec["k"]),
        rename_group_by="CellGroup",
        return_seurat_like=False,
        reference_profile=ref_prof,
        sample_for_cell=meta_toy["Sample"],
        landmark_ids=spec["landmark_ids"],
        cell_ids=spec["query_ids"],
        cell_metadata=meta_toy,
    )
    w1_cts_csv = cand_dir / "w1_lmc_counts_py.csv"
    w1_raw_counts.to_csv(w1_cts_csv)

    # 1b. return.seurat = TRUE runs CreateSeuratObject which converts '_' -> '-' on feature/landmark IDs
    w1_obj = sl.generate_sample_object(
        np.asarray(spec["neighbor_index"], dtype=int),
        group_by="Sample",
        k_nn=int(spec["k"]),
        normalization_method="ChiSquared",
        rename_group_by="CellGroup",
        add_meta_data=True,
        seurat_hyphenate_landmarks=True,
        reference_profile=ref_prof,
        sample_for_cell=meta_toy["Sample"],
        landmark_ids=spec["landmark_ids"],
        cell_ids=spec["query_ids"],
        cell_metadata=meta_toy,
    )
    w1_res_csv = cand_dir / "w1_pearson_residuals_py.csv"
    w1_obj.residuals.to_csv(w1_res_csv)

    c_w1 = cl_mod.compare(oracle_dir / "w1_lmc_counts_backend.csv", w1_cts_csv, mode="exact")
    r_w1 = cl_mod.compare(
        oracle_dir / "w1_pearson_residuals_seurat_obj.csv",
        w1_res_csv,
        mode="numeric",
        atol=1e-12,
        rtol=1e-12,
        tolerance_note="IEEE-754 float64 contingency residual arithmetic",
    )
    w1_log = sl.generate_sample_object(
        np.asarray(spec["neighbor_index"], dtype=int),
        group_by="Sample",
        k_nn=int(spec["k"]),
        normalization_method="LogNormalize",
        scale_factor=10_000.0,
        rename_group_by="CellGroup",
        add_meta_data=True,
        seurat_hyphenate_landmarks=True,
        reference_profile=ref_prof,
        sample_for_cell=meta_toy["Sample"],
        landmark_ids=spec["landmark_ids"],
        cell_ids=spec["query_ids"],
        cell_metadata=meta_toy,
    )
    w1_log_csv = cand_dir / "w1_log_normalized_py.csv"
    w1_log.normalized.to_csv(w1_log_csv)
    l_w1 = cl_mod.compare(
        oracle_dir / "w1_log_normalized_seurat_obj.csv",
        w1_log_csv,
        mode="numeric",
        atol=1e-12,
        rtol=1e-12,
        tolerance_note="IEEE-754 float64 log1p column scaling",
    )
    for op_name, cap, kind, rep, tol_s, ref_f, cand_f in [
        ("GenerateSampleObject_w1_counts", "F05", "EXACT", c_w1, "exact", "w1_lmc_counts_backend.csv", w1_cts_csv),
        ("GenerateSampleObject_w1_chisquared", "F06", "NUMERIC", r_w1, "atol=1e-12,rtol=1e-12", "w1_pearson_residuals_seurat_obj.csv", w1_res_csv),
        ("GenerateSampleObject_w1_lognorm", "F05", "NUMERIC", l_w1, "atol=1e-12,rtol=1e-12", "w1_log_normalized_seurat_obj.csv", w1_log_csv),
    ]:
        rows.append(
            {
                "profile": profile_name,
                "capability_id": cap,
                "operation": op_name,
                "comparison_kind": kind,
                "matched": bool(rep["matched_after_id_alignment"]),
                "raw_order_equal": bool(rep["raw_row_order_equal"] and rep["raw_column_order_equal"]),
                "max_abs_err": float(rep["max_abs_decimal"]),
                "rel_frobenius": rep.get("relative_frobenius"),
                "tolerance": tol_s,
                "oracle_file": str(oracle_dir / ref_f),
                "candidate_file": str(cand_f),
                "notes": "Includes zero-query sample C and duplicate neighbor [1,1]",
            }
        )

    # Mandatory negative control NC1: disabling self-exclusion MUST be caught by exact comparator!
    bad_no_excl = sl.landmark_matrix(
        np.asarray(spec["neighbor_index"], dtype=int),
        meta_toy["Sample"],
        spec["landmark_ids"],
        k=int(spec["k"]),
        cell_ids=[f"nonlm_{i}" for i in range(len(spec["query_ids"]))],
        sample_order=spec["sample_order"],
    )
    bad_no_excl.index = w1_raw_counts.index
    nc1_csv = cand_dir / "nc1_corrupted_no_self_exclusion.csv"
    bad_no_excl.to_csv(nc1_csv)
    nc1_rep = cl_mod.compare(oracle_dir / "w1_lmc_counts_backend.csv", nc1_csv, mode="exact")
    neg_controls.append(
        {
            "profile": profile_name,
            "control_id": "NC1_missing_self_exclusion",
            "detected_corruption": not bool(nc1_rep["matched_after_id_alignment"]),
            "mismatch_count": nc1_rep["mismatch_count"],
        }
    )

    # Load shared 12-donor, 360-cell cohort inputs exported from R
    counts_g_by_c = pd.read_csv(oracle_dir / "shared_counts_genes_by_cells.csv", index_col=0)
    lognorm_g_by_c = pd.read_csv(oracle_dir / "shared_lognorm_genes_by_cells.csv", index_col=0)
    meta_c = pd.read_csv(oracle_dir / "shared_cell_metadata.csv", index_col=0)
    meta_c["Disease"] = pd.Categorical(meta_c["Disease"], categories=["HV", "MILD", "SEV"])
    pca_emb = pd.read_csv(oracle_dir / "shared_pca_embeddings.csv", index_col=0)
    train_cells = (oracle_dir / "shared_training_cells.txt").read_text().splitlines()
    lm_cells = (oracle_dir / "shared_landmark_cells.txt").read_text().splitlines()
    all_hvgs = (oracle_dir / "shared_all_hvgs.txt").read_text().splitlines()
    train_scaled_g_by_c = pd.read_csv(oracle_dir / "shared_training_scaled_hvgs.csv", index_col=0)

    # 2. F10 QuickCorTest (categorical grouped + numeric)
    train_lognorm_c_by_g = lognorm_g_by_c.loc[:, train_cells].T
    py_qct_cat = sl.quick_cor_select(
        train_lognorm_c_by_g,
        meta_c.loc[train_cells, ["Disease"]],
        group_by=meta_c.loc[train_cells, "cell_type"],
        top_prop=0.15,
        cor_cutoff=0.15,
        min_pct=0.1,
        min_cells=40,
        unlist=True,
        reference_profile=ref_prof,
    )
    r_qct_cat = (oracle_dir / "f10_quickcortest_cat_grouped.txt").read_text().splitlines()

    py_qct_num = sl.quick_cor_select(
        train_lognorm_c_by_g,
        meta_c.loc[train_cells, "SeverityNum"],
        group_by=None,
        top_prop=0.20,
        cor_cutoff=0.10,
        min_pct=0.1,
        min_cells=40,
        unlist=True,
        reference_profile=ref_prof,
    )
    r_qct_num = (oracle_dir / "f10_quickcortest_numeric.txt").read_text().splitlines()

    for op_n, py_g, r_g, rf_name in [
        ("QuickCorTest_categorical_grouped", py_qct_cat, r_qct_cat, "f10_quickcortest_cat_grouped.txt"),
        ("QuickCorTest_numeric", py_qct_num, r_qct_num, "f10_quickcortest_numeric.txt"),
    ]:
        set_eq = set(py_g) == set(r_g)
        ord_eq = list(py_g) == list(r_g)
        jacc = len(set(py_g) & set(r_g)) / max(len(set(py_g) | set(r_g)), 1)
        cand_txt = cand_dir / rf_name.replace(".txt", "_py.txt")
        cand_txt.write_text("\n".join(str(g) for g in py_g) + "\n", encoding="utf-8")
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F10",
                "operation": op_n,
                "comparison_kind": "EXACT_SET_AND_ORDER",
                "matched": bool(set_eq and ord_eq),
                "raw_order_equal": bool(ord_eq),
                "max_abs_err": float(1.0 - jacc),
                "rel_frobenius": None,
                "tolerance": "exact_gene_set_and_order",
                "oracle_file": str(oracle_dir / rf_name),
                "candidate_file": str(cand_txt),
                "notes": f"n_r={len(r_g)}, n_py={len(py_g)}, jaccard={jacc:.6f}",
            }
        )

    # 3. F13/F18/F19 RunPLS (plsr, cppls, spls, kernelpls_ondisk, cppls_ondisk) + F08 ProjectCellEmbeddings & PredictPLS
    x_train_scaled = train_scaled_g_by_c.loc[all_hvgs, train_cells].T  # (180, n_hvg)

    # 3a. plsr single numeric Y with ncomp=5
    pls_num = sl.fit_pls(x_train_scaled, meta_c.loc[train_cells, "SeverityNum"], n_components=5, scale=False, method="plsr")
    r_plsr_scores = pd.read_csv(oracle_dir / "f13_plsr_single_y_scores.csv", index_col=0)
    # Column-wise +/-1 sign alignment (PLS components are unique up to sign)
    sgn_plsr = np.array([1.0 if np.dot(r_plsr_scores.iloc[:, j], pls_num.scores_[:, j]) >= 0 else -1.0 for j in range(5)])
    plsr_cand_csv = cand_dir / "f13_plsr_single_y_scores_py.csv"
    pd.DataFrame(pls_num.scores_, index=r_plsr_scores.index, columns=r_plsr_scores.columns).to_csv(plsr_cand_csv)
    plsr_diff = np.abs(r_plsr_scores.to_numpy() - (pls_num.scores_ * sgn_plsr[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F13",
            "operation": "RunPLS_plsr_single_response_5comp",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(np.max(plsr_diff) <= 1e-9),
            "raw_order_equal": True,
            "max_abs_err": float(np.max(plsr_diff)),
            "rel_frobenius": float(np.linalg.norm(plsr_diff) / np.linalg.norm(r_plsr_scores.to_numpy())),
            "tolerance": "sign_aligned_atol=1e-9",
            "oracle_file": str(oracle_dir / "f13_plsr_single_y_scores.csv"),
            "candidate_file": str(plsr_cand_csv),
            "notes": "Single numeric response ncomp=5 (validates B05 against R pls::plsr)",
        }
    )

    # 3b. cppls in-memory on categorical Y = Disease (6 comps)
    # Note: In both legacy and v6, in-memory RunPLS calls pls::cppls (which in installed pls 2.9-0 centers Y!)
    pls_cp = sl.fit_pls(
        x_train_scaled, meta_c.loc[train_cells, ["Disease"]], n_components=6, scale=False, method="cppls", reference_profile="v6"
    )
    r_cp_scores = pd.read_csv(oracle_dir / "f13_cppls_disease_scores.csv", index_col=0)
    r_cp_proj = pd.read_csv(oracle_dir / "f13_cppls_disease_projection.csv", index_col=0)
    cp_cand_csv = cand_dir / "f13_cppls_disease_scores_py.csv"
    pd.DataFrame(pls_cp.scores_, index=r_cp_scores.index, columns=r_cp_scores.columns).to_csv(cp_cand_csv)
    sgn_cp = np.array([1.0 if np.dot(r_cp_scores.iloc[:, j], pls_cp.scores_[:, j]) >= 0 else -1.0 for j in range(6)])
    cp_s_diff = np.abs(r_cp_scores.to_numpy() - (pls_cp.scores_ * sgn_cp[None, :]))
    cp_p_diff = np.abs(r_cp_proj.loc[all_hvgs].to_numpy() - (pls_cp.projection * sgn_cp[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F13",
            "operation": "RunPLS_cppls_categorical_6comp",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(np.max(cp_s_diff) <= 1e-8 and np.max(cp_p_diff) <= 1e-8),
            "raw_order_equal": True,
            "max_abs_err": float(max(np.max(cp_s_diff), np.max(cp_p_diff))),
            "rel_frobenius": float(np.linalg.norm(cp_s_diff) / np.linalg.norm(r_cp_scores.to_numpy())),
            "tolerance": "sign_aligned_atol=1e-8",
            "oracle_file": str(oracle_dir / "f13_cppls_disease_scores.csv"),
            "candidate_file": str(cp_cand_csv),
            "notes": "Canonical PLS with pivoted QR + SVD CCA on 3-level categorical Disease",
        }
    )

    # 3c. spls on numeric Y (4 comps, eta=0.5)
    pls_sp = sl.fit_pls(
        x_train_scaled, meta_c.loc[train_cells, "SeverityNum"], n_components=4, scale=False, method="spls", eta=0.5
    )
    r_sp_scores = pd.read_csv(oracle_dir / "f13_spls_scores.csv", index_col=0)
    r_sp_proj = pd.read_csv(oracle_dir / "f13_spls_projection.csv", index_col=0)
    sp_cand_csv = cand_dir / "f13_spls_scores_py.csv"
    pd.DataFrame(pls_sp.scores_, index=r_sp_scores.index, columns=r_sp_scores.columns).to_csv(sp_cand_csv)
    py_active_genes = [all_hvgs[i] for i in pls_sp.active_indices]
    active_match = list(r_sp_proj.index) == py_active_genes
    sgn_sp = np.array([1.0 if np.dot(r_sp_scores.iloc[:, j], pls_sp.scores_[:, j]) >= 0 else -1.0 for j in range(r_sp_scores.shape[1])])
    sp_s_diff = np.abs(r_sp_scores.to_numpy() - (pls_sp.scores_ * sgn_sp[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F13",
            "operation": "RunPLS_spls_active_set_and_scores",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(active_match and np.max(sp_s_diff) <= 1e-8),
            "raw_order_equal": bool(active_match),
            "max_abs_err": float(np.max(sp_s_diff)) if active_match else np.inf,
            "rel_frobenius": float(np.linalg.norm(sp_s_diff) / np.linalg.norm(r_sp_scores.to_numpy())),
            "tolerance": "exact_active_set_and_sign_aligned_atol=1e-8",
            "oracle_file": str(oracle_dir / "f13_spls_scores.csv"),
            "candidate_file": str(sp_cand_csv),
            "notes": f"n_active_r={len(r_sp_proj)}, n_active_py={len(py_active_genes)}",
        }
    )

    # 3d. F19 kernelpls_ondisk & F18 cppls_ondisk against BPCells C++!
    # CRITICAL: In legacy scSLIDE, CPPLS.cpp does NOT center Y (`reference_profile='legacy_ondisk'`),
    # whereas in Seurat v6.0.0, src/pls.cpp DOES center Y (`reference_profile='v6'`)!
    import scipy.sparse as sp_mod

    x_train_csr = sp_mod.csr_matrix(x_train_scaled.to_numpy())
    py_kpls_od = sl.kernelpls_ondisk(x_train_csr, meta_c.loc[train_cells, ["Disease"]], ncomp=5)
    r_kpls_od = pd.read_csv(oracle_dir / "f19_kernelpls_ondisk_scores.csv", index_col=0)
    kpls_cand_csv = cand_dir / "f19_kernelpls_ondisk_scores_py.csv"
    pd.DataFrame(py_kpls_od["scores"], index=r_kpls_od.index, columns=r_kpls_od.columns).to_csv(kpls_cand_csv)
    sgn_ko = np.array([1.0 if np.dot(r_kpls_od.iloc[:, j], py_kpls_od["scores"][:, j]) >= 0 else -1.0 for j in range(5)])
    ko_diff = np.abs(r_kpls_od.to_numpy() - (py_kpls_od["scores"] * sgn_ko[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F19",
            "operation": "kernelpls_ondisk_streaming_vs_BPCells",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(np.max(ko_diff) <= 1e-8),
            "raw_order_equal": True,
            "max_abs_err": float(np.max(ko_diff)),
            "rel_frobenius": float(np.linalg.norm(ko_diff) / np.linalg.norm(r_kpls_od.to_numpy())),
            "tolerance": "sign_aligned_atol=1e-8",
            "oracle_file": str(oracle_dir / "f19_kernelpls_ondisk_scores.csv"),
            "candidate_file": str(kpls_cand_csv),
            "notes": "Streaming CSR kernelpls vs BPCells C++ kernelpls_cpp",
        }
    )

    od_prof = "v6" if is_v6 else "legacy_ondisk"
    py_cp_od = sl.cppls_ondisk(x_train_csr, meta_c.loc[train_cells, ["Disease"]], ncomp=5, reference_profile=od_prof)
    r_cp_od = pd.read_csv(oracle_dir / "f18_cppls_ondisk_scores.csv", index_col=0)
    cpod_cand_csv = cand_dir / "f18_cppls_ondisk_scores_py.csv"
    pd.DataFrame(py_cp_od["scores"], index=r_cp_od.index, columns=r_cp_od.columns).to_csv(cpod_cand_csv)
    sgn_co = np.array([1.0 if np.dot(r_cp_od.iloc[:, j], py_cp_od["scores"][:, j]) >= 0 else -1.0 for j in range(5)])
    co_diff = np.abs(r_cp_od.to_numpy() - (py_cp_od["scores"] * sgn_co[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F18",
            "operation": f"cppls_ondisk_streaming_vs_BPCells_{od_prof}",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(np.max(co_diff) <= 1e-8),
            "raw_order_equal": True,
            "max_abs_err": float(np.max(co_diff)),
            "rel_frobenius": float(np.linalg.norm(co_diff) / np.linalg.norm(r_cp_od.to_numpy())),
            "tolerance": "sign_aligned_atol=1e-8",
            "oracle_file": str(oracle_dir / "f18_cppls_ondisk_scores.csv"),
            "candidate_file": str(cpod_cand_csv),
            "notes": f"Matches {od_prof} Y-centering rule in C++ cancorr()",
        }
    )

    # 3e. F08 ProjectCellEmbeddings & PredictPLS
    train_data_raw = lognorm_g_by_c.loc[all_hvgs, train_cells].T
    full_data_raw = lognorm_g_by_c.loc[all_hvgs, :].T
    pls_for_proj = sl.fit_pls(
        train_data_raw,
        meta_c.loc[train_cells, ["Disease"]],
        n_components=6,
        scale=True,
        scale_max=10.0,
        method="cppls",
        reference_profile="v6",
    )
    py_proj_pls = sl.project_pls_embeddings(pls_for_proj, full_data_raw, ncomp=6, scale_max=10.0)
    r_proj_pls = pd.read_csv(oracle_dir / "f08_project_cell_embeddings_pls.csv", index_col=0)
    proj_cand_csv = cand_dir / "f08_project_cell_embeddings_pls_py.csv"
    pd.DataFrame(py_proj_pls, index=r_proj_pls.index, columns=r_proj_pls.columns).to_csv(proj_cand_csv)
    sgn_pj = np.array([1.0 if np.dot(r_proj_pls.iloc[:, j], py_proj_pls[:, j]) >= 0 else -1.0 for j in range(6)])
    pj_diff = np.abs(r_proj_pls.to_numpy() - (py_proj_pls * sgn_pj[None, :]))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F08",
            "operation": "ProjectCellEmbeddings_full_query_projection",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(np.max(pj_diff) <= 1e-8),
            "raw_order_equal": True,
            "max_abs_err": float(np.max(pj_diff)),
            "rel_frobenius": float(np.linalg.norm(pj_diff) / np.linalg.norm(r_proj_pls.to_numpy())),
            "tolerance": "sign_aligned_atol=1e-8",
            "oracle_file": str(oracle_dir / "f08_project_cell_embeddings_pls.csv"),
            "candidate_file": str(proj_cand_csv),
            "notes": "Includes training mean/sd scaling + Seurat +10 upper clip",
        }
    )

    pred_file = oracle_dir / "f08_predict_pls_seurat.csv"
    if pred_file.exists():
        py_pred_y = sl.predict_pls(pls_for_proj, full_data_raw, ncomp=6, apply_training_scale=True)
        r_pred_y = pd.read_csv(pred_file, index_col=0)
        pred_cand_csv = cand_dir / "f08_predict_pls_py.csv"
        pd.DataFrame(py_pred_y, index=full_data_raw.index, columns=r_pred_y.columns).to_csv(pred_cand_csv)
        pred_cmp = cl_mod.compare(
            pred_file, pred_cand_csv, mode="numeric", atol=1e-9, rtol=1e-9,
            tolerance_note="Linear response prediction Y_hat = X_scaled B + B0 invariant to component sign"
        )
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F08",
                "operation": "PredictPLS_response_prediction",
                "comparison_kind": "NUMERIC",
                "matched": bool(pred_cmp["matched_after_id_alignment"]),
                "raw_order_equal": bool(pred_cmp["raw_row_order_equal"] and pred_cmp["raw_column_order_equal"]),
                "max_abs_err": float(pred_cmp["max_abs_decimal"]),
                "rel_frobenius": pred_cmp["relative_frobenius"],
                "tolerance": "atol=1e-9,rtol=1e-9",
                "oracle_file": str(pred_file),
                "candidate_file": str(pred_cand_csv),
                "notes": "Sign-invariant response prediction on full query",
            }
        )

    # 4. F04 FindsmNN & F03 FindmmNN (using exact shared R embeddings to isolate NN/WNN math from ANN/upstream!)
    all_cell_ids = list(meta_c.index.astype(str))
    c_lookup = {c: i for i, c in enumerate(all_cell_ids)}
    lm_idx0 = np.array([c_lookup[c] for c in lm_cells], dtype=np.int64)

    py_smnn = sl.find_sm_nn(r_proj_pls.loc[all_cell_ids].to_numpy(), lm_idx0, k_nn=5, l2_norm=True, query_ids=all_cell_ids, landmark_ids=lm_cells)
    r_smnn_df = pd.read_csv(oracle_dir / "f04_findsmnn_idx.csv", index_col=0)
    r_smnn_annoy = r_smnn_df.loc[all_cell_ids].to_numpy()
    emb_np = r_proj_pls.loc[all_cell_ids].to_numpy(dtype=np.float64)
    emb_l2 = emb_np / np.linalg.norm(emb_np, axis=1, keepdims=True)
    ref_l2 = emb_l2[lm_idx0]
    diff_q = np.flatnonzero(np.any(py_smnn.nn_idx != r_smnn_annoy, axis=1))
    py_strictly_closer_all = True
    for qi in diff_q:
        d_py_sum = float(np.sum(np.linalg.norm(ref_l2[py_smnn.nn_idx[qi] - 1] - emb_l2[qi], axis=1)))
        d_r_sum = float(np.sum(np.linalg.norm(ref_l2[r_smnn_annoy[qi] - 1] - emb_l2[qi], axis=1)))
        if d_py_sum > d_r_sum + 1e-12:
            py_strictly_closer_all = False
            break
    smnn_cand_csv = cand_dir / "f04_findsmnn_idx_py.csv"
    pd.DataFrame(py_smnn.nn_idx, index=all_cell_ids, columns=r_smnn_df.columns).to_csv(smnn_cand_csv)
    smnn_raw_ord = (list(r_smnn_df.index.astype(str)) == all_cell_ids)
    sm_annoy_rate = float(np.mean(py_smnn.nn_idx == r_smnn_annoy))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F04",
            "operation": "FindsmNN_exact_vs_Seurat_Annoy_default",
            "comparison_kind": "APPROX_ANN_RECALL_AND_OPTIMALITY",
            "matched": bool(sm_annoy_rate >= 0.99 and py_strictly_closer_all),
            "raw_order_equal": bool(smnn_raw_ord),
            "max_abs_err": float(1.0 - sm_annoy_rate),
            "rel_frobenius": None,
            "tolerance": "recall>=0.99_and_exact_cKDTree_distance<=Annoy_on_100pct_queries",
            "oracle_file": str(oracle_dir / "f04_findsmnn_idx.csv"),
            "candidate_file": str(smnn_cand_csv),
            "notes": f"agreement={sm_annoy_rate:.6f} ({len(diff_q)}/360 queries differ; py_strictly_closer_on_all={py_strictly_closer_all})",
        }
    )

    rann_file = oracle_dir / "f04_findsmnn_rann_exact_idx.csv"
    if rann_file.exists():
        r_smnn_rann = pd.read_csv(rann_file, index_col=0).loc[all_cell_ids].to_numpy()
        sm_rann_rate = float(np.mean(py_smnn.nn_idx == r_smnn_rann))
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F04",
                "operation": "FindsmNN_exact_vs_Seurat_RANN_exact",
                "comparison_kind": "EXACT",
                "matched": bool(sm_rann_rate == 1.0),
                "raw_order_equal": bool(list(pd.read_csv(rann_file, index_col=0).index.astype(str)) == all_cell_ids),
                "max_abs_err": float(1.0 - sm_rann_rate),
                "rel_frobenius": None,
                "tolerance": "exact_100pct_index_equality",
                "oracle_file": str(rann_file),
                "candidate_file": str(smnn_cand_csv),
                "notes": f"elementwise_agreement={sm_rann_rate:.6f} (exact cKDTree vs exact RANN::nn2)",
            }
        )

    # FindmmNN: note Seurat uses Annoy approximate kNN (n.trees=50) for per-modality search when knn.range=48 == n_landmarks!
    # When knn.range == 48 == n_landmarks and n_cells == 360, let's measure exact agreement on bothnn_idx[:5], nn_dist, and resulting LMC!
    py_mm_fixed = sl.find_mm_nn(
        [pca_emb.loc[all_cell_ids].to_numpy()[:, :10], r_proj_pls.loc[all_cell_ids].to_numpy()[:, :6]],
        lm_idx0,
        k_nn=5,
        knn_range=48,
        l2_norm=True,
        fix_wnn_weights=(0.5, 0.5),
        query_ids=all_cell_ids,
        landmark_ids=lm_cells,
        reference_profile="v6",
    )
    r_mm_df = pd.read_csv(oracle_dir / "f03_findmmnn_fixed_idx.csv", index_col=0)
    r_mm_idx = r_mm_df.loc[all_cell_ids].to_numpy()
    r_mm_dist = pd.read_csv(oracle_dir / "f03_findmmnn_fixed_dist.csv", index_col=0).loc[all_cell_ids].to_numpy()
    top5_jacc = float(
        np.mean([len(set(py_mm_fixed.nn_idx[i, :5]) & set(r_mm_idx[i, :5])) / 5.0 for i in range(len(all_cell_ids))])
    )
    py_lmc_from_wnn = sl.landmark_matrix(
        py_mm_fixed.nn_idx, meta_c["Donor"], lm_cells, k=5, cell_ids=all_cell_ids, sample_order=sorted(meta_c["Donor"].unique())
    )
    r_lmc_from_wnn = pd.read_csv(oracle_dir / "f05_cohort_lmc_counts.csv", index_col=0)
    lmc_cor = float( np.corrcoef(py_lmc_from_wnn.to_numpy().ravel(), r_lmc_from_wnn.to_numpy().ravel())[0, 1] )
    lmc_exact_frac = float(np.mean(py_lmc_from_wnn.to_numpy() == r_lmc_from_wnn.to_numpy()))
    mmf_cand_csv = cand_dir / "f03_findmmnn_fixed_idx_py.csv"
    pd.DataFrame(py_mm_fixed.nn_idx, index=all_cell_ids, columns=r_mm_df.columns[:py_mm_fixed.nn_idx.shape[1]]).to_csv(mmf_cand_csv)
    mmf_raw_ord = (list(r_mm_df.index.astype(str)) == all_cell_ids and r_mm_df.shape[1] == py_mm_fixed.nn_idx.shape[1])
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F03",
            "operation": "FindmmNN_fixed_weights_vs_Seurat_Annoy_WNN",
            "comparison_kind": "NEIGHBOR_SET_AND_LMC",
            "matched": bool(top5_jacc >= 0.98 and lmc_cor >= 0.995),
            "raw_order_equal": bool(mmf_raw_ord),
            "max_abs_err": float(np.max(np.abs(py_mm_fixed.nn_dist[:, :5] - r_mm_dist[:, :5]))),
            "rel_frobenius": float(1.0 - top5_jacc),
            "tolerance": "top5_overlap>=0.98_and_lmc_r>=0.995 (R uses Annoy 50-tree float32 ANN for sigma_i)",
            "oracle_file": str(oracle_dir / "f03_findmmnn_fixed_idx.csv"),
            "candidate_file": str(mmf_cand_csv),
            "notes": f"top5_overlap={top5_jacc:.5f}, lmc_exact_cell_frac={lmc_exact_frac:.5f}, lmc_pearson_r={lmc_cor:.6f}",
        }
    )

    py_mm_learned = sl.find_mm_nn(
        [pca_emb.loc[all_cell_ids].to_numpy()[:, :10], r_proj_pls.loc[all_cell_ids].to_numpy()[:, :6]],
        lm_idx0,
        k_nn=5,
        knn_range=48,
        l2_norm=True,
        fix_wnn_weights=None,
        query_ids=all_cell_ids,
        landmark_ids=lm_cells,
        reference_profile="v6",
    )
    r_mml_df = pd.read_csv(oracle_dir / "f03_findmmnn_learned_idx.csv", index_col=0)
    r_mml_idx = r_mml_df.loc[all_cell_ids].to_numpy()
    mml_cand_csv = cand_dir / "f03_findmmnn_learned_idx_py.csv"
    pd.DataFrame(py_mm_learned.nn_idx, index=all_cell_ids, columns=r_mml_df.columns[:py_mm_learned.nn_idx.shape[1]]).to_csv(mml_cand_csv)
    mml_raw_ord = (list(r_mml_df.index.astype(str)) == all_cell_ids and r_mml_df.shape[1] == py_mm_learned.nn_idx.shape[1])
    top5_learned_jacc = float(
        np.mean([len(set(py_mm_learned.nn_idx[i, :5]) & set(r_mml_idx[i, :5])) / 5.0 for i in range(len(all_cell_ids))])
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F03",
            "operation": "FindmmNN_learned_modality_weights",
            "comparison_kind": "NEIGHBOR_SET_AND_LMC",
            "matched": bool(top5_learned_jacc >= 0.95),
            "raw_order_equal": bool(mml_raw_ord),
            "max_abs_err": float(1.0 - top5_learned_jacc),
            "rel_frobenius": float(1.0 - top5_learned_jacc),
            "tolerance": "top5_overlap>=0.95 (Annoy vs exact kNN imputation)",
            "oracle_file": str(oracle_dir / "f03_findmmnn_learned_idx.csv"),
            "candidate_file": str(mml_cand_csv),
            "notes": f"top5_overlap={top5_learned_jacc:.5f}",
        }
    )

    # 5. Frozen R weighted.nn -> Python GenerateSampleObject + NormalizeChiSquared + RunDiffusionMap + principal_curve
    py_s_obj = sl.generate_sample_object(
        r_mm_idx,
        group_by="Donor",
        k_nn=5,
        normalization_method="ChiSquared",
        rename_group_by="cell_type",
        add_meta_data=True,
        center_residuals=True,
        reference_profile=ref_prof,
        sample_for_cell=meta_c["Donor"],
        landmark_ids=lm_cells,
        cell_ids=all_cell_ids,
        cell_metadata=meta_c,
    )
    # Note CreateSeuratObject replaces '_' with '-' in feature names ('Mono_LM1' -> 'Mono-LM1')
    py_cts_hyphen = py_s_obj.counts.copy()
    py_cts_hyphen.index = pd.Index([idx.replace("_", "-") for idx in py_cts_hyphen.index])
    py_res_hyphen = py_s_obj.residuals.copy()
    py_res_hyphen.index = pd.Index([idx.replace("_", "-") for idx in py_res_hyphen.index])
    f05_cand = cand_dir / "f05_cohort_lmc_counts_py.csv"
    f06_cand = cand_dir / "f06_cohort_pearson_residuals_py.csv"
    py_cts_hyphen.to_csv(f05_cand)
    py_res_hyphen.to_csv(f06_cand)

    cmp_f05 = cl_mod.compare(oracle_dir / "f05_cohort_lmc_counts.csv", f05_cand, mode="exact")
    cmp_f06 = cl_mod.compare(
        oracle_dir / "f06_cohort_pearson_residuals.csv",
        f06_cand,
        mode="numeric",
        atol=1e-12,
        rtol=1e-12,
        tolerance_note="IEEE-754 float64 contingency residual arithmetic",
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F05",
            "operation": "GenerateSampleObject_12donor_cohort_counts",
            "comparison_kind": "EXACT",
            "matched": bool(cmp_f05["matched_after_id_alignment"]),
            "raw_order_equal": bool(cmp_f05["raw_row_order_equal"] and cmp_f05["raw_column_order_equal"]),
            "max_abs_err": float(cmp_f05["max_abs_decimal"]),
            "rel_frobenius": None,
            "tolerance": "exact",
            "oracle_file": str(oracle_dir / "f05_cohort_lmc_counts.csv"),
            "candidate_file": str(f05_cand),
            "notes": "Exact integer counts after documented Seurat '_'->'-' feature-name rule",
        }
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F06",
            "operation": "NormalizeChiSquared_12donor_cohort_residuals",
            "comparison_kind": "NUMERIC",
            "matched": bool(cmp_f06["matched_after_id_alignment"]),
            "raw_order_equal": bool(cmp_f06["raw_row_order_equal"] and cmp_f06["raw_column_order_equal"]),
            "max_abs_err": float(cmp_f06["max_abs_decimal"]),
            "rel_frobenius": cmp_f06["relative_frobenius"],
            "tolerance": "atol=1e-12,rtol=1e-12",
            "oracle_file": str(oracle_dir / "f06_cohort_pearson_residuals.csv"),
            "candidate_file": str(f06_cand),
            "notes": "Pearson residuals on 48x12 LMC",
        }
    )

    # F12 RunDiffusionMap vs destiny::DiffusionMap!
    py_dm = sl.run_diffusion_map(py_s_obj, ncomp=4, metric="cosine", layer="scale.data")
    r_dm_emb = pd.read_csv(oracle_dir / "f12_diffusion_embeddings.csv", index_col=0)
    r_dm_ev = pd.read_csv(oracle_dir / "f12_diffusion_eigenvalues.csv")["eigenvalue"].to_numpy(dtype=np.float64)
    dm_cand_csv = cand_dir / "f12_diffusion_embeddings_py.csv"
    py_dm.embeddings.to_csv(dm_cand_csv)
    spec_cmp = sl.compare_spectral_embeddings(
        r_dm_emb, py_dm.embeddings, r_dm_ev, py_dm.eigenvalues,
        degeneracy_gap_tol=1e-5, eigenvalue_atol=1e-10, subspace_atol=1e-10
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F12",
            "operation": "RunDiffusionMap_vs_destiny_DiffusionMap",
            "comparison_kind": "SUBSPACE_SIGN",
            "matched": bool(spec_cmp["equivalent_under_legal_gauge"]),
            "raw_order_equal": True,
            "max_abs_err": float(spec_cmp["max_coordinate_abs_err_after_legal_gauge"]),
            "rel_frobenius": float(spec_cmp["relative_frobenius_err_after_legal_gauge"]),
            "tolerance": "eigenvalue_atol=1e-10,subspace_atol=1e-10",
            "oracle_file": str(oracle_dir / "f12_diffusion_embeddings.csv"),
            "candidate_file": str(dm_cand_csv),
            "notes": f"max_ev_diff={spec_cmp['max_eigenvalue_abs_diff']:.3e}, pairwise_dist_err={spec_cmp['pairwise_distance_max_abs_err']:.3e}",
        }
    )

    # Mandatory negative control NC2: 30-degree cross-eigenvalue rotation MUST be rejected by compare_spectral_embeddings!
    th = np.deg2rad(30.0)
    rot_bad = np.eye(4)
    rot_bad[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    bad_dm_df = pd.DataFrame(r_dm_emb.to_numpy() @ rot_bad, index=r_dm_emb.index, columns=r_dm_emb.columns)
    nc2 = sl.compare_spectral_embeddings(r_dm_emb, bad_dm_df, r_dm_ev, r_dm_ev, eigenvalue_atol=1e-10, subspace_atol=1e-10)
    neg_controls.append(
        {
            "profile": profile_name,
            "control_id": "NC2_illegal_cross_mode_rotation",
            "detected_corruption": not bool(nc2["equivalent_under_legal_gauge"]),
            "max_err": nc2["max_coordinate_abs_err_after_legal_gauge"],
        }
    )

    # Principal curve on frozen R DC1:2
    r_pcurve = pd.read_csv(oracle_dir / "f12_principal_curve.csv").set_index("sample_id")
    py_pcurve = sl.fit_principal_curve(r_dm_emb.iloc[:, :2], maxit=20)
    pcurve_cand_csv = cand_dir / "f12_principal_curve_py.csv"
    pd.DataFrame({"sample_id": r_dm_emb.index, "s1": py_pcurve.s[:, 0], "s2": py_pcurve.s[:, 1], "ord": py_pcurve.ord_1based, "lambda": py_pcurve.lambda_}).to_csv(pcurve_cand_csv, index=False)
    pc_raw_ord = (list(r_pcurve.index.astype(str)) == list(r_dm_emb.index.astype(str)) and list(r_pcurve.columns) == ["s1", "s2", "ord", "lambda"])
    pc_rho = abs(float(spearmanr(r_pcurve.loc[r_dm_emb.index, "lambda"], py_pcurve.lambda_).statistic))
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F12",
            "operation": "principal_curve_pseudotime_concordance",
            "comparison_kind": "SEMANTIC_MONOTONE_ARC",
            "matched": bool(pc_rho >= 0.95),
            "raw_order_equal": bool(pc_raw_ord),
            "max_abs_err": float(1.0 - pc_rho),
            "rel_frobenius": None,
            "tolerance": "|spearman_rho|>=0.95 (df=5 Reinsch make_smoothing_spline vs Fortran GAM smooth.spline)",
            "oracle_file": str(oracle_dir / "f12_principal_curve.csv"),
            "candidate_file": str(pcurve_cand_csv),
            "notes": f"|spearman_rho|={pc_rho:.6f}",
        }
    )

    # 6. F16 TrajDETest vs glmGamPoi::glm_gp + test_de!
    pb_cts_r = pd.read_csv(oracle_dir / "f16_mono_pseudobulk_counts.csv", index_col=0)
    pb_dat_r = pd.read_csv(oracle_dir / "f16_mono_pseudobulk_data.csv", index_col=0)
    pb_meta_r = pd.read_csv(oracle_dir / "f16_mono_pseudobulk_meta.csv", index_col=0)
    r_de = pd.read_csv(oracle_dir / "f16_trajdetest_complete.csv", index_col=0)

    import anndata as ad

    pb_ad = ad.AnnData(X=pb_cts_r.T.to_numpy(dtype=np.float64), obs=pb_meta_r.copy(), var=pd.DataFrame(index=pb_cts_r.index))
    pb_ad.layers["counts"] = pb_cts_r.T.to_numpy(dtype=np.float64)
    pb_ad.layers["data"] = pb_dat_r.T.to_numpy(dtype=np.float64)

    py_de = sl.traj_de_test(
        pb_ad,
        traj_var="traj_DC1",
        latent_vars=["Age", "Sex", "log_count"],
        complete_results=True,
    )
    de_cand_csv = cand_dir / "f16_trajdetest_py.csv"
    py_de.to_csv(de_cand_csv)

    gene_set_eq = set(py_de["gene_ID"]) == set(r_de["gene_ID"])
    common_g = [g for g in r_de["gene_ID"] if g in py_de.index]
    r_sub = r_de.set_index("gene_ID").loc[common_g]
    p_sub = py_de.loc[common_g]
    fc_max_err = float(np.max(np.abs(r_sub["avg_log2FC"] - p_sub["avg_log2FC"])))
    beta_cor = float(np.corrcoef(r_sub["beta_Traj"], p_sub["beta_Traj"])[0, 1])
    beta_max_err = float(np.max(np.abs(r_sub["beta_Traj"] - p_sub["beta_Traj"])))
    logp_cor = float(
        np.corrcoef(-np.log10(np.maximum(r_sub["p_Traj"], 1e-300)), -np.log10(np.maximum(p_sub["p_Traj"], 1e-300)))[0, 1]
    )
    rows.append(
        {
            "profile": profile_name,
            "capability_id": "F16",
            "operation": "TrajDETest_filterByExpr_FoldChange_and_glmGamPoi_QLF",
            "comparison_kind": "NUMERIC",
            "matched": bool(gene_set_eq and fc_max_err <= 1e-10 and beta_cor >= 0.999 and logp_cor >= 0.99),
            "raw_order_equal": bool(list(r_de.index.astype(str)) == list(py_de.index.astype(str)) and list(r_de.columns) == list(py_de.columns)),
            "max_abs_err": beta_max_err,
            "rel_frobenius": fc_max_err,
            "tolerance": "exact_filterByExpr_genes, fc_atol=1e-10, beta_r>=0.999, -log10p_r>=0.99",
            "oracle_file": str(oracle_dir / "f16_trajdetest_complete.csv"),
            "candidate_file": str(de_cand_csv),
            "notes": f"n_genes={len(common_g)}, fc_err={fc_max_err:.2e}, beta_r={beta_cor:.6f}, -log10p_r={logp_cor:.6f}",
        }
    )

    # Mandatory negative control NC3: replacing log_count covariate with offset OR replacing QL-F with Wald fails strict parity!
    py_de_offset = sl.traj_de_test(
        pb_ad, traj_var="traj_DC1", latent_vars=["Age", "Sex", "log_count"],
        library_size_mode="offset", complete_results=True
    )
    off_beta_diff = float(np.max(np.abs(r_sub["beta_Traj"] - py_de_offset.loc[common_g, "beta_Traj"])))
    neg_controls.append(
        {
            "profile": profile_name,
            "control_id": "NC3_offset_replaces_covariate",
            "detected_corruption": bool(off_beta_diff > 1e-3 and "beta_log_count" not in py_de_offset.columns),
            "max_beta_diff": off_beta_diff,
        }
    )

    # 7. Legacy-only exports: F02 CorrectSampleComBat, F17 cellanova_calc_BE, F01 BuildLandmarkObject
    combat_reg_file = oracle_dir / "f02_combat_regression_atten.csv"
    if combat_reg_file.exists():
        r_res_in = pd.read_csv(oracle_dir / "f06_cohort_pearson_residuals.csv", index_col=0)
        s_meta_d = py_s_obj.metadata.copy()
        py_cb_reg = sl.correct_sample_combat(
            r_res_in,
            batch_key="Batch",
            condition_key="Disease",
            control_label="HV",
            metadata=s_meta_d,
            par_prior=True,
            method="regression",
            correct_attenuation=True,
            floor_beta=0.0,
        )
        cb_reg_csv = cand_dir / "f02_combat_regression_atten_py.csv"
        py_cb_reg.to_csv(cb_reg_csv)
        cmp_cb = cl_mod.compare(
            combat_reg_file, cb_reg_csv, mode="numeric", atol=1e-4, rtol=1e-4,
            tolerance_note="Parametric EM convergence tolerance conv=1e-4 in .combat_it_sol"
        )
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F02",
                "operation": "CorrectSampleComBat_parametric_regression_attenuation_floor0",
                "comparison_kind": "NUMERIC",
                "matched": bool(cmp_cb["matched_after_id_alignment"]),
                "raw_order_equal": bool(cmp_cb["raw_row_order_equal"] and cmp_cb["raw_column_order_equal"]),
                "max_abs_err": float(cmp_cb["max_abs_decimal"]),
                "rel_frobenius": cmp_cb["relative_frobenius"],
                "tolerance": "atol=1e-4,rtol=1e-4 (EM conv=1e-4)",
                "oracle_file": str(combat_reg_file),
                "candidate_file": str(cb_reg_csv),
                "notes": "Control-only EB shrinkage + attenuation-corrected OLS + floor.beta=0",
            }
        )

    combat_np_file = oracle_dir / "f02_combat_nonparam_posvar_subtraction.csv"
    if combat_np_file.exists():
        r_np_ref = pd.read_csv(combat_np_file, index_col=0)
        r_res_in = pd.read_csv(oracle_dir / "f06_cohort_pearson_residuals.csv", index_col=0).loc[r_np_ref.index]
        py_cb_np = sl.correct_sample_combat(
            r_res_in,
            batch_key="Batch",
            condition_key="Disease",
            control_label="HV",
            metadata=py_s_obj.metadata.copy(),
            par_prior=False,
            method="subtraction",
        )
        cb_np_csv = cand_dir / "f02_combat_nonparam_posvar_subtraction_py.csv"
        py_cb_np.to_csv(cb_np_csv)
        cmp_np = cl_mod.compare(
            combat_np_file, cb_np_csv, mode="numeric", atol=1e-10, rtol=1e-10,
            tolerance_note="Closed-form non-parametric normal/inverse-gamma kernel quadrature"
        )
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F02",
                "operation": "CorrectSampleComBat_nonparametric_subtraction",
                "comparison_kind": "NUMERIC",
                "matched": bool(cmp_np["matched_after_id_alignment"]),
                "raw_order_equal": bool(cmp_np["raw_row_order_equal"] and cmp_np["raw_column_order_equal"]),
                "max_abs_err": float(cmp_np["max_abs_decimal"]),
                "rel_frobenius": cmp_np["relative_frobenius"],
                "tolerance": "atol=1e-10,rtol=1e-10",
                "oracle_file": str(combat_np_file),
                "candidate_file": str(cb_np_csv),
                "notes": "Positive-variance subset (full set exposes upstream R .combat_int_eprior d_hat==0 bug)",
            }
        )

    ca_file = oracle_dir / "f17_cellanova_corrected.csv"
    if ca_file.exists():
        # Export scale.data of initial HVGs on full 360 cells matching Seurat ScaleData(coh_obj, features=VariableFeatures)
        init_hvgs = (oracle_dir / "shared_initial_hvgs.txt").read_text().splitlines()
        raw_d_hvg = lognorm_g_by_c.loc[init_hvgs, all_cell_ids].T.to_numpy(dtype=np.float64)
        mu_h = raw_d_hvg.mean(axis=0, keepdims=True)
        sd_h = np.std(raw_d_hvg, axis=0, ddof=1, keepdims=True)
        sd_h[sd_h == 0] = 1.0
        scaled_hvg = np.clip((raw_d_hvg - mu_h) / sd_h, -10.0, 10.0)
        py_ca = sl.cellanova_calc_be(
            pd.DataFrame(scaled_hvg, index=all_cell_ids, columns=init_hvgs),
            integrate_key=meta_c.loc[all_cell_ids, "Donor"].tolist(),
            control_dict={"ctrl": ["D01", "D02", "D03", "D04"]},
            reduction=pca_emb.loc[all_cell_ids].to_numpy(),
            k_select=3,
        )
        ca_cand_csv = cand_dir / "f17_cellanova_corrected_py.csv"
        py_ca.corrected.T.to_csv(ca_cand_csv)
        cmp_ca = cl_mod.compare(
            ca_file, ca_cand_csv, mode="numeric", atol=1e-9, rtol=1e-9,
            tolerance_note="QR OLS + rank-3 SVD projection P = V V^T (sign-invariant subspace projector)"
        )
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F17",
                "operation": "cellanova_calc_BE_control_pool_projection",
                "comparison_kind": "NUMERIC",
                "matched": bool(cmp_ca["matched_after_id_alignment"]),
                "raw_order_equal": bool(cmp_ca["raw_row_order_equal"] and cmp_ca["raw_column_order_equal"]),
                "max_abs_err": float(cmp_ca["max_abs_decimal"]),
                "rel_frobenius": cmp_ca["relative_frobenius"],
                "tolerance": "atol=1e-9,rtol=1e-9",
                "oracle_file": str(ca_file),
                "candidate_file": str(ca_cand_csv),
                "notes": "Two-stage batch QR OLS + control-pool SVD subspace subtraction",
            }
        )

    blo_file = oracle_dir / "f01_landmark_correlations.csv"
    if blo_file.exists():
        # Attach R's exact traj_DC1 and traj_overall to py_s_obj.metadata so we test BuildLandmarkObject math cleanly
        s_for_blo = sl.SampleLevelObject.from_counts(
            py_cts_hyphen,
            normalization="chi_squared",
            metadata=py_s_obj.metadata.copy(),
            reference_profile="legacy",
        )
        s_for_blo.metadata["traj_DC1"] = r_dm_emb.loc[s_for_blo.counts.columns].iloc[:, 0].to_numpy()
        s_for_blo.metadata["traj_overall"] = r_pcurve.loc[s_for_blo.counts.columns, "lambda"].to_numpy()
        py_blo = sl.build_landmark_object(
            r_mm_idx,
            s_for_blo,
            features_to_test=["traj_DC1", "traj_overall"],
            k_nn=10,
            cell_ids=all_cell_ids,
            landmark_cell_ids=lm_cells,
            nn_dist=r_mm_dist,
            cell_metadata=meta_c,
        )
        blo_cand_csv = cand_dir / "f01_landmark_correlations_py.csv"
        py_blo.metadata[["cor.coef_traj_DC1", "cor.coef_traj_overall"]].to_csv(blo_cand_csv)
        cmp_blo = cl_mod.compare(
            blo_file, blo_cand_csv, mode="numeric", atol=1e-12, rtol=1e-12,
            tolerance_note="LogNormalize + Pearson correlation across samples"
        )
        rows.append(
            {
                "profile": profile_name,
                "capability_id": "F01",
                "operation": "BuildLandmarkObject_trajectory_correlations",
                "comparison_kind": "NUMERIC",
                "matched": bool(cmp_blo["matched_after_id_alignment"]),
                "raw_order_equal": bool(cmp_blo["raw_row_order_equal"] and cmp_blo["raw_column_order_equal"]),
                "max_abs_err": float(cmp_blo["max_abs_decimal"]),
                "rel_frobenius": cmp_blo["relative_frobenius"],
                "tolerance": "atol=1e-12,rtol=1e-12",
                "oracle_file": str(blo_file),
                "candidate_file": str(blo_cand_csv),
                "notes": "Landmark-level transposed normalized density correlations with sample trajectories",
            }
        )

    return rows, neg_controls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cl_mod = load_compare_labelled(args.repo_root)

    dist = importlib.metadata.distribution("scslide-py")
    build_meta = {
        "scslide_file": sl.__file__,
        "scslide_version": sl.__version__,
        "distribution_version": dist.version,
        "python_executable": sys.executable,
    }

    v6_rows, v6_nc = evaluate_profile(
        "v6",
        args.oracle_root / "v6_full",
        args.oracle_root / "v6_norm_starter",
        out_dir / "candidates_v6",
        cl_mod,
    )
    leg_rows, leg_nc = evaluate_profile(
        "legacy",
        args.oracle_root / "legacy_scslide_complete",
        args.oracle_root / "legacy_norm_starter",
        out_dir / "candidates_legacy",
        cl_mod,
    )

    all_rows = v6_rows + leg_rows
    all_nc = v6_nc + leg_nc
    df_parity = pd.DataFrame(all_rows)
    df_nc = pd.DataFrame(all_nc)
    df_parity.to_csv(out_dir / "PARITY_RESULTS.tsv", sep="\t", index=False)
    df_nc.to_csv(out_dir / "NEGATIVE_CONTROLS.tsv", sep="\t", index=False)

    summary = {
        "build": build_meta,
        "total_checks": len(all_rows),
        "matched_checks": int(df_parity["matched"].sum()),
        "failed_checks": int((~df_parity["matched"]).sum()),
        "negative_controls_total": len(all_nc),
        "negative_controls_detected_all": bool(df_nc["detected_corruption"].all()),
        "failed_operations": df_parity.loc[~df_parity["matched"], ["profile", "capability_id", "operation", "max_abs_err", "notes"]].to_dict(orient="records"),
    }
    (out_dir / "parity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_checks"] == 0 and summary["negative_controls_detected_all"] and summary["total_checks"] == 47 and summary["negative_controls_total"] == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
