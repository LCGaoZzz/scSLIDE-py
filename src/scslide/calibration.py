"""Known-answer synthetic calibration used by the bundled campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .plot import plot_calibration
from .sample import expected_counts, landmark_matrix, pearson_residuals


def build_synthetic_cohort_fixture(
    *,
    seed: int = 20261116,
    n_subjects: int = 12,
    n_repeat_subjects: int = 2,
    cells_per_sample: int = 30,
    n_genes: int = 60,
    as_anndata: bool = True,
) -> object:
    """Build a small, license-clean synthetic cohort fixture for portable onboarding & R-migration checks.

    This fixture is strictly synthetic and carries explicit ``synthetic_demo_only=True`` metadata so
    demonstration outputs are never confused with biological findings. When ``as_anndata=False``,
    returns a core-only dictionary of NumPy/SciPy/pandas objects without requiring ``anndata``.
    """
    import scipy.sparse as sp

    if int(n_subjects) < 6:
        raise ValueError("n_subjects must be >= 6")
    if int(n_repeat_subjects) < 0 or int(n_repeat_subjects) > int(n_subjects):
        raise ValueError("n_repeat_subjects must lie in 0..n_subjects")
    if int(cells_per_sample) < 6 or int(n_genes) < 12:
        raise ValueError("cells_per_sample >= 6 and n_genes >= 12 are required")

    rng = np.random.default_rng(int(seed))
    draws: list[str] = []
    subjects: list[str] = []
    diseases: list[str] = []
    ages: list[float] = []
    sexes: list[str] = []
    visit_days: list[float] = []
    batches: list[str] = []

    disease_cycle = ["HV", "MILD", "SEV", "CRIT"]
    for i in range(int(n_subjects)):
        sid = f"S{i + 1:02d}"
        dis = "HV" if i < max(2, int(n_subjects) // 3) else disease_cycle[1 + (i % 3)]
        age = float(40.0 + (34.0 * i) / max(int(n_subjects) - 1, 1))
        sex = "F" if i % 2 == 0 else "M"
        batch = "B1" if i % 2 == 0 else "B2"
        n_vis = 2 if i < int(n_repeat_subjects) else 1
        for v in range(1, n_vis + 1):
            draws.append(f"{sid}-V{v}")
            subjects.append(sid)
            diseases.append(dis)
            ages.append(age)
            sexes.append(sex)
            visit_days.append(0.0 if dis == "HV" else float(3 * v + (i % 5)))
            batches.append(batch)

    n_samples = len(draws)
    n_cells = n_samples * int(cells_per_sample)
    cell_ids = [f"cell_{i:04d}" for i in range(n_cells)]
    cell_draw = np.repeat(draws, int(cells_per_sample))
    cell_subj = np.repeat(subjects, int(cells_per_sample))
    cell_dis = np.repeat(diseases, int(cells_per_sample))
    cell_age = np.repeat(ages, int(cells_per_sample))
    cell_sex = np.repeat(sexes, int(cells_per_sample))
    cell_day = np.repeat(visit_days, int(cells_per_sample))
    cell_batch = np.repeat(batches, int(cells_per_sample))
    cell_types = np.resize(["CD14_Mono", "CD4_T", "NK"], n_cells)

    sev_num = np.where(cell_dis == "HV", 0.0, np.where(cell_dis == "MILD", 1.0, np.where(cell_dis == "SEV", 2.0, 3.0)))
    rates = np.full((n_cells, int(n_genes)), 1.1, dtype=np.float64)
    mono_m = cell_types == "CD14_Mono"
    rates[mono_m, :8] += 2.0 + 1.5 * sev_num[mono_m, None]
    rates[cell_types == "CD4_T", 8:14] += 2.6
    rates[cell_types == "NK", 14:20] += 2.6
    counts_csr = sp.csr_matrix(rng.poisson(rates).astype(np.float64))

    obs = pd.DataFrame(
        {
            "Donor": cell_draw,
            "subject_id": cell_subj,
            "Disease": pd.Categorical(cell_dis, categories=["HV", "MILD", "SEV", "CRIT"]),
            "Age": cell_age,
            "Sex": cell_sex,
            "TimeSinceOnset": cell_day,
            "Batch": cell_batch,
            "cell_type": cell_types,
        },
        index=cell_ids,
    )
    gene_names = [f"GENE_{g:03d}" for g in range(int(n_genes))]
    rs = np.asarray(counts_csr.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 10_000.0 / np.maximum(rs, 1.0), 0.0)
    data_csr = (sp.diags(inv) @ counts_csr).tocsr()
    data_csr.data = np.log1p(data_csr.data)
    u_pca, sv_pca, _ = np.linalg.svd(data_csr.toarray(), full_matrices=False)
    n_pcs = min(8, u_pca.shape[1])
    x_pca = u_pca[:, :n_pcs] * sv_pca[:n_pcs]

    demo_meta = {
        "synthetic_demo_only": True,
        "not_biological_discovery": True,
        "license": "MIT",
        "seed": int(seed),
        "n_subjects": int(n_subjects),
        "n_samples": int(n_samples),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    }
    if not as_anndata:
        return {
            "counts": counts_csr,
            "data": data_csr,
            "obs": obs,
            "var_names": gene_names,
            "X_pca": x_pca,
            "scslide_demo": demo_meta,
        }

    from .anndata import _require_anndata
    from .pipeline import normalize_and_hvg

    ad = _require_anndata()
    adata = ad.AnnData(
        X=counts_csr.copy(),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["counts"] = counts_csr.copy()
    normalize_and_hvg(adata, n_top_genes=min(40, int(n_genes)))
    adata.obsm["X_pca"] = x_pca
    adata.uns["scslide_demo"] = demo_meta
    return adata


def build_fixture(seed: int = 20261110) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_genes, n_cells, n_landmarks, n_samples = 40, 60, 8, 6
    cells_per_sample, stored_k, k_nn = n_cells // n_samples, 10, 5
    landmark_ids = np.array([f"LM{i+1:02d}" for i in range(n_landmarks)])
    # The first query rows are landmark cells themselves.  Naming those rows
    # with the landmark IDs makes the self-exclusion rule explicit and testable.
    cell_ids = np.array(list(landmark_ids) + [f"cell{i+1:03d}" for i in range(n_cells - n_landmarks)])
    sample_ids = np.array([f"D{i+1}" for i in range(n_samples)])
    sample_for_cell = np.repeat(sample_ids, cells_per_sample)
    expression = rng.poisson(4.0, size=(n_genes, n_cells)).astype(int)
    expression[:4, : n_cells // 2] += rng.poisson(3.0, size=(4, n_cells // 2))
    nn_idx = np.zeros((n_cells, stored_k), dtype=int)
    for i in range(n_cells):
        if i < n_landmarks:
            nn_idx[i] = rng.integers(1, n_landmarks + 1, size=stored_k)
        else:
            base = (i - n_landmarks) % n_landmarks + 1
            nn_idx[i] = np.concatenate([[base], rng.integers(1, n_landmarks + 1, size=stored_k - 1)])
            rng.shuffle(nn_idx[i])
    counts = landmark_matrix(nn_idx, sample_for_cell, landmark_ids, k=k_nn, cell_ids=cell_ids)
    # Independent known-answer construction mirrors the R aggregation contract;
    # this is the oracle used to measure implementation error.
    expected_known = np.zeros_like(counts.to_numpy())
    for q in range(n_landmarks, n_cells):
        j = int(np.flatnonzero(sample_ids == sample_for_cell[q])[0])
        for landmark in nn_idx[q, :k_nn] - 1:
            expected_known[landmark, j] += 1
    expected = pd.DataFrame(expected_known, index=counts.index, columns=counts.columns)
    independence_expected = expected_counts(counts)
    residuals = pearson_residuals(counts)
    centered = residuals.to_numpy().T - residuals.to_numpy().T.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    embedding = pd.DataFrame(u[:, :2] * s[:2], index=sample_ids, columns=["component_1", "component_2"])
    self_landmark_assignments = int(np.asarray(nn_idx[:n_landmarks, :k_nn]).size)
    return {
        "seed": seed, "expression": expression, "neighbor_index": nn_idx,
        "cell_ids": cell_ids, "sample_for_cell": sample_for_cell,
        "counts": counts, "expected": expected, "independence_expected": independence_expected,
        "residuals": residuals, "embedding": embedding,
        "metrics": {
            "landmark_self_exclusion": self_landmark_assignments,
            "lmc_total_count": int(counts.to_numpy().sum()),
            "expected_lmc_total_count": int(k_nn * (n_cells - n_landmarks)),
            "lmc_known_answer_rmse": float(np.sqrt(np.mean((counts.to_numpy() - expected.to_numpy()) ** 2))),
            "chi_squared_finite": int(np.isfinite(residuals.to_numpy()).all()),
            "chi_squared_total_residual_abs": float(abs(residuals.to_numpy().sum())),
            "sample_embedding_components": 2,
        },
    }


def run(output_dir: str | Path, *, seed: int = 20261110, source_input: str | Path | None = None, campaign_entry: str | Path | None = None) -> dict[str, object]:
    root = Path(output_dir)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    result = build_fixture(seed)
    result["counts"].to_csv(root / "tables/lmc_counts.csv")
    result["expected"].to_csv(root / "tables/lmc_expected_counts.csv")
    result["residuals"].to_csv(root / "tables/pearson_residuals.csv")
    result["embedding"].to_csv(root / "tables/sample_level_svd_diagnostic.csv")
    pd.DataFrame(result["neighbor_index"], index=result["cell_ids"], columns=[f"neighbor_{i+1}" for i in range(10)]).to_csv(root / "tables/neighbor_index_1based.csv", index_label="cell")
    pd.DataFrame({"cell": result["cell_ids"], "sample": result["sample_for_cell"]}).to_csv(root / "tables/cell_sample_map.csv", index=False)
    metrics = pd.DataFrame([{"metric": k, "value": v} for k, v in result["metrics"].items()])
    metrics.to_csv(root / "calibration_metrics.tsv", sep="\t", index=False)
    plot_calibration(result["counts"], result["residuals"], result["embedding"], root / "figures/scslide_calibration")
    from . import __version__

    provenance = {
        "package": "scslide-py", "version": __version__, "seed": seed,
        "input_contract": "synthetic known-answer fixture; expression array is an identity witness",
        "source_input": str(source_input) if source_input else None,
        "source_input_sha256": hashlib.sha256(Path(source_input).read_bytes()).hexdigest() if source_input and Path(source_input).is_file() else None,
        "campaign_entry": str(campaign_entry) if campaign_entry else None,
        "normalization": "(observed - expected) / sqrt(expected)",
        "embedding": "centered SVD diagnostic; not destiny::DiffusionMap",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "source_adapted_calibration_completed", "seed": seed, "metrics": result["metrics"], "figure": "figures/scslide_calibration.png"}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--campaign-entry", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, seed=args.seed, source_input=args.source_input, campaign_entry=args.campaign_entry), ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
