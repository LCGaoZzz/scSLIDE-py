"""Publication-grade scSLIDE plotting utilities with automatic source-data CSV sidecars (F07, F14).

Implements native Python counterparts of:
- ``PlotLandmarkObject`` (:func:`plot_landmark_object` — F07)
- ``SampleLevelDimPlot`` (:func:`sample_level_dim_plot` — F14)
- ``PlotAnnotatedHeatmap`` (:func:`plot_annotated_heatmap` — Seurat v6 / Scillus trajectory heatmap)
- Case & diagnostic views (:func:`plot_sample_diffusion`, :func:`plot_clinical_scatter`,
  :func:`plot_severity_ridges`, :func:`plot_trajectory_bins`, and v0.1.1 :func:`plot_calibration`)

Design guarantees:
- Never mutates global ``matplotlib.rcParams`` (all styling runs inside ``plt.rc_context``).
- Exports vector-editable text (``pdf.fonttype=42``, ``ps.fonttype=42``, ``svg.fonttype='none'``).
- Uses colorblind-safe Okabe-Ito categorical palettes and symmetric zero-centered diverging colormaps.
- Automatically writes ``<stem>.csv`` source-data tables alongside saved figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kruskal, pearsonr, spearmanr

from .landmark import LandmarkObject, build_landmark_object
from .sample import SampleLevelObject


OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

DIVERGING_BWR = mcolors.LinearSegmentedColormap.from_list(
    "scslide_bwr",
    ["#053061", "#4393c3", "#f7f7f7", "#d6604d", "#67001f"],
)


def _resolve_local_sans_font() -> str:
    installed = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Liberation Sans", "Arimo", "Nimbus Sans", "Noto Sans", "DejaVu Sans"):
        if candidate in installed:
            return candidate
    return "DejaVu Sans"


def _publication_rc() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [_resolve_local_sans_font(), "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _category_palette(categories: Sequence[str], palette: Mapping[str, str] | Sequence[str] | None = None) -> dict[str, str]:
    cats = [str(c) for c in categories]
    if isinstance(palette, Mapping):
        return {c: str(palette.get(c, OKABE_ITO[i % len(OKABE_ITO)])) for i, c in enumerate(cats)}
    seq = list(palette) if palette is not None else OKABE_ITO
    return {c: str(seq[i % len(seq)]) for i, c in enumerate(cats)}


def _save_figure_bundle(
    fig: plt.Figure,
    output: str | Path | None,
    source_df: pd.DataFrame | None = None,
    *,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
    close: bool = False,
) -> list[Path]:
    written: list[Path] = []
    if output is not None:
        stem = Path(output)
        if stem.suffix.lower() in (".png", ".pdf", ".svg"):
            stem = stem.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in formats:
            p = stem.with_suffix(f".{ext.lstrip('.')}")
            kw: dict[str, Any] = {"dpi": dpi} if ext.lstrip(".").lower() == "png" else {}
            fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
            written.append(p)
        if source_df is not None:
            csv_p = stem.with_suffix(".csv")
            source_df.to_csv(csv_p, index=False)
            written.append(csv_p)
    if close:
        plt.close(fig)
    return written


def plot_calibration(
    counts: pd.DataFrame,
    residuals: pd.DataFrame,
    embedding: pd.DataFrame,
    output: str | Path,
    *,
    title: str = "scSLIDE source-adapted calibration",
) -> None:
    """Render a clean three-panel calibration figure and save PNG/PDF/SVG + CSV sidecar (preserves v0.1.1)."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_publication_rc()):
        fig = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], width_ratios=[1.15, 1])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
        for ax, matrix, label, cmap, cbar_label in [
            (axes[0], counts, "A  Landmark-by-sample counts", "viridis", "count"),
            (axes[1], residuals, "B  Pearson residuals", DIVERGING_BWR, "residual"),
        ]:
            data = matrix.to_numpy(float)
            lim = max(float(np.max(np.abs(data))), 1.0)
            im = ax.imshow(
                data,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-lim if label.startswith("B") else None,
                vmax=lim if label.startswith("B") else None,
            )
            ax.set_title(label, loc="left", fontweight="bold")
            ax.set_xlabel("sample / donor")
            ax.set_ylabel("landmark")
            ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=0)
            ax.set_yticks(range(matrix.shape[0]), matrix.index)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(cbar_label, rotation=270, labelpad=10)
            ax.tick_params(length=2)

        ax = axes[2]
        xy = embedding.to_numpy(float)
        colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(embedding)))
        ax.scatter(xy[:, 0], xy[:, 1], s=52, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
        for (x_v, y_v), label in zip(xy, embedding.index):
            ax.annotate(str(label), (x_v, y_v), xytext=(5, 2), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#c8c8c8", lw=0.6)
        ax.axvline(0, color="#c8c8c8", lw=0.6)
        ax.set_title("C  Sample-level centered SVD diagnostic", loc="left", fontweight="bold")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.text(0.01, 0.03, "Diagnostic only; not destiny::DiffusionMap", transform=ax.transAxes, fontsize=7.2, color="#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")

        src = embedding.copy()
        src.insert(0, "sample_id", embedding.index.astype(str))
        _save_figure_bundle(fig, output_path, src, formats=("png", "pdf", "svg"), dpi=600, close=True)


def plot_landmark_object(
    landmark_obj: LandmarkObject,
    *,
    order: bool = True,
    pt_size: float = 8.0,
    alpha: float = 0.85,
    plot_featureplot: bool = False,
    ncol: int | None = None,
    seed: int = 42,
    output: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Native implementation of ``PlotLandmarkObject`` (F07, ``R/visualization.R``)."""
    cor_cols = [c for c in landmark_obj.metadata.columns if str(c).startswith("cor.coef_")]
    if len(cor_cols) == 0:
        raise ValueError("No 'cor.coef_*' columns found in landmark_obj.metadata")

    umap_df = landmark_obj.umap.loc[landmark_obj.density_by_sample.index]
    meta_cor = landmark_obj.metadata.loc[umap_df.index, cor_cols]
    n_panels = len(cor_cols)
    n_cols = min(n_panels, int(ncol)) if ncol is not None else n_panels
    n_rows = int(np.ceil(n_panels / max(n_cols, 1)))

    rng = np.random.default_rng(seed)
    source_rows: list[pd.DataFrame] = []

    with plt.rc_context(_publication_rc()):
        fig1, axes1 = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols + 0.6, 3.2 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        vmax = max(float(np.nanmax(np.abs(meta_cor.to_numpy(dtype=np.float64)))), 0.1)

        last_sc = None
        for idx, col in enumerate(cor_cols):
            r_i, c_i = divmod(idx, n_cols)
            ax = axes1[r_i, c_i]
            vals = meta_cor[col].to_numpy(dtype=np.float64)
            if order:
                ord_p = np.argsort(np.abs(vals), kind="mergesort")
            else:
                ord_p = rng.permutation(len(vals))
            x_p = umap_df["UMAP_1"].to_numpy(dtype=np.float64)[ord_p]
            y_p = umap_df["UMAP_2"].to_numpy(dtype=np.float64)[ord_p]
            v_p = vals[ord_p]
            last_sc = ax.scatter(
                x_p, y_p, c=v_p, cmap=DIVERGING_BWR, vmin=-vmax, vmax=vmax, s=pt_size, alpha=alpha, linewidths=0
            )
            ax.set_title(col.replace("cor.coef_", "r with "), fontweight="bold")
            ax.set_xlabel("wnnUMAP 1")
            ax.set_ylabel("wnnUMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for sp_name in ("top", "right"):
                ax.spines[sp_name].set_visible(False)

            source_rows.append(
                pd.DataFrame(
                    {
                        "landmark_id": umap_df.index.to_numpy()[ord_p],
                        "UMAP_1": x_p,
                        "UMAP_2": y_p,
                        "feature": col,
                        "correlation": v_p,
                    }
                )
            )

        for extra in range(n_panels, n_rows * n_cols):
            r_i, c_i = divmod(extra, n_cols)
            axes1[r_i, c_i].set_visible(False)

        if last_sc is not None:
            cb = fig1.colorbar(last_sc, ax=axes1.ravel().tolist(), shrink=0.82, pad=0.02)
            cb.set_label("Pearson r")

        src_df = pd.concat(source_rows, ignore_index=True)
        written = _save_figure_bundle(fig1, output, src_df, close=not show and output is not None)

        fig2 = None
        if plot_featureplot:
            samples = list(landmark_obj.density_by_sample.columns)
            nc2 = max(1, int(np.ceil(np.sqrt(len(samples)))))
            nr2 = int(np.ceil(len(samples) / nc2))
            fig2, axes2 = plt.subplots(nr2, nc2, figsize=(2.6 * nc2, 2.5 * nr2), squeeze=False, constrained_layout=True)
            dens = landmark_obj.density_by_sample.loc[umap_df.index]
            d_max = max(float(np.max(dens.to_numpy())), 1e-6)
            for s_idx, s_name in enumerate(samples):
                r_i, c_i = divmod(s_idx, nc2)
                ax2 = axes2[r_i, c_i]
                sv = dens[s_name].to_numpy(dtype=np.float64)
                ord_s = np.argsort(sv, kind="mergesort")
                ax2.scatter(
                    umap_df["UMAP_1"].to_numpy()[ord_s],
                    umap_df["UMAP_2"].to_numpy()[ord_s],
                    c=sv[ord_s],
                    cmap="viridis",
                    vmin=0.0,
                    vmax=d_max,
                    s=max(3.0, pt_size * 0.7),
                    alpha=alpha,
                    linewidths=0,
                )
                ax2.set_title(str(s_name), fontsize=8, fontweight="bold")
                ax2.set_xticks([])
                ax2.set_yticks([])
            for extra in range(len(samples), nr2 * nc2):
                r_i, c_i = divmod(extra, nc2)
                axes2[r_i, c_i].set_visible(False)
            if output is not None:
                stem2 = Path(output).with_suffix("")
                _save_figure_bundle(fig2, f"{stem2}_donors", None, close=not show)

    return {"p1": fig1, "p2": fig2, "source_data": src_df, "written_files": written}


def sample_level_dim_plot(
    cell_adata_or_nn: Any,
    sample_obj: SampleLevelObject | Any,
    *,
    features_to_test: Sequence[str],
    nn_name: str = "weighted.nn",
    landmark_assay_name: str = "LANDMARK",
    rename_sample: Sequence[str] | None = None,
    order: bool = True,
    pt_size: float = 8.0,
    alpha: float = 0.85,
    ncol: int | None = None,
    return_landmark_obj: bool = True,
    k_nn: int = 20,
    min_dist: float = 0.3,
    seed: int = 42,
    umap_backend: str = "auto",
    allow_fallback: bool = False,
    output: str | Path | None = None,
    show: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Native implementation of ``SampleLevelDimPlot`` (F14, ``R/visualization.R``)."""
    lm_obj = build_landmark_object(
        cell_adata_or_nn,
        sample_obj,
        features_to_test=features_to_test,
        nn_name=nn_name,
        landmark_assay_name=landmark_assay_name,
        rename_sample=rename_sample,
        k_nn=k_nn,
        min_dist=min_dist,
        seed=seed,
        umap_backend=umap_backend,
        allow_fallback=allow_fallback,
        **kwargs,
    )
    plots = plot_landmark_object(
        lm_obj,
        order=order,
        pt_size=pt_size,
        alpha=alpha,
        ncol=ncol,
        seed=seed,
        output=output,
        show=show,
    )
    plots["umap_info"] = lm_obj.attrs.get("umap_info")
    if return_landmark_obj:
        return {"landmark_obj": lm_obj, **plots}
    return plots


def plot_sample_diffusion(
    sample_obj: SampleLevelObject | Any,
    *,
    dims: tuple[int, int] = (1, 2),
    group_by: str = "Disease",
    reduction_name: str = "DiffMap",
    curve_coords: np.ndarray | None = None,
    curve_ord: np.ndarray | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    output: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot sample-level Diffusion Map embedding with group colors and optional principal curve."""
    from .anndata import sample_object_from_anndata

    if not isinstance(sample_obj, SampleLevelObject):
        sample_obj = sample_object_from_anndata(sample_obj)
    if reduction_name not in sample_obj.embeddings:
        raise ValueError(f"Reduction {reduction_name!r} not found in sample_obj.embeddings")

    emb = sample_obj.embeddings[reduction_name]
    d1, d2 = int(dims[0]) - 1, int(dims[1]) - 1
    x_col, y_col = emb.columns[d1], emb.columns[d2]
    meta = sample_obj.metadata.loc[emb.index] if sample_obj.metadata is not None else pd.DataFrame(index=emb.index)
    groups = meta[group_by].astype(str) if group_by in meta.columns else pd.Series(["sample"] * len(emb), index=emb.index)
    if group_by in meta.columns and isinstance(meta[group_by].dtype, pd.CategoricalDtype):
        cat_order = [str(c) for c in meta[group_by].cat.categories if str(c) in set(groups)]
    else:
        cat_order = list(pd.unique(groups))
    cmap_dict = _category_palette(cat_order, palette=palette)

    with plt.rc_context(_publication_rc()):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.4, 3.8), constrained_layout=True)
        else:
            fig = ax.figure

        for cat in cat_order:
            m = (groups == cat).to_numpy()
            ax.scatter(
                emb.loc[m, x_col],
                emb.loc[m, y_col],
                c=cmap_dict[cat],
                label=f"{cat} (n={int(m.sum())})",
                s=46,
                edgecolor="white",
                linewidth=0.65,
                alpha=0.92,
                zorder=3,
            )

        if curve_coords is not None:
            if hasattr(curve_coords, "curve_dense") and curve_coords.curve_dense is not None:
                cc_plot = np.asarray(curve_coords.curve_dense, dtype=np.float64)
                ax.plot(cc_plot[:, 0], cc_plot[:, 1], color="#111111", lw=1.8, zorder=4, label="Principal curve")
            else:
                cc = np.asarray(curve_coords, dtype=np.float64)
                ord_c = np.arange(len(cc)) if curve_ord is None else np.asarray(curve_ord, dtype=np.int64)
                ax.plot(cc[ord_c, 0], cc[ord_c, 1], color="#111111", lw=1.8, zorder=4, label="Principal curve")

        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_title(title or f"Sample Diffusion Map ({x_col} vs {y_col})", loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

        src_df = pd.DataFrame(
            {
                "sample_id": emb.index.astype(str),
                x_col: emb[x_col].to_numpy(),
                y_col: emb[y_col].to_numpy(),
                group_by: groups.to_numpy(),
            }
        )
        _save_figure_bundle(fig, output, src_df)
    return fig, ax, src_df


def plot_clinical_scatter(
    metadata: pd.DataFrame,
    x: str,
    y: str,
    *,
    group_by: str | None = "Disease",
    group_order: Sequence[str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    output: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot sample trajectory vs clinical variable with Pearson r and Spearman rho annotations."""
    df = metadata.copy()
    xv = pd.to_numeric(df[x], errors="coerce")
    yv = pd.to_numeric(df[y], errors="coerce")
    valid = xv.notna() & yv.notna()
    df_v = df.loc[valid].copy()
    x_arr = xv.loc[valid].to_numpy(dtype=np.float64)
    y_arr = yv.loc[valid].to_numpy(dtype=np.float64)

    r_p, p_p = pearsonr(x_arr, y_arr) if len(x_arr) >= 3 else (np.nan, np.nan)
    r_s, p_s = spearmanr(x_arr, y_arr) if len(x_arr) >= 3 else (np.nan, np.nan)

    with plt.rc_context(_publication_rc()):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.4, 3.8), constrained_layout=True)
        else:
            fig = ax.figure

        if group_by and group_by in df_v.columns:
            present_cats = set(df_v[group_by].astype(str))
            if group_order is not None:
                cats = [str(c) for c in group_order if str(c) in present_cats]
            elif isinstance(df_v[group_by].dtype, pd.CategoricalDtype):
                cats = [str(c) for c in df_v[group_by].cat.categories if str(c) in present_cats]
            else:
                cats = list(pd.unique(df_v[group_by].astype(str)))
            cmap_dict = _category_palette(cats, palette=palette)
            for cat in cats:
                m = (df_v[group_by].astype(str) == cat).to_numpy()
                ax.scatter(
                    x_arr[m], y_arr[m], c=cmap_dict[cat], label=f"{cat} (n={int(m.sum())})",
                    s=46, edgecolor="white", linewidth=0.65, alpha=0.92, zorder=3
                )
            ax.legend(frameon=False, loc="best")
        else:
            ax.scatter(x_arr, y_arr, c=OKABE_ITO[0], s=46, edgecolor="white", linewidth=0.65, zorder=3)

        if len(x_arr) >= 3 and np.std(x_arr) > 0:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            xs = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
            ax.plot(xs, intercept + slope * xs, color="#333333", ls="--", lw=1.2, zorder=2)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(
            title or f"{x} vs {y} (r={r_p:.2f}, p={p_p:.2g}; rho={r_s:.2f}, p={p_s:.2g})",
            loc="left",
            fontweight="bold",
            fontsize=8.8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        src_df = pd.DataFrame({"sample_id": df_v.index.astype(str), x: x_arr, y: y_arr})
        if group_by and group_by in df_v.columns:
            src_df[group_by] = df_v[group_by].astype(str).to_numpy()
        _save_figure_bundle(fig, output, src_df)
    return fig, ax, src_df


def plot_severity_ridges(
    metadata: pd.DataFrame,
    value_col: str,
    group_col: str = "Disease",
    *,
    group_order: Sequence[str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    output: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Ridge density + individual donor rug/points matching COVID ``ggridges`` severity panel."""
    df = metadata.loc[metadata[value_col].notna() & metadata[group_col].notna()].copy()
    vals_all = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    grps_all = df[group_col].astype(str).to_numpy()

    cats = list(group_order) if group_order is not None else list(pd.unique(grps_all))
    cats = [c for c in cats if np.any(grps_all == c)]
    cmap_dict = _category_palette(cats, palette=palette)

    groups_data = [vals_all[grps_all == c] for c in cats if np.sum(grps_all == c) >= 1]
    kw_p = kruskal(*groups_data).pvalue if len(groups_data) >= 2 and all(len(g) >= 2 for g in groups_data) else np.nan

    with plt.rc_context(_publication_rc()):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
        else:
            fig = ax.figure

        span = max(float(vals_all.max() - vals_all.min()), 1e-3)
        x_grid = np.linspace(float(vals_all.min()) - 0.30 * span, float(vals_all.max()) + 0.30 * span, 250)

        for idx, cat in enumerate(cats):
            v_c = vals_all[grps_all == cat]
            y_base = float(idx)
            col = cmap_dict[cat]
            if len(v_c) >= 3 and np.std(v_c) > 1e-8:
                kde = gaussian_kde(v_c)
                dens = kde(x_grid)
                dens_scaled = 0.72 * dens / max(float(dens.max()), 1e-12)
                ax.fill_between(x_grid, y_base, y_base + dens_scaled, color=col, alpha=0.55, zorder=2 + idx)
                ax.plot(x_grid, y_base + dens_scaled, color=col, lw=1.2, zorder=3 + idx)
            ax.scatter(
                v_c,
                np.full_like(v_c, y_base + 0.08),
                color=col,
                edgecolor="white",
                linewidth=0.6,
                s=34,
                zorder=4 + idx,
            )

        ax.set_yticks(range(len(cats)), [f"{c} (n={int(np.sum(grps_all == c))})" for c in cats])
        ax.set_xlabel(value_col)
        ax.set_ylabel(group_col)
        kw_txt = f" (Kruskal-Wallis P={kw_p:.2g})" if np.isfinite(kw_p) else ""
        ax.set_title(title or f"{value_col} across {group_col}{kw_txt}", loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        src_df = pd.DataFrame({"sample_id": df.index.astype(str), group_col: grps_all, value_col: vals_all})
        _save_figure_bundle(fig, output, src_df)
    return fig, ax, src_df


def plot_trajectory_bins(
    metadata: pd.DataFrame,
    traj_col: str,
    score_col: str,
    *,
    group_col: str | None = "Disease",
    n_bins: int = 5,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    seed: int = 42,
    output: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot quantile-ranked trajectory bins vs clinical/neuropathology score (SEA-AD CPS boxplot + donor points)."""
    df = metadata.loc[metadata[traj_col].notna() & metadata[score_col].notna()].copy()
    tv = pd.to_numeric(df[traj_col], errors="coerce").to_numpy(dtype=np.float64)
    sv = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float64)
    ranks = pd.Series(tv).rank(method="average").to_numpy()
    labels = [f"traj-bin_{i + 1}" for i in range(int(n_bins))]
    bins = pd.cut(ranks, bins=int(n_bins), labels=labels)
    df["traj_bin"] = bins.astype(str)

    r_s, p_s = spearmanr(tv, sv) if len(tv) >= 3 else (np.nan, np.nan)
    rng = np.random.default_rng(seed)

    with plt.rc_context(_publication_rc()):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
        else:
            fig = ax.figure

        box_data = [sv[df["traj_bin"].to_numpy() == lb] for lb in labels]
        bp = ax.boxplot(
            box_data,
            positions=np.arange(len(labels)),
            widths=0.52,
            showfliers=False,
            patch_artist=True,
            zorder=2,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#f2f4f7")
            patch.set_edgecolor("#444444")
            patch.set_linewidth(0.9)
        for med in bp["medians"]:
            med.set_color("#111111")
            med.set_linewidth(1.4)

        if group_col and group_col in df.columns:
            cats = list(pd.unique(df[group_col].astype(str)))
            cmap_dict = _category_palette(cats, palette=palette)
            for _c_idx, cat in enumerate(cats):
                m_cat = (df[group_col].astype(str) == cat).to_numpy()
                x_pos = np.array([labels.index(b) for b in df.loc[m_cat, "traj_bin"]], dtype=np.float64)
                x_jit = x_pos + rng.uniform(-0.15, 0.15, size=len(x_pos))
                ax.scatter(
                    x_jit,
                    sv[m_cat],
                    c=cmap_dict[cat],
                    marker="o",
                    s=34,
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.9,
                    label=f"{cat} (n={int(m_cat.sum())})",
                    zorder=4,
                )
            ax.legend(frameon=False, loc="best")
        else:
            x_pos = np.array([labels.index(b) for b in df["traj_bin"]], dtype=np.float64)
            ax.scatter(x_pos + rng.uniform(-0.15, 0.15, size=len(x_pos)), sv, c=OKABE_ITO[0], s=34, zorder=4)

        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_xlabel("Pseudo-trajectory bins")
        ax.set_ylabel(score_col)
        ax.set_title(
            title or f"Pseudo-trajectory vs {score_col} (Spearman rho={r_s:.2f}, P={p_s:.2g})",
            loc="left",
            fontweight="bold",
            fontsize=8.8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        src_df = pd.DataFrame(
            {
                "sample_id": df.index.astype(str),
                traj_col: tv,
                "traj_bin": df["traj_bin"].to_numpy(),
                score_col: sv,
            }
        )
        if group_col and group_col in df.columns:
            src_df[group_col] = df[group_col].astype(str).to_numpy()
        _save_figure_bundle(fig, output, src_df)
    return fig, ax, src_df


def plot_annotated_heatmap(
    adata_or_matrix: Any,
    features: Sequence[str],
    *,
    sort_by: str | Sequence[str],
    anno_var: str | Sequence[str] | None = "Disease",
    metadata: pd.DataFrame | None = None,
    layer: str = "scale.data",
    hm_limit: tuple[float, float, float] = (-2.0, 0.0, 2.0),
    palette: Mapping[str, str] | Sequence[str] | None = None,
    show_column_names: bool = False,
    title: str | None = None,
    output: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, plt.Axes], pd.DataFrame]:
    """Native implementation of Seurat v6 ``PlotAnnotatedHeatmap`` / ``Scillus::plot_heatmap``."""
    is_adata = hasattr(adata_or_matrix, "obs") and hasattr(adata_or_matrix, "var_names")
    if is_adata:
        adata = adata_or_matrix
        meta_df = metadata.copy() if metadata is not None else adata.obs.copy()
        avail = [str(v) for v in adata.var_names]
        v_map = {v: i for i, v in enumerate(avail)}
        req = [str(f) for f in dict.fromkeys(features) if str(f) in v_map]
        if len(req) == 0:
            raise ValueError("None of the requested heatmap features were found in AnnData.var_names")
        cols = np.array([v_map[f] for f in req], dtype=np.int64)

        if layer in adata.layers:
            raw = adata.layers[layer][:, cols]
            mat_s_by_g = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw, dtype=np.float64)
        else:
            # Compute row-wise gene z-scores across samples from 'data' layer (matching ScaleData)
            base_l = adata.layers["data"][:, cols] if "data" in adata.layers else adata.X[:, cols]
            arr = base_l.toarray() if hasattr(base_l, "toarray") else np.asarray(base_l, dtype=np.float64)
            mu = arr.mean(axis=0, keepdims=True)
            sd = np.std(arr, axis=0, ddof=1, keepdims=True)
            sd[sd == 0] = 1.0
            mat_s_by_g = (arr - mu) / sd
        mat_df = pd.DataFrame(mat_s_by_g.T, index=req, columns=adata.obs_names.astype(str))
    else:
        if not isinstance(adata_or_matrix, pd.DataFrame) or metadata is None:
            raise ValueError("When passing a matrix instead of AnnData, supply a genes x samples DataFrame and metadata")
        req = [str(f) for f in dict.fromkeys(features) if str(f) in adata_or_matrix.index]
        mat_df = adata_or_matrix.loc[req].astype(np.float64).copy()
        meta_df = metadata.loc[mat_df.columns].copy()

    sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
    meta_sorted = meta_df.loc[mat_df.columns].sort_values(sort_cols, kind="mergesort")
    ordered_samples = list(meta_sorted.index.astype(str))
    mat_ord = mat_df.loc[:, ordered_samples]
    clipped = np.clip(mat_ord.to_numpy(dtype=np.float64), float(hm_limit[0]), float(hm_limit[2]))

    anno_cols = [] if anno_var is None else ([anno_var] if isinstance(anno_var, str) else list(anno_var))
    anno_cols = [c for c in anno_cols if c in meta_sorted.columns]

    n_genes, n_samps = clipped.shape
    fig_h = max(3.6, min(9.5, 0.16 * n_genes + 1.4))
    fig_w = max(5.2, min(9.2, 0.11 * n_samps + 2.4))

    with plt.rc_context(_publication_rc()):
        hr = [0.22] * len(anno_cols) + [0.22, max(2.5, 0.14 * n_genes)]
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        gs = fig.add_gridspec(len(hr), 1, height_ratios=hr, hspace=0.04)

        axes_dict: dict[str, plt.Axes] = {}
        for a_idx, acol in enumerate(anno_cols):
            ax_a = fig.add_subplot(gs[a_idx, 0])
            s_vals = meta_sorted[acol]
            if pd.api.types.is_numeric_dtype(s_vals) and not isinstance(s_vals.dtype, pd.CategoricalDtype):
                arr_a = s_vals.to_numpy(dtype=np.float64)[None, :]
                ax_a.imshow(arr_a, aspect="auto", cmap="viridis")
            else:
                if isinstance(palette, Mapping):
                    cats = [c for c in palette.keys() if c in set(s_vals.astype(str))]
                    if not cats:
                        cats = list(pd.unique(s_vals.astype(str)))
                else:
                    cats = list(pd.unique(s_vals.astype(str)))
                c_map = _category_palette(cats, palette=palette)
                rgb = np.array([[mcolors.to_rgb(c_map[str(v)]) for v in s_vals.astype(str)]])
                ax_a.imshow(rgb, aspect="auto")
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=c_map[c], edgecolor="#333333", lw=0.4, label=str(c)) for c in cats]
                ax_a.legend(
                    handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.05),
                    ncol=min(len(cats), 5), frameon=False, fontsize=7.0, handlelength=1.0, columnspacing=0.9
                )
            ax_a.set_yticks([0], [acol], fontsize=7.8, fontweight="bold")
            ax_a.set_xticks([])
            for sp_n in ("top", "right", "bottom", "left"):
                ax_a.spines[sp_n].set_linewidth(0.5)
            axes_dict[f"anno_{acol}"] = ax_a

        # Continuous trajectory bar
        ax_t = fig.add_subplot(gs[len(anno_cols), 0])
        t_vals = pd.to_numeric(meta_sorted[sort_cols[0]], errors="coerce").to_numpy(dtype=np.float64)[None, :]
        ax_t.imshow(t_vals, aspect="auto", cmap="cividis")
        ax_t.set_yticks([0], [sort_cols[0]], fontsize=7.8, fontweight="bold")
        ax_t.set_xticks([])
        axes_dict["trajectory_bar"] = ax_t

        ax_hm = fig.add_subplot(gs[-1, 0])
        im = ax_hm.imshow(
            clipped,
            aspect="auto",
            interpolation="nearest",
            cmap=DIVERGING_BWR,
            vmin=float(hm_limit[0]),
            vmax=float(hm_limit[2]),
        )
        ax_hm.set_yticks(range(n_genes), list(mat_ord.index), fontsize=max(6.2, min(8.2, 180.0 / max(n_genes, 1))))
        if show_column_names:
            ax_hm.set_xticks(range(n_samps), ordered_samples, rotation=90, fontsize=6.5)
        else:
            ax_hm.set_xticks([])
            ax_hm.set_xlabel(f"Donors / samples ordered by {sort_cols[0]} (n={n_samps})")
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.028, pad=0.02)
        cb.set_label("Scaled expression (z)")
        axes_dict["heatmap"] = ax_hm

        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10.0)

        src_df = mat_ord.copy()
        src_df.insert(0, "gene_ID", mat_ord.index.astype(str))
        _save_figure_bundle(fig, output, src_df)
    return fig, axes_dict, src_df
