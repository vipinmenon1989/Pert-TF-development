#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # HPC/headless safe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import scanpy as sc


def read_ids(txt_path: Path):
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def get_umap_xy(adata, coord_key: str):
    # coord_key can be "X_umap" or "X_umap_scgpt" (stored in .obsm)
    if coord_key in adata.obsm:
        xy = adata.obsm[coord_key]
        if xy is None or xy.shape[1] < 2:
            raise ValueError(f"{coord_key} exists but is invalid shape={None if xy is None else xy.shape}")
        return np.asarray(xy[:, :2])
    raise KeyError(f"Could not find adata.obsm['{coord_key}']. Available obsm keys: {list(adata.obsm.keys())}")


def ensure_cats_and_colors(adata, key: str):
    """
    Keep category order + colors stable (important for overlaying PDFs).
    If adata.uns[f"{key}_colors"] exists, we reuse it.
    """
    if key not in adata.obs.columns:
        raise KeyError(f"Missing adata.obs['{key}']")

    adata.obs[key] = adata.obs[key].astype("category")

    colors_key = f"{key}_colors"
    colors = None
    if colors_key in adata.uns and adata.uns[colors_key] is not None:
        colors = list(adata.uns[colors_key])

    cats = list(adata.obs[key].cat.categories)
    return cats, colors


def plot_umap_matplotlib(
    xy,
    labels,
    categories,
    colors=None,
    title="",
    out_pdf=None,
    out_png=None,
    xlim=None,
    ylim=None,
    s=2.0,
    alpha=0.9,
    rasterized=True,
    legend=True,
):
    # Map labels -> integers over full category list (keeps colors stable even if some cats absent)
    cat_to_i = {c: i for i, c in enumerate(categories)}
    lab = pd.Series(labels).astype(str)

    # unseen labels become NaN (won't be plotted)
    idx = lab.map(cat_to_i).to_numpy()
    keep = ~pd.isna(idx)
    idx = idx[keep].astype(int)
    xy2 = xy[keep]

    # colors
    if colors is None:
        # fallback palette
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(categories))]
    cmap = ListedColormap(colors[: len(categories)])

    fig, ax = plt.subplots(figsize=(10, 7))
    sca = ax.scatter(
        xy2[:, 0], xy2[:, 1],
        c=idx,
        cmap=cmap,
        s=s,
        alpha=alpha,
        linewidths=0,
        rasterized=rasterized,
    )

    ax.set_title(title, fontsize=20, pad=12)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend:
        # custom legend handles (one per category) using same palette
        handles = []
        for i, c in enumerate(categories):
            h = plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                           markerfacecolor=colors[i], markeredgecolor="none", label=c)
            handles.append(h)
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)

    fig.tight_layout()

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Remove a list of cell IDs and replot UMAPs with identical coordinates.")
    ap.add_argument("--h5ad", required=True, type=Path, help="Input h5ad containing UMAP coords in .obsm")
    ap.add_argument("--tail-ids", required=True, type=Path, help="Text file: one cell_id (obs_name) per line")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--coord-key", default=None, help="obsm key for coordinates. If omitted: try X_umap_scgpt then X_umap")
    ap.add_argument("--celltype-key", default="celltype_2")
    ap.add_argument("--genotype-key", default="genotype")
    ap.add_argument("--prefix", default="umap_compare")
    ap.add_argument("--also-png", action="store_true", help="Also write PNG copies")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.h5ad}")
    ad = sc.read_h5ad(str(args.h5ad))

    # pick coordinate key
    if args.coord_key is None:
        if "X_umap_scgpt" in ad.obsm:
            coord_key = "X_umap_scgpt"
        elif "X_umap" in ad.obsm:
            coord_key = "X_umap"
        else:
            raise KeyError(f"No X_umap_scgpt or X_umap found in obsm. Keys: {list(ad.obsm.keys())}")
    else:
        coord_key = args.coord_key

    xy = get_umap_xy(ad, coord_key)

    # fixed axis limits from ORIGINAL coords (so PDFs can be overlaid)
    pad = 0.5
    xlim = (float(np.min(xy[:, 0]) - pad), float(np.max(xy[:, 0]) + pad))
    ylim = (float(np.min(xy[:, 1]) - pad), float(np.max(xy[:, 1]) + pad))

    # load tail IDs and compute mask
    tail_ids = set(read_ids(args.tail_ids))
    obs_names = pd.Index(ad.obs_names)
    present = obs_names.isin(list(tail_ids))
    n_present = int(present.sum())
    print(f"[ids] tail list={len(tail_ids)} present_in_adata={n_present} / n_obs={ad.n_obs}")

    # Summaries of the removed cells
    removed = ad.obs.loc[present, [args.celltype_key, args.genotype_key]].copy()
    removed.to_csv(args.outdir / f"{args.prefix}.removed_cells.tsv", sep="\t")

    removed[args.genotype_key].astype(str).value_counts().to_csv(
        args.outdir / f"{args.prefix}.removed_genotype_counts.tsv", sep="\t", header=False
    )
    removed[args.celltype_key].astype(str).value_counts().to_csv(
        args.outdir / f"{args.prefix}.removed_celltype_counts.tsv", sep="\t", header=False
    )

    # Make a filtered view (lightweight: only obs + obsm needed)
    keep_mask = ~present
    ad_f = sc.AnnData(
        X=None,
        obs=ad.obs.loc[keep_mask].copy(),
        obsm={coord_key: xy[keep_mask]},
        uns=ad.uns.copy(),  # keep *_colors if present
    )

    # category order + colors (from original, for consistent legends/palettes)
    ct_cats, ct_colors = ensure_cats_and_colors(ad, args.celltype_key)
    gt_cats, gt_colors = ensure_cats_and_colors(ad, args.genotype_key)

    # IMPORTANT: keep same categories in filtered object even if some missing
    ad_f.obs[args.celltype_key] = ad_f.obs[args.celltype_key].astype("category")
    ad_f.obs[args.celltype_key] = ad_f.obs[args.celltype_key].cat.set_categories(ct_cats)

    ad_f.obs[args.genotype_key] = ad_f.obs[args.genotype_key].astype("category")
    ad_f.obs[args.genotype_key] = ad_f.obs[args.genotype_key].cat.set_categories(gt_cats)

    # export filtered IDs too (useful for sanity)
    pd.Series(ad_f.obs_names).to_csv(args.outdir / f"{args.prefix}.kept_cell_ids.txt", index=False, header=False)

    # ---- PLOTS (PDF; optional PNG) ----
    def outpaths(tag):
        pdf = args.outdir / f"{args.prefix}.{tag}.pdf"
        png = (args.outdir / f"{args.prefix}.{tag}.png") if args.also_png else None
        return pdf, png

    # ORIGINAL celltype
    pdf, png = outpaths("ORIG_celltype")
    plot_umap_matplotlib(
        xy=xy,
        labels=ad.obs[args.celltype_key].astype(str).values,
        categories=ct_cats,
        colors=ct_colors,
        title=f"ORIGINAL UMAP colored by {args.celltype_key} ({coord_key})",
        out_pdf=pdf,
        out_png=png,
        xlim=xlim, ylim=ylim,
        s=2.0, alpha=0.9, rasterized=True, legend=True,
    )

    # FILTERED celltype
    pdf, png = outpaths("FILTERED_celltype_removedTail")
    plot_umap_matplotlib(
        xy=ad_f.obsm[coord_key],
        labels=ad_f.obs[args.celltype_key].astype(str).values,
        categories=ct_cats,
        colors=ct_colors,
        title=f"FILTERED (tail removed) colored by {args.celltype_key} ({coord_key})",
        out_pdf=pdf,
        out_png=png,
        xlim=xlim, ylim=ylim,
        s=2.0, alpha=0.9, rasterized=True, legend=True,
    )

    # ORIGINAL genotype
    pdf, png = outpaths("ORIG_genotype")
    plot_umap_matplotlib(
        xy=xy,
        labels=ad.obs[args.genotype_key].astype(str).values,
        categories=gt_cats,
        colors=gt_colors,
        title=f"ORIGINAL UMAP colored by {args.genotype_key} ({coord_key})",
        out_pdf=pdf,
        out_png=png,
        xlim=xlim, ylim=ylim,
        s=2.0, alpha=0.9, rasterized=True, legend=True,
    )

    # FILTERED genotype
    pdf, png = outpaths("FILTERED_genotype_removedTail")
    plot_umap_matplotlib(
        xy=ad_f.obsm[coord_key],
        labels=ad_f.obs[args.genotype_key].astype(str).values,
        categories=gt_cats,
        colors=gt_colors,
        title=f"FILTERED (tail removed) colored by {args.genotype_key} ({coord_key})",
        out_pdf=pdf,
        out_png=png,
        xlim=xlim, ylim=ylim,
        s=2.0, alpha=0.9, rasterized=True, legend=True,
    )

    print("[done] wrote to:", args.outdir)


if __name__ == "__main__":
    main()
