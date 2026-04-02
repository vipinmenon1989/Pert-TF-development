#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def read_ids(path: Path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow CSV-like or whitespace
            s = s.split(",")[0].strip()
            ids.append(s)
    return pd.Index(ids)


def plot_group(ax, X, labels, title, s=2, alpha=0.8, legend=True):
    labels = pd.Series(labels).astype(str)
    cats = pd.Categorical(labels).categories

    # stable ordering for overlay (so colors consistent within a figure)
    # matplotlib will cycle colors; we keep order fixed by category order
    for cat in cats:
        m = (labels.values == cat)
        if m.sum() == 0:
            continue
        ax.scatter(X[m, 0], X[m, 1], s=s, alpha=alpha, label=cat)

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_aspect("equal", adjustable="box")

    if legend:
        # Put legend outside for readability
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            markerscale=3,
            ncol=1,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, type=Path, help="AnnData used for original UMAPs")
    ap.add_argument("--tail-ids", required=True, type=Path, help="tail_esc_cell_ids.txt")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--umap-key", default="X_umap_scgpt", help="obsm key for UMAP coords")
    ap.add_argument("--celltype-key", default="celltype_2", help="obs key for celltype labels")
    ap.add_argument("--genotype-key", default="genotype", help="obs key for genotype labels")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--also-write-filtered-h5ad", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ad = sc.read_h5ad(str(args.h5ad))
    if args.umap_key not in ad.obsm:
        raise KeyError(f"Missing ad.obsm['{args.umap_key}']. Use the same h5ad that produced the UMAPs.")
    for k in [args.celltype_key, args.genotype_key]:
        if k not in ad.obs.columns:
            raise KeyError(f"Missing ad.obs['{k}']")

    tail_ids = read_ids(args.tail_ids)

    # Intersect with adata index (avoid crashes)
    tail_ids = tail_ids.intersection(ad.obs_names)
    if len(tail_ids) == 0:
        raise RuntimeError("No tail IDs matched adata.obs_names (wrong h5ad or ID format).")

    keep_mask = ~ad.obs_names.isin(tail_ids)
    tail_mask = ad.obs_names.isin(tail_ids)

    ad_keep = ad[keep_mask].copy()
    ad_tail = ad[tail_mask].copy()

    # Use EXACT SAME UMAP axis limits from the full adata, so PDFs overlay perfectly
    X_full = np.asarray(ad.obsm[args.umap_key])
    xmin, xmax = np.nanmin(X_full[:, 0]), np.nanmax(X_full[:, 0])
    ymin, ymax = np.nanmin(X_full[:, 1]), np.nanmax(X_full[:, 1])

    def make_pdf(ad_sub, tag):
        X = np.asarray(ad_sub.obsm[args.umap_key])
        celltype = ad_sub.obs[args.celltype_key].astype(str).values
        genotype = ad_sub.obs[args.genotype_key].astype(str).values

        # --- 1) Celltype PDF ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        plot_group(
            ax, X, celltype,
            title=f"{tag} UMAP colored by {args.celltype_key}",
            s=args.point_size, alpha=args.alpha,
            legend=(not args.no_legend),
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        out1 = args.outdir / f"{tag}.celltype.pdf"
        fig.savefig(out1, bbox_inches="tight")
        plt.close(fig)

        # --- 2) Genotype PDF ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        plot_group(
            ax, X, genotype,
            title=f"{tag} UMAP colored by {args.genotype_key}",
            s=args.point_size, alpha=args.alpha,
            legend=(not args.no_legend),
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        out2 = args.outdir / f"{tag}.genotype.pdf"
        fig.savefig(out2, bbox_inches="tight")
        plt.close(fig)

        return out1, out2

    # 0) Original (for reference overlay)
    orig1, orig2 = make_pdf(ad, "ORIGINAL")

    # 1) Removed tail ESC cells
    rm1, rm2 = make_pdf(ad_keep, "REMOVED_tailESC")

    # 2) Tail only (useful sanity)
    tail1, tail2 = make_pdf(ad_tail, "ONLY_tailESC")

    print("Wrote PDFs:")
    print("  ", orig1)
    print("  ", orig2)
    print("  ", rm1)
    print("  ", rm2)
    print("  ", tail1)
    print("  ", tail2)

    # Optional: save filtered h5ad (so downstream plots match exactly)
    if args.also_write_filtered_h5ad:
        out_keep = args.outdir / "adata_removed_tailESC.h5ad"
        out_tail = args.outdir / "adata_only_tailESC.h5ad"
        ad_keep.write_h5ad(out_keep, compression="gzip")
        ad_tail.write_h5ad(out_tail, compression="gzip")
        print("Wrote filtered h5ad:")
        print("  ", out_keep)
        print("  ", out_tail)

    # Also write a genotype enrichment table for the tail cells
    vc = ad_tail.obs[args.genotype_key].astype(str).value_counts()
    vc.to_csv(args.outdir / "ONLY_tailESC.genotype_counts.tsv", sep="\t")
    print("Wrote:", args.outdir / "ONLY_tailESC.genotype_counts.tsv")


if __name__ == "__main__":
    main()
