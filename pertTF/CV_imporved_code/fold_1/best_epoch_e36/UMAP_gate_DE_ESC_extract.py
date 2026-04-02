#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

def pick_umap_key(adata):
    # prefer scGPT UMAP if present
    for k in ["X_umap_scgpt", "X_umap", "X_umap_scgpt_next", "X_umap_scgptNEXT"]:
        if k in adata.obsm:
            return k
    raise KeyError("No UMAP found in adata.obsm. Expected one of: X_umap_scgpt, X_umap, ...")

def knn_indices(X, k=30):
    # sklearn-free KNN using brute force (OK for ~80k with k small? borderline).
    # Prefer sklearn if available.
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto", metric="euclidean")
        nn.fit(X)
        _, idx = nn.kneighbors(X, return_distance=True)
        return idx[:, 1:]  # drop self
    except Exception:
        # brute force fallback (slow for large n; avoid unless you must)
        n = X.shape[0]
        idx = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            d = np.sum((X - X[i]) ** 2, axis=1)
            d[i] = np.inf
            idx[i] = np.argpartition(d, k)[:k]
        return idx

def safe_cat(s):
    return s.astype(str).replace({"nan": np.nan, "None": np.nan})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--celltype-key", default="celltype_2")
    ap.add_argument("--genotype-key", default="genotype")
    ap.add_argument("--esc-name", default="ESC")
    ap.add_argument("--de-name", default="DE")
    ap.add_argument("--k", type=int, default=30, help="kNN on UMAP")
    ap.add_argument("--de-tail-q", type=float, default=0.90, help="quantile cutoff for DE_tail by ESC-neighbor-frac")
    ap.add_argument("--esc-tail-q", type=float, default=0.90, help="quantile cutoff for ESC tail by DE_tail-neighbor-frac")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ad = sc.read_h5ad(str(args.h5ad))
    ct_key = args.celltype_key
    gt_key = args.genotype_key

    if ct_key not in ad.obs.columns:
        raise KeyError(f"Missing obs['{ct_key}']")
    if gt_key not in ad.obs.columns:
        print(f"[warn] Missing obs['{gt_key}'] — genotype enrichment will be skipped.")

    umap_key = pick_umap_key(ad)
    X = np.asarray(ad.obsm[umap_key], dtype=np.float32)

    ct = safe_cat(ad.obs[ct_key])
    is_esc = (ct.values == args.esc_name)
    is_de  = (ct.values == args.de_name)

    print(f"[info] Using UMAP key: {umap_key}")
    print(f"[info] n_obs={ad.n_obs}  ESC={is_esc.sum()}  DE={is_de.sum()}  k={args.k}")

    idx = knn_indices(X, k=args.k)

    # neighbor fractions
    neigh_is_esc = is_esc[idx]  # (n,k)
    neigh_is_de  = is_de[idx]

    esc_neigh_frac = neigh_is_esc.mean(axis=1)
    de_neigh_frac  = neigh_is_de.mean(axis=1)

    # ---------- Stage 1: define DE_tail (DE cells bordering/protruding into ESC) ----------
    de_esc_frac = esc_neigh_frac[is_de]
    if de_esc_frac.size == 0:
        raise RuntimeError("No DE cells found; check labels.")
    de_tail_thr = np.quantile(de_esc_frac, args.de_tail_q)
    de_tail = np.zeros(ad.n_obs, dtype=bool)
    de_tail[np.where(is_de)[0]] = (de_esc_frac >= de_tail_thr)

    # ---------- Stage 2: define ESC near DE_tail ----------
    neigh_is_de_tail = de_tail[idx]
    de_tail_neigh_frac = neigh_is_de_tail.mean(axis=1)

    esc_de_tail_frac = de_tail_neigh_frac[is_esc]
    esc_tail_thr = np.quantile(esc_de_tail_frac, args.esc_tail_q)
    esc_tail = np.zeros(ad.n_obs, dtype=bool)
    esc_tail[np.where(is_esc)[0]] = (esc_de_tail_frac >= esc_tail_thr)

    print(f"[gate] DE_tail: thr={de_tail_thr:.3f}  n={de_tail.sum()}")
    print(f"[gate] ESC_tail_near_DEtail: thr={esc_tail_thr:.3f}  n={esc_tail.sum()}")

    # save per-cell table
    df = pd.DataFrame({
        "cell_id": ad.obs_names,
        "celltype": ct.values,
        "umap1": X[:, 0],
        "umap2": X[:, 1],
        "esc_neigh_frac": esc_neigh_frac,
        "de_neigh_frac": de_neigh_frac,
        "de_tail_neigh_frac": de_tail_neigh_frac,
        "is_DE_tail": de_tail,
        "is_ESC_tail": esc_tail,
    }).set_index("cell_id", drop=False)

    if gt_key in ad.obs.columns:
        df["genotype"] = safe_cat(ad.obs[gt_key]).values

    df_out = args.outdir / "tail_gates_per_cell.csv"
    df.to_csv(df_out, index=False)
    print(f"[write] {df_out}")

    # write ID lists
    (args.outdir / "DE_tail_cell_ids.txt").write_text("\n".join(df.loc[df["is_DE_tail"], "cell_id"]) + "\n")
    (args.outdir / "ESC_tail_cell_ids.txt").write_text("\n".join(df.loc[df["is_ESC_tail"], "cell_id"]) + "\n")
    print(f"[write] DE_tail_cell_ids.txt / ESC_tail_cell_ids.txt")

    # genotype enrichment: tail ESC vs all ESC
    if gt_key in ad.obs.columns:
        esc_all = df[df["celltype"] == args.esc_name].copy()
        esc_t   = df[df["is_ESC_tail"]].copy()

        c_all = esc_all["genotype"].value_counts()
        c_t   = esc_t["genotype"].value_counts()

        enr = pd.DataFrame({
            "tail_count": c_t,
            "esc_count": c_all,
        }).fillna(0).astype(int)

        enr["tail_frac"] = enr["tail_count"] / max(1, enr["tail_count"].sum())
        enr["esc_frac"]  = enr["esc_count"]  / max(1, enr["esc_count"].sum())
        enr["log2_enrichment"] = np.log2((enr["tail_frac"] + 1e-9) / (enr["esc_frac"] + 1e-9))
        enr = enr.sort_values("log2_enrichment", ascending=False)

        enr_out = args.outdir / "genotype_enrichment_ESCtail_vs_ESC.csv"
        enr.to_csv(enr_out)
        print(f"[write] {enr_out}")

    # headless debug plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        # ESC + DE background
        ax.scatter(df.loc[df["celltype"] == args.esc_name, "umap1"],
                   df.loc[df["celltype"] == args.esc_name, "umap2"],
                   s=4, alpha=0.15, label=f"{args.esc_name} (all)")
        ax.scatter(df.loc[df["celltype"] == args.de_name, "umap1"],
                   df.loc[df["celltype"] == args.de_name, "umap2"],
                   s=6, alpha=0.35, label=f"{args.de_name} (all)")

        # DE tail and ESC tail overlays
        ax.scatter(df.loc[df["is_DE_tail"], "umap1"],
                   df.loc[df["is_DE_tail"], "umap2"],
                   s=14, alpha=0.9, label="DE_tail")
        ax.scatter(df.loc[df["is_ESC_tail"], "umap1"],
                   df.loc[df["is_ESC_tail"], "umap2"],
                   s=18, alpha=0.95, label="ESC near DE_tail")

        ax.set_title("Auto-gated DE tail + ESC near DE tail (UMAP)")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(loc="best")
        fig.tight_layout()
        out_png = args.outdir / "tail_gate_debug_DEtail_then_ESCtail.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[write] {out_png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

if __name__ == "__main__":
    main()
