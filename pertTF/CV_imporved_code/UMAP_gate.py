#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

def get_umap(adata, key):
    if key in adata.obsm:
        return np.asarray(adata.obsm[key])
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])
    raise KeyError(f"No UMAP found in obsm['{key}'] or obsm['X_umap'].")

def genotype_enrichment(obs_esc, tail_mask_esc):
    # obs_esc is obs filtered to ESC only; tail_mask_esc is boolean over ESC rows
    g_all = obs_esc["genotype"].astype(str)
    g_tail = g_all[tail_mask_esc]

    all_counts = g_all.value_counts()
    tail_counts = g_tail.value_counts()

    out = pd.DataFrame({"count_in_tail": tail_counts, "count_in_ESC": all_counts}).fillna(0).astype(int)
    out["frac_in_tail"] = out["count_in_tail"] / max(1, int(tail_mask_esc.sum()))
    out["frac_in_ESC"]  = out["count_in_ESC"] / max(1, len(g_all))
    out["enrichment_ratio"] = (out["frac_in_tail"] + 1e-12) / (out["frac_in_ESC"] + 1e-12)
    return out.sort_values(["count_in_tail", "enrichment_ratio"], ascending=[False, False])

def parse_args():
    ap = argparse.ArgumentParser("Auto-gate ESC tail near DE using UMAP neighbor composition")
    ap.add_argument("--h5ad", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--celltype-key", default="celltype_2")
    ap.add_argument("--genotype-key", default="genotype")
    ap.add_argument("--umap-key", default="X_umap_scgpt")

    ap.add_argument("--esc-label", default="ESC")
    ap.add_argument("--de-label", default="DE")

    ap.add_argument("--k", type=int, default=50, help="kNN in UMAP space")
    ap.add_argument("--min-de-frac", type=float, default=0.20, help="min fraction of DE neighbors")
    ap.add_argument("--top-frac", type=float, default=0.05, help="also keep top X fraction by score (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", default=True)
    return ap.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(args.h5ad))
    obs = adata.obs.copy()

    if args.celltype_key not in obs.columns:
        raise ValueError(f"Missing obs['{args.celltype_key}']")
    if args.genotype_key not in obs.columns:
        raise ValueError(f"Missing obs['{args.genotype_key}']")
    xy = get_umap(adata, args.umap_key)

    ct = obs[args.celltype_key].astype(str).to_numpy()
    gt = obs[args.genotype_key].astype(str).to_numpy()

    is_esc = (ct == args.esc_label)
    is_de  = (ct == args.de_label)

    if is_esc.sum() == 0 or is_de.sum() == 0:
        raise ValueError(f"Need both ESC and DE present in {args.celltype_key}")

    # ----- kNN in UMAP space (no sklearn needed) -----
    # brute-force is too slow; use scanpy neighbors on UMAP coords
    ad_knn = sc.AnnData(X=xy)
    sc.pp.neighbors(ad_knn, n_neighbors=args.k, use_rep="X", random_state=args.seed, key_added="umap_knn")

    # adjacency in CSR; for each ESC cell compute fraction of DE in its neighbors
    A = ad_knn.obsp["umap_knn_connectivities"].tocsr()
    de_mask = is_de.astype(np.float32)

    # de_score[i] = sum_j A[i,j]*DE[j] / sum_j A[i,j]
    denom = np.asarray(A.sum(axis=1)).reshape(-1)
    numer = np.asarray(A.dot(de_mask)).reshape(-1)
    de_frac = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    # distance-to-DE-centroid term (to favor “near DE”)
    de_centroid = xy[is_de].mean(axis=0)
    dist = np.linalg.norm(xy - de_centroid, axis=1)
    # scale by DE spread
    sigma = np.median(np.linalg.norm(xy[is_de] - de_centroid, axis=1)) + 1e-9
    score = de_frac * np.exp(-dist / sigma)

    esc_score = score[is_esc]
    esc_de_frac = de_frac[is_esc]

    # threshold gate
    tail = np.zeros(adata.n_obs, dtype=bool)
    base_gate = is_esc & (de_frac >= args.min_de_frac)
    tail |= base_gate

    if args.top_frac and args.top_frac > 0:
        n_top = max(1, int(args.top_frac * is_esc.sum()))
        top_idx_esc = np.argsort(-esc_score)[:n_top]
        esc_indices = np.where(is_esc)[0]
        tail[esc_indices[top_idx_esc]] = True

    # outputs
    tail_ids = adata.obs_names[tail].to_list()
    (args.outdir / "tail_esc_cell_ids.txt").write_text("\n".join(map(str, tail_ids)) + "\n")
    adata[tail].copy().write_h5ad(args.outdir / "tail_esc_subset.h5ad", compression="gzip")

    # genotype enrichment within ESC only
    obs_esc = obs.loc[is_esc].copy()
    obs_esc["genotype"] = obs_esc[args.genotype_key].astype(str)
    tail_mask_esc = tail[is_esc]
    enr = genotype_enrichment(obs_esc, tail_mask_esc)
    enr.to_csv(args.outdir / "genotype_enrichment_tailESC_vs_allESC.csv")

    # save per-cell table for debugging
    dbg = pd.DataFrame({
        "cell_id": adata.obs_names,
        "celltype": ct,
        "genotype": gt,
        "de_neighbor_frac": de_frac,
        "dist_to_de_centroid": dist,
        "score": score,
        "selected_tail": tail,
    })
    dbg.loc[is_esc].to_csv(args.outdir / "esc_tail_scoring_table.csv", index=False)

    print(f"[OK] Selected tail ESC cells: {tail.sum()}")
    print(f"[OK] Wrote: {args.outdir.resolve()}")
    print("[Top genotypes in tail ESC]:")
    print(enr.head(15)[["count_in_tail", "frac_in_tail", "enrichment_ratio"]].to_string())

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 7))
        ax = plt.gca()
        ax.scatter(xy[is_esc, 0], xy[is_esc, 1], s=3, alpha=0.15, label="ESC (all)")
        ax.scatter(xy[is_de, 0],  xy[is_de, 1],  s=6, alpha=0.35, label="DE (all)")
        ax.scatter(xy[tail, 0],   xy[tail, 1],   s=8, alpha=0.9,  label="ESC tail (selected)")
        ax.set_title("Auto-gated ESC tail near DE (UMAP)")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.legend(loc="best")
        fig.savefig(args.outdir / "tail_gate_debug.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
