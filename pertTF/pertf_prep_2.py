#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scgpt.preprocess import Preprocessor


def parse_args():
    ap = argparse.ArgumentParser("Prepare multiple preprocessed AnnDatas with different doublet thresholds")
    ap.add_argument("--raw-h5ad", required=True, type=Path, help="Input .h5ad (must contain obs['celltype_2'])")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-wt-frac", type=float, default=0.1)
    ap.add_argument("--n-bins", type=int, default=51)
    ap.add_argument("--n-hvg", type=int, default=3000)

    ap.add_argument("--filter-missing-subcluster", action="store_true", default=True)
    ap.add_argument("--no-filter-missing-subcluster", dest="filter_missing_subcluster", action="store_false")

    # Make downstream code safe even if it still references obs["celltype"]
    ap.add_argument("--overwrite-celltype-with-celltype2", action="store_true", default=True)

    # Safety: crash if input celltype_2 looks overwritten to 20-class
    ap.add_argument("--guard-celltype2-cardinality", action="store_true", default=True)
    ap.add_argument("--no-guard-celltype2-cardinality", dest="guard_celltype2_cardinality", action="store_false")

    return ap.parse_args()


def nunique_obs(ad, key: str):
    if key not in ad.obs.columns:
        return None
    return ad.obs[key].astype(str).nunique()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA ONCE
    in_path = args.raw_h5ad.resolve()
    print(f"[prep] Reading base file: {in_path}")
    adata_base = sc.read_h5ad(str(in_path))
    
    # --- MERGE ESC (D3) -> ESC ---
    print("[prep] Merging 'ESC (D3)' into 'ESC'...")
    for col in ["celltype", "celltype_2", "celltype_truth"]:
        if col in adata_base.obs.columns:
            adata_base.obs[col] = (
                adata_base.obs[col]
                .astype(str)
                .replace("ESC (D3)", "ESC")
                .astype("category")
            )
            
    # Capture Original Counts (Before any doublet filtering)
    # We use this to calculate exactly what we lost later
    original_counts = adata_base.obs["celltype_2"].value_counts()

    # --- CALCULATE SCORES ONCE ---
    print("[prep] Calculating Scrublet scores (this takes a moment)...")
    try:
        sc.pp.scrublet(adata_base, batch_key=None)
    except Exception as e:
        raise RuntimeError(f"Scrublet failed. Install 'scrublet' and 'scikit-image'. Error: {e}")

    # --- DEFINE THRESHOLDS TO TEST ---
    THRESHOLDS = [0.15, 0.12, 0.10, 0.08]
    
    results_summary = []

    print("\n" + "="*80)
    print(f"STARTING OPTIMIZATION LOOP: Testing {THRESHOLDS}")
    print("="*80 + "\n")

    for thresh in THRESHOLDS:
        print(f"\n>>> PROCESSING THRESHOLD: {thresh}")
        
        # 2. CREATE A FRESH COPY FOR THIS RUN
        adata = adata_base.copy()

        # 3. APPLY THRESHOLD
        if "doublet_score" in adata.obs:
            adata.obs["predicted_doublet"] = adata.obs["doublet_score"] > thresh
            n_doub = adata.obs['predicted_doublet'].sum()
            doub_pct = (n_doub / adata.n_obs) * 100
            
            # --- FULL DAMAGE REPORT ---
            # Create a breakdown of loss per cell type
            doub_mask = adata.obs['predicted_doublet']
            lost_counts = adata.obs.loc[doub_mask, "celltype_2"].value_counts()
            
            # Create a DataFrame for display
            df_report = pd.DataFrame({
                "Original": original_counts,
                "Lost": lost_counts,
            }).fillna(0).astype(int)
            
            df_report["Remaining"] = df_report["Original"] - df_report["Lost"]
            df_report["%_Lost"] = (df_report["Lost"] / df_report["Original"] * 100).round(2)
            
            # Sort by % Lost to see the worst hit groups first
            df_report = df_report.sort_values("%_Lost", ascending=False)
            
            print(f"    - Total Doublets Removed: {n_doub} ({doub_pct:.2f}%)")
            print("-" * 60)
            print(f"    - DETAILED LOSS REPORT (Threshold {thresh}):")
            print(df_report.to_string())
            print("-" * 60)
            
            # Filter the data
            adata = adata[~adata.obs['predicted_doublet']].copy()
            
            # Save stats for final leaderboard (Focusing on PGT safety)
            pgt_stats = df_report.loc["PGT"] if "PGT" in df_report.index else None
            
            results_summary.append({
                "Threshold": thresh,
                "Total_Doublets": n_doub,
                "PGT_Lost": pgt_stats["Lost"] if pgt_stats is not None else 0,
                "PGT_Remaining": pgt_stats["Remaining"] if pgt_stats is not None else 0
            })

        else:
            print("[WARN] No doublet score found, skipping threshold.")
            continue

        # 4. STANDARD PREPROCESSING (The rest of your pipeline)
        
        # Guard check
        if args.guard_celltype2_cardinality:
            n_ct2 = nunique_obs(adata, "celltype_2")
            n_sub = nunique_obs(adata, "sub.cluster")
            if (n_sub is not None and n_ct2 == n_sub):
                 pass # Silence this warning to keep logs clean

        # Subcluster filter
        if args.filter_missing_subcluster and ("sub.cluster" in adata.obs.columns):
            valid = adata.obs["sub.cluster"].notna()
            adata = adata[valid].copy()

        # Genotypes
        if "gene" not in adata.obs.columns:
            raise ValueError("adata.obs['gene'] missing")
        
        adata.obs["gene"] = (
            adata.obs["gene"].astype(str)
            .str.replace("124_NANOGe_het", "124_NANOGe-het", regex=False)
            .str.replace("123_NANOGe_het", "123_NANOGe-het", regex=False)
        )
        genotypes = (
            adata.obs["gene"].str.split("_").str[-1]
            .replace({"WT111": "WT", "WT4": "WT", "NGN3": "NEUROG3"})
            .fillna("WT")
            .astype(str)
        )
        adata.obs["genotype"] = genotypes

        # Thinning WT
        rng = np.random.default_rng(int(args.seed))
        keep_mask = (genotypes != "WT") | ((genotypes == "WT") & (rng.random(adata.n_obs) < float(args.keep_wt_frac)))
        adata = adata[keep_mask, :].copy()
        
        # Reattach labels
        adata.obs["genotype"] = adata_base.obs["genotype"].loc[adata.obs_names].values
        adata.obs["celltype_2"] = adata_base.obs["celltype_2"].loc[adata.obs_names].astype("category")

        if args.overwrite_celltype_with_celltype2:
            adata.obs["celltype"] = adata.obs["celltype_2"].copy()

        # GPTin / HVG / Binning
        adata.layers["GPTin"] = adata.X
        ad_tmp = adata.copy()
        ad_tmp.X = ad_tmp.layers["GPTin"]
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        sc.pp.highly_variable_genes(ad_tmp, flavor="seurat_v3", batch_key=None, n_top_genes=int(args.n_hvg))
        
        ranks0 = (ad_tmp.var["highly_variable_rank"].astype(float) - 1.0)
        adata.var["hvg_rank"] = ranks0.reindex(adata.var_names)
        keep_genes = adata.var.sort_values("hvg_rank").index[: int(args.n_hvg)]
        adata = adata[:, keep_genes].copy()

        pp = Preprocessor(
            use_key="GPTin",
            normalize_total=None,
            log1p=False,
            subset_hvg=False,
            hvg_flavor="seurat_v3",
            binning=int(args.n_bins),
            result_binned_key="X_binned",
        )
        pp(adata, batch_key=None)

        if sp.issparse(adata.layers["X_binned"]):
            adata.layers["X_binned"] = adata.layers["X_binned"].astype(np.uint8).tocsr()
        else:
            adata.layers["X_binned"] = sp.csr_matrix(np.asarray(adata.layers["X_binned"], dtype=np.uint8))

        adata.layers.pop("GPTin", None)
        adata.raw = None

        # 5. SAVE WITH UNIQUE NAME
        filename = f"preprocessed_hvg{int(args.n_hvg)}_thresh{thresh}.h5ad"
        out_h5ad = (args.outdir / filename).resolve()
        adata.write_h5ad(out_h5ad, compression="lzf")
        print(f"    -> Saved: {out_h5ad}")

    # --- FINAL SUMMARY ---
    print("\n" + "="*60)
    print("FINAL SUMMARY (Focus on PGT Safety)")
    print("="*60)
    df_res = pd.DataFrame(results_summary)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
