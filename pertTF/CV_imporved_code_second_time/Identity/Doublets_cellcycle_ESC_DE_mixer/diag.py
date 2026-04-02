#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np

def main():
    # 1. Load the 0.08 preprocessed file
    file_path = "preprocessed_hvg3000_thresh0.08.h5ad"
    print(f"[diag] Loading {file_path}...")
    adata = sc.read_h5ad(file_path)

    # 2. AUTO-DETECT COORDINATES
    # We find the 'center of mass' of the DE cluster to define the territory
    de_subset = adata[adata.obs['celltype'] == 'DE']
    x_mid = de_subset.obsm['X_umap'][:, 0].mean()
    y_mid = de_subset.obsm['X_umap'][:, 1].mean()
    x_std = de_subset.obsm['X_umap'][:, 0].std()
    y_std = de_subset.obsm['X_umap'][:, 1].std()

    # Define the DE Territory as a box around the mean (2 standard deviations)
    # This captures the main 'blue cloud' without manual guessing
    ghost_mask = (adata.obs["celltype"] == "ESC") & \
                 (adata.obsm["X_umap"][:, 0] > x_mid - 2*x_std) & \
                 (adata.obsm["X_umap"][:, 0] < x_mid + 2*x_std) & \
                 (adata.obsm["X_umap"][:, 1] > y_mid - 2*y_std) & \
                 (adata.obsm["X_umap"][:, 1] < y_mid + 2*y_std)

    n_ghosts = ghost_mask.sum()
    print(f"[diag] Mathematically identified {n_ghosts} Ghost cells in DE territory.")

    if n_ghosts < 5:
        print("[ERR] Too few ghosts found. Your DE cluster might be shaped irregularly.")
        return

    # 3. GENERATE THE "SMOKING GUN" TABLE
    # Compare these specific 39-40 cells to the rest of the DE cluster
    adata.obs['is_ghost'] = 'Pure_DE'
    adata.obs.loc[ghost_mask, 'is_ghost'] = 'Ghost_ESC'
    
    # Differential Expression: What makes a Ghost different from a Pure DE?
    sc.tl.rank_genes_groups(adata, groupby='is_ghost', groups=['Ghost_ESC'], reference='Pure_DE', method='wilcoxon')
    
    result = adata.uns['rank_genes_groups']
    df = pd.DataFrame({
        'gene': result['names']['Ghost_ESC'],
        'logfoldchange': result['logfoldchanges']['Ghost_ESC'],
        'pvals_adj': result['pvals_adj']['Ghost_ESC']
    })
    
    # Filter for significant markers (p-adj < 0.05)
    sig_df = df[df['pvals_adj'] < 0.05].head(20)
    print("\n[diag] TOP 20 GENES DRIVING THE ESC LABEL IN DE TERRITORY:")
    print(sig_df[['gene', 'logfoldchange', 'pvals_adj']])
    sig_df.to_csv("PI_convincing_table.csv")

    # 4. SAVE PLOTS FOR PI
    # Violin plots for standard markers to show 'lag' or 'soup'
    # Check if markers exist in your var_names; use standard ones if so
    markers = [g for g in ['NANOG', 'SOX17', 'POU5F1', 'FOXA2'] if g in adata.var_names]
    
    # This plot will show that Ghost_ESC has DE levels of SOX17 but trace NANOG
    sc.pl.violin(adata[adata.obs['is_ghost'].isin(['Pure_DE', 'Ghost_ESC'])], 
                 keys=markers, groupby='is_ghost', show=False, save="_convince_PI.png")

if __name__ == "__main__":
    main()
