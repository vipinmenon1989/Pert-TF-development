#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_path = "preprocessed_hvg3000_thresh0.08.h5ad"
    adata = sc.read_h5ad(file_path)

    # 1. Re-run scoring to ensure we have the data
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51VC', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
    s_genes = [g for g in s_genes if g in adata.var_names]
    g2m_genes = [g for g in g2m_genes if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # 2. Define Groups
    de_subset = adata[adata.obs['celltype'] == 'DE']
    x_mid, y_mid = de_subset.obsm['X_umap'][:, 0].mean(), de_subset.obsm['X_umap'][:, 1].mean()
    x_std, y_std = de_subset.obsm['X_umap'][:, 0].std(), de_subset.obsm['X_umap'][:, 1].std()
    ghost_mask = (adata.obs["celltype"] == "ESC") & \
                 (np.abs(adata.obsm["X_umap"][:, 0] - x_mid) < 2*x_std) & \
                 (np.abs(adata.obsm["X_umap"][:, 1] - y_mid) < 2*y_std)

    adata.obs['pi_groups'] = 'Other'
    adata.obs.loc[adata.obs['celltype'] == 'DE', 'pi_groups'] = 'Pure_DE'
    adata.obs.loc[ghost_mask, 'pi_groups'] = 'Ghost_ESC (Proliferating)'
    
    # 3. Create the 3-Panel Visual Proof
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # PANEL A: UMAP Highlight (The "Where" proof)
    sc.pl.umap(adata, color='pi_groups', groups=['Pure_DE', 'Ghost_ESC (Proliferating)'], 
               title="UMAP: Shared Transcriptomic Territory", ax=axes[0], show=False)

    # PANEL B: Lineage Markers (The "What" proof)
    sc.pl.violin(adata[adata.obs['pi_groups'] != 'Other'], keys=['SOX17'], 
                 groupby='pi_groups', ax=axes[1], show=False)
    axes[1].set_title("Lineage: Both are SOX17+")

    # PANEL C: Proliferation Score (The "Why" proof)
    sc.pl.violin(adata[adata.obs['pi_groups'] != 'Other'], keys=['S_score'], 
                 groupby='pi_groups', ax=axes[2], show=False)
    axes[2].set_title("Artifact Source: High S-Phase Score")

    plt.tight_layout()
    plt.savefig("FINAL_PI_PROOF.png", dpi=300)
    print("[plot] Saved: FINAL_PI_PROOF.png")

if __name__ == "__main__":
    main()
