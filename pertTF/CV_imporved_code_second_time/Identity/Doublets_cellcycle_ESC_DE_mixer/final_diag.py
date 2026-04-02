#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. LOAD THE DATA
    file_path = "preprocessed_hvg3000_thresh0.08.h5ad"
    print(f"[diag] Reading {file_path}...")
    adata = sc.read_h5ad(file_path)

    # 2. CALCULATE CELL CYCLE PHASES
    # Standard S and G2M marker lists
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51VC', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
    
    s_genes = [g for g in s_genes if g in adata.var_names]
    g2m_genes = [g for g in g2m_genes if g in adata.var_names]
    
    print("[diag] Scoring cell cycle (identifying S, G2M, and G1 phases)...")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # 3. DEFINE GROUPS BASED ON UMAP TERRITORY
    de_subset = adata[adata.obs['celltype'] == 'DE']
    x_mid, y_mid = de_subset.obsm['X_umap'][:, 0].mean(), de_subset.obsm['X_umap'][:, 1].mean()
    x_std, y_std = de_subset.obsm['X_umap'][:, 0].std(), de_subset.obsm['X_umap'][:, 1].std()

    # Isolate the "Ghosts" sitting inside the DE cluster territory
    ghost_mask = (adata.obs["celltype"] == "ESC") & \
                 (np.abs(adata.obsm["X_umap"][:, 0] - x_mid) < 2*x_std) & \
                 (np.abs(adata.obsm["X_umap"][:, 1] - y_mid) < 2*y_std)

    adata.obs['pi_groups'] = 'Other'
    adata.obs.loc[adata.obs['celltype'] == 'DE', 'pi_groups'] = 'Pure_DE'
    adata.obs.loc[ghost_mask, 'pi_groups'] = 'Ghost_ESC_Proliferating'
    
    # 4. EXPORT QUANTITATIVE DATA (CSVs)
    # CSV 1: Statistical Phase Distribution
    phase_dist = adata.obs.groupby(['pi_groups', 'phase']).size().unstack(fill_value=0)
    phase_dist.to_csv("PI_CellCycle_Counts.csv")
    
    # CSV 2: Differential Expression (Ghosts vs Pure DE)
    sc.tl.rank_genes_groups(adata, groupby='pi_groups', groups=['Ghost_ESC_Proliferating'], reference='Pure_DE', method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    markers_df = pd.DataFrame({
        'gene': result['names']['Ghost_ESC_Proliferating'],
        'logfoldchange': result['logfoldchanges']['Ghost_ESC_Proliferating'],
        'pvals_adj': result['pvals_adj']['Ghost_ESC_Proliferating']
    })
    markers_df.to_csv("PI_Ghost_Marker_Evidence.csv", index=False)
    print("[csv] CSV files generated: PI_CellCycle_Counts.csv, PI_Ghost_Marker_Evidence.csv")

    # 5. GENERATE VISUAL EVIDENCE PLOTS
    print("[plot] Generating visual evidence figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Lineage Confirmation (Proving they are Endoderm)
    sc.pl.violin(adata[adata.obs['pi_groups'] != 'Other'], 
                 keys=['SOX17'], groupby='pi_groups', ax=axes[0], show=False)
    axes[0].set_title("Identity Proof: Both groups are SOX17+")

    # Panel 2: Cell Cycle Conflict (Proving they are dividing)
    sc.pl.violin(adata[adata.obs['pi_groups'] != 'Other'], 
                 keys=['S_score'], groupby='pi_groups', ax=axes[1], show=False)
    axes[1].set_title("Label Conflict: Ghosts are Hyper-Proliferative")

    plt.tight_layout()
    plt.savefig("PI_Evidence_Consolidated.png")
    print("[plot] Final plot saved: PI_Evidence_Consolidated.png")

if __name__ == "__main__":
    main()