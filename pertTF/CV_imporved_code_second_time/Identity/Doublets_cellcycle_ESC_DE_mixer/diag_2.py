#!/usr/bin/env python3
import scanpy as sc
import pandas as pd

def main():
    # 1. Load the 0.08 preprocessed file
    file_path = "preprocessed_hvg3000_thresh0.08.h5ad"
    print(f"[diag] Loading {file_path}...")
    adata = sc.read_h5ad(file_path)

    # 2. DEFINE CELL CYCLE GENES
    # Standard lists (S-phase and G2M-phase)
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51VC', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

    # Filter for genes present in your data
    s_genes = [g for g in s_genes if g in adata.var_names]
    g2m_genes = [g for g in g2m_genes if g in adata.var_names]

    # 3. CALCULATE SCORES
    print("[diag] Scoring cell cycle phases...")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # 4. ANALYZE GHOSTS vs PURE DE
    # Use the same 'is_ghost' logic from before
    x_mid = adata[adata.obs['celltype'] == 'DE'].obsm['X_umap'][:, 0].mean()
    y_mid = adata[adata.obs['celltype'] == 'DE'].obsm['X_umap'][:, 1].mean()
    x_std = adata[adata.obs['celltype'] == 'DE'].obsm['X_umap'][:, 0].std()
    y_std = adata[adata.obs['celltype'] == 'DE'].obsm['X_umap'][:, 1].std()

    ghost_mask = (adata.obs["celltype"] == "ESC") & \
                 (adata.obsm["X_umap"][:, 0] > x_mid - 2*x_std) & \
                 (adata.obsm["X_umap"][:, 0] < x_mid + 2*x_std) & \
                 (adata.obsm["X_umap"][:, 1] > y_mid - 2*y_std) & \
                 (adata.obsm["X_umap"][:, 1] < y_mid + 2*y_std)

    adata.obs['is_ghost'] = 'Pure_DE'
    adata.obs.loc[ghost_mask, 'is_ghost'] = 'Ghost_ESC'

    # 5. STATISTICAL COMPARISON
    summary = adata.obs.groupby('is_ghost')[['S_score', 'G2M_score']].mean()
    print("\n[diag] Average Cell Cycle Scores:")
    print(summary)
    summary.to_csv("cell_cycle_ghost_comparison.csv")

    # 6. SAVE PLOTS FOR PI
    # Show if Ghosts are just the dividing cells
    sc.pl.violin(adata[adata.obs['is_ghost'].isin(['Pure_DE', 'Ghost_ESC'])], 
                 keys=['S_score', 'G2M_score'], groupby='is_ghost', show=False, save="_cell_cycle_check.png")
    
    # Plot phase distribution
    phase_counts = adata.obs.groupby(['is_ghost', 'phase']).size().unstack(fill_value=0)
    print("\n[diag] Phase Distribution:")
    print(phase_counts)
    phase_counts.to_csv("cell_cycle_phase_distribution.csv")

if __name__ == "__main__":
    main()
