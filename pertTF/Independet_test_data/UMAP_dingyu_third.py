import matplotlib
# CRITICAL: Must be set before importing pyplot for HPC
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scanpy as sc
import numpy as np

# --- 1. CONFIGURATION ---
print("Loading data...")
adata = sc.read_h5ad('18clones_seurat.h5ad')
sc.set_figure_params(dpi=300, vector_friendly=True)

# Define genes
genes_to_plot = [
    'POU5F1', 'SOX2', 'NANOG', 'EOMES', 'SOX17', 'GATA6', 'HNF1B', 
    'FOXA2', 'ONECUT1', 'PDX1', 'SOX9', 'NKX6-1', 'NEUROG3', 'NKX2-2', 
    'CHGA', 'INS', 'GCG', 'ARX', 'SST', 'TPH1', 'SLC18A1', 'HES1', 
    'GLIS3', 'YAP1', 'KLF5', 'CDX2', 'AFP', 'HNF4A', 'COL1A1', 
    'COL3A1', 'LEF1', 'FLT1', 'PLVAP'
]

# Ensure cluster column is string
adata.obs['seurat_clusters'] = adata.obs['seurat_clusters'].astype(str)

# --- 2. PRE-CALCULATE ALL CENTROIDS ---
# We calculate the center of EVERY cluster (0, 1, 2... 15)
# This allows us to stamp the number on the map for every gene.
print("Calculating label positions for all clusters...")
cluster_centroids = {}
all_clusters = adata.obs['seurat_clusters'].unique()

for cluster in all_clusters:
    # Extract UMAP coordinates for cells in this cluster
    coords = adata[adata.obs['seurat_clusters'] == cluster].obsm['X_umap']
    # Calculate mean position
    centroid = np.mean(coords, axis=0)
    cluster_centroids[cluster] = centroid

# --- 3. GENERATE PDF ---
output_filename = 'feature_plots_global_expression.pdf'
print(f"Generating PDF: {output_filename}")

with PdfPages(output_filename) as pdf:
    
    # === PAGE 1: Reference Map (Clusters Labels) ===
    print("Plotting Reference Map...")
    fig_ref = plt.figure(figsize=(7, 7))
    sc.pl.umap(
        adata, 
        color='seurat_clusters', 
        legend_loc='on data',       # Labels all clusters automatically
        add_outline=True,           # Draws borders
        title="Reference Map: All Clusters",
        ax=plt.gca(),
        show=False
    )
    pdf.savefig(fig_ref, bbox_inches='tight')
    plt.close()

    # === PAGES 2+: Gene Expression (Global) ===
    print("Plotting genes across all clusters...")
    
    for gene in genes_to_plot:
        # Check raw data for gene
        if gene not in adata.raw.var_names:
            print(f"  Skipping {gene}: not found.")
            continue
            
        fig = plt.figure(figsize=(7, 7))
        
        # Plot expression for the WHOLE dataset
        sc.pl.umap(
            adata, 
            color=gene, 
            use_raw=True, 
            cmap='Reds',       # Red = High expression
            vmin=0, vmax='p99', 
            show=False, 
            title=f"{gene} Expression (Global)",
            ax=plt.gca(),
            frameon=False
        )
        
        # OVERLAY ALL CLUSTER NUMBERS
        # This stamps '0', '1', '2'... on top of the red/white map
        ax = plt.gca()
        for cluster, (x, y) in cluster_centroids.items():
            ax.text(
                x, y, 
                cluster, 
                fontsize=8,                # Smaller font since there are many clusters
                weight='bold', 
                color='black',
                ha='center', va='center',
                # Semi-transparent box to make text readable over expression
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", alpha=0.6, ec="none")
            )
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print("Done! Open 'feature_plots_global_expression.pdf'")
