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

# Define the clusters to HIGHLIGHT (others will be grey)
clusters_to_highlight = ['8', '9', '13', '15']

# Create a subset for just the highlighted clusters
adata_subset = adata[adata.obs['seurat_clusters'].isin(clusters_to_highlight)].copy()

# --- 2. CALCULATE POSITIONS (Centroids) ---
# We need centroids for the highlight clusters to label them on the gene plots
print("Calculating label positions...")
cluster_centroids = {}
for cluster in clusters_to_highlight:
    coords = adata_subset[adata_subset.obs['seurat_clusters'] == cluster].obsm['X_umap']
    cluster_centroids[cluster] = np.mean(coords, axis=0)

# --- 3. GENERATE PDF ---
output_filename = 'feature_plots_global_context_grey.pdf'
print(f"Generating PDF: {output_filename}")

with PdfPages(output_filename) as pdf:
    
    # === PAGE 1: Reference Map (ALL CLUSTERS) ===
    print("Plotting Global Reference Map...")
    fig_ref = plt.figure(figsize=(7, 7))
    sc.pl.umap(
        adata, 
        color='seurat_clusters', 
        legend_loc='on data',       # Labels all clusters (0-15)
        add_outline=True, 
        title="Global Reference Map (All Clusters)",
        ax=plt.gca(),
        show=False
    )
    pdf.savefig(fig_ref, bbox_inches='tight')
    plt.close()

    # === PAGES 2+: Gene Expression (Highlight Mode) ===
    print("Plotting genes with grey background...")
    
    # Get global UMAP coordinates for the grey background
    # We use pure matplotlib for the background to ensure it's simple grey
    all_umap_x = adata.obsm['X_umap'][:, 0]
    all_umap_y = adata.obsm['X_umap'][:, 1]
    
    for gene in genes_to_plot:
        if gene not in adata.raw.var_names:
            print(f"  Skipping {gene}: not found.")
            continue
            
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # LAYER 1: The Grey Background (All Cells)
        # s=10 sets the dot size. Adjust if too big/small.
        ax.scatter(all_umap_x, all_umap_y, c='lightgrey', s=20, edgecolors='none', label='Other Clusters')
        
        # LAYER 2: The Expression Data (Only Clusters 8,9,13,15)
        # We plot the subset ON TOP of the grey background
        sc.pl.umap(
            adata_subset, 
            color=gene, 
            use_raw=True, 
            cmap='Reds',       # Expression is Red
            vmin=0, vmax='p99', 
            show=False, 
            ax=ax,             # Plot on the existing axes
            title=f"{gene} (Highlighter Mode)",
            colorbar_loc='right',
            frameon=False,
            size=20            # Match size with background
        )
        
        # LAYER 3: The Labels (Only 8,9,13,15)
        for cluster, (x, y) in cluster_centroids.items():
            ax.text(
                x, y, 
                cluster, 
                fontsize=10, 
                weight='bold', 
                color='black',
                ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", alpha=0.8, ec="black", lw=0.5)
            )

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print("Done! Open 'feature_plots_global_context_grey.pdf'")
