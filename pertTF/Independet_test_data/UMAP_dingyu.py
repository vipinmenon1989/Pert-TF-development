import matplotlib
# CRITICAL: Must be set before importing pyplot for HPC/Headless environments
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scanpy as sc
import numpy as np

# --- 1. CONFIGURATION & DATA LOADING ---
print("Loading data...")
# Load your specific file
adata = sc.read_h5ad('18clones_seurat.h5ad')

# Set Scanpy settings for higher quality output
sc.set_figure_params(dpi=300, vector_friendly=True)

# Define the full gene list from your screenshot
genes_to_plot = [
    'POU5F1', 'SOX2', 'NANOG', 'EOMES', 'SOX17', 'GATA6', 'HNF1B', 
    'FOXA2', 'ONECUT1', 'PDX1', 'SOX9', 'NKX6-1', 'NEUROG3', 'NKX2-2', 
    'CHGA', 'INS', 'GCG', 'ARX', 'SST', 'TPH1', 'SLC18A1', 'HES1', 
    'GLIS3', 'YAP1', 'KLF5', 'CDX2', 'AFP', 'HNF4A', 'COL1A1', 
    'COL3A1', 'LEF1', 'FLT1', 'PLVAP'
]

# --- 2. SUBSET DATA (Clusters 8, 9, 13, 15) ---
print("Subsetting for clusters 8, 9, 13, 15...")

# Ensure cluster column is categorical/string
adata.obs['seurat_clusters'] = adata.obs['seurat_clusters'].astype(str)

# Filter
clusters_of_interest = ['8', '9', '13', '15']
adata_subset = adata[adata.obs['seurat_clusters'].isin(clusters_of_interest)].copy()

# --- 3. PRE-CALCULATE CLUSTER CENTROIDS ---
# We calculate the center (mean X, Y) of each cluster to place the label accurately later
print("Calculating cluster label positions...")
cluster_centroids = {}
for cluster in clusters_of_interest:
    # Extract UMAP coordinates for cells in this cluster
    coords = adata_subset[adata_subset.obs['seurat_clusters'] == cluster].obsm['X_umap']
    # Calculate mean position
    centroid = np.mean(coords, axis=0)
    cluster_centroids[cluster] = centroid

# --- 4. GENERATE PDF ---
output_filename = 'feature_plots_clusters_8_9_13_15_labeled.pdf'
print(f"Generating PDF: {output_filename}")

with PdfPages(output_filename) as pdf:
    
    # === PAGE 1: The Reference Map ===
    # Plots clusters with distinct colors and outlines
    print("Plotting Reference Map...")
    fig_ref = plt.figure(figsize=(6, 6))
    sc.pl.umap(
        adata_subset, 
        color='seurat_clusters', 
        legend_loc='on data',       # Puts labels on the blobs
        add_outline=True,           # key for boundaries
        title="Reference Map: Clusters 8, 9, 13, 15",
        palette='tab10',            
        ax=plt.gca(),
        show=False
    )
    pdf.savefig(fig_ref, bbox_inches='tight')
    plt.close()

    # === PAGES 2+: Gene Expression with Overlays ===
    print("Plotting genes...")
    for gene in genes_to_plot:
        # Check if gene is in the raw data (full gene list)
        if gene not in adata_subset.raw.var_names:
            print(f"  Warning: Gene '{gene}' not found in data. Skipping.")
            continue
            
        fig = plt.figure(figsize=(6, 6))
        
        # Plot the gene expression (Red color scale)
        sc.pl.umap(
            adata_subset, 
            color=gene, 
            use_raw=True, 
            cmap='Reds',       # White -> Red
            vmin=0, vmax='p99', # Cut off top 1% outliers for better contrast
            show=False, 
            title=f"{gene} Expression",
            ax=plt.gca(),
            frameon=False,      # Remove the square box frame
            colorbar_loc='right'
        )
        
        # Overlay the cluster numbers manually
        ax = plt.gca()
        for cluster, (x, y) in cluster_centroids.items():
            ax.text(
                x, y, 
                cluster, 
                fontsize=12, 
                weight='bold', 
                color='black',
                ha='center', va='center',
                # Add a semi-transparent white box so text is readable over red dots
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none") 
            )
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close() # clear memory

print("Success! Script completed.")
