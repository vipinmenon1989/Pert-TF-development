import matplotlib
matplotlib.use('Agg') # Essential for HPC
import matplotlib.pyplot as plt
import scanpy as sc

# 1. Global Figure Settings
# Adjusting dpi_save ensures the PNG is high resolution.
# vector_friendly=True ensures the PDF text is editable (not outlined).
sc.set_figure_params(dpi=300, dpi_save=300, vector_friendly=True)

# 2. Load Data
adata = sc.read_h5ad('18clones_seurat.h5ad')

# 3. Define Genes
genes_to_plot = [
    'POU5F1', 'SOX2', 'NANOG', 'EOMES', 'SOX17', 'GATA6', 'HNF1B', 
    'FOXA2', 'ONECUT1', 'PDX1', 'SOX9', 'NKX6-1', 'NEUROG3', 'NKX2-2', 
    'CHGA', 'INS', 'GCG', 'ARX', 'SST', 'TPH1', 'SLC18A1', 'HES1', 
    'GLIS3', 'YAP1', 'KLF5', 'CDX2', 'AFP', 'HNF4A', 'COL1A1', 
    'COL3A1', 'LEF1', 'FLT1', 'PLVAP'
]

# 4. Subset Data (Clusters 8, 9, 13, 15)
adata.obs['seurat_clusters'] = adata.obs['seurat_clusters'].astype(str)
clusters_of_interest = ['8', '9', '13', '15']
adata_subset = adata[adata.obs['seurat_clusters'].isin(clusters_of_interest)].copy()

# 5. Generate Plot
# We store the returned DotPlot object in 'dp'
dp = sc.pl.dotplot(
    adata_subset, 
    var_names=genes_to_plot, 
    groupby='seurat_clusters',     
    standard_scale='var',      
    cmap='RdBu_r',             
    use_raw=True,              
    swap_axes=True,            # Genes on Y-axis
    show=False,                
    return_fig=True            
)

# 6. Save as PDF and PNG
# bbox_inches='tight' is crucial to prevent cutting off the long gene list
dp.savefig('dotplot_clusters_8_9_13_15.pdf', bbox_inches='tight')
dp.savefig('dotplot_clusters_8_9_13_15.png', bbox_inches='tight')

print("Saved 'dotplot_clusters_8_9_13_15.pdf' and 'dotplot_clusters_8_9_13_15.png'")
