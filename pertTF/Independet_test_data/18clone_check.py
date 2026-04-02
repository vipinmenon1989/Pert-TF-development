import os
import scanpy as sc
import pandas as pd

# ----------------------
# CONFIG
# ----------------------
h5ad_path = "18clones_seurat.h5ad"

# Optionally force the column names (set to None to auto-detect)
cluster_col_override = None   # e.g. "leiden" or "seurat_clusters"
celltype_col_override = None  # e.g. "celltype" or "cell_type"

outdir = "scanpy_umap_outputs"
os.makedirs(outdir, exist_ok=True)
sc.settings.figdir = outdir
sc.settings.verbosity = 2

# ----------------------
# LOAD
# ----------------------
adata = sc.read_h5ad(h5ad_path)
print(f"\nLoaded: {h5ad_path}")
print(f"Original Shape: {adata.n_obs} cells × {adata.n_vars} genes")

# ----------------------
# MODIFY: RENAME & FILTER
# ----------------------
# 1. Rename 'gene' column to 'genotype' if it exists in .obs
if 'gene' in adata.obs.columns:
    print("\nRenaming adata.obs['gene'] -> 'genotype'...")
    adata.obs.rename(columns={'gene': 'genotype'}, inplace=True)
else:
    print("\n[Warning] Column 'gene' not found in adata.obs. Skipping rename.")

# 2. Filter out rows where genotype is 'CCDC6'
target_col = 'genotype'  # This is the new name
remove_val = 'CCDC6'

if target_col in adata.obs.columns:
    n_before = adata.n_obs
    
    # Check if we actually have this value
    if remove_val in adata.obs[target_col].values:
        print(f"Filtering out cells where {target_col} == '{remove_val}'...")
        
        # logical vector: Keep rows that are NOT 'CCDC6'
        keep_mask = adata.obs[target_col] != remove_val
        adata = adata[keep_mask].copy()
        
        n_after = adata.n_obs
        print(f"Removed {n_before - n_after} cells. New Shape: {adata.n_obs} cells.")
    else:
        print(f"Value '{remove_val}' not found in column '{target_col}'. No cells removed.")
else:
    print(f"[Warning] Column '{target_col}' not found. Cannot filter.")

# ----------------------
# INSPECT CONTENTS (Updated)
# ----------------------
print("\n=== adata.obs columns (Current) ===")
print(list(adata.obs.columns))

# ----------------------
# PICK COLUMNS (cluster + celltype)
# ----------------------
def pick_col(obs: pd.DataFrame, candidates):
    for c in candidates:
        if c in obs.columns:
            return c
    return None

cluster_candidates = [
    "leiden", "louvain", "seurat_clusters", "clusters", "cluster", "snn_res", "resolution"
]
# Added 'genotype' to candidates in case you want to plot it as the cell type
celltype_candidates = [
    "celltype_2", "cell_type", "CellType", "celltypes", "annotation", "annot", "genotype"
]

cluster_col = cluster_col_override or pick_col(adata.obs, cluster_candidates)
celltype_col = celltype_col_override or pick_col(adata.obs, celltype_candidates)

print(f"\nUsing cluster_col = {cluster_col}")
print(f"Using celltype_col = {celltype_col}")

# Make sure they are categorical (prettier legends + stable coloring)
for col in [cluster_col, celltype_col]:
    if col is not None:
        # If it's not categorical, make it so. 
        # Also re-cast to remove unused categories (like 'CCDC6') from the index
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype("category")
        else:
            # Drop unused categories in case 'CCDC6' is still in the category list but with 0 cells
            adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

# ----------------------
# UMAP: use existing or compute
# ----------------------
if "X_umap" in adata.obsm.keys():
    print("\nUMAP embedding found in adata.obsm['X_umap'] -> will plot directly.")
else:
    print("\nNo UMAP found -> computing neighbors+UMAP.")
    if "X_pca" not in adata.obsm.keys():
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

# ----------------------
# PLOTS
# ----------------------
colors = [c for c in [cluster_col, celltype_col] if c is not None]

if colors:
    # Two-panel UMAP (cluster + celltype)
    sc.pl.umap(
        adata,
        color=colors,
        ncols=min(2, len(colors)),
        wspace=0.35,
        frameon=False,
        show=True,
        save="_cluster_genotype_filtered.png"
    )
else:
    print("[!] No suitable columns found for plotting.")

print(f"\nSaved figures to: {outdir}/")
