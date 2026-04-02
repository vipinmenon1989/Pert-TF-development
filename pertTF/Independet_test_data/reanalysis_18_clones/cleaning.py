import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
input_path = "18clones_annotated_final.h5ad"
output_h5ad = "18clones_cleaned_final.h5ad"
output_csv = "18clones_coordinates_metadata.csv"
exclude_genotypes = ['UBA6', 'NA']

# 1. LOAD DATA
adata = sc.read_h5ad(input_path)

# 2. FILTERING
initial_count = adata.n_obs
adata = adata[~adata.obs['genotype'].isin(exclude_genotypes)].copy()
adata = adata[adata.obs['genotype'].notna()].copy()

# Refresh categories to clean up legends
adata.obs['genotype'] = adata.obs['genotype'].cat.remove_unused_categories()
adata.obs['celltype'] = adata.obs['celltype'].cat.remove_unused_categories()

# 3. EXPORT COORDINATES & METADATA TO CSV
# This gives you the actual UMAP numbers (Cell_ID, UMAP_1, UMAP_2, Celltype, Genotype)
df = adata.obs.copy()
df['UMAP_1'] = adata.obsm['X_umap'][:, 0]
df['UMAP_2'] = adata.obsm['X_umap'][:, 1]
df.to_csv(output_csv)
print(f"✅ CSV Exported: {output_csv}")

# 4. GENERATE DUAL FORMAT PLOTS (PNG & PDF)
sc.settings.figdir = "final_plots"
os.makedirs(sc.settings.figdir, exist_ok=True)

formats = ['.png', '.pdf']

for fmt in formats:
    print(f"Generating {fmt} plots...")
    sc.pl.umap(
        adata, 
        color=['celltype', 'genotype'], 
        save=f"_cleaned{fmt}", 
        show=False,
        title=["Celltype (Cleaned)", "Genotype (Cleaned)"],
        frameon=False
    )

# 5. SAVE CLEANED H5AD
adata.write(output_h5ad)
print(f"\n[COMPLETE]")
print(f"Total cells remaining: {adata.n_obs} (Removed: {initial_count - adata.n_obs})")
