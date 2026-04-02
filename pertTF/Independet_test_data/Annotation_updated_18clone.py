import scanpy as sc
import pandas as pd
import os

# CONFIG
h5ad_input = "18clones_seurat.h5ad"
annotation_path = "Annotation.csv"
h5ad_output = "18clones_annotated_final.h5ad"

# 1. LOAD DATA
adata = sc.read_h5ad(h5ad_input)
ann_df = pd.read_csv(annotation_path)

# 2. COLUMN RENAMING & CLEANING (Genotype)
if 'gene' in adata.obs.columns:
    # Rename first
    adata.obs.rename(columns={'gene': 'genotype'}, inplace=True)
    
    # Logic: Split string by '_' and keep the part after the first underscore
    # We use .str.split('_').str[1] to extract 'CCD6' from '46_CCD6'
    adata.obs['genotype'] = adata.obs['genotype'].astype(str).str.split('_').str[1]
    
    print("Renamed 'gene' to 'genotype' and cleaned string prefixes.")
elif 'genotype' in adata.obs.columns:
    # If already named genotype but needs cleaning
    adata.obs['genotype'] = adata.obs['genotype'].astype(str).str.split('_').str[1]
    print("Cleaned 'genotype' string prefixes.")
else:
    print("[!] Neither 'gene' nor 'genotype' column found. Skipping cleaning.")

# 3. ANNOTATION MAPPING
# Ensure seurat_clusters exists in both
if 'seurat_clusters' not in adata.obs.columns:
    raise KeyError("Column 'seurat_clusters' not found in adata.obs.")

# Force string type for robust matching
adata.obs['seurat_clusters'] = adata.obs['seurat_clusters'].astype(str)
ann_df['seurat_clusters'] = ann_df['seurat_clusters'].astype(str)

# Determine the source column in CSV (case-insensitive check)
csv_celltype_col = None
for col in ['celltype', 'Celltype', 'cell_type']:
    if col in ann_df.columns:
        csv_celltype_col = col
        break

if csv_celltype_col is None:
    raise KeyError(f"Could not find a celltype column in {annotation_path}")

# Create mapping and apply to 'celltype' column
mapping = dict(zip(ann_df['seurat_clusters'], ann_df[csv_celltype_col]))
adata.obs['celltype'] = adata.obs['seurat_clusters'].map(mapping)

# Ensure categorical for visualization
adata.obs['celltype'] = adata.obs['celltype'].astype('category')

# ---------------------------------------------------------
# 4. INSPECTION & VERIFICATION
# ---------------------------------------------------------
print("\n=== UPDATED ADATA.OBS COLUMNS ===")
print(adata.obs.columns.tolist())

print("\n=== CELLTYPE DISTRIBUTION ===")
print(adata.obs['celltype'].value_counts())

# Added Genotype Distribution
if 'genotype' in adata.obs.columns:
    print("\n=== GENOTYPE DISTRIBUTION ===")
    print(adata.obs['genotype'].value_counts())
else:
    print("\n[!] 'genotype' column missing; cannot print distribution.")

# Check for unmapped cells
missing = adata.obs['celltype'].isna().sum()
if missing > 0:
    print(f"\n[!] WARNING: {missing} cells failed to map. Check cluster IDs in CSV.")

# ---------------------------------------------------------
# 5. GENERATE UMAPs
# ---------------------------------------------------------
sc.settings.figdir = "scanpy_umap_outputs"
os.makedirs(sc.settings.figdir, exist_ok=True)

sc.pl.umap(
    adata,
    color=['seurat_clusters', 'celltype', 'genotype'],
    ncols=3,
    wspace=0.5,
    frameon=False,
    save="_final_annotation_plots.png",
    show=True
)

# 6. SAVE
adata.write(h5ad_output)
print(f"\n[COMPLETE] New file saved as: {h5ad_output}")