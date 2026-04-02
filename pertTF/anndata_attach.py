import pandas as pd
import scanpy as sc

# 1. Define filenames
csv_path = 'box_gate_cell_ids_celltype.csv'
original_h5ad = 'object_integrated_assay3_annotated_final.cleaned.updated.celltype2_only.h5ad'
new_h5ad = 'object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad'

# 2. Load data
df = pd.read_csv(csv_path)
adata = sc.read_h5ad(original_h5ad)

# 3. Print 'Before' counts
print("--- Celltype Counts BEFORE Update ---")
if 'celltype_2' in adata.obs.columns:
    print(adata.obs['celltype_2'].value_counts())
    
    # CRITICAL: If 'celltype' is categorical, convert to string to allow editing
    if adata.obs['celltype_2'].dtype.name == 'category':
        adata.obs['celltype_2'] = adata.obs['celltype_2'].astype(str)
else:
    print("Column 'celltype_2' does not exist yet. Creating placeholder.")
    adata.obs['celltype_2'] = 'Unknown' # Initialize with a default value

# 4. Update Logic
# Ensure both IDs are strings to avoid type mismatches (e.g., '101' vs 101)
adata.obs.index = adata.obs.index.astype(str)
target_ids = df['cell_id'].astype(str)

# Find matches
mask = adata.obs.index.isin(target_ids)

# Apply change
adata.obs.loc[mask, 'celltype_2'] = 'ESC (D3)'

# 5. Print 'After' counts
print("\n--- Celltype Counts AFTER Update ---")
print(adata.obs['celltype_2'].value_counts())

# 6. Save as NEW file
# compression='gzip' is optional but recommended to save disk space
adata.write(new_h5ad, compression='gzip') 

print(f"\nSuccessfully saved modified AnnData to: {new_h5ad}")
