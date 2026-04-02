import scanpy as sc
import pandas as pd

# 1. Load your existing h5ad file
adata = sc.read_h5ad('18clones_annotated_final.h5ad')

# 2. Remove the numeric prefixes (e.g., "44_", "33_")
adata.obs['genotype'] = (
    adata.obs['genotype']
    .astype(str)                         
    .str.replace(r'^\d+_', '', regex=True) 
    .astype('category')                  
)

# 3. Remove the 'CCDC6' genotype
print(f"Cells before filtering: {adata.n_obs}")
adata = adata[adata.obs['genotype'] != 'CCDC6'].copy()
print(f"Cells after removing CCDC6: {adata.n_obs}")

# 4. Verify the change
print("Remaining genotypes:", adata.obs['genotype'].unique())

# 5. Save to a new file
adata.write('18clones_annotated_final_cleaned.h5ad')
