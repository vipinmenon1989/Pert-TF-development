import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
FILE_PATH = "adata_best_validation.h5ad"
OUTPUT_CSV = "umap_metadata_extraction.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Could not find {FILE_PATH} in current directory.")

# --- 2. LOAD DATA ---
print(f"Loading {FILE_PATH}...")
adata = sc.read_h5ad(FILE_PATH)

# --- 3. DATA EXTRACTION ---
print("Extracting coordinates and metadata...")

# Create the dataframe from .obs (Metadata)
df = adata.obs.copy()

# Add UMAP coordinates from .obsm
# Scanpy stores these in a (N_cells, 2) numpy array
df['UMAP_1'] = adata.obsm['X_umap'][:, 0]
df['UMAP_2'] = adata.obsm['X_umap'][:, 1]

# Ensure the index (Cell IDs) is a named column in the CSV
df.index.name = 'Cell_ID'

# Save to CSV
df.to_csv(OUTPUT_CSV)
print(f"✅ CSV saved successfully to: {OUTPUT_CSV}")

# --- 4. RECONSTRUCTION PLOTS ---
print("Generating verification plots...")

# Create a side-by-side figure to match your 'e98' validation images
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Cell Type
sc.pl.umap(adata, color='celltype', ax=ax[0], show=False, 
           title="Reconstructed: Cell Type", frameon=False)

# Plot 2: Genotype
sc.pl.umap(adata, color='genotype', ax=ax[1], show=False, 
           title="Reconstructed: Genotype", frameon=False)

plt.tight_layout()
plt.savefig("reconstruction_check.png", dpi=300)
print("✅ Verification plot saved to: reconstruction_check.png")
plt.show()
