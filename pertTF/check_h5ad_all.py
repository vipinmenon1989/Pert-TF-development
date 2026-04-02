import scanpy as sc
import pandas as pd

# Load your file (update the path)
adata = sc.read_h5ad("object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad")

print("--- CARDINALITY SUMMARY ---")
print(f"Total Cells: {adata.n_obs}")
print(f"Total Genes: {adata.n_vars}")

print("\n--- CELLTYPE_2 DISTRIBUTION ---")
if "celltype_2" in adata.obs:
    print(adata.obs["celltype_2"].value_counts())
else:
    print("WARNING: 'celltype_2' column not found in .obs")

print("\n--- GENOTYPE DISTRIBUTION ---")
if "genotype" in adata.obs:
    print(adata.obs["genotype"].value_counts())
else:
    print("WARNING: 'genotype' column not found in .obs")
