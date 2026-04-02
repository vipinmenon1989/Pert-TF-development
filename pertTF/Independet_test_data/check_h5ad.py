#!/usr/bin/env python3
import scanpy as sc
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_h5ad.py <file.h5ad>")
    sys.exit(1)

path = sys.argv[1]
print(f"\n=== Inspecting {path} ===")

# --- Basic summary ---
adata = sc.read_h5ad(path)
print(f"Cells (n_obs): {adata.n_obs:,}")
print(f"Genes (n_vars): {adata.n_vars:,}")
print("Unique cell types:", adata.obs["cell_type_annotation"].nunique())
print("Cell type names:", adata.obs["cell_type_annotation"].unique().tolist())
# --- obs (cell metadata) ---
print("\n--- adata.obs columns ---")
print(list(adata.obs.columns))

if "celltype_2" in adata.obs:
    print(f"\n[celltype_2] unique count: {adata.obs['celltype_2'].nunique()}")
    print(f"Top 10 values:\n{adata.obs['celltype_2'].value_counts().head(10)}")
    print(f"\n[celltype] unique count: {adata.obs['celltype'].nunique()}")
    print(f"Top 10 values:\n{adata.obs['celltype'].value_counts().head(10)}")
elif "cell_type_annotation" in adata.obs:
    print(f"\n[celltype] unique count: {adata.obs['celltype'].nunique()}")
    print(f"Top 10 values:\n{adata.obs['celltype'].value_counts().head(10)}")
else:
    print("\nNo 'celltype' or 'celltype_2' column found!")

if "gene" in adata.obs:
    print(f"\n[gene] unique count: {adata.obs['gene'].nunique()}")
    print(f"Top 10 values:\n{adata.obs['gene'].value_counts().head(10)}")
else:
    print("\nNo 'gene' column in obs!")

if "genotype" in adata.obs:
    print(f"\n[genotype] unique count: {adata.obs['genotype'].nunique()}")
    print(f"Top 10 values:\n{adata.obs['genotype'].value_counts().head(10)}")
else:
    print("\nNo 'genotype' column in obs!")

# --- var (gene metadata) ---
print("\n--- adata.var columns ---")
print(list(adata.var.columns))

# --- layers ---
print("\n--- adata.layers ---")
print(list(adata.layers.keys()))

# --- misc ---
if hasattr(adata, "uns"):
    print("\n--- adata.uns keys ---")
    print(list(adata.uns.keys()))

print("\nInspection complete.\n")
