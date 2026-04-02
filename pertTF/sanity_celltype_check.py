import os
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe (HPC)

# ----------------------
# CONFIG
# ----------------------
h5ad_path = "object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad"
outdir = "scanpy_umap_celltypes_outputs"
os.makedirs(outdir, exist_ok=True)

sc.settings.figdir = outdir
sc.settings.verbosity = 2

celltype_cols = ["celltype", "celltype_2"]  # force these

# ----------------------
# LOAD
# ----------------------
adata = sc.read_h5ad(h5ad_path)
print(f"\nLoaded: {h5ad_path}")
print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")

# sanity check columns exist
missing = [c for c in celltype_cols if c not in adata.obs.columns]
if missing:
    raise ValueError(f"Missing obs columns: {missing}\nAvailable: {list(adata.obs.columns)}")

# categorical for stable colors / nicer legends
for c in celltype_cols:
    if not pd.api.types.is_categorical_dtype(adata.obs[c]):
        adata.obs[c] = adata.obs[c].astype("category")

# ----------------------
# UMAP: use existing or compute
# ----------------------
if "X_umap" in adata.obsm_keys():
    print("\nUMAP embedding found -> plotting directly.")
else:
    print("\nNo UMAP found -> computing neighbors+UMAP.")
    # Prefer existing PCA if present; otherwise compute a minimal pipeline
    if "X_pca" not in adata.obsm_keys():
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata)

# ----------------------
# SAVE BOTH PNG + PDF
# ----------------------
# Scanpy saves as:  outdir/umap_<save>.png  (and .pdf when ext="pdf")
base = "_celltype_vs_celltype2"

# PNG
sc.set_figure_params(dpi=200)
sc.pl.umap(
    adata,
    color=celltype_cols,
    ncols=2,
    wspace=0.35,
    frameon=False,
    show=False,
    save=f"{base}.png",
)

# PDF (vector)
sc.pl.umap(
    adata,
    color=celltype_cols,
    ncols=2,
    wspace=0.35,
    frameon=False,
    show=False,
    save=f"{base}.pdf",
)

print(f"\nSaved figures into: {outdir}/")
print(f"Expected files: umap{base}.png and umap{base}.pdf")
