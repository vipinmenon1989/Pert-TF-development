import os
import scanpy as sc
import pandas as pd

# ----------------------
# CONFIG
# ----------------------
h5ad_path = "18clones_annotated_final.h5ad"

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
print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")

# ----------------------
# INSPECT CONTENTS
# ----------------------
print("\n=== adata.obs columns ===")
print(list(adata.obs.columns))

print("\n=== adata.var columns ===")
print(list(adata.var.columns))

print("\n=== adata.uns keys ===")
print(list(adata.uns.keys()))

print("\n=== adata.obsm keys (embeddings like X_umap, X_pca) ===")
print(list(adata.obsm.keys()))

print("\n=== adata.obsp keys (graphs/distances/connectivities) ===")
print(list(adata.obsp.keys()))

print("\n=== adata.layers keys ===")
print(list(adata.layers.keys()))

print("\n=== adata.X info ===")
print(type(adata.X), getattr(adata.X, "shape", None))
print(f"adata.raw present? {adata.raw is not None}")

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
celltype_candidates = [
    "celltype_2", "cell_type", "CellType", "celltypes", "annotation", "annot", "cell_type_final"
]

cluster_col = cluster_col_override or pick_col(adata.obs, cluster_candidates)
celltype_col = celltype_col_override or pick_col(adata.obs, celltype_candidates)

def suggest_cols(obs: pd.DataFrame, needle: str):
    needle = needle.lower()
    hits = [c for c in obs.columns if needle in c.lower()]
    return hits[:30]

if cluster_col is None:
    print("\n[!] Could not auto-detect a cluster column.")
    print("    Suggestions containing 'leid', 'louv', 'clust':")
    print("    ", suggest_cols(adata.obs, "leid") + suggest_cols(adata.obs, "louv") + suggest_cols(adata.obs, "clust"))

if celltype_col is None:
    print("\n[!] Could not auto-detect a celltype column.")
    print("    Suggestions containing 'cell', 'type', 'annot':")
    print("    ", suggest_cols(adata.obs, "cell") + suggest_cols(adata.obs, "type") + suggest_cols(adata.obs, "annot"))

print(f"\nUsing cluster_col = {cluster_col}")
print(f"Using celltype_col = {celltype_col}")

# Make sure they are categorical (prettier legends + stable coloring)
for col in [cluster_col, celltype_col]:
    if col is not None and not pd.api.types.is_categorical_dtype(adata.obs[col]):
        adata.obs[col] = adata.obs[col].astype("category")

# ----------------------
# UMAP: use existing or compute
# ----------------------
if "X_umap" in adata.obsm_keys():
    print("\nUMAP embedding found in adata.obsm['X_umap'] -> will plot directly.")
else:
    print("\nNo UMAP found -> computing neighbors+UMAP.")
    # Minimal, reasonably safe default pipeline
    # If you already have X_pca, we can reuse it.
    if "X_pca" not in adata.obsm_keys():
        # If your object is already normalized/logged, this still usually works,
        # but it *does* change adata.X. Comment these 3 lines if you want zero-touch.
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
if not colors:
    raise ValueError("No cluster/celltype columns found. Set cluster_col_override/celltype_col_override to correct obs column names.")

# Two-panel UMAP (cluster + celltype)
sc.pl.umap(
    adata,
    color=colors,
    ncols=min(2, len(colors)),
    wspace=0.35,
    frameon=False,
    show=True,
    save="_cluster_celltype.png"
)

# Optionally: put cluster labels on the embedding (works best with <= ~30 clusters)
if cluster_col is not None:
    sc.pl.umap(
        adata,
        color=cluster_col,
        legend_loc="on data",
        frameon=False,
        show=True,
        save="_cluster_ondata.png"
    )

print(f"\nSaved figures to: {outdir}/ (Scanpy appends names under sc.settings.figdir)")

