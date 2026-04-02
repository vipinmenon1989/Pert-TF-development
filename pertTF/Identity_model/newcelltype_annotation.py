import pandas as pd
import anndata as ad

# === paths ===
H5AD_IN  = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/fold_5/best_epoch_e34/adata_best_validation.h5ad"   # change to your file
CSV_META = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/metadata_celltype_celltype2_barcode.csv"  # the CSV we created from Seurat
H5AD_OUT = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/fold_5/best_epoch_e34/adata_best_validation_updated.h5ad"

# === load data ===
adata = ad.read_h5ad(H5AD_IN)

# Robust CSV read: headerless with explicit names
df = pd.read_csv(
    CSV_META,
    header=None,
    names=["celltype", "celltype_2", "barcode"],
    dtype=str
)

# Basic sanity: drop duplicate barcodes in CSV (keep first)
if df["barcode"].duplicated().any():
    dup_n = df["barcode"].duplicated().sum()
    print(f"[WARN] Found {dup_n} duplicated barcodes in CSV. Keeping first occurrence.")
    df = df[~df["barcode"].duplicated(keep="first")].copy()

# Index by barcode and align to adata.obs_names
df = df.set_index("barcode")

# Sanity check: overlap
overlap = df.index.intersection(adata.obs_names)
if len(overlap) == 0:
    raise ValueError(
        "No overlap between CSV barcodes and adata.obs_names. "
        "Check that barcodes match exactly (case, suffixes like '-1', etc.)."
    )

# Reindex to adata order (introduces NaN for missing)
df_aligned = df.reindex(adata.obs_names)

# Optional: if you want to see how many cells get values
matched = df_aligned["celltype_2"].notna().sum()
print(f"[INFO] Matched celltype_2 for {matched} / {adata.n_obs} cells")

# Make pandas categoricals to keep .h5ad size sane
for col in ["celltype", "celltype_2"]:
    if col in df_aligned.columns:
        # Replace NaN with empty string (or a placeholder like 'NA')
        series = df_aligned[col].fillna("")
        adata.obs[col] = pd.Categorical(series)  # categorical is compact and scanpy-friendly

# Write out
adata.write(H5AD_OUT, compression="gzip")
print(f"[OK] Wrote {H5AD_OUT}")
