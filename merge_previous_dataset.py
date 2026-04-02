import anndata as ad
import pandas as pd

# --- input files ---
A_PATH = "object_integrated_assay3_annotated_nounk_raw.cleaned.h5ad"
B_PATH = "object_integrated_assay3_annotated_nounk_corrected_KO_matched_WT_July10.h5ad"
OUT    = "object_integrated_assay3_annotated_final.cleaned.h5ad"

# --- load ---
a = ad.read_h5ad(A_PATH)
b = ad.read_h5ad(B_PATH)

# --- sanity checks ---
if "celltype_2" not in b.obs.columns:
    raise KeyError("`celltype_2` not found in b.obs")

# Cells we want: exactly B's barcodes, but keep only those that exist in A
barcodes_b = pd.Index(b.obs_names)
keep = barcodes_b[barcodes_b.isin(a.obs_names)]  # preserves B's order

missing_in_a = barcodes_b.difference(keep)
if len(missing_in_a) > 0:
    print(f"Warning: {len(missing_in_a)} cells in B not found in A (they'll be dropped).")

# Subset A to those cells, preserving B's order
new = a[keep].copy()

# Bring over B's celltype_2 aligned to the new obs order
new.obs["celltype_2"] = b.obs.loc[new.obs_names, "celltype_2"].astype(b.obs["celltype_2"].dtype)

# Optional: if you want to ensure categorical levels are tidy
if pd.api.types.is_categorical_dtype(new.obs["celltype_2"]):
    new.obs["celltype_2"] = new.obs["celltype_2"].cat.remove_unused_categories()

# --- write ---
new.write_h5ad(OUT, compression="gzip")
print(f"Done. Wrote {new.n_obs} cells and {new.n_vars} genes to: {OUT}")
