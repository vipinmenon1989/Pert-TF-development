import scanpy as sc
import pandas as pd

H5AD_PATH = "object_integrated_assay3_annotated_final.cleaned.h5ad"
CSV_PATH  = "metadata_celltype_celltype2_barcode.csv"   # no header
OUT_PATH  = "object_integrated_assay3_annotated_final.cleaned.updated.h5ad"

adata = sc.read_h5ad(H5AD_PATH)

# IMPORTANT: force strings so labels don't become ints/floats
df = pd.read_csv(CSV_PATH, header=None, dtype=str)

label_col   = 1  # column 2 in the CSV = new celltype_2
barcode_col = 2  # column 3 in the CSV = barcode / cell_id

# normalize ids + labels
df[barcode_col] = df[barcode_col].astype(str).str.strip()
df[label_col]   = df[label_col].astype(str).str.strip()
adata.obs_names = adata.obs_names.astype(str).str.strip()

# OPTIONAL: fix common 10x suffix mismatch
# def drop_10x_suffix(s): return s[:-2] if s.endswith("-1") else s
# df[barcode_col] = df[barcode_col].map(drop_10x_suffix)
# adata.obs_names = adata.obs_names.map(drop_10x_suffix)

# drop junk and duplicates
df = df.dropna(subset=[barcode_col, label_col]).drop_duplicates(subset=[barcode_col], keep="first")

# -----------------------------
# OPTIONAL AUTO-FIX:
# If "celltype_2" column is just numeric codes, use another column that looks like names.
# (Assumes there are at least 2 non-barcode columns.)
# -----------------------------
def looks_numeric_series(s: pd.Series) -> bool:
    s2 = s.dropna().astype(str).str.strip()
    if s2.empty:
        return False
    # if almost everything is digits / floats / NA-ish, it's probably codes
    numericish = s2.str.fullmatch(r"[-+]?\d+(\.\d+)?").mean()
    return numericish > 0.95

if looks_numeric_series(df[label_col]):
    # try alternative label column(s) except barcode_col
    candidate_label_cols = [c for c in df.columns if c != barcode_col and c != label_col]
    if candidate_label_cols:
        alt = candidate_label_cols[0]
        if not looks_numeric_series(df[alt]):
            print(f"[WARN] CSV label_col={label_col} looks numeric-coded; switching to alt label_col={alt}")
            label_col = alt
            df[label_col] = df[label_col].astype(str).str.strip()
        else:
            print(f"[WARN] CSV label_col={label_col} looks numeric-coded and alt col {alt} also looks numeric. Keeping {label_col}.")

# mapping: barcode -> new celltype_2
mapping = pd.Series(df[label_col].values, index=df[barcode_col]).to_dict()

target = "celltype_2"

# keep a snapshot of old values for comparison
old_vals = (
    adata.obs[target].astype(object).copy()
    if target in adata.obs.columns
    else pd.Series([None] * adata.n_obs, index=adata.obs_names)
)

# backup old labels (WRITE-SAFE)
if target in adata.obs.columns:
    adata.obs[target + "_old"] = adata.obs[target].astype(object)

# update (WRITE-SAFE: object)
new_vals = pd.Index(adata.obs_names).map(mapping)
adata.obs[target] = pd.Series(new_vals, index=adata.obs_names).astype(object)

# -----------------------------
# DEBUG: verify mapping worked
# -----------------------------
n_mapped = pd.Series(new_vals).notna().sum()
print(f"[DEBUG] Mapped (non-NA) from CSV: {n_mapped} / {adata.n_obs} ({n_mapped/adata.n_obs:.1%})")

tmp = pd.DataFrame({
    "barcode": adata.obs_names,
    "old": old_vals.reindex(adata.obs_names).astype(object).values,
    "new": adata.obs[target].astype(object).values,
}).set_index("barcode")

print("\n[DEBUG] First 15 mapped examples (old -> new):")
print(tmp[tmp["new"].notna()].head(15))

changed = (tmp["new"].notna()) & (tmp["old"].astype(str) != tmp["new"].astype(str))
print(f"\n[DEBUG] Changed among mapped cells: {changed.sum()}")

print("\n[DEBUG] Top labels BEFORE (old):")
print(pd.Series(old_vals.astype(object)).value_counts(dropna=False).head(10))

print("\n[DEBUG] Top labels AFTER (new, before fillna):")
print(tmp["new"].value_counts(dropna=False).head(10))
# -----------------------------

# keep old values for unmatched cells (recommended)
if target + "_old" in adata.obs.columns:
    adata.obs[target] = adata.obs[target].where(adata.obs[target].notna(), adata.obs[target + "_old"])

# FINAL GUARANTEE: ensure labels are strings (prevents numeric-looking categories)
adata.obs[target] = adata.obs[target].astype(str)

# convert to category for nicer plotting + smaller on disk
adata.obs[target] = adata.obs[target].astype("category")
if target + "_old" in adata.obs.columns:
    adata.obs[target + "_old"] = adata.obs[target + "_old"].astype(str).astype("category")

# sanity
matched = pd.Index(adata.obs_names).isin(df[barcode_col]).sum()
print(f"\nCells in h5ad: {adata.n_obs}")
print(f"Unique IDs in CSV: {df[barcode_col].nunique()}")
print(f"Matched cells: {matched} ({matched/adata.n_obs:.1%})")

print("\nTop labels (FINAL in adata.obs['celltype_2']):")
print(adata.obs[target].value_counts(dropna=False).head(10))

adata.write_h5ad(OUT_PATH)
print("Saved:", OUT_PATH)