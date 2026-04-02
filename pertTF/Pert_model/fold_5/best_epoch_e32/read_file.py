import anndata as ad
adata = ad.read_h5ad("adata_best_validation_withNEXT_from_same_head_updated.h5ad")
print (adata.obs.columns)
for col in adata.obs.columns:
    print(f"\n---- {col} ----")
    print(adata.obs[col].unique()[:50])  # limit to first 50 to avoid flooding
    print(f"Total unique: {adata.obs[col].nunique()}")

print (adata.obsm.keys())
