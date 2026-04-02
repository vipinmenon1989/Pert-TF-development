import scanpy as sc
ad = sc.read_h5ad("object_integrated_assay3_annotated_final.cleaned.updated.h5ad")
for k in ["celltype", "celltype_2", "celltype_2_old", "sub.cluster"]:
    if k in ad.obs:
        print(k, "nunique=", ad.obs[k].astype(str).nunique())
    else:
        print(k, "MISSING")
