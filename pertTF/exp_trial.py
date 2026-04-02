import anndata as ad

adata_eval = ad.read_h5ad("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_epoch_e44/adata_best_validation.h5ad")

print("OBS COLS:", list(adata_eval.obs.columns)[:30])
print("Has genotype_next?:", "genotype_next" in adata_eval.obs.columns)

print("LAYERS:", list(adata_eval.layers.keys()))
print("OBSM  :", list(adata_eval.obsm.keys())[:30])
print("OBSM shapes:", {k: getattr(v, 'shape', None) for k,v in adata_eval.obsm.items()})
