#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import anndata as ad

# ---- INPUT / OUTPUT ----
INPUT  = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_5/best_epoch_e32/adata_best_validation.h5ad"
OUTPUT = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_5/best_epoch_e32/adata_best_validation_withNEXT_from_same_head.h5ad"

# keys we will use
PERT_PROB_KEY    = "X_pert_pred_probs"        # existing “now” genotype probs (the one that produced the PNGs)
NEXT_PROB_KEY    = "X_pertNEXT_pred_probs"    # we will create this
NEXT_LABEL_TRUE  = "genotype_next"            # ground-truth NEXT labels
PRED_GENOTYPE    = "predicted_genotype"       # to recover the head's class ordering
PRED_NEXT_LABEL  = "predicted_genotypeNEXT"   # we will create this too

def ensure_cat_series(x, categories=None):
    """
    Return a pandas Series with categorical dtype and stable categories.
    Works whether x is Series or array-like.
    """
    s = x.astype(str) if isinstance(x, pd.Series) else pd.Series(np.asarray(x).astype(str))
    if categories is None:
        categories = sorted(pd.unique(s))
    return s.astype(pd.CategoricalDtype(categories=categories, ordered=False))

def main():
    adata = ad.read_h5ad(INPUT)

    # --- sanity checks
    if PERT_PROB_KEY not in adata.obsm:
        raise KeyError(f"Missing obsm['{PERT_PROB_KEY}']")
    if NEXT_LABEL_TRUE not in adata.obs.columns:
        raise KeyError(f"Missing obs['{NEXT_LABEL_TRUE}']")
    if PRED_GENOTYPE not in adata.obs.columns:
        raise KeyError(f"Missing obs['{PRED_GENOTYPE}'] (needed to recover head class ordering)")

    # 1) Exact class ordering used by the genotype head
    pred_geno_cat = ensure_cat_series(adata.obs[PRED_GENOTYPE])
    class_order = list(pred_geno_cat.cat.categories)

    # 2) Copy the SAME probability matrix into a NEXT slot
    P = np.asarray(adata.obsm[PERT_PROB_KEY], dtype=float)
    # normalize rows just in case
    row_sum = P.sum(axis=1, keepdims=True)
    np.divide(P, np.where(row_sum == 0.0, 1.0, row_sum), out=P)

    # If for any reason prob width != class list length, fix it deterministically.
    C = P.shape[1]
    if len(class_order) < C:
        # pad unseen tails with synthetic names so indices stay aligned
        class_order = class_order + [f"__unused_{i}" for i in range(len(class_order), C)]
    elif len(class_order) > C:
        # trim to the head width
        class_order = class_order[:C]

    adata.obsm[NEXT_PROB_KEY] = P

    # 3) Create predicted_genotypeNEXT via argmax on the SAME probs
    hard_idx = P.argmax(axis=1)
    hard_labels = [class_order[i] for i in hard_idx]
    adata.obs[PRED_NEXT_LABEL] = pd.Categorical(hard_labels, categories=class_order, ordered=False)

    # 4) Stash class-order hints for downstream plotting
    adata.uns["genotypeNEXT_class_order"] = list(class_order)
    # keep these too for any downstream script that checks them
    adata.uns["pert_class_order"] = list(class_order)
    adata.uns["genotype_class_order"] = list(class_order)

    # 5) Write out
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    adata.write_h5ad(OUTPUT)
    print(f"[WRITE] {OUTPUT}")
    print("[OK] Use THIS file as INPUT_PATH for your ROC/PR/UMAP script. "
          "NEXT curves now read from X_pertNEXT_pred_probs with the same head class order, "
          "so your PDFs should match the original PNG tallies.")

if __name__ == "__main__":
    main()
