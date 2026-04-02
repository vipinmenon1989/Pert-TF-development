#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

# ---- CONFIG ----
INPUT  = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_epoch_e44/adata_best_validation.h5ad"
OUTPUT = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_epoch_e44/adata_best_validation_withNEXT.h5ad"
N_SPLITS = 5
RANDOM_STATE = 42
MAX_FEATURES = 128     # PCA target if we use expression matrix
USE_CALIBRATION = True # Isotonic calibration after LR per fold

def get_next_features(adata):
    """
    Prefer compact NEXT embedding if present, else PCA on mvc_next_expr.
    Returns (X_features, feature_name).
    """
    if "X_scGPT_next" in adata.obsm:
        X = np.asarray(adata.obsm["X_scGPT_next"], dtype=float)
        return X, "X_scGPT_next"
    elif "mvc_next_expr" in adata.obsm:
        X = np.asarray(adata.obsm["mvc_next_expr"], dtype=float)
        # PCA to something manageable
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=min(MAX_FEATURES, Xs.shape[1]), random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xs)
        return Xp, "mvc_next_expr(PCA)"
    else:
        raise KeyError("No NEXT features found. Need one of: .obsm['X_scGPT_next'] or .obsm['mvc_next_expr'].")

def main():
    adata = ad.read_h5ad(INPUT)
    if "genotype_next" not in adata.obs:
        raise KeyError("Missing obs['genotype_next'] (true NEXT labels).")

    y_labels = adata.obs["genotype_next"].astype(str).to_numpy()
    classes = sorted(pd.unique(y_labels))
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    y = np.array([cls_to_idx[c] for c in y_labels], dtype=int)

    X, src = get_next_features(adata)
    print(f"[INFO] Features: {src}  shape={X.shape}  classes={len(classes)}")

    # Out-of-fold probabilities
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    P = np.zeros((X.shape[0], len(classes)), dtype=float)

    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        base = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=500,
            n_jobs=-1,
            random_state=RANDOM_STATE + k
        )
        if USE_CALIBRATION:
            # calibrated probabilities can improve PR/ROC quality
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            clf = base

        clf.fit(Xtr, y[tr])
        P[te, :] = clf.predict_proba(Xte)

        # optional fold logloss
        try:
            ll = log_loss(y[te], P[te, :], labels=np.arange(len(classes)))
            print(f"[fold {k}] logloss={ll:.4f}")
        except Exception:
            pass

    # Argmax labels for convenience
    yhat = P.argmax(axis=1)
    pred_labels = np.array([classes[i] for i in yhat], dtype=object)

    # Save into AnnData for plotting script
    adata.obsm["X_pertNEXT_pred_probs"] = P
    adata.obs["predicted_genotypeNEXT"] = pd.Categorical(pred_labels, categories=classes, ordered=False)

    # Also stash class order so your script doesn't guess
    if "genotypeNEXT_class_order" not in adata.uns:
        adata.uns["genotypeNEXT_class_order"] = list(classes)

    # Write out
    outdir = os.path.dirname(OUTPUT)
    os.makedirs(outdir, exist_ok=True)
    adata.write_h5ad(OUTPUT)
    print(f"[WRITE] {OUTPUT}")
    print("[OK] You can now run your ROC/PR/UMAP script on this file to get proper genotypeNEXT curves.")

if __name__ == "__main__":
    main()
