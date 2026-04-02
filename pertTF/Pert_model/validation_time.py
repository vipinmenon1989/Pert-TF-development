#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Timepoint-stratified validation:

For each timepoint (obs['time']):

- UMAPs:
    * celltype (true)
    * genotype (true)
    * predicted_celltype
    * predicted_genotype

- ROC + PR curves:
    * celltype vs X_cls_pred_probs
    * genotype vs X_pert_pred_probs

- Per-class metrics:
    * precision, recall, F1
    * AUC(ROC), AP(AUPR)
    * one-vs-rest accuracy
    * overall accuracy (per timepoint, repeated for each class row)

Outputs:
    - Vector PDFs (+ PNGs) into OUTDIR
    - metrics_celltype_by_time.csv
    - metrics_genotype_by_time.csv
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)

# --------------------
# CONFIG – EDIT THESE
# --------------------
INPUT_PATH = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_2/best_epoch_e36/adata_with_collapsed_preds.h5ad"
#OUTDIR     = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1/best_epoch_e36/VAL_PDFS_BY_TIME"
OUTDIR     = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_2/best_epoch_e36/VAL_PDFS_BY_TIME_NEXT"
# keys in obsm
CELLTYPE_PROB_KEY        = "X_cls_pred_probs"
GENOTYPE_PROB_KEY        = "X_pert_pred_probs"
GENOTYPE_NEXT_PROB_KEY   = "X_pertNEXT_pred_probs"     # added


# label columns
CELLTYPE_TRUTH_COL   = "celltype_2"
GENOTYPE_TRUTH_COL   = "genotype"
CELLTYPE_PRED_COL    = "predicted_celltype_collapsed"
GENOTYPE_PRED_COL    = "predicted_genotype_collapsed"
GENOTYPE_NEXT_TRUTH_COL  = "genotype_next"
GENOTYPE_NEXT_PRED_COL  = "predicted_genotypeNEXT_collapsed"   # added
# time column – prefer 'time', else fallback to 'time_point'
TIME_COL_PRIMARY   = "time"
TIME_COL_FALLBACK  = "time_point"

# embedding keys
UMAP_KEY = "X_umap_scgpt"   # if missing, will compute from REP_KEY
REP_KEY  = "X_scGPT"

os.makedirs(OUTDIR, exist_ok=True)

# --------------------
# Small utils
# --------------------
def save_pdf_png(fig, outbase):
    pdf = f"{outbase}.pdf"
    png = f"{outbase}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {pdf}")

def pick_palette(n):
    base = plt.get_cmap("tab20").colors
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def ensure_cat_series(x, categories=None):
    s = x.astype(str) if isinstance(x, pd.Series) else pd.Series(np.asarray(x).astype(str))
    if categories is None:
        categories = sorted(pd.unique(s))
    return s.astype(pd.CategoricalDtype(categories=categories, ordered=False))

def get_umap(adata, use_key=UMAP_KEY, rep_key=REP_KEY, seed=0):
    # If UMAP already present, use it
    if use_key in adata.obsm:
        return adata.obsm[use_key]
    # Fallback: if vanilla X_umap exists, just alias it
    if "X_umap" in adata.obsm:
        adata.obsm[use_key] = adata.obsm["X_umap"].copy()
        return adata.obsm[use_key]
    # Else compute from REP_KEY
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{rep_key}' to compute UMAP.")
    sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=12, key_added="scgpt", random_state=seed)
    sc.tl.umap(adata, neighbors_key="scgpt", random_state=seed, min_dist=0.5)
    adata.obsm[use_key] = adata.obsm["X_umap"].copy()
    return adata.obsm[use_key]

def sanitize_for_fname(x):
    x = str(x)
    x = x.replace(" ", "_")
    x = re.sub(r"[^0-9A-Za-z_\-\.]", "_", x)
    return x

def plot_umap_category_with_mask(adata, umap_coords, category, mask, title, outbase):
    """
    Plot UMAP for a subset of cells (mask) colored by `category`.
    """
    # subset
    coords_sub = umap_coords[mask]
    labels_sub = adata.obs[category].iloc[mask]

    cat_s = ensure_cat_series(labels_sub)
    classes = list(cat_s.cat.categories)
    colors = pick_palette(len(classes))

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    X = coords_sub
    for i, cls in enumerate(classes):
        idx = (cat_s == cls).to_numpy()
        if idx.sum() == 0:
            continue
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=6,
            alpha=0.8,
            label=str(cls),
            linewidths=0,
            c=[colors[i]],
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(markerscale=2.0, fontsize=8, loc="best", ncol=1, frameon=False)
    save_pdf_png(fig, outbase)

# --------------------
# Recover head order globally (no time split)
# --------------------
def recover_head_order_from_probs(adata, prob_key: str, label_col: str):
    """
    Align probability columns to the observed labels in label_col
    via Hungarian assignment on mean probabilities.

    Returns:
        class_order (list of old label names)
        probs_ordered (N x C matrix with columns aligned to class_order)
    """
    if prob_key not in adata.obsm:
        raise KeyError(f"obsm['{prob_key}'] missing")

    y_prob = np.asarray(adata.obsm[prob_key], dtype=float)
    labels = pd.Series(adata.obs[label_col].astype(str).values)
    classes = sorted(labels.unique())
    C = len(classes)
    J = y_prob.shape[1]

    S = np.zeros((C, J), dtype=float)
    for i, cls in enumerate(classes):
        idx = (labels == cls).values
        if idx.any():
            S[i, :] = y_prob[idx, :].mean(axis=0)

    row_ind, col_ind = linear_sum_assignment(-S)
    col_for_class = np.full(C, -1, dtype=int)
    for i, j in zip(row_ind, col_ind):
        if 0 <= i < C:
            col_for_class[i] = j
    for i in range(C):
        if col_for_class[i] < 0:
            col_for_class[i] = int(np.argmax(S[i, :]))

    y_prob_ordered = y_prob[:, col_for_class]
    return list(classes), y_prob_ordered

# --------------------
# Multiclass ROC/PR + per-class metrics (assumes probs already aligned)
# --------------------
def curves_multiclass_aligned(y_true_labels, y_prob, class_names, prefix, outdir):
    """
    y_true_labels : (N,) ground-truth class names (strings)
    y_prob        : (N, C) probs, already aligned to class_names
    class_names   : list of length C
    prefix        : string used in filenames
    outdir        : where to save plots

    Returns:
        df_metrics (per-class metrics DataFrame)
        y_pred_idx (hard argmax indices)
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2:
        raise ValueError(f"[{prefix}] y_prob must be 2D, got {y_prob.shape}")
    C = y_prob.shape[1]

    if len(class_names) != C:
        raise ValueError(
            f"[{prefix}] class_names length ({len(class_names)}) != prob width ({C}); "
            f"this function assumes probs are already aligned."
        )

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    y_idx = np.array([name_to_idx.get(str(lbl), -1) for lbl in y_true_labels], dtype=int)
    keep = y_idx >= 0
    y_idx = y_idx[keep]
    y_prob = y_prob[keep]

    if len(y_idx) == 0:
        print(f"[{prefix}] No valid rows after label filtering; skipping.")
        return None, None

    # Hard preds
    y_pred_idx = y_prob.argmax(axis=1)

    # Overall accuracy
    accuracy_overall = (y_pred_idx == y_idx).mean()

    # One-vs-rest per-class accuracy
    acc_ovr = []
    for cls in range(C):
        pos = (y_idx == cls)
        neg = ~pos
        tp = ((y_pred_idx == cls) & pos).sum()
        tn = ((y_pred_idx != cls) & neg).sum()
        acc_cls = (tp + tn) / len(y_idx)
        acc_ovr.append(acc_cls)

    # Binarize for curves
    y_true_bin = label_binarize(y_idx, classes=np.arange(C))
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)

    fpr, tpr, roc_auc = {}, {}, {}
    prec_curve, rec_curve, ap = {}, {}, {}
    for i in range(C):
        pos = int(y_true_bin[:, i].sum())
        if pos == 0:
            fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), np.nan
            prec_curve[i], rec_curve[i], ap[i] = np.array([1.0]), np.array([0.0]), np.nan
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec_curve[i], rec_curve[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")

    valid_aucs = [roc_auc[i] for i in range(C) if not np.isnan(roc_auc[i])]
    valid_aps  = [ap[i]      for i in range(C) if not np.isnan(ap[i])]
    roc_auc_macro = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    ap_macro      = float(np.mean(valid_aps))  if valid_aps  else np.nan

    # Per-class PR/REC/F1 via hard preds
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        y_idx, y_pred_idx, labels=np.arange(C), average=None, zero_division=0
    )

    # Plots
    def save(fig, name): save_pdf_png(fig, os.path.join(outdir, name))

    # ROC
    fig = plt.figure(figsize=(8, 7))
    for i, cname in enumerate(class_names):
        if np.isnan(roc_auc.get(i, np.nan)):
            continue
        plt.plot(fpr[i], tpr[i], lw=1.2, label=f"{cname} (AUC={roc_auc[i]:.3f})", alpha=0.9)
    plt.plot(fpr_micro, tpr_micro, lw=2.0, linestyle="--", label=f"micro (AUC={roc_auc_micro:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {prefix} (micro={roc_auc_micro:.3f}, macro={roc_auc_macro:.3f})")
    plt.legend(fontsize=8, loc="lower right")
    save(fig, f"{prefix}_roc")

    # PR
    baseline = float(np.mean(y_true_bin))
    fig = plt.figure(figsize=(8, 7))
    for i, cname in enumerate(class_names):
        if np.isnan(ap.get(i, np.nan)):
            continue
        plt.plot(rec_curve[i], prec_curve[i], lw=1.2, label=f"{cname} (AP={ap[i]:.3f})", alpha=0.9)
    plt.plot([0, 1], [baseline, baseline], "k--", lw=1, label=f"baseline={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR — {prefix} (micro={ap_micro:.3f}, macro={ap_macro:.3f})")
    plt.legend(fontsize=8, loc="lower left")
    save(fig, f"{prefix}_pr")

    print(
        f"[{prefix}] micro-AUC={roc_auc_micro:.4f}, macro-AUC={roc_auc_macro:.4f} | "
        f"micro-AP={ap_micro:.4f}, macro-AP={ap_macro:.4f}"
    )

    df = pd.DataFrame({
        "class": class_names,
        "support": support_c.astype(int),
        "precision": prec_c,
        "recall": rec_c,
        "f1": f1_c,
        "auc_roc": [roc_auc.get(i, np.nan) for i in range(C)],
        "ap":      [ap.get(i, np.nan) for i in range(C)],
        "accuracy_one_vs_rest": acc_ovr,
        "accuracy_overall": [accuracy_overall] * C,
    })
    df["class"] = df["class"].astype(str)
    return df, y_pred_idx

# --------------------
# MAIN (time-stratified)
# --------------------
def main():
    adata = ad.read_h5ad(INPUT_PATH)
    print(f"[LOAD] AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    # figure out time column
    if TIME_COL_PRIMARY in adata.obs.columns:
        time_col = TIME_COL_PRIMARY
    elif TIME_COL_FALLBACK in adata.obs.columns:
        time_col = TIME_COL_FALLBACK
    else:
        raise KeyError(f"Neither '{TIME_COL_PRIMARY}' nor '{TIME_COL_FALLBACK}' found in obs.")

    # sanity keys
    for k in [CELLTYPE_TRUTH_COL, GENOTYPE_TRUTH_COL, GENOTYPE_NEXT_TRUTH_COL,CELLTYPE_PRED_COL,GENOTYPE_PRED_COL,GENOTYPE_NEXT_PRED_COL]:
        if k not in adata.obs.columns:
            raise KeyError(f"Missing obs column: {k}")
    for k in [CELLTYPE_PROB_KEY, GENOTYPE_PROB_KEY, GENOTYPE_NEXT_PROB_KEY]:
        if k not in adata.obsm:
            raise KeyError(f"Missing probability matrix in .obsm: {k}")

    # global UMAP
    seed_val = int(adata.uns.get("seed", 0)) if isinstance(adata.uns.get("seed", 0), (int, float)) else 0
    umap_coords = get_umap(adata, use_key=UMAP_KEY, rep_key=REP_KEY, seed=seed_val)

    # global head alignment (celltype & genotype)
    ct_classes, P_ct = recover_head_order_from_probs(
        adata, prob_key=CELLTYPE_PROB_KEY, label_col=CELLTYPE_PRED_COL
    )
    gt_classes, P_gt = recover_head_order_from_probs(
        adata, prob_key=GENOTYPE_PROB_KEY, label_col=GENOTYPE_PRED_COL
    )
    # global alignment: genotype NEXT
    gt_next_classes, P_gt_next = recover_head_order_from_probs(adata,prob_key=GENOTYPE_NEXT_PROB_KEY,label_col=GENOTYPE_NEXT_PRED_COL)

    times = sorted(pd.unique(adata.obs[time_col].astype(str)))
    print(f"[INFO] Found {len(times)} timepoints in '{time_col}': {times}")

    all_ct_rows = []
    all_gt_rows = []
    all_gt_next_rows = []

    for t in times:
        mask = (adata.obs[time_col].astype(str) == t).values
        n_tp = int(mask.sum())
        if n_tp < 5:
            print(f"[WARN] time={t}: only {n_tp} cells; skipping metrics/plots.")
            continue

        t_safe = sanitize_for_fname(t)
        tp_dir = os.path.join(OUTDIR, t_safe)  # Create subdirectory
        os.makedirs(tp_dir, exist_ok=True)

        print(f"\n[time={t}] n_cells={n_tp} -> saving to {tp_dir}")

        # ------- UMAPs for this timepoint -------
        plot_umap_category_with_mask(adata, umap_coords, CELLTYPE_TRUTH_COL, mask,title=f"UMAP — Celltype (true) — time={t}",outbase=os.path.join(tp_dir, f"umap_celltype_true_time-{t_safe}"))
        plot_umap_category_with_mask(adata, umap_coords, GENOTYPE_TRUTH_COL, mask,title=f"UMAP — Genotype (true) — time={t}",outbase=os.path.join(tp_dir, f"umap_genotype_true_time-{t_safe}"))
        plot_umap_category_with_mask(adata, umap_coords, CELLTYPE_PRED_COL, mask,title=f"UMAP — Celltype (predicted, collapsed) — time={t}",outbase=os.path.join(tp_dir, f"umap_celltype_pred_collapsed_time-{t_safe}"))
        plot_umap_category_with_mask(adata, umap_coords, GENOTYPE_PRED_COL, mask,title=f"UMAP — Genotype (predicted) — time={t}",outbase=os.path.join(tp_dir, f"umap_genotype_pred_time-{t_safe}"))

        # ------- Metrics: Celltype -------
        y_true_ct_t = adata.obs[CELLTYPE_TRUTH_COL].astype(str).values[mask]
        P_ct_t = P_ct[mask, :]
        prefix_ct = os.path.join(tp_dir, f"celltype_time-{t_safe}")
        df_ct_t, _ = curves_multiclass_aligned(y_true_ct_t, P_ct_t, ct_classes, prefix_ct, tp_dir)
        if df_ct_t is not None:
            df_ct_t["timepoint"] = t
            all_ct_rows.append(df_ct_t)

        # ------- Metrics: genotype -------
        y_true_gt_t = adata.obs[GENOTYPE_TRUTH_COL].astype(str).values[mask]
        P_gt_t = P_gt[mask, :]
        prefix_gt = os.path.join(tp_dir, f"genotype_time-{t_safe}")
        df_gt_t, _ = curves_multiclass_aligned(y_true_gt_t, P_gt_t, gt_classes, prefix_gt, tp_dir)
        if df_gt_t is not None:
            df_gt_t["timepoint"] = t
            all_gt_rows.append(df_gt_t)

        # ------- Metrics: genotype NEXT -------
        y_true_gt_next_t = adata.obs[GENOTYPE_NEXT_TRUTH_COL].astype(str).values[mask]
        P_gt_next_t = P_gt_next[mask, :]
        prefix_gt_next = os.path.join(tp_dir, f"genotypeNEXT_time-{t_safe}")
        df_gt_next_t, _ = curves_multiclass_aligned(y_true_gt_next_t,P_gt_next_t,gt_next_classes,prefix_gt_next,tp_dir)

        if df_gt_next_t is not None:
            df_gt_next_t["timepoint"] = t
            all_gt_next_rows.append(df_gt_next_t)

        # ------- Write CSVs across all timepoints -------
    if all_ct_rows:
        df_ct = pd.concat(all_ct_rows, ignore_index=True)
        ct_csv = os.path.join(OUTDIR, "metrics_celltype_by_time.csv")
        df_ct.to_csv(ct_csv, index=False)
        print(f"[WRITE] {ct_csv}")

    if all_gt_rows:
        df_gt = pd.concat(all_gt_rows, ignore_index=True)
        gt_csv = os.path.join(OUTDIR, "metrics_genotype_by_time.csv")
        df_gt.to_csv(gt_csv, index=False)
        print(f"[WRITE] {gt_csv}")
    
    if all_gt_next_rows:
        df_gt_next = pd.concat(all_gt_next_rows, ignore_index=True)
        gt_next_csv = os.path.join(OUTDIR, "metrics_genotypeNEXT_by_time.csv")
        df_gt_next.to_csv(gt_next_csv, index=False)
        print(f"[WRITE] {gt_next_csv}")

    print("[DONE] Timepoint-stratified PDFs, PNGs, and CSVs written to:", OUTDIR)


if __name__ == "__main__":
    main()
