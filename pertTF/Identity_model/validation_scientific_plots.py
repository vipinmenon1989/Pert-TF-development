#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation figures in one pass (with taxonomy collapse for celltype):
- Celltype: Align old-head probs -> collapse to new taxonomy -> ROC/PR + UMAPs + CSVs
- Genotype: ROC/PR + UMAPs + CSVs (unchanged)
- GenotypeNEXT: same as before

Key points:
* AUROC-based Hungarian alignment of old head to old labels
* Collapse old classes -> new taxonomy (Celltype_final / celltype_2 / etc.)
* Safe obs assignments using keep-masks
* Auto-detect new celltype column; override via env CELLTYPE_LABEL_OVERRIDE

Outputs vector PDFs (and PNGs) under OUTDIR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, roc_auc_score
)
from scipy.optimize import linear_sum_assignment  # Hungarian assignment

# --------------------
# CONFIG (EDIT THESE)
# --------------------
INPUT_PATH = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/fold_5/best_epoch_e34/adata_best_validation_updated.h5ad"
OUTDIR     = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/fold_5/best_epoch_e34/VAL_PDFS"

os.makedirs(OUTDIR, exist_ok=True)

# Probability keys
CELLTYPE_PROB_KEY = "X_cls_pred_probs"
GENOTYPE_PROB_KEY = "X_pert_pred_probs"
NEXT_PROB_KEYS = [
    "X_pertNEXT_pred_probs",
    "X_pert_pred_probs_next",
    "X_genotypeNEXT_pred_probs",
    "X_pred_probs_genotypeNEXT",
]

# Embedding keys
UMAP_KEY = "X_umap_scgpt"
REP_KEY  = "X_scGPT"

# Log
LOG_PATH = os.path.join(OUTDIR, "validation_all_tasks_log.txt")

# ---------------------------------------
# Logging helpers
# ---------------------------------------
def log(msg: str):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

def reset_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

# ---------------------------------------
# Small utils
# ---------------------------------------
def save_pdf_png(fig, outbase):
    pdf = f"{outbase}.pdf"
    png = f"{outbase}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[WRITE] {pdf}")

def pick_palette(n):
    base = plt.get_cmap("tab20").colors
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def ensure_cat_series(series, categories=None):
    if isinstance(series, pd.Series):
        s = series.astype(str)
    else:
        s = pd.Series(np.asarray(series).astype(str))
    if categories is None:
        categories = sorted(pd.unique(s))
    return s.astype(pd.CategoricalDtype(categories=categories, ordered=False))

def get_umap(adata, use_key=UMAP_KEY, rep_key=REP_KEY, seed=0):
    if use_key in adata.obsm:
        return adata.obsm[use_key]
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{rep_key}' to compute UMAP.")
    sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=12, key_added="scgpt", random_state=seed)
    sc.tl.umap(adata, neighbors_key="scgpt", random_state=seed, min_dist=0.5)
    adata.obsm[use_key] = adata.obsm["X_umap"].copy()
    return adata.obsm[use_key]

def plot_umap_category(adata, umap_coords, category, title, outbase):
    cat_s = ensure_cat_series(adata.obs[category])
    classes = list(cat_s.cat.categories)
    colors = pick_palette(len(classes))

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    X = umap_coords
    for i, cls in enumerate(classes):
        idx = (cat_s == cls).to_numpy()
        if idx.sum() == 0:
            continue
        ax.scatter(X[idx, 0], X[idx, 1], s=6, alpha=0.8,
                   label=str(cls), linewidths=0, c=[colors[i]])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(markerscale=2.0, fontsize=8, loc="best", ncol=1, frameon=False)
    save_pdf_png(fig, outbase)

# ---------------------------------------
# Column detection helpers
# ---------------------------------------
CELLTYPE_PREF_ORDER = [
    "Celltype_final", "celltype_final",
    "Celltype_updated", "celltype_updated",
    "celltype_2", "Celltype_2",
    "celltype_v2", "Celltype_v2",
    "CelltypeFinal"
]

def pick_obs_column(adata, preferred_keys, fallback=None, env_var=None, task_name=""):
    """
    Choose an obs column:
    1) If env_var is set and exists in obs, use it.
    2) First key in preferred_keys that exists and has any non-null values.
    3) Fallback if provided and exists.
    Returns (key or None).
    """
    # 1) explicit override
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            if override in adata.obs.columns:
                nonnull = int(adata.obs[override].notna().sum())
                log(f"[{task_name}] Using override obs['{override}'] (nonnull={nonnull}) via ${env_var}.")
                return override
            else:
                log(f"[{task_name}] WARNING: override obs['{override}'] not found; ignoring.")

    # 2) preferred list
    for k in preferred_keys:
        if k in adata.obs.columns:
            nonnull = int(adata.obs[k].notna().sum())
            if nonnull > 0:
                log(f"[{task_name}] Selected obs['{k}'] from preferred list (nonnull={nonnull}).")
                return k
            else:
                log(f"[{task_name}] NOTE: obs['{k}'] exists but is empty; skipping.")

    # 3) fallback
    if fallback and fallback in adata.obs.columns:
        nonnull = int(adata.obs[fallback].notna().sum())
        log(f"[{task_name}] Falling back to obs['{fallback}'] (nonnull={nonnull}).")
        return fallback

    log(f"[{task_name}] ERROR: No suitable obs column found.")
    return None

def normalize_obs_str(adata, col):
    """Trim spaces and cast to string categorical to avoid duplicate categories with whitespace."""
    s = adata.obs[col].astype(str).str.strip()
    adata.obs[col] = pd.Categorical(s, categories=sorted(pd.unique(s)))

# ---------------------------------------
# OLD→NEW taxonomy collapse helpers
# ---------------------------------------
def recover_head_order_from_probs(adata, prob_key: str, label_col: str):
    """
    Align probability columns (old head) to observed labels in `label_col`
    using per-class AUROC and Hungarian assignment.
    Returns (old_classes_sorted, probs_ordered_old).
    """
    if prob_key not in adata.obsm:
        raise KeyError(f"obsm['{prob_key}'] missing")
    if label_col not in adata.obs.columns:
        raise KeyError(f"obs['{label_col}'] missing")

    y_prob = np.asarray(adata.obsm[prob_key], dtype=float)
    labels = adata.obs[label_col].astype(str).values
    old_classes = sorted(pd.unique(labels))
    C = len(old_classes); J = y_prob.shape[1]

    name_to_idx = {c: i for i, c in enumerate(old_classes)}
    y_idx_all = np.array([name_to_idx.get(str(lbl), -1) for lbl in labels], dtype=int)
    keep = (y_idx_all >= 0)
    y_idx = y_idx_all[keep]
    y_prob_keep = y_prob[keep]

    # binarize
    y_true_bin = label_binarize(y_idx, classes=np.arange(C))
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)

    # AUROC score matrix
    S = np.zeros((C, J), dtype=float)
    for i in range(C):
        yt = y_true_bin[:, i]
        if yt.sum() == 0:
            S[i, :] = 0.5
            continue
        for j in range(J):
            try:
                S[i, j] = roc_auc_score(yt, y_prob_keep[:, j])
            except Exception:
                S[i, j] = 0.5

    cost = 1.0 - S
    _, col_ind = linear_sum_assignment(cost)
    aligned_cols = col_ind[:C]

    # IMPORTANT: use full (unmasked) rows to preserve N
    P_old_aligned = y_prob[:, aligned_cols]
    return old_classes, P_old_aligned

def infer_old_to_new_map(adata, old_col: str, new_col: str) -> dict:
    """
    Build mapping old_class -> new_class via majority vote on rows having both labels.
    """
    if old_col not in adata.obs.columns:
        raise KeyError(f"obs['{old_col}'] missing")
    if new_col not in adata.obs.columns:
        raise KeyError(f"obs['{new_col}'] missing")

    old = adata.obs[old_col].astype(str)
    new = adata.obs[new_col].astype(str)
    table = pd.crosstab(old, new)  # rows: old, cols: new

    mapping = {}
    for oc in table.index:
        if table.loc[oc].sum() == 0:
            continue
        nc = table.loc[oc].idxmax()
        mapping[oc] = nc
    return mapping

def collapse_probs_to_new_taxonomy(adata, prob_key: str, old_label_col: str, new_truth_col: str):
    """
    1) Recover old head order -> P_old_aligned (N x Cold)
    2) Infer mapping old -> new (majority vote)
    3) Collapse columns into new classes; row-wise renormalize
    Returns: (P_new, new_classes_sorted, mapping_dict)
    """
    old_classes, P_old = recover_head_order_from_probs(adata, prob_key=prob_key, label_col=old_label_col)
    mapping = infer_old_to_new_map(adata, old_col=old_label_col, new_col=new_truth_col)

    # Any old class not seen → map to itself (defensive)
    for oc in old_classes:
        if oc not in mapping:
            mapping[oc] = oc

    new_classes = sorted(pd.unique(pd.Series(list(mapping.values()))))
    K = len(new_classes)
    name_to_oldidx = {n: i for i, n in enumerate(old_classes)}

    groups = {nc: [] for nc in new_classes}
    for oc, nc in mapping.items():
        if oc in name_to_oldidx:
            groups[nc].append(name_to_oldidx[oc])

    P_new = np.zeros((P_old.shape[0], K), dtype=float)
    for k, nc in enumerate(new_classes):
        cols = groups.get(nc, [])
        if len(cols) == 0:
            continue
        P_new[:, k] = P_old[:, cols].sum(axis=1)

    rowsum = P_new.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    P_new = P_new / rowsum

    return P_new, new_classes, mapping

# ---------------------------------------
# Metrics/plots with alignment (generic)
# ---------------------------------------
def align_prob_columns(y_true_labels, y_prob, class_names):
    """
    Align columns of y_prob to target class_names by maximizing AUROC.
    Returns (y_prob_aligned, class_names_aligned, info_dict, keep_mask).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    classes = list(class_names)
    n_true = len(classes)
    n_cols = y_prob.shape[1]

    name_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx_all = np.array([name_to_idx.get(str(lbl), -1) for lbl in y_true_labels], dtype=int)
    keep = (y_idx_all >= 0) & np.isfinite(y_idx_all)
    if not np.all(keep):
        dropped = int((~keep).sum())
        log(f"[align] Dropping {dropped} rows with labels not in class_names.")
    y_idx = y_idx_all[keep]
    y_prob_keep = y_prob[keep]

    y_true_bin = label_binarize(y_idx, classes=np.arange(n_true))
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)

    S = np.zeros((n_true, n_cols), dtype=float)
    for i in range(n_true):
        yt = y_true_bin[:, i]
        if yt.sum() == 0:
            S[i, :] = 0.5
            continue
        for j in range(n_cols):
            try:
                S[i, j] = roc_auc_score(yt, y_prob_keep[:, j])
            except Exception:
                S[i, j] = 0.5

    cost = 1.0 - S
    _, col_ind = linear_sum_assignment(cost)
    aligned_cols = col_ind[:n_true]

    y_prob_aligned = y_prob_keep[:, aligned_cols]
    perm = np.array(aligned_cols, dtype=int)
    identity = np.array_equal(perm, np.arange(n_true))
    if not identity:
        mapping = ", ".join([f"{classes[i]} ← col{perm[i]}" for i in range(n_true)])
        log(f"[align] Prob columns re-ordered by assignment: {mapping}")
    else:
        log("[align] Prob column order matches class_names (identity).")

    return y_prob_aligned, classes[:n_true], {"perm": perm, "score_matrix": S}, keep

def curves_multiclass_with_csv(y_true_labels, y_prob, class_names, prefix, outdir):
    y_prob_aligned, class_names_aligned, _info, keep = align_prob_columns(y_true_labels, y_prob, class_names)

    y_prob = np.asarray(y_prob_aligned, dtype=float)
    C = y_prob.shape[1]
    name_to_idx = {n: i for i, n in enumerate(class_names_aligned)}

    y_idx_all = np.array([name_to_idx.get(str(lbl), -1) for lbl in y_true_labels], dtype=int)
    y_idx = y_idx_all[keep]

    y_pred_idx = y_prob.argmax(axis=1)
    accuracy_overall = (y_pred_idx == y_idx).mean() if len(y_idx) else np.nan

    acc_ovr = []
    if len(y_idx):
        for cls in range(C):
            pos = (y_idx == cls)
            neg = ~pos
            tp = ((y_pred_idx == cls) & pos).sum()
            tn = ((y_pred_idx != cls) & neg).sum()
            acc_cls = (tp + tn) / len(y_idx)
            acc_ovr.append(acc_cls)
    else:
        acc_ovr = [np.nan] * C

    if len(y_idx):
        y_true_bin = label_binarize(y_idx, classes=np.arange(C))
        if y_true_bin.ndim == 1:
            y_true_bin = y_true_bin.reshape(-1, 1)
    else:
        y_true_bin = np.zeros((0, C), dtype=int)

    fpr, tpr, roc_auc = {}, {}, {}
    prec_curve, rec_curve, ap = {}, {}, {}
    for i in range(C):
        if len(y_idx) == 0 or int(y_true_bin[:, i].sum()) == 0:
            fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), np.nan
            prec_curve[i], rec_curve[i], ap[i] = np.array([1.0]), np.array([0.0]), np.nan
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec_curve[i], rec_curve[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    if len(y_idx):
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")
        baseline = float(np.mean(y_true_bin)) if y_true_bin.size else 0.0
    else:
        fpr_micro = np.array([0, 1]); tpr_micro = np.array([0, 1]); roc_auc_micro = np.nan
        ap_micro = np.nan; baseline = 0.0

    valid_aucs = [roc_auc[i] for i in range(C) if not np.isnan(roc_auc[i])]
    valid_aps  = [ap[i]      for i in range(C) if not np.isnan(ap[i])]
    roc_auc_macro = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    ap_macro      = float(np.mean(valid_aps))  if valid_aps  else np.nan

    def save(fig, name): save_pdf_png(fig, os.path.join(outdir, name))

    fig = plt.figure(figsize=(8, 7))
    for i, cname in enumerate(class_names_aligned):
        if np.isnan(roc_auc.get(i, np.nan)): continue
        plt.plot(fpr[i], tpr[i], lw=1.2, label=f"{cname} (AUC={roc_auc[i]:.3f})", alpha=0.9)
    if not np.isnan(roc_auc_micro):
        plt.plot(fpr_micro, tpr_micro, lw=2.0, linestyle="--", label=f"micro (AUC={roc_auc_micro:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {prefix} (micro={roc_auc_micro:.3f}, macro={roc_auc_macro:.3f})")
    plt.legend(fontsize=8, loc="lower right")
    save(fig, f"{prefix}_roc")

    fig = plt.figure(figsize=(8, 7))
    for i, cname in enumerate(class_names_aligned):
        if np.isnan(ap.get(i, np.nan)): continue
        plt.plot(rec_curve[i], prec_curve[i], lw=1.2, label=f"{cname} (AP={ap[i]:.3f})", alpha=0.9)
    plt.plot([0, 1], [baseline, baseline], "k--", lw=1, label=f"baseline={baseline:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {prefix} (micro={ap_micro:.3f}, macro={ap_macro:.3f})")
    plt.legend(fontsize=8, loc="lower left")
    save(fig, f"{prefix}_pr")

    log(f"[{prefix}] micro-AUC={roc_auc_micro:.4f}, macro-AUC={roc_auc_macro:.4f} | micro-AP={ap_micro:.4f}, macro-AP={ap_macro:.4f})")

    if len(y_idx):
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_idx, y_pred_idx, labels=np.arange(C), average=None, zero_division=0
        )
    else:
        prec_c = rec_c = f1_c = np.array([np.nan] * C); support_c = np.zeros(C, dtype=int)

    df = pd.DataFrame({
        "class": class_names_aligned,
        "support": support_c.astype(int),
        "precision": prec_c,
        "recall": rec_c,
        "f1": f1_c,
        "auc_roc": [roc_auc.get(i, np.nan) for i in range(C)],
        "ap": [ap.get(i, np.nan) for i in range(C)],
        "accuracy_one_vs_rest": acc_ovr,
        "accuracy_overall": [accuracy_overall] * C,
    })
    df["class"] = df["class"].astype(str)

    return df, y_pred_idx, class_names_aligned, keep

# ---------------------------------------
# TASK RUNNERS
# ---------------------------------------
def run_celltype(adata, outdir, umap_coords):
    # Choose TRUE label column (new taxonomy) with robust detection + env override
    true_label_col = pick_obs_column(
        adata,
        preferred_keys=CELLTYPE_PREF_ORDER,
        fallback="celltype",
        env_var="CELLTYPE_LABEL_OVERRIDE",
        task_name="celltype"
    )
    if true_label_col is None:
        log("[celltype] SKIP: no true label column could be resolved.")
        return

    normalize_obs_str(adata, true_label_col)

    # OLD taxonomy label column for aligning the head
    old_label_candidates = ["predicted_celltype", "celltype"]
    old_label_col = None
    for c in old_label_candidates:
        if c in adata.obs.columns:
            normalize_obs_str(adata, c)
            old_label_col = c
            break
    if old_label_col is None:
        # last resort: use the new truth itself (works if head already matches it)
        old_label_col = true_label_col
    log(f"[celltype] truth='{true_label_col}' | old_label_for_alignment='{old_label_col}'")

    if CELLTYPE_PROB_KEY not in adata.obsm:
        log(f"[celltype] SKIP: obsm['{CELLTYPE_PROB_KEY}'] missing.")
        # Still plot labels if helpful
        plot_umap_category(adata, umap_coords, true_label_col,
                           f"UMAP — Celltype (true: {true_label_col})",
                           os.path.join(outdir, "umap_celltype_true"))
        return

    # ---- Collapse old-head probabilities to new taxonomy ----
    P_new, new_classes, mapping = collapse_probs_to_new_taxonomy(
        adata, prob_key=CELLTYPE_PROB_KEY,
        old_label_col=old_label_col,
        new_truth_col=true_label_col
    )
    adata.obsm["X_cls_pred_probs_collapsed"] = P_new
    adata.uns["celltype_new_classes"] = list(new_classes)
    adata.uns["celltype_old2new_map"] = mapping

    # Save mapping for auditability
    map_df = pd.DataFrame(sorted(mapping.items()), columns=["old_class", "new_class"])
    map_path = os.path.join(outdir, "celltype_old2new_map.csv")
    map_df.to_csv(map_path, index=False); log(f"[WRITE] {map_path}")

    # Metrics & plots on the NEW taxonomy
    y_true = adata.obs[true_label_col].astype(str).values
    df, y_pred_idx, class_order, keep = curves_multiclass_with_csv(
        y_true, P_new, new_classes, "celltype", outdir
    )

    # Stamp the source label column, class counts, class order, and optional crosstab
    df.insert(0, "label_source", true_label_col)

    cls_counts = (
        adata.obs[true_label_col].astype(str)
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="count")
    )
    cls_counts_path = os.path.join(outdir, f"celltype_classes_used__{true_label_col}.csv")
    cls_counts.to_csv(cls_counts_path, index=False); log(f"[WRITE] {cls_counts_path}")

    used_classes_path = os.path.join(outdir, f"celltype_class_order_for_metrics__{true_label_col}.txt")
    with open(used_classes_path, "w") as fh:
        for c in class_order:
            fh.write(str(c) + "\n")
    log(f"[WRITE] {used_classes_path}")

    if "celltype" in adata.obs.columns and true_label_col != "celltype":
        ct_tab = pd.crosstab(
            adata.obs["celltype"].astype(str).str.strip(),
            adata.obs[true_label_col].astype(str).str.strip(),
            dropna=False
        )
        ct_tab_path = os.path.join(outdir, f"celltype_crosstab_old_vs_{true_label_col}.csv")
        ct_tab.to_csv(ct_tab_path); log(f"[WRITE] {ct_tab_path}")

    # Write metrics (generic + label-specific)
    csv_path = os.path.join(outdir, "metrics_celltype.csv")
    csv_labeled_path = os.path.join(outdir, f"metrics_celltype__{true_label_col}.csv")
    df.to_csv(csv_path, index=False)
    df.to_csv(csv_labeled_path, index=False)
    log(f"[WRITE] {csv_path}")
    log(f"[WRITE] {csv_labeled_path}")

    # Safe predicted labels (collapsed) with keep-mask
    full_pred = np.array([None] * adata.n_obs, dtype=object)
    full_pred[keep] = np.array([class_order[i] for i in y_pred_idx], dtype=object)
    adata.obs["predicted_celltype_collapsed"] = pd.Categorical(full_pred, categories=class_order, ordered=False)

    # UMAPs (new taxonomy)
    plot_umap_category(adata, umap_coords, true_label_col,
                       f"UMAP — Celltype (true: {true_label_col})",
                       os.path.join(outdir, "umap_celltype_true"))
    plot_umap_category(adata, umap_coords, "predicted_celltype_collapsed",
                       "UMAP — Celltype (predicted: collapsed)",
                       os.path.join(outdir, "umap_celltype_pred_collapsed"))

def run_genotype(adata, outdir, umap_coords):
    if "genotype" not in adata.obs.columns:
        log("[genotype] SKIP: obs['genotype'] missing.")
        return
    if GENOTYPE_PROB_KEY not in adata.obsm:
        log(f"[genotype] SKIP: obsm['{GENOTYPE_PROB_KEY}'] missing.")
        return

    gt_order = sorted(pd.unique(adata.obs["genotype"].astype(str)))
    y_true = adata.obs["genotype"].astype(str).values
    y_prob = np.asarray(adata.obsm[GENOTYPE_PROB_KEY], dtype=float)

    log(f"[genotype] classes={len(gt_order)}; prob_key={GENOTYPE_PROB_KEY}")
    df, y_pred_idx, class_order, keep = curves_multiclass_with_csv(y_true, y_prob, gt_order, "genotype", outdir)

    csv_path = os.path.join(outdir, "metrics_genotype.csv")
    df.to_csv(csv_path, index=False); log(f"[WRITE] {csv_path}")

    full_pred = np.array([None] * adata.n_obs, dtype=object)
    full_pred[keep] = np.array([class_order[i] for i in y_pred_idx], dtype=object)
    adata.obs["predicted_genotype_collapsed"] = pd.Categorical(full_pred, categories=class_order, ordered=False)

    plot_umap_category(adata, umap_coords, "genotype", "UMAP — Genotype (true)",
                       os.path.join(outdir, "umap_genotype_true"))
    if "predicted_genotype" in adata.obs.columns:
        plot_umap_category(adata, umap_coords, "predicted_genotype", "UMAP — Genotype (predicted)",
                           os.path.join(outdir, "umap_genotype_pred"))
    if "predicted_genotype_collapsed" in adata.obs.columns:
        plot_umap_category(adata, umap_coords, "predicted_genotype_collapsed",
                           "UMAP — Genotype (predicted, collapsed)",
                           os.path.join(outdir, "umap_genotype_pred_collapsed"))

def run_genotypeNEXT(adata, outdir, umap_coords):
    if "genotype_next" not in adata.obs.columns:
        log("[genotypeNEXT] SKIP: obs['genotype_next'] missing.")
        return

    y_true_next = adata.obs["genotype_next"].astype(str).values
    class_order_true = sorted(pd.unique(adata.obs["genotype_next"].astype(str)))

    next_prob_key = None
    for k in NEXT_PROB_KEYS:
        if k in adata.obsm:
            next_prob_key = k
            break

    if next_prob_key is not None:
        y_prob_next = np.asarray(adata.obsm[next_prob_key], dtype=float)
        order = class_order_true
        log(f"[genotypeNEXT] Using probs from obsm['{next_prob_key}']; classes={len(order)}")

        df, y_pred_idx, class_order, keep = curves_multiclass_with_csv(y_true_next, y_prob_next, order, "genotypeNEXT", outdir)

        csv_path = os.path.join(outdir, "metrics_genotypeNEXT.csv")
        df.to_csv(csv_path, index=False); log(f"[WRITE] {csv_path}")

        full_pred = np.array([None] * adata.n_obs, dtype=object)
        full_pred[keep] = np.array([class_order[i] for i in y_pred_idx], dtype=object)
        adata.obs["predicted_genotypeNEXT_collapsed"] = pd.Categorical(full_pred, categories=class_order, ordered=False)

    else:
        # derive hard NEXT labels via next_cell_id -> predicted_genotype
        if "next_cell_id" not in adata.obs.columns or "predicted_genotype" not in adata.obs.columns:
            log("[genotypeNEXT] SKIP: cannot derive predicted NEXT; no curves/UMAP for NEXT.")
            return
        pred_now = adata.obs["predicted_genotype"].astype(str)
        idx_to_pred = pd.Series(pred_now.values, index=adata.obs_names)
        next_ids = adata.obs["next_cell_id"].astype(str).values
        derived = []
        missing = 0
        for nid in next_ids:
            if nid in idx_to_pred.index:
                derived.append(idx_to_pred.loc[nid])
            else:
                derived.append(np.nan); missing += 1
        if missing > 0:
            log(f"[genotypeNEXT] derived: {missing} rows had next_cell_id not found.")
        derived = pd.Series(derived, index=adata.obs_names, name="predicted_genotypeNEXT")

        cats = sorted(pd.unique(pd.concat([
            adata.obs["genotype_next"].astype(str),
            derived.astype(str)
        ], ignore_index=True)))
        y_prob_next = np.zeros((adata.n_obs, len(cats)), dtype=float)
        name_to_idx = {n: i for i, n in enumerate(cats)}
        for r, lbl in enumerate(derived.astype(str).values):
            j = name_to_idx.get(lbl, None)
            if j is not None:
                y_prob_next[r, j] = 1.0

        df, y_pred_idx, class_order, keep = curves_multiclass_with_csv(y_true_next, y_prob_next, cats, "genotypeNEXT", outdir)

        csv_path = os.path.join(outdir, "metrics_genotypeNEXT.csv")
        df.to_csv(csv_path, index=False); log(f"[WRITE] {csv_path}")

        full_pred = np.array([None] * adata.n_obs, dtype=object)
        full_pred[keep] = np.array([class_order[i] for i in y_pred_idx], dtype=object)
        adata.obs["predicted_genotypeNEXT_collapsed"] = pd.Categorical(full_pred, categories=class_order, ordered=False)

    plot_umap_category(adata, umap_coords, "genotype_next",
                       "UMAP — Genotype NEXT (true)",
                       os.path.join(outdir, "umap_genotypeNEXT_true"))
    if "predicted_genotypeNEXT_collapsed" in adata.obs.columns:
        plot_umap_category(adata, umap_coords, "predicted_genotypeNEXT_collapsed",
                           "UMAP — Genotype NEXT (predicted, collapsed)",
                           os.path.join(outdir, "umap_genotypeNEXT_pred"))

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    reset_log()
    adata = ad.read_h5ad(INPUT_PATH)
    log(f"[LOAD] AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    log(f"[OBS COLS] {list(adata.obs.columns)}")
    log(f"[OBSM KEYS] {list(adata.obsm.keys())}")

    for key in [CELLTYPE_PROB_KEY, GENOTYPE_PROB_KEY]:
        if key not in adata.obsm:
            log(f"[WARN] Missing probability matrix in .obsm: {key}")

    seed_val = int(adata.uns.get("seed", 0)) if isinstance(adata.uns.get("seed", 0), (int, float)) else 0
    umap_coords = get_umap(adata, use_key=UMAP_KEY, rep_key=REP_KEY, seed=seed_val)

    run_celltype(adata, OUTDIR, umap_coords)
    run_genotype(adata, OUTDIR, umap_coords)
#    run_genotypeNEXT(adata, OUTDIR, umap_coords)

    out_h5ad = os.path.join(OUTDIR, "adata_with_collapsed_preds.h5ad")
    try:
        adata.write_h5ad(out_h5ad, compression="gzip")
        log(f"[WRITE] {out_h5ad}")
    except Exception as e:
        log(f"[WARN] Could not write updated AnnData: {e}")

    log(f"[DONE] PDFs, PNGs, and CSVs written to: {OUTDIR}")

if __name__ == "__main__":
    main()