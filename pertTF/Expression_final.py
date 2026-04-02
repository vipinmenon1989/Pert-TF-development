import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.sparse as sp
from numpy.linalg import norm
import pandas as pd
from pathlib import Path

# -----------------------
# Load AnnData for fold 1
# -----------------------
adata_path = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_3/best_epoch_e45/adata_best_validation.h5ad"
adata_eval = ad.read_h5ad(adata_path)

# 1. ground-truth "future"/"next" expression for each cell
true_next = adata_eval.layers["next_expr"]
if sp.issparse(true_next):
    true_next = true_next.toarray()
true_next = np.asarray(true_next, dtype=float)

# 2. model-predicted "future"/"next" expression
pred_next = adata_eval.obsm["mvc_next_expr"]
pred_next = np.asarray(pred_next, dtype=float)

print("true_next shape:", true_next.shape)
print("pred_next shape:", pred_next.shape)
assert true_next.shape == pred_next.shape, "shape mismatch between true and predicted next expression"

n_cells, n_genes = true_next.shape

# -----------------------
# Helper functions
# -----------------------

def safe_pearson(a, b):
    # returns NaN if vector is constant or invalid
    try:
        r, _ = pearsonr(a, b)
    except Exception:
        r = np.nan
    return r

def safe_spearman(a, b):
    try:
        rho, _ = spearmanr(a, b)
    except Exception:
        rho = np.nan
    return rho

def safe_cosine(a, b):
    # cosine similarity = (a·b)/(|a||b|)
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

# -----------------------
# 0. Global Pearson over EVERYTHING
# -----------------------
flat_true = true_next.reshape(-1)
flat_pred = pred_next.reshape(-1)
global_pearson, _ = pearsonr(flat_true, flat_pred)
print("global Pearson (all cell*gene entries):", float(global_pearson))

# -----------------------
# 1. Per-cell metrics
#    - Pearson per cell
#    - Cosine per cell (directional similarity of full profile)
# -----------------------
cell_pearsons = []
cell_cosines  = []

for i in range(n_cells):
    t = true_next[i, :]
    p = pred_next[i, :]

    cell_pearsons.append(safe_pearson(t, p))
    cell_cosines.append(safe_cosine(t, p))

mean_cell_pearson   = float(np.nanmean(cell_pearsons))
median_cell_pearson = float(np.nanmedian(cell_pearsons))
mean_cell_cosine    = float(np.nanmean(cell_cosines))
median_cell_cosine  = float(np.nanmedian(cell_cosines))

print("\n=== Per-cell agreement ===")
print("mean per-cell Pearson:",  mean_cell_pearson)
print("median per-cell Pearson:", median_cell_pearson)
print("mean per-cell Cosine similarity:",  mean_cell_cosine)
print("median per-cell Cosine similarity:", median_cell_cosine)

# -----------------------
# 2. Per-gene metrics
#    - Pearson per gene (absolute level recovery)
#    - Spearman per gene (rank / monotonic recovery)
# -----------------------
gene_pearsons  = []
gene_spearmans = []

for g in range(n_genes):
    t = true_next[:, g]
    p = pred_next[:, g]

    gene_pearsons.append(safe_pearson(t, p))
    gene_spearmans.append(safe_spearman(t, p))

mean_gene_pearson    = float(np.nanmean(gene_pearsons))
median_gene_pearson  = float(np.nanmedian(gene_pearsons))
mean_gene_spearman   = float(np.nanmean(gene_spearmans))
median_gene_spearman = float(np.nanmedian(gene_spearmans))

print("\n=== Per-gene agreement ===")
print("mean per-gene Pearson:",  mean_gene_pearson)
print("median per-gene Pearson:", median_gene_pearson)
print("mean per-gene Spearman:",  mean_gene_spearman)
print("median per-gene Spearman:", median_gene_spearman)

# -----------------------
# 3. Pattern-only agreement (gene-wise z-scored)
#    We z-score each gene across cells in true and predicted separately,
#    then compare cell profiles again.
#
#    Intuition: "Did I get the structure of the program right,
#    even if my amplitudes are off?"
# -----------------------
true_next_z = true_next.copy()
pred_next_z = pred_next.copy()

# z-score each gene across cells, separately for true and predicted
true_means = true_next_z.mean(axis=0, keepdims=True)
true_stds  = true_next_z.std(axis=0, keepdims=True) + 1e-8
true_next_z = (true_next_z - true_means) / true_stds

pred_means = pred_next_z.mean(axis=0, keepdims=True)
pred_stds  = pred_next_z.std(axis=0, keepdims=True) + 1e-8
pred_next_z = (pred_next_z - pred_means) / pred_stds

cell_pearsons_z = []
cell_cosines_z  = []

for i in range(n_cells):
    t = true_next_z[i, :]
    p = pred_next_z[i, :]

    cell_pearsons_z.append(safe_pearson(t, p))
    cell_cosines_z.append(safe_cosine(t, p))

mean_cell_pearson_z   = float(np.nanmean(cell_pearsons_z))
median_cell_pearson_z = float(np.nanmedian(cell_pearsons_z))
mean_cell_cosine_z    = float(np.nanmean(cell_cosines_z))
median_cell_cosine_z  = float(np.nanmedian(cell_cosines_z))

print("\n=== Per-cell agreement AFTER gene-wise z-score (pattern-only) ===")
print("mean per-cell Pearson (z-scored):",  mean_cell_pearson_z)
print("median per-cell Pearson (z-scored):", median_cell_pearson_z)
print("mean per-cell Cosine (z-scored):",   mean_cell_cosine_z)
print("median per-cell Cosine (z-scored):", median_cell_cosine_z)

# -----------------------
# 4. Save summary for fold 1
# -----------------------
summary = {
    "fold": 1,
    "n_cells": n_cells,
    "n_genes": n_genes,

    # global
    "global_pearson_all_entries": float(global_pearson),

    # per-cell raw
    "mean_cell_pearson":  mean_cell_pearson,
    "median_cell_pearson": median_cell_pearson,
    "mean_cell_cosine":   mean_cell_cosine,
    "median_cell_cosine": median_cell_cosine,

    # per-gene
    "mean_gene_pearson":  mean_gene_pearson,
    "median_gene_pearson": median_gene_pearson,
    "mean_gene_spearman": mean_gene_spearman,
    "median_gene_spearman": median_gene_spearman,

    # per-cell z-scored (pattern-only)
    "mean_cell_pearson_z":  mean_cell_pearson_z,
    "median_cell_pearson_z": median_cell_pearson_z,
    "mean_cell_cosine_z":   mean_cell_cosine_z,
    "median_cell_cosine_z": median_cell_cosine_z,
}

print("\n=== SUMMARY (fold 1) ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# make sure output dir exists (same fold_3 dir where your best epoch data lives)
out_dir = Path("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_3/best_epoch_e45")
out_dir.mkdir(parents=True, exist_ok=True)
 
out_csv = out_dir / "fold3_expression_metrics.csv"
pd.DataFrame([summary]).to_csv(out_csv, index=False)

print(f"\nSaved summary metrics to: {out_csv}")