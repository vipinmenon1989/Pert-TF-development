import torch
import scanpy as sc
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.sparse import issparse

# --- CRITICAL IMPORTS ---
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score,
    roc_curve, 
    precision_recall_curve, 
    auc,
    classification_report
)
from sklearn.preprocessing import label_binarize

# Framework Imports
from perttf.model.pertTF import PerturbationTFModel
from perttf.utils.custom_tokenizer import tokenize_and_pad_batch

# --- 1. SETUP ---
H5AD_PATH = "18clones_annotated_final_cleaned.h5ad"
MODEL_DIR = Path("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/fold_1")
OUT_DIR = Path("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Independet_test_data/reanalysis_18_clones/analytical_results_corrected") # New dir to keep things clean
OUT_DIR.mkdir(exist_ok=True)
sc.settings.figdir = OUT_DIR

# Check GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ GPU Detected. Enabling FP16.")
else:
    raise RuntimeError("❌ No GPU detected.")

# --- 2. LOAD INFRASTRUCTURE ---
print("Loading model artifacts...")
with open(MODEL_DIR / "vocab.json", "r") as f:
vocab = json.load(f)
with open(MODEL_DIR / "config.json", "r") as f:
config = json.load(f)
run_params = torch.load(MODEL_DIR / "running_parameters.pt", weights_only=False)

# --- 3. EXTRACT GENES ---
print("\n--- EXTRACTING GENES ---")
all_tokens = list(vocab.keys())
special_tokens = {'<pad>', '<cls>', '<eos>', '<unk>', '<mask >', '<mask >'} 
model_genes = [t for t in all_tokens if t not in special_tokens and not t.startswith('<')]

# --- 4. ALIGN DATA ---
print("\n--- ALIGNING DATA ---")
adata = sc.read_h5ad(H5AD_PATH)
data_genes = np.array(adata.var_names).astype(str)
model_genes = np.array(model_genes).astype(str)

common = np.intersect1d(model_genes, data_genes)
if len(common) == 0:
    print("Direct match 0. Trying UPPER CASE...")
    data_upper = np.char.upper(data_genes)
    model_upper = np.char.upper(model_genes)
    common = np.intersect1d(model_upper, data_upper)
    if len(common) > 0:
        print(f"Found {len(common)} matches via UPPER CASE.")
        adata.var_names = data_upper 
        model_genes = model_upper    
    else:
        raise ValueError("CRITICAL: 0 matching genes.")
else:
    print(f"Found {len(common)} exact matches.")

X_full = np.zeros((adata.n_obs, len(model_genes)), dtype=np.float32)
data_idx_map = {g: i for i, g in enumerate(adata.var_names)}
model_idx_map = {g: i for i, g in enumerate(model_genes)}

valid_m_idx = [model_idx_map[g] for g in common]
valid_d_idx = [data_idx_map[g] for g in common]

if issparse(adata.X):
    X_full[:, valid_m_idx] = adata.X[:, valid_d_idx].toarray()
else:
    X_full[:, valid_m_idx] = adata.X[:, valid_d_idx]

# --- 5. MODEL INIT (FP16) ---
celltype_map = {v: k for k, v in run_params['cell_type_to_index'].items()}
genotype_map = {v: k for k, v in run_params['genotype_to_index'].items()}
d_model = config.get('layer_size', config.get('d_model', 64))

model = PerturbationTFModel(
    len(genotype_map), 3, 1,
    ntoken=len(vocab),
    d_model=d_model,
    nhead=config.get('nhead', 4),
    d_hid=d_model,
    nlayers=config.get('nlayers', 4),
    vocab=vocab,
    dropout=config.get('dropout', 0.1),
    pad_token=config.get('pad_token', "<pad>"),
    pad_value=config.get('pad_value', -2),
    do_mvc=config.get('GEPC', False),
    n_cls=len(celltype_map),
    use_fast_transformer=True,
    use_batch_labels=config.get('use_batch_label', False),
    num_batch_labels=2,
    explicit_zero_prob=config.get('explicit_zero_prob', True)
)

model = model.to(DEVICE).half() 
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=True))
model.eval()

# --- 6. INFERENCE ---
print("\n--- RUNNING INFERENCE ---")
batch_size = config.get('batch_size', 32)
results = {"ct_probs": [], "gt_probs": []}
pad_token_id = vocab.get(config.get('pad_token', "<pad>"), 0)
model_gene_ids = np.array([vocab[g] for g in model_genes], dtype=np.int64)

with torch.no_grad():
    for i in range(0, len(X_full), batch_size):
        batch_X = X_full[i : i+batch_size]
        tokenized = tokenize_and_pad_batch(
            batch_X, model_gene_ids, max_len=config.get('max_seq_len', 3000),
            vocab=vocab, pad_token=config.get('pad_token', "<pad>"),
            pad_value=config.get('pad_value', -2), append_cls=True, include_zero_gene=True
        )[0]
        
        gene_ids = tokenized["genes"].to(DEVICE)
        values = tokenized["values"].to(DEVICE).half()
        output = model(
            gene_ids, values, src_key_padding_mask=gene_ids.eq(pad_token_id),
            CLS=True, PERTPRED=True
        )
        results["ct_probs"].append(torch.softmax(output["cls_output"], dim=1).float().cpu().numpy())
        results["gt_probs"].append(torch.softmax(output["pert_output"], dim=1).float().cpu().numpy())

ct_probs = np.concatenate(results["ct_probs"])
gt_probs = np.concatenate(results["gt_probs"])
ct_pred_idx = np.argmax(ct_probs, axis=1)
gt_pred_idx = np.argmax(gt_probs, axis=1)

adata.obs['pred_celltype'] = [celltype_map.get(i, "Unknown") for i in ct_pred_idx]
adata.obs['pred_genotype'] = [genotype_map.get(i, "Unknown") for i in gt_pred_idx]

# --- INSERT DIAGNOSTIC AUDIT HERE ---
print("\n--- DIAGNOSTIC AUDIT: THE FATE OF 'DE' CELLS ---")

# 1. Hard Classification Check
# We isolate cells that are Ground Truth 'DE' and see what the MODEL labeled them.
de_mask = adata.obs['celltype'] == 'DE'
if de_mask.sum() > 0:
    de_cells_pred = adata.obs.loc[de_mask, 'pred_celltype']
    print(f"\nTotal Ground Truth DE Cells: {sum(de_mask)}")
    print("Model predictions for these cells (Top 5):")
    print(de_cells_pred.value_counts().head(5))

    # 2. Soft Probability Check
    # Compare the score for 'DE' vs the score for the winner (likely PFG/Liver)
    de_col_idx = [k for k, v in celltype_map.items() if v == 'DE'][0]
    
    # Probability model gave to the 'DE' label for these cells
    probs_assigned_to_DE_class = ct_probs[de_mask, de_col_idx]
    # Probability model gave to the WINNING label for these cells
    top_class_probs = np.max(ct_probs[de_mask], axis=1)
    
    print(f"\n--- SCORE ANALYSIS ---")
    print(f"Avg probability of the WINNER (e.g. PFG): {np.mean(top_class_probs):.4f}")
    print(f"Avg probability of 'DE' (The loser):      {np.mean(probs_assigned_to_DE_class):.4f}")
    
    if np.mean(probs_assigned_to_DE_class) > 0.1: # Threshold check
        print("\n✅ LOGIC CHECK: The model sees the DE signal (High AUPR explanation).")
        print("It just ranks another label slightly higher.")
    else:
        print("\n❌ LOGIC CHECK: The model is completely missing the signal.")
else:
    print("No DE cells found in Ground Truth column.")
# ------------------------------------

# --- VISUAL PROOF OF AUPR (BOXPLOT) ---
import seaborn as sns # Ensure seaborn is installed/imported

print("\n--- GENERATING AUPR EXPLANATION PLOT ---")

# 1. Prepare Data for Plotting
# We isolate the score the model specifically gave to the 'DE' label
de_col_idx = [k for k, v in celltype_map.items() if v == 'DE'][0]
adata.obs['DE_Score'] = ct_probs[:, de_col_idx]
adata.obs['Is_True_DE'] = adata.obs['celltype'] == 'DE'

# 2. Generate Boxplot
fig, ax = plt.subplots(figsize=(6, 8))
# We use a log scale because the differences might be small (e.g. 0.004 vs 0.00001)
sns.boxplot(data=adata.obs, x='Is_True_DE', y='DE_Score', ax=ax, showfliers=False, palette="Set2")
ax.set_yscale('log') 
ax.set_title("Why AUPR is 0.71:\nModel 'DE' Scores (Log Scale)")
ax.set_ylabel("Probability Assigned to 'DE' Label")
ax.set_xticklabels(["Other Cells", "True DE Cells"])

# 3. Save
plt.savefig(OUT_DIR / "AUPR_Explanation_Boxplot.png", dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / "AUPR_Explanation_Boxplot.pdf", bbox_inches='tight')
print("✅ Saved AUPR proof plot to AUPR_Explanation_Boxplot.png")


# --- 7. METRICS (ROC + AUPR) ---
def save_dual(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)

def evaluate_detailed(true_col, pred_col, probs, mapping, name):
    mask = adata.obs[true_col].isin(mapping.values())
    if not mask.any(): return

    y_true = adata.obs.loc[mask, true_col].astype(str)
    y_pred = adata.obs.loc[mask, pred_col].astype(str)
    y_probs = probs[mask]
    unique_labels = np.unique(y_true)
    
    # 1. Classification Report
    report = classification_report(y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / f"Metrics_Per_Class_{name}.csv")

    # Prepare for Curves
    n_cls = len(mapping)
    y_true_indices = [list(mapping.values()).index(x) for x in y_true]
    y_bin = label_binarize(y_true_indices, classes=range(n_cls))
    if y_bin.shape[1] == 1: y_bin = np.hstack((1 - y_bin, y_bin))

    # 2. ROC Plot
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    for label_name in unique_labels:
        model_idx = [k for k, v in mapping.items() if v == label_name][0]
        fpr, tpr, _ = roc_curve(y_bin[:, model_idx], y_probs[:, model_idx])
        ax_roc.plot(fpr, tpr, label=f'{label_name} (AUC={auc(fpr, tpr):.2f})')
    
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title(f'ROC - {name} (Strict)')
    ax_roc.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    save_dual(fig_roc, f"ROC_{name}_Strict")

    # 3. AUPR Plot (Precision-Recall)
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    for label_name in unique_labels:
        model_idx = [k for k, v in mapping.items() if v == label_name][0]
        p, r, _ = precision_recall_curve(y_bin[:, model_idx], y_probs[:, model_idx])
        # Calculate AP explicitly for label
        ap = average_precision_score(y_bin[:, model_idx], y_probs[:, model_idx])
        ax_pr.plot(r, p, label=f'{label_name} (AP={ap:.2f})')
        
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Precision-Recall - {name} (Strict)')
    ax_pr.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    save_dual(fig_pr, f"AUPR_{name}_Strict")

print("\n--- CALCULATING STRICT METRICS ---")
evaluate_detailed('celltype', 'pred_celltype', ct_probs, celltype_map, "Celltype")
evaluate_detailed('genotype', 'pred_genotype', gt_probs, genotype_map, "Genotype")

# --- 8. FIXED UMAP PLOTTING ---
print("\n--- GENERATING UMAPS ---")
if "X_umap" not in adata.obsm:
    print("Computing Neighbors and UMAP...")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

def plot_strict_vs_novel(true_col, pred_col, title_prefix):
    # 1. Comparison Prep
    true_labels = set(adata.obs[true_col].unique())
    pred_labels = set(adata.obs[pred_col].unique())
    all_labels = list(true_labels.union(pred_labels))
    all_labels.sort() 

    # 2. Error Column Prep
    adata.obs[f'{pred_col}_errors'] = adata.obs.apply(
        lambda row: "Correct Match" if row[pred_col] == row[true_col] else row[pred_col], 
        axis=1
    )

    # --- PLOT 1: DIRECT COMPARISON ---
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    sc.pl.umap(adata, color=true_col, ax=ax[0], show=False, 
               title=f"Ground Truth {title_prefix}", 
               legend_loc='right margin', 
               palette='tab20', # Standard palette
               s=30, frameon=False)
    
    sc.pl.umap(adata, color=pred_col, ax=ax[1], show=False, 
               title=f"Model Prediction {title_prefix} (Raw)", 
               legend_loc='right margin', 
               palette='tab20', # Matching palette
               s=30, frameon=False)
    
    plt.tight_layout()
    save_dual(fig, f"UMAP_{title_prefix}_Comparison_Unified")

    # --- PLOT 2: ERROR IDENTIFICATION (FIXED) ---
    # Get all unique labels in the error column
    unique_errors = adata.obs[f'{pred_col}_errors'].unique()
    
    # Generate a robust list of valid colors (combining Tableau and CSS4 to ensure enough colors)
    valid_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    # Manually build the palette dictionary
    error_palette = {}
    c_idx = 0
    for label in unique_errors:
        if label == "Correct Match":
            error_palette[label] = "lightgrey" # Explicitly grey out correct ones
        else:
            # Assign a valid color from our list
            error_palette[label] = valid_colors[c_idx % len(valid_colors)]
            c_idx += 1
            
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.umap(adata, color=f'{pred_col}_errors', ax=ax, show=False, 
               title=f"Mismatch Detail: {title_prefix}",
               palette=error_palette, # Now strictly valid strings, no None
               legend_loc='right margin', 
               s=30, frameon=False)
    
    save_dual(fig, f"UMAP_{title_prefix}_Mismatch_Detail")

# Run
plot_strict_vs_novel('celltype', 'pred_celltype', "Celltype")
plot_strict_vs_novel('genotype', 'pred_genotype', "Genotype")

# Run the new function
plot_strict_vs_novel('celltype', 'pred_celltype', "Celltype")
plot_strict_vs_novel('genotype', 'pred_genotype', "Genotype")

plot_strict_vs_novel('celltype', 'pred_celltype', "Celltype")
plot_strict_vs_novel('genotype', 'pred_genotype', "Genotype")

print(f"\n✅ DONE! Results in: {OUT_DIR.absolute()}")
