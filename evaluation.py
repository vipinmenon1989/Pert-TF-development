import copy
import json
import time
import os
import logging
import numpy as np
import resource
import psutil
os.environ["WANDB_API_KEY"]= "3f42c1f651e5c0658618383b0a787f06656bd550"
os.environ["KMP_WARNINGS"] = "off"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from tqdm import tqdm
import gseapy as gp
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from collections import Counter, defaultdict
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm
import csv
import pandas as pd
from pertTF import PerturbationTFModel
from config_gen import generate_config
from train_data_gen import produce_training_datasets
from train_function import train,wrapper_train,eval_testdata
import wandb, random
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("eval_only")

# === USER CONFIG ===
BASE_SAVE_DIR = "/local/projects-t3/lilab/vmenon/Pert-TF-model/dev_pancreatic-Jul31-16-49"  # contains fold_1 .. fold_5
#HELDOUT_H5AD = "heldout.h5ad"  # your held-out data (genotype/celltype omitted during training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================

eval_dir = Path(BASE_SAVE_DIR) / "evaluate_test"
eval_dir.mkdir(parents=True, exist_ok=True)

# 1. Load full data used for reconstructing reference_data: 
#    This should be the same adata you had after removing holdout genotype/celltype and after preprocessing.
#    We assume you still have the base .h5ad (without the held-out cells) or can recreate it here.
#    For simplicity, we'll load the same held-out file and then invert the mask to get the training part.

# Load original (full) AnnData and reconstruct holdout split exactly like training script did:
# You need the raw filtered adata that excludes the holdouts.
# Example: if you only have the original big adata, redo the holdout split here.
adata_full = sc.read_h5ad("/local/projects-t3/lilab/vmenon/Pert-TF-model/object_integrated_assay3_annotated_nounk_raw.cleaned.h5ad")
# apply the same preprocessing / filtering as in training before holdout
# (You must mirror whatever you did originally: subsetting, cleaning, etc.)
# Here we assume the holdout was defined by genotype or celltype as you described.
# Example for genotype + celltype holdout (adjust to your actual):
holdout_mask = holdout_mask = ((adata_full.obs["genotype"].str.contains("FOXA1|RFX6|PBX1")))

adata_holdout = adata_full[holdout_mask].copy()
adata_train_pool = adata_full[~holdout_mask].copy()

# Recreate preprocessing on training pool and held-out separately
# Load the config from one fold to know preprocessing params
fold1_config_path = Path(BASE_SAVE_DIR) / "fold_1" / "config.json"
with open(fold1_config_path) as f:
    cfg_dict = json.load(f)
class C: pass
config = C()
for k, v in cfg_dict.items():
    setattr(config, k, v)

# Build preprocessor same as training
preprocessor = Preprocessor(
    use_key="GPTin",  # the key in adata.layers to use as raw data
    #filter_gene_by_counts=3,  # step 1
    #filter_cell_by_counts=False,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    #result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    #result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes; use n_hvg default
    hvg_flavor="seurat_v3", # if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    #binning=0,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
# You need to ensure layers/keys exist like in training (e.g., GPTin); if not, replicate that pipeline before this.

preprocessor(adata_train_pool, batch_key=None)
preprocessor(adata_holdout, batch_key=None)

# Reconstruct reference_data once (from the training pool)
reference_data = produce_training_datasets(adata_train_pool, config, next_cell_pred='identity')
logger.info("Reconstructed reference_data from training pool (used across folds)")

# 2. Load held-out data for evaluation (already preprocessed above)
# Use adata_holdout as the target to evaluate.

# 3. Iterate folds
eval_results = []
for fold in range(1, 6):
    fold_dir = Path(BASE_SAVE_DIR) / f"fold_{fold}"
    logger.info(f"Evaluating fold {fold}")
    model_path = fold_dir / "best_model.pt"
    vocab_path = fold_dir / "vocab.pt"
    config_path = fold_dir / "config.json"

    if not model_path.exists() or not vocab_path.exists() or not config_path.exists():
        logger.warning(f"Missing files for fold {fold}, skipping.")
        continue

    # Load config again (if needed) to ensure coherence
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = C()
    for k, v in cfg_dict.items():
        setattr(config, k, v)

    # Preprocess held-out AGAIN if necessary (should be already done)
    # preprocessor(adata_holdout, batch_key=None)  # skip if already applied

    # Load vocab & model
    vocab = torch.load(vocab_path)
    ntokens = len(vocab)
    model = PerturbationTFModel(
        n_pert=reference_data['n_perturb'],
        nlayers_pert=3,
        n_ps=1,
        ntoken=ntokens,
        d_model=config.layer_size,
        nhead=config.nhead,
        d_hid=config.layer_size,
        nlayers=config.nlayers,
        nlayers_cls=3,
        n_cls=reference_data['n_cls'],
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=(config.dab_weight > 0),
        use_batch_labels=config.use_batch_label,
        num_batch_labels=reference_data['num_batch_types'],
        domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Avoid KeyError: drop held-out cells with unseen genotype/perturbation
    genotype_to_index = reference_data.get("genotype_to_index")
    if genotype_to_index is None:
        raise KeyError("reference_data missing 'genotype_to_index'")

    valid_mask = adata_holdout.obs["genotype"].isin(list(genotype_to_index.keys()))
    if (~valid_mask).any():
        logger.warning(f"Fold {fold}: dropping {np.sum(~valid_mask)} held-out cells with unseen genotype")
    adata_eval_input = adata_holdout[valid_mask].copy()

    # Run evaluation
    results = eval_testdata(
        model=model,
        adata_t=adata_eval_input,
        gene_ids=reference_data["gene_ids"],
        train_data_dict=reference_data,
        config=config,
        eval_key=f"fold{fold}_heldout",
        epoch=None,
        make_plots=True,
    )
    adata_eval = results["adata"]
    # Save the evaluated h5ad
    adata_eval.write_h5ad(eval_dir / f"heldout_fold{fold}.h5ad")

    # Save figures (UMAP etc.)
    for name, obj in results.items():
        if isinstance(obj, plt.Figure):
            obj.savefig(eval_dir / f"{name}_fold{fold}.png", dpi=300, bbox_inches="tight")
            plt.close(obj)

    # Compute metrics
    cls_acc = np.mean(adata_eval.obs.get("predicted_celltype", "") == adata_eval.obs.get("celltype", ""))
    pert_acc = np.mean(adata_eval.obs.get("predicted_genotype", "") == adata_eval.obs.get("genotype", ""))
    if "X_scGPT" in adata_eval.obsm and "X_scGPT" in adata_holdout.obsm:
        mse = np.mean((adata_eval.obsm["X_scGPT"] - adata_holdout.obsm["X_scGPT"]) ** 2)
    else:
        mse = np.nan

    eval_results.append({
        "fold": fold,
        "cls_accuracy": cls_acc,
        "pert_accuracy": pert_acc,
        "mse": mse,
    })

# 4. Summary
df_eval = pd.DataFrame(eval_results)
df_eval.to_csv(eval_dir / "heldout_evaluation_summary.csv", index=False)
avg_row = {
    "fold": "avg",
    "cls_accuracy": df_eval["cls_accuracy"].mean(),
    "pert_accuracy": df_eval["pert_accuracy"].mean(),
    "mse": df_eval["mse"].mean(),
}
df_eval_with_avg = pd.concat([df_eval, pd.DataFrame([avg_row])], ignore_index=True)
df_eval_with_avg.to_csv(eval_dir / "heldout_evaluation_summary_with_avg.csv", index=False)
logger.info("Evaluation complete. Summary saved.")
