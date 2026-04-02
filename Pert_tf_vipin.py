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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_script")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logger.info(f"Seeding completed with seed={SEED}")

mem_limit_bytes = 300 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
def log_memory(stage: str):
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    vmem = psutil.virtual_memory()
    logger.info(f"[{stage}] RSS={mem.rss / 1e9:.2f}GB, VMS={mem.vms / 1e9:.2f}GB, avail={vmem.available / 1e9:.2f}GB")

log_memory("startup")

hyperparameter_defaults = dict(
    seed=42,
    #dataset_name="PBMC_10K", # Dataset name
    dataset_name="pancreatic",
    do_train=True, # Flag to indicate whether to do update model parameters during training
    #load_model="/content/drive/MyDrive/Colab Notebooks/scGPT/pretrain_blood", # Path to pre-trained model
    load_model=None,
    GEPC=True,  # Gene expression modelling for cell objective
    ecs_thres=0.7,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight= 0.0, #2000.0, # DAR objective weight for batch correction; if has batch, set to 1.0
    this_weight = 1.0, # weight for predicting the expression of current cell
    next_weight = 00.0, # weight for predicting the next cell
    n_rounds = 1, # number of rounds for generating the next cell
    next_cell_pred_type = 'identity', # the method to predict the next cell, either "pert" (random next cell within the same cell type) or "identity" (the same cell). If "identity", set next_weight=0
    #
    ecs_weight = 1.0, # weight for predicting the similarity of cells
    cell_type_classifier=True, #  do we need the trasnformer to separate cell types?
    cell_type_classifier_weight = 1.0,
    perturbation_classifier_weight = 10.0,
    perturbation_input = False, # use perturbation as input?
    CCE = False, # Contrastive cell embedding objective
    mask_ratio=0.15, # Default mask ratio
    epochs=60, # Default number of epochs for fine-tuning
    n_bins=51, # Default number of bins for value binning in data pre-processing
    #lr=1e-4, # Default learning rate for fine-tuning
    lr=1e-3, # learning rate for learning de novo
    batch_size=32, # Default batch size for fine-tuning
    layer_size=32, # defalut 32
    nlayers=2,
    nhead=4, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.4, # Default dropout rate during model fine-tuning
    schedule_ratio=0.99,  # Default rate for learning rate decay
    save_eval_interval=5, # Default model evaluation interval
    log_interval=60, # Default log interval
    fast_transformer=True, # Default setting
    pre_norm=False, # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision

    do_sample_in_train = False , # sample the bernoulli in training
    ADV = False, # Adversarial training for batch correction
    adv_weight=10000,
    adv_E_delay_epochs = 2, # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 2,
    lr_ADV = 1e-3, # learning rate for discriminator, used when ADV is True

    DSBN =False, # True if (config.dab_weight >0 or config.ADV ) else False  # Domain-spec batchnorm; default is True
    per_seq_batch_sample =  False, # DSBN # default True
    use_batch_label = False, # default: equal to DSBN
    schedule_interval = 1,

    explicit_zero_prob = True,  # whether explicit bernoulli for zeros
    n_hvg = 3000,  # number of highly variable genes

    mask_value = -1,
    pad_value = -2,
    pad_token = "<pad>",
    ps_weight = 0.0
)


sc.set_figure_params(figsize=(4, 4))
logger.info("Generating configuration from defaults...")
config, run_session = generate_config(hyperparameter_defaults,wandb_mode="online")
logger.info(f"Config generated: dataset={config.dataset_name}, epochs={config.epochs}")
# Persist config as JSON
safe_cfg = {}
for k, v in config.__dict__.items():
    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
        safe_cfg[k] = v

with open("config.json", "w") as cf:
    json.dump(safe_cfg, cf, indent=2)

logger.info("Saved JSON-safe config.json")

log_memory("post-config-save")
dataset_name = config.dataset_name
save_dir = Path(f"/local/projects-t3/lilab/vmenon/Pert-TF-model/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Outputs and checkpoints will be saved to {save_dir}")

adata0=sc.read_h5ad("object_integrated_assay3_annotated_nounk_raw.cleaned.h5ad")
logger.info(f"Loaded AnnData: n_obs={adata0.n_obs}, n_vars={adata0.n_vars}")
log_memory("post-load")
if 'sub.cluster' in adata0.obs.columns:
    valid = adata0.obs['sub.cluster'].notna()
    dropped = np.sum(~valid)
    adata0 = adata0[valid].copy()
    logger.info(f"Dropped {dropped} cells missing sub.cluster.")

# Map 'sub.cluster' to 'celltype'
adata0.obs['celltype'] = adata0.obs['sub.cluster']
logger.info("Mapped sub.cluster → celltype.")

# Preserve raw matrix for binning
adata0.layers['GPTin'] = adata0.X.copy()

adata0.obs['gene'] = (
    adata0.obs['gene']
    .str.replace('124_NANOGe_het','124_NANOGe-het', regex=False)
    .str.replace('123_NANOGe_het','123_NANOGe-het', regex=False)
)
genotypes = (
    adata0.obs['gene'].str.split('_').str[-1]
    .replace({'WT111':'WT','WT4':'WT','NGN3':'NEUROG3'})
    .fillna('WT')
)
mask = (genotypes != 'WT') | (np.random.rand(adata0.n_obs) < 0.01)
adata = adata0[mask, :].copy()
logger.info(f"Subsampled: retained {adata.n_obs} cells out of {adata0.n_obs}.")
log_memory("after-subsample")


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
preprocessor(adata, batch_key= None)
logger.info("Preprocessing complete.")
log_memory("after-preprocess")
#logger.info("Producing training datasets...")
#data_produced = produce_training_datasets(adata, config, next_cell_pred='identity')
#logger.info("Training datasets ready.")
#print (data_produced['adata_sorted'].obs)
data_barcode = adata.obs_names.to_list()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 🧪 Hold out specific genotypes/perturbs for final validation
logger.info("Creating hold-out validation set from specific perturb/genotype")
holdout_mask = (
    (adata.obs["genotype"].str.contains("FOXA1|RFX6|PBX1"))
)
adata_holdout = adata[holdout_mask].copy()
adata = adata[~holdout_mask].copy()

# ✅ Preprocess the held-out data with the same preprocessor
preprocessor(adata_holdout, batch_key=None)

logger.info(f"Held out {adata_holdout.n_obs} cells for final validation")

# ✅ Continue to use the rest of the already preprocessed adata


# 🔁 Rewritten Cross-Validation Training Loop — With Comments on Moved Blocks

kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)
adata.obs['split'] = 'unused'  # init

obs_names = np.array(adata.obs_names)
fold_results = []
logger.info(f"5-fold cross validation starts")
for fold, (train_idx, val_idx) in enumerate(kf.split(obs_names)):
    logger.info(f"===== Fold {fold + 1}/5 =====")

    
    save_dir_fold = save_dir / f"fold_{fold + 1}"
    save_dir_fold.mkdir(parents=True, exist_ok=True)

    # ──────── Define train/val splits ────────
    logger.info(f"Train/Val splits")
    train_barcodes = obs_names[train_idx]
    val_barcodes   = obs_names[val_idx]
    adata.obs['split'] = 'unused'
    adata.obs.loc[train_barcodes, 'split'] = 'train'
    adata.obs.loc[val_barcodes, 'split']   = 'validation'

    # ✅ Subset data *before* producing fold_data to avoid full matrix
    adata_fold = adata[adata.obs['split'].isin(['train', 'validation'])]

    # 🔁 Now use the subset, not full adata
    fold_data = produce_training_datasets(adata_fold, config, next_cell_pred='identity')

    # ──────── NEW: Save vocab here per fold ────────
    vocab = fold_data['vocab']
    with open(save_dir_fold / 'vocab.json', 'w') as vf:
        json.dump(vocab.get_stoi(), vf, indent=2)
    torch.save(vocab, save_dir_fold / 'vocab.pt')

    # ──────── Compute ntokens now ────────
    ntokens = len(vocab)
    n_perturb = fold_data['n_perturb']
    n_body_layers = 3                    # your “3” from before

    # ──────── Define model per fold ────────
    model = PerturbationTFModel(
        n_pert=fold_data['n_perturb'],
        nlayers_pert=n_body_layers,
        n_ps=1,
        ntoken=ntokens,
        d_model=config.layer_size,
        nhead=config.nhead,
        d_hid=config.layer_size,
        nlayers=config.nlayers,
        nlayers_cls=3,
        n_cls=fold_data['n_cls'],
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=(config.dab_weight > 0),
        use_batch_labels=config.use_batch_label,
        num_batch_labels=fold_data['num_batch_types'],
        domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    ).to(device)

    # ──────── Patch encode_batch method conditionally ────────
    if config.next_cell_pred_type == "identity":
        orig_encode = model.encode_batch_with_perturb
        def encode_batch_force_t0(
            src, values, src_key_padding_mask, batch_size,
            batch_labels=None, pert_labels=None, pert_labels_next=None,
            output_to_cpu=True, time_step=0, return_np=False
        ):
            return orig_encode(
                src, values, src_key_padding_mask, batch_size,
                batch_labels, pert_labels, pert_labels_next,
                output_to_cpu, 0, return_np
            )
        model.encode_batch_with_perturb = encode_batch_force_t0

    # ──────── Train model ────────
    logger.info(f"Training model")
    train_result = wrapper_train(
        model, config, fold_data,
        eval_adata_dict={'validation': fold_data['adata_sorted']},
        save_dir=save_dir_fold,
        fold=fold + 1
    )
    best_model = train_result["model"]

    # ──────── Save metrics for this fold ────────
    logger.info(f"Save the fold model")
    fold_results.append({
        "fold": fold + 1,
        "best_model_epoch": train_result["best_model_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "save_path": str(save_dir_fold / "best_model.pt"),
    })

    # ──────── Save model artifact to WandB ────────
    artifact = wandb.Artifact(f"best_model_fold{fold+1}", type="model")
    artifact.add_file(str(save_dir_fold / "best_model.pt"))
    run_session.log_artifact(artifact)
# ✅ Free memory after fold is done
    del model, train_result, fold_data, adata_fold, vocab
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# Create directory to save evaluation results
eval_dir = save_dir / "evaluate_test"
eval_dir.mkdir(parents=True, exist_ok=True)

# Load held-out data (you can customize this)
#adata_holdout = sc.read_h5ad("your_heldout_data.h5ad")  # 🔁 Replace with your actual path
preprocessor(adata_holdout, batch_key=None)

eval_results = []
reference_data = produce_training_datasets(adata, config, next_cell_pred='identity')
for result in fold_results:
    fold = result["fold"]
    model_path = result["save_path"]
    save_dir_fold = Path(model_path).parent
    vocab_path = save_dir_fold / "vocab.pt"

    logger.info(f"Evaluating fold {fold} model on held-out data")

    # Load vocab and model
    vocab = torch.load(vocab_path)
    ntokens = len(vocab)

    model = PerturbationTFModel(
        n_pert=reference_data['n_perturb'],  # Assuming same
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
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate
    results = eval_testdata(
        model=model,
        adata_t=adata_holdout,
        gene_ids=reference_data["gene_ids"],
        train_data_dict=reference_data,
        config=config,
        eval_key=f"fold{fold}_heldout",
        epoch=result["best_model_epoch"],
        make_plots=True,
    )

    adata_eval = results["adata"]
    eval_h5ad_path = eval_dir / f"heldout_fold{fold}.h5ad"
    adata_eval.write_h5ad(eval_h5ad_path)

    # Save UMAPs
    for k, v in results.items():
        if isinstance(v, plt.Figure):
            fig_path = eval_dir / f"{k}_fold{fold}.png"
            v.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(v)

    # Compute accuracy & MSE
    cls_acc = np.mean(adata_eval.obs["predicted_celltype"] == adata_eval.obs["celltype"])
    pert_acc = np.mean(adata_eval.obs["predicted_genotype"] == adata_eval.obs["genotype"])
    if "X_scGPT" in adata_eval.obsm and "X_scGPT" in adata_holdout.obsm:
        mse = np.mean((adata_eval.obsm["X_scGPT"] - adata_holdout.obsm["X_scGPT"]) ** 2)
    else:
        logger.warning(f"X_scGPT missing in fold {fold}; skipping MSE.")
        mse = np.nan

    eval_results.append({
        "fold": fold,
        "best_model_epoch": result["best_model_epoch"],
        "mse": mse,
        "cls_accuracy": cls_acc,
        "pert_accuracy": pert_acc,
    })

# Save per-fold evaluation summary
df_eval = pd.DataFrame(eval_results)
summary_path = eval_dir / "heldout_evaluation_summary.csv"
df_eval.to_csv(summary_path, index=False)
logger.info(f"Saved per-fold summary to {summary_path}")

# Add average row to new DataFrame
avg_row = {
    "fold": "avg",
    "best_model_epoch": np.nan,
    "mse": df_eval["mse"].mean(),
    "cls_accuracy": df_eval["cls_accuracy"].mean(),
    "pert_accuracy": df_eval["pert_accuracy"].mean(),
}
df_eval_with_avg = pd.concat([df_eval, pd.DataFrame([avg_row])], ignore_index=True)
summary_path_avg = eval_dir / "heldout_evaluation_summary_with_avg.csv"
df_eval_with_avg.to_csv(summary_path_avg, index=False)
logger.info(f"Saved summary with average row to {summary_path_avg}")


# ──────── Save overall fold summary ────────
pd.DataFrame(fold_results).to_csv(save_dir / "fold_summary.csv", index=False)
logger.info("Saved fold_summary.csv with best epochs and val losses.")

# ──────── Finalize WandB ────────
run_session.finish()
wandb.finish()
logger.info("WandB run finished. Script done.")