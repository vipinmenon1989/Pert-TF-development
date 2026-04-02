import copy
import json
import time
import os
import logging
import numpy as np
import resource
import psutil
import sys, scipy.sparse as sp
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
#from torchtext._torchtext import Vocab 
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
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function_vipin import train, wrapper_train, eval_testdata
import wandb, random
from sklearn.model_selection import KFold
from typing import Optional
import optuna
from pathlib import Path
import json
import multiprocessing as mp
from perttf.utils.safe_config import SafeConfig

# ---- basic logging ----
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("cv_train_eval")
# ---------- Fast-tune knobs (env-overridable) ----------
FAST_TUNE = bool(int(os.environ.get("FAST_TUNE", "1")))  # 1=tune fast, 0=full
TUNE_N_SPLITS = int(os.environ.get("TUNE_N_SPLITS", "5"))  # folds per trial (fast)
TUNE_EPOCHS = int(os.environ.get("TUNE_EPOCHS", "65"))     # max epochs per trial (fast)
TUNE_MAX_CELLS = os.environ.get("TUNE_MAX_CELLS", "80000") # None or int
TUNE_MAX_CELLS = None if str(TUNE_MAX_CELLS).lower() in {"", "none"} else int(TUNE_MAX_CELLS)
NHVG_CANDIDATES = [3000]

def _wandb_safe_set(cfg, updates: dict):
    """
    Safe setter for W&B (or plain) configs.
    Works whether cfg is a wandb.Config or a simple object.
    """
    try:
        if hasattr(cfg, "update"):
            cfg.update(updates, allow_val_change=True)
            return
    except Exception:
        pass
    # Fallback to attribute set
    for k, v in updates.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass

# --- NVML helpers to map local indices -> PCI bus IDs / UUIDs ---
def _init_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception as e:
        raise RuntimeError(f"NVML init failed: {e}")

def get_visible_devices_info():
    """
    Return [{'local_index': i, 'bus_id': '00000000:19:00.0', 'uuid': 'GPU-xxxx', 'name': 'NVIDIA H200'}, ...]
    Works across pynvml versions where fields may be bytes or str.
    """
    import torch
    pynvml = _init_nvml()
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)

        # pci info
        pci = pynvml.nvmlDeviceGetPciInfo(h)
        bus_id = pci.busId
        if isinstance(bus_id, (bytes, bytearray)):
            bus_id = bus_id.decode()
        else:
            bus_id = str(bus_id)

        # uuid
        try:
            uuid = pynvml.nvmlDeviceGetUUID(h)
        except AttributeError:
            # very old NVML fallback
            uuid = pynvml.nvmlDeviceGetBoardPartNumber(h)
        if isinstance(uuid, (bytes, bytearray)):
            uuid = uuid.decode()
        else:
            uuid = str(uuid)

        # name
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, (bytes, bytearray)):
            name = name.decode()
        else:
            name = str(name)

        info.append({'local_index': i, 'bus_id': bus_id, 'uuid': uuid, 'name': name})
    return info


def _as_list(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [v]


def materialize_config_space(space: dict) -> dict:
    return {k: _as_list(v) for k, v in space.items()}


def choose_config(
    space: dict,
    keys_for_search=None,
    trial_index: Optional[int] = None,
    policy: str = "first",
    rng_seed: int = 1234,
) -> dict:
    """Pick one concrete config from dict-of-lists."""
    space = materialize_config_space(space)
    rng = np.random.default_rng(int(rng_seed))
    if keys_for_search is None:
        keys_for_search = [k for k, v in space.items() if len(v) > 1]
    prod_keys = [k for k in keys_for_search if len(space.get(k, [])) > 1]
    prod_lists = [space[k] for k in prod_keys]
    radices = [len(lst) for lst in prod_lists]
    picked = {}

    def _decode_index(idx, sizes):
        if not sizes:
            return []
        out = []
        for base in reversed(sizes):
            out.append(idx % base)
            idx //= base
        return list(reversed(out))

    if trial_index is not None and len(prod_keys) > 0:
        total = 1
        for r in radices:
            total *= r
        if trial_index < 0 or trial_index >= total:
            raise ValueError(f"trial_index={trial_index} out of range [0, {total-1}]")
        idxs = _decode_index(trial_index, radices)
        for k, i in zip(prod_keys, idxs):
            picked[k] = space[k][i]
    else:
        for k in prod_keys:
            picked[k] = space[k][rng.integers(0, len(space[k]))] if policy == "random" else space[k][0]
    for k, lst in space.items():
        if k in picked:
            continue
        picked[k] = lst[rng.integers(0, len(lst))] if (policy == "random" and len(lst) > 1) else lst[0]
    return picked

# ---------- NEW: run heavy prep ONCE in parent, then reuse ----------
def prepare_and_persist_adata(raw_h5ad_path: str, save_dir: Path, config) -> Path:
    """
    Load the raw AnnData once, perform thinning/HVG/binning ONCE,
    write a preprocessed .h5ad, and return its path.
    """
    logger.info(f"[parent] Loading raw AnnData: {raw_h5ad_path}")
    adata0 = sc.read_h5ad(raw_h5ad_path)
    logger.info(f"[parent] Loaded AnnData: n_obs={adata0.n_obs}, n_vars={adata0.n_vars}")

    # Optional: drop NA sub.cluster
    if getattr(config, "filter_missing_subcluster", False) and ("sub.cluster" in adata0.obs.columns):
        valid = adata0.obs["sub.cluster"].notna()
        adata0 = adata0[valid].copy()
        logger.info(f"[parent] Dropped NA sub.cluster -> n_obs={adata0.n_obs}")

    # Tidy genotype labels
    if "gene" not in adata0.obs.columns:
        raise ValueError("adata0.obs['gene'] not found; cannot derive genotypes.")
    adata0.obs["gene"] = (
        adata0.obs["gene"]
        .astype(str)
        .str.replace("124_NANOGe_het", "124_NANOGe-het", regex=False)
        .str.replace("123_NANOGe_het", "123_NANOGe-het", regex=False)
    )
    genotypes = (
        adata0.obs["gene"]
        .str.split("_")
        .str[-1]
        .replace({"WT111": "WT", "WT4": "WT", "NGN3": "NEUROG3"})
        .fillna("WT")
    )
    adata0.obs['genotype'] = genotypes.astype(str)

    # Keep all non-WT + fraction of WT
    KEEP_WT_FRAC = float(getattr(config, "KEEP_WT_FRAC", 0.01))
    rng = np.random.default_rng(int(config.seed))
    mask = (genotypes != "WT") | ((genotypes == "WT") & (rng.random(adata0.n_obs) < KEEP_WT_FRAC))
    adata = adata0[mask, :].copy()
    adata.obs["genotype"] = genotypes.loc[adata.obs_names].astype(str).values
    logger.info(f"[parent] After thinning: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    # celltype + GPTin layer
    adata.obs["celltype_2"] = adata.obs.get("sub.cluster", "")
    if "GPTin" not in adata.layers:
        adata.layers["GPTin"] = adata.X

    # HVG selection (optional)
    USE_HVG = bool(getattr(config, "USE_HVG", True))
    if USE_HVG:
        ad_tmp = adata.copy()
        ad_tmp.X = ad_tmp.layers["GPTin"]
        sc.pp.normalize_total(ad_tmp, target_sum=1e4)
        sc.pp.log1p(ad_tmp)
        # compute seurat_v3 scores on the FULL matrix
        sc.pp.highly_variable_genes(ad_tmp, flavor="seurat_v3", batch_key=None, n_top_genes=ad_tmp.shape[1],)
        if "highly_variable_rank" not in ad_tmp.var:
            raise RuntimeError("Scanpy did not produce 'highly_variable_rank' with seurat_v3.")
        
        ranks0 = (ad_tmp.var["highly_variable_rank"].astype(float) - 1.0)
        adata.var["hvg_rank"] = ranks0.reindex(adata.var_names)
        logger.info("[parent] HVG ranks saved in var['hvg_rank']; not subsetting here.")
    else:
        logger.info("[parent] HVG disabled; no ranks stored; workers will keep all genes.")

    # Binning
    preprocessor = Preprocessor(
        use_key="GPTin",
        normalize_total=None,
        log1p=False,
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=int(getattr(config, "n_bins", 51)),
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=None)
    assert "X_binned" in adata.layers, "Binning did not produce X_binned."
    logger.info("[parent] Binning complete.")

    # --- NEW: make X_binned compact ---
    Xb = adata.layers["X_binned"]
    if sp.issparse(Xb):
        adata.layers["X_binned"] = Xb.astype(np.uint8).tocsr()
    else:
        Xb = np.asarray(Xb, dtype=np.uint8)
        adata.layers["X_binned"] = sp.csr_matrix(Xb)

    # optional slimming (keeps obs/var columns; removes heavy extras)
    adata.layers.pop("GPTin", None)
    adata.raw = None
    # If you know you don't need big slots:
    # adata.obsm.clear(); adata.varm.clear(); adata.uns.clear()

    # Persist
    preproc_path = save_dir / "preprocessed.h5ad"
    adata.write_h5ad(preproc_path, compression="lzf")
    logger.info(f"[parent] Wrote preprocessed file: {preproc_path}")
    return preproc_path

# ========== Hyperparameter defaults ==========
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="pancreatic",
    do_train=True,
    load_model=None,
    GEPC=True,
    ecs_thres=0.7,
    dab_weight=0.0,
    this_weight=1.0,
    next_weight=10.0,
    n_rounds=1,
    next_cell_pred_type="pert",
    ecs_weight=1.0,
    cell_type_classifier=True,
    cell_type_classifier_weight=1.0,
    perturbation_classifier_weight=50.0,
    perturbation_input=False,
    CCE=False,
    mask_ratio=0.15,
    epochs=65,
    n_bins=51,
    lr=1e-3,
    batch_size=64,
    layer_size=64,
    nlayers=4,
    nhead=4,
    dropout=0.4,
    schedule_ratio=0.99,
    save_eval_interval=5,
    log_interval=60,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    do_sample_in_train=False,
    ADV=False,
    adv_weight=10000,
    adv_E_delay_epochs=2,
    adv_D_delay_epochs=2,
    lr_ADV=1e-3,
    DSBN=False,
    per_seq_batch_sample=False,
    use_batch_label=False,
    schedule_interval=1,
    explicit_zero_prob=True,
    USE_HVG=True,
    n_hvg=3000,
    mask_value=-1,
    pad_value=-2,
    pad_token="<pad>",
    ps_weight=0.0,
    max_cells=None,
    max_cells_seed=1337,
    filter_missing_subcluster=True,
    KEEP_WT_FRAC=0.1,
    early_stop_patience=10,
    early_stop_min_delta=0.01,
    optuna_enable=[False],
    optuna_n_trials=1,
    optuna_timeout=[None],
)

SEARCH_KEYS = [
    "ecs_thres",
    "perturbation_classifier_weight",
    "lr",
    "layer_size",
    "nlayers",
    "nhead",
    "dropout",
    "n_hvg",
    "KEEP_WT_FRAC",
    "early_stop_patience",
    "early_stop_min_delta",
    "ADV",
    "adv_weight",
    "lr_ADV",
    "batch_size",
]

TRIAL_INDEX = os.environ.get("TRIAL_INDEX")
TRIAL_INDEX = int(TRIAL_INDEX) if TRIAL_INDEX else None
CHOICE_POLICY = os.environ.get("CHOICE_POLICY", "first")
CHOICE_SEED = int(os.environ.get("CHOICE_SEED", "1337"))

# ---- LOCK BEST HYPERPARAMS (insert above choose_config) ----
BEST_LR = 0.0012273661723331349
BEST_PERTURB_W = 50
# If you already have a best dropout, set it here; otherwise it will keep your default (0.4)
BEST_DROPOUT = 0.15579754426081674  # e.g., 0.32

# Clamp the defaults to single values so _suggest_* returns fixed numbers.
if BEST_DROPOUT is None:
    _dropout_val = hyperparameter_defaults.get("dropout", 0.4)
else:
    _dropout_val = float(BEST_DROPOUT)

hyperparameter_defaults.update({
    "lr": [float(BEST_LR)],
    "perturbation_classifier_weight": [float(BEST_PERTURB_W)],
    "dropout": [float(_dropout_val)],
    "optuna_n_trials": 1,  # ensure we run just 1 trial
})

chosen_defaults = choose_config(
    hyperparameter_defaults, keys_for_search=SEARCH_KEYS, trial_index=TRIAL_INDEX, policy=CHOICE_POLICY, rng_seed=CHOICE_SEED
)

# Real config (scalars)
config, run_session = generate_config(chosen_defaults, wandb_mode="disabled")
config = SafeConfig(config)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#logger.info(f"Using device: {device}")

# ---- dirs ----
dataset_name = config.dataset_name
save_dir = Path(f"/local/projects-t3/lilab/vmenon/Pert-TF-model/cv_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Outputs and checkpoints will be saved to {save_dir}")

# ========== Worker: one fold per process ==========
def run_single_fold_worker(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    save_dir: Path,
    device_uuid: str,
    chosen_defaults: dict,
    run_session_project: Optional[str],
    preprocessed_h5ad_path: Path,   # << path not AnnData
    result_file: Optional[Path] = None,
    wandb_mode: str = "disabled",   # "disabled" for Optuna trials; "online" for final
):
    import wandb, os, torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_uuid)
    print(f"[fold {fold}] binding UUID={device_uuid} | "f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}") 
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    print(f"[fold {fold}] device={dev} |"
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} |"
          f"current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
        )
    cfg, _ = generate_config(chosen_defaults, wandb_mode="disabled")
    cfg = SafeConfig(cfg)
    if wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    if wandb_mode == "disabled" and FAST_TUNE:
        if hasattr(cfg, "epochs"):
            cfg.update({"epochs": min(int(getattr(cfg, "epochs", 100)), int(TUNE_EPOCHS))}, allow_val_change=True)
        else:
            cfg.epochs = int(TUNE_EPOCHS)
        if TUNE_MAX_CELLS is not None:
            if hasattr(cfg, "update"):
                cfg.update({"max_cells": int(TUNE_MAX_CELLS)}, allow_val_change=True)
            else:
                cfg.max_cells = int(TUNE_MAX_CELLS)
                

    # Open preprocessed path
    ad_backed = sc.read_h5ad(str(preprocessed_h5ad_path), backed="r")
    obs_df = ad_backed.obs.copy()
    obs_names = np.array(obs_df.index)
    obs_df["split"] = "unused"
    train_barcodes = obs_names[train_idx]
    val_barcodes = obs_names[val_idx]
    obs_df.loc[train_barcodes, "split"] = "train"
    obs_df.loc[val_barcodes, "split"] = "validation"
    del ad_backed

    adata_fold = sc.read_h5ad(str(preprocessed_h5ad_path))
    adata_fold.obs = obs_df
    logger.info(f"[fold {fold}] Using pre-subset file {preprocessed_h5ad_path.name} (n_vars={adata_fold.n_vars})")
    if wandb_mode == "disabled" and FAST_TUNE and (TUNE_MAX_CELLS is not None) and (adata_fold.n_obs > TUNE_MAX_CELLS):
        train_mask = (adata_fold.obs["split"].values == "train")
        val_mask   = (adata_fold.obs["split"].values == "validation")
        n_val = int(val_mask.sum())
        budget_for_train = max(0, int(TUNE_MAX_CELLS) - n_val)
        if budget_for_train < train_mask.sum():
            rng = np.random.default_rng(int(getattr(cfg, "seed", 42)))
            train_idx_all = np.where(train_mask)[0]
            keep_train = rng.choice(train_idx_all, size=budget_for_train, replace=False)
            keep_idx = np.concatenate([np.where(val_mask)[0], keep_train])
            keep_idx.sort()
            adata_fold = adata_fold[keep_idx].copy()


     #Per-trial HVG subsetting
    
    fold_data = produce_training_datasets(adata_fold, cfg, next_cell_pred="pert")
    ad_eval = fold_data['adata_sorted']
    missing = [k for k in ['celltype_2','genotype'] if k not in ad_eval.obs.columns]
    if missing:
        raise RuntimeError(f"[Fold {fold}] Missing required labels in eval AnnData: {missing}")

    # log quick counts so you see what eval will compute
    ct_counts = ad_eval.obs['celltype_2'].value_counts()
    gt_counts = ad_eval.obs['genotype'].value_counts()
    logger.info(f"[Fold {fold}] eval label counts — celltype_2: {dict(ct_counts.head(10))}")
    logger.info(f"[Fold {fold}] eval label counts — genotype: {dict(gt_counts.head(10))}")
    # --- sanity: classification metrics need ≥2 classes ---
    if gt_counts.nunique() <= 1:
        logger.warning(f"[Fold {fold}] Genotype has <=1 class in validation; ROC/AUPR will be undefined.")
    if ct_counts.nunique() <= 1:
        logger.warning(f"[Fold {fold}] Celltype has <=1 class in validation; ROC/AUPR will be undefined.")
    
    save_dir_fold = save_dir / f"fold_{fold}"
    save_dir_fold.mkdir(parents=True, exist_ok=True)

    vocab = fold_data["vocab"]
    def vocab_to_stoi(v):
        if hasattr(v, "get_stoi"):
            return dict(v.get_stoi())
        if hasattr(v, "stoi"):
            return dict(v.stoi)
        if hasattr(v, "token2id"):
            return dict(v.token2id)
        if hasattr(v, "get_itos"):
            itos = list(v.get_itos())
        else:
            itos = list(v)
        return {tok: idx for idx, tok in enumerate(itos)}
    (save_dir_fold / "vocab.json").write_text(json.dumps(vocab_to_stoi(vocab), indent=2))
    torch.save(vocab, save_dir_fold / "vocab.pt")

    ntokens = len(vocab)
    model = PerturbationTFModel(
        n_pert=fold_data["n_perturb"],
        nlayers_pert=3,
        n_ps=1,
        ntoken=ntokens,
        d_model=cfg.layer_size,
        nhead=cfg.nhead,
        d_hid=cfg.layer_size,
        nlayers=cfg.nlayers,
        nlayers_cls=3,
        n_cls=fold_data["n_cls"],
        vocab=vocab,
        dropout=cfg.dropout,
        pad_token=cfg.pad_token,
        pad_value=cfg.pad_value,
        do_mvc=cfg.GEPC,
        do_dab=(cfg.dab_weight > 0),
        use_batch_labels=cfg.use_batch_label,
        num_batch_labels=fold_data["num_batch_types"],
        domain_spec_batchnorm=cfg.DSBN,
        n_input_bins=cfg.n_bins,
        ecs_threshold=cfg.ecs_thres,
        explicit_zero_prob=cfg.explicit_zero_prob,
        use_fast_transformer=cfg.fast_transformer,
        pre_norm=cfg.pre_norm,
    ).to(dev)

    if cfg.next_cell_pred_type == "pert":
        orig_encode = model.encode_batch_with_perturb

        def encode_batch_force_t0(
            src,
            values,
            src_key_padding_mask,
            batch_size,
            batch_labels=None,
            pert_labels=None,
            pert_labels_next=None,
            perturbation=None,
            pert_scale=None,
            output_to_cpu=True,
            time_step=0,
            return_np=False,
            predict_expr=False,
            mvc_src=None,
            **kw,
        ):
            # force time_step=0, but keep every other arg aligned by NAME
            return orig_encode(
                src=src,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
                batch_size=batch_size,
                batch_labels=batch_labels,
                pert_labels=pert_labels,
                pert_labels_next=pert_labels_next,
                perturbation=perturbation,
                pert_scale=pert_scale,
                output_to_cpu=output_to_cpu,
                time_step=0,  # <-- THIS is the only override we actually want
                return_np=return_np,
                predict_expr=predict_expr,
                mvc_src=mvc_src,
                **kw,
            )

        model.encode_batch_with_perturb = encode_batch_force_t0

    fold_run = wandb.init(
        project=run_session_project,
        name=f"cv_fold_{fold}",
        group="cross_validation",
        reinit=True,
        config=cfg.as_dict() if hasattr(cfg, "as_dict") else None,
        mode=wandb_mode,
    )
    fold_run.config.update({"fold": fold})

    train_result = wrapper_train(
        model, cfg, fold_data,
        eval_adata_dict={"validation": fold_data["adata_sorted"]},
        save_dir=save_dir_fold,
        fold=fold,
        run=fold_run,
        trial=None,
        use_early_stopping=True,
    )
    best_model = train_result["model"]
    best_val_loss = float(train_result["best_val_loss"])
    torch.save(best_model.state_dict(), save_dir_fold / "best_model.pt")

    if wandb_mode != "disabled":
        artifact = wandb.Artifact(f"best_model_fold{fold}", type="model")
        artifact.add_file(str(save_dir_fold / "best_model.pt"))
        fold_run.log_artifact(artifact)
    fold_run.finish()

    if result_file is not None:
        try:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps({"fold": fold, "best_val_loss": best_val_loss}))
        except Exception as e:
            print(f"[fold {fold}] failed to write result_file {result_file}: {e}")

    del model, best_model, fold_data, vocab, adata_fold
    torch.cuda.empty_cache()

def _suggest_float_or_fixed(trial, name, low=None, high=None, *, log=False):
    """If defaults[name] is list -> search, if scalar -> return fixed value."""
    base = hyperparameter_defaults.get(name, None)

    # list => search (unless length 1)
    if isinstance(base, (list, tuple, np.ndarray)):
        vals = list(base)
        if len(vals) == 1:
            return float(vals[0])
        lo = float(min(vals)) if low is None else float(low)
        hi = float(max(vals)) if high is None else float(high)
        return trial.suggest_float(name, lo, hi, log=log)

    # scalar => fixed
    if base is not None:
        return float(base)

    # no default => require explicit low/high
    if low is None or high is None:
        raise ValueError(f"{name}: need low/high or default in hyperparameter_defaults.")
    return trial.suggest_float(name, float(low), float(high), log=log)

def _suggest_categorical_or_fixed(trial, name):

    """If defaults[name] is list -> categorical search; if scalar -> return scalar."""
    base = hyperparameter_defaults.get(name, None)
    if isinstance(base, (list, tuple, np.ndarray)):
        vals = list(dict.fromkeys(base))
        if len(vals) == 1:
            return vals[0]
        return trial.suggest_categorical(name, vals)
    return base

# =============== Optuna objective (5-fold CV per trial) ===============
def make_objective(preprocessed_h5ad_path: Path, base_config, base_defaults, base_save_dir: Path):
    def objective(trial: optuna.trial.Trial) -> float:
        """Suggest hyperparams safely; run 5-fold CV in parallel; return mean best_val_loss."""
        # ---- sample safely from your defaults ----
        td = dict(base_defaults)

        # lr
        lr_list = _as_list(hyperparameter_defaults["lr"])
        #td["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        td["lr"]  = _suggest_float_or_fixed(trial, "lr", log=True)  # stays fixed at 1e-3
        

        # batch size
        bs_list = _as_list(hyperparameter_defaults["batch_size"])
        td["batch_size"] = _suggest_categorical_or_fixed(trial, "batch_size")

        # transformer depth/width
        layer_list = _as_list(hyperparameter_defaults["layer_size"])
        td["layer_size"] = _suggest_categorical_or_fixed(trial, "layer_size")

        nl_list = _as_list(hyperparameter_defaults["nlayers"])
        td["nlayers"]    = _suggest_categorical_or_fixed(trial, "nlayers")

        # heads must divide d_model
        head_candidates = _as_list(hyperparameter_defaults["nhead"])
        div_ok_heads = [h for h in head_candidates if td["layer_size"] % h == 0]
        if not div_ok_heads:
            trial.report(1e9, step=0)
            raise optuna.TrialPruned()
        if len(div_ok_heads) > 1:
            td["nhead"] = int(trial.suggest_categorical("nhead", div_ok_heads))
        else:
            td["nhead"] = int(div_ok_heads[0])
        # dropout
        dr_vals = _as_list(hyperparameter_defaults["dropout"])
        
        td["dropout"]    = _suggest_float_or_fixed(trial, "dropout")
        #td["dropout"] = trial.suggest_float("dropout", 0.10, 0.50)
        # perturbation weight
        pcw_list = _as_list(hyperparameter_defaults["perturbation_classifier_weight"])  
        #td["perturbation_classifier_weight"] = trial.suggest_categorical("perturbation_classifier_weight", [10, 20, 50])
        td["perturbation_classifier_weight"] = _suggest_float_or_fixed(trial, "perturbation_classifier_weight")
        # ecs_thres
        ecs_vals = _as_list(hyperparameter_defaults["ecs_thres"])
        td["ecs_thres"]  = _suggest_float_or_fixed(trial, "ecs_thres")
        

        # KEEP_WT_FRAC
        td["KEEP_WT_FRAC"] = _suggest_float_or_fixed(trial, "KEEP_WT_FRAC")

        # early stopping
        td["early_stop_patience"] = int(_suggest_categorical_or_fixed(trial, "early_stop_patience"))
        td["early_stop_min_delta"] = _suggest_float_or_fixed(trial, "early_stop_min_delta", log=True)
        
        # HVG (keep same as base unless you want to search it)
        v = base_defaults.get("n_hvg", getattr(base_config, "n_hvg", 3000))
       # td["n_hvg"] = trial.suggest_categorical("n_hvg", v) if isinstance(v, (list, tuple)) and len(v)>1 else int(v)
        td["n_hvg"] = trial.suggest_categorical("n_hvg", NHVG_CANDIDATES)

        # sanity
        if td["layer_size"] % td["nhead"] != 0:
            trial.report(1e9, step=0)
            raise optuna.TrialPruned()

        # ---- Build folds once (from preprocessed file) ----
        ad_backed = sc.read_h5ad(str(preprocessed_h5ad_path), backed="r")
        obs_names = np.array(ad_backed.obs_names)
        var_df = ad_backed.var.copy()
        del ad_backed
        # Create/use per-trial HVG subset file
        trial_hvg = int(td["n_hvg"])
        subset_path = base_save_dir / f"preprocessed_hvg{trial_hvg}.h5ad"
        if not subset_path.exists():
            if "hvg_rank" in var_df:
                keep = var_df.sort_values("hvg_rank").index[:trial_hvg]
            else:
                keep = var_df.index[:trial_hvg]
            
            ad_mem = sc.read_h5ad(str(preprocessed_h5ad_path))
            ad_mem = ad_mem[:, keep].copy()
            # ensure layers stays compact
            if "X_binned" in ad_mem.layers:
                Xb = ad_mem.layers["X_binned"]
                if sp.issparse(Xb):
                    ad_mem.layers["X_binned"] = Xb.astype(np.uint8).tocsr()
                else:
                    ad_mem.layers["X_binned"] = sp.csr_matrix(np.asarray(Xb, dtype=np.uint8))
            ad_mem.write_h5ad(subset_path, compression="lzf")
            del ad_mem
        preprocessed_for_trial = subset_path
                
        N_SPLITS = TUNE_N_SPLITS if FAST_TUNE else 5
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=int(base_config.seed))
        folds = [(fold, tr, va) for fold, (tr, va) in enumerate(kf.split(obs_names), start=1)]
        import torch 
        dev_info = get_visible_devices_info() if torch.cuda.is_available() else []
        max_fold_gpus = int(os.environ.get("MAX_FOLD_GPUS", "5"))
        dev_info = dev_info[:max_fold_gpus] if dev_info else []
        if not dev_info:
            uuids, bus_ids = [], []
            gpu_slots = 1
        else:
            uuids = [d["uuid"] for d in dev_info]
            bus_ids = [d["bus_id"] for d in dev_info]
            gpu_slots = len(uuids)
        avail_gb = psutil.virtual_memory().available / 1e9
        cells = len(obs_names)
        n_hvg = int(td["n_hvg"])
        mem_per_fold_gb = max(1.0, (cells * n_hvg) / 1e9 * 1.5)
        ram_slots = max(1, int(avail_gb // mem_per_fold_gb))
        num_slots = min(gpu_slots, ram_slots)
        # Build mapping: fold -> GPU index 0..num_slots-1
        gpu_map = {f: (uuids[i % num_slots] if uuids else None)
                   for i, (f, _, _) in enumerate(folds)}
        
        print(f"[trial {trial.number}] slots={num_slots} | bus_ids={bus_ids} | gpu_map={gpu_map}")

        # ---- Launch folds in waves (one per GPU) ----
        from multiprocessing import Process
        trial_dir = base_save_dir / f"optuna_trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        run_session_project = run_session.project if hasattr(run_session, "project") else None

        def launch_wave(wave_folds):
            procs, files = [], []
            for (fold, train_idx, val_idx) in wave_folds:
                rf = trial_dir / f"fold_{fold}_result.json"
                p = Process(target=run_single_fold_worker,args=(fold, train_idx, val_idx, base_save_dir, gpu_map[fold],td, run_session_project, preprocessed_for_trial, rf, "disabled"),)
                p.start()
                procs.append(p); files.append(rf)
            for p in procs: p.join()
            return files

        result_files = []
        for i in range(0, len(folds), num_slots):
            wave = folds[i:i+num_slots]
            result_files.extend(launch_wave(wave))
    
        # collect fold losses (unchanged)
        fold_losses = []
        for rf in result_files:
            try:
                obj = json.loads(Path(rf).read_text())
                fold_losses.append(float(obj["best_val_loss"]))
            except Exception:
                fold_losses.append(1e9)
        mean_loss = float(np.mean(fold_losses))
        trial.report(mean_loss, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_loss
    return objective
# ========================= MAIN =========================
if __name__ == "__main__":
    import getpass, tempfile

    user = getpass.getuser() or "user"
    # Prefer shared memory; fall back to local /tmp
    local_base_candidates = [f"/dev/shm/{user}", f"/tmp/{user}"]
    LOCAL_BASE = None
    for p in local_base_candidates:
        try:
            os.makedirs(p, exist_ok=True)
            LOCAL_BASE = p
            break
        except Exception:
            continue
    if LOCAL_BASE is None:
        LOCAL_BASE = os.getcwd()

    LOCAL_TMP = os.path.join(LOCAL_BASE, f"perttf_tmp_{os.getpid()}")
    os.makedirs(LOCAL_TMP, exist_ok=True)

    # Make ALL temp users honor this (multiprocessing, numpy, matplotlib, etc.)
    os.environ["TMPDIR"] = LOCAL_TMP
    os.environ["TEMP"]   = LOCAL_TMP
    os.environ["TMP"]    = LOCAL_TMP

    # Also tell Python's tempfile module directly (affects multiprocessing internals)
    tempfile.tempdir = LOCAL_TMP

    # Matplotlib font cache / config (also noisy on NFS)
    os.environ["MPLCONFIGDIR"] = os.path.join(LOCAL_TMP, "mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    # (Optional) Scanpy cache off NFS too
    os.environ.setdefault("SCANPY_CACHE_DIR", os.path.join(LOCAL_TMP, "scanpy_cache"))
    os.makedirs(os.environ["SCANPY_CACHE_DIR"], exist_ok=True)

    # Sanity print so you can verify at runtime
    print(f"[tmp] tempfile.gettempdir() = {tempfile.gettempdir()}")

    mp.set_start_method("spawn", force=True)

    RAW_H5AD = "object_integrated_assay3_annotated_final.cleaned.h5ad"

    # 1) Preprocess ONCE
    preprocessed_h5ad_path = prepare_and_persist_adata(RAW_H5AD, save_dir, config)

    # 2) Optuna setup (persistent SQLite DB)
    optuna_dir = save_dir / "optuna_results"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    db_path = (optuna_dir / "optuna.db").resolve().as_posix()
    storage_url = f"sqlite:////{db_path.lstrip('/')}" if db_path.startswith("/") else f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"perttf_{config.dataset_name}",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(config.seed)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    objective = make_objective(preprocessed_h5ad_path, config, chosen_defaults, save_dir)
    n_trials = int(getattr(config, "optuna_n_trials", 25))
    study.optimize(objective, n_trials=n_trials)

    # 3) Persist Optuna results
    logger.info(f"[OPTUNA] Best value:  {study.best_value:.6f}")
    logger.info(f"[OPTUNA] Best params: {study.best_params}")

    (optuna_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    pd.DataFrame([dict(**study.best_params, best_value=study.best_value)]).to_csv(optuna_dir / "best_params.csv", index=False)
    try:
        study.trials_dataframe().to_csv(optuna_dir / "all_trials.csv", index=False)
    except Exception as e:
        logger.warning(f"Could not export trials_dataframe: {e}")

    # 4) Final model on near-full data with best params
    try:
        import wandb

        best_defaults = dict(chosen_defaults)
        best_defaults.update(study.best_params)
        final_config, _ = generate_config(best_defaults, wandb_mode="online")

        # Build tiny val split for early stopping
        ad_full = sc.read_h5ad(str(preprocessed_h5ad_path))
        ad_full.obs["split"] = "train"
        rng = np.random.default_rng(int(final_config.seed))
        idx = np.arange(ad_full.n_obs)
        if ad_full.n_obs >= 10:
            val_size = max(1, int(0.10 * ad_full.n_obs))
            val_idx = rng.choice(idx, size=val_size, replace=False)
            ad_full.obs.iloc[val_idx, ad_full.obs.columns.get_loc("split")] = "validation"

        final_data = produce_training_datasets(ad_full, final_config, next_cell_pred="identity")

        final_dir = save_dir / "final_full_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        final_vocab = final_data["vocab"]
        ntokens = len(final_vocab)

        final_model = PerturbationTFModel(
            n_pert=final_data["n_perturb"],
            nlayers_pert=3,
            n_ps=1,
            ntoken=ntokens,
            d_model=final_config.layer_size,
            nhead=final_config.nhead,
            d_hid=final_config.layer_size,
            nlayers=final_config.nlayers,
            nlayers_cls=3,
            n_cls=final_data["n_cls"],
            vocab=final_vocab,
            dropout=final_config.dropout,
            pad_token=final_config.pad_token,
            pad_value=final_config.pad_value,
            do_mvc=final_config.GEPC,
            do_dab=(final_config.dab_weight > 0),
            use_batch_labels=final_config.use_batch_label,
            num_batch_labels=final_data["num_batch_types"],
            domain_spec_batchnorm=final_config.DSBN,
            n_input_bins=final_config.n_bins,
            ecs_threshold=final_config.ecs_thres,
            explicit_zero_prob=final_config.explicit_zero_prob,
            use_fast_transformer=getattr(final_config, "fast_transformer", False),
            pre_norm=getattr(final_config, "pre_norm", False),
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if final_config.next_cell_pred_type == "pert":
           _orig_encode = final_model.encode_batch_with_perturb

        def _encode_force_t0(
            src,
            values,
            src_key_padding_mask,
            batch_size,
            batch_labels=None,
            pert_labels=None,
            pert_labels_next=None,
            perturbation=None,
            pert_scale=None,
            output_to_cpu=True,
            time_step=0,
            return_np=False,
            predict_expr=False,
            mvc_src=None,
            **kw,
        ):
        # Call the original encode using NAMED args so nothing gets misaligned.
        # We override only time_step -> 0 to force "next_cell_pred_type = identity".
            return _orig_encode(
                src=src,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
                batch_size=batch_size,
                batch_labels=batch_labels,
                pert_labels=pert_labels,
                pert_labels_next=pert_labels_next,
                perturbation=perturbation,
                pert_scale=pert_scale,
                output_to_cpu=output_to_cpu,
                time_step=0,
                return_np=return_np,
                predict_expr=predict_expr,
                mvc_src=mvc_src,
                **kw,
            )

        final_model.encode_batch_with_perturb = _encode_force_t0

        final_run = wandb.init(
            project=(run_session.project if hasattr(run_session, "project") else None),
            name="final_full_model",
            group="final",
            reinit=True,
            config=final_config.as_dict() if hasattr(final_config, "as_dict") else None,
        )
        final_run.config.update({"final_with_best_optuna": True})

        final_result = wrapper_train(
            final_model, final_config, final_data,
            eval_adata_dict={"validation": final_data["adata_sorted"]},
            save_dir=final_dir,
            fold=0,
            run=final_run,
            trial=None,
            use_early_stopping=True,
        )

        torch.save(final_result["model"].state_dict(), final_dir / "best_model.pt")
        torch.save(final_data["vocab"], final_dir / "vocab.pt")

        final_artifact = wandb.Artifact("best_model_final_full", type="model")
        final_artifact.add_file(str(final_dir / "best_model.pt"))
        final_artifact.add_file(str(final_dir / "vocab.pt"))
        final_run.log_artifact(final_artifact)
        final_run.finish()

        logger.info(
            f"[FINAL] best_epoch={final_result['best_model_epoch']} | "
            f"best_val_loss={final_result['best_val_loss']:.6f} | saved in {final_dir}"
        )
    except Exception as e:
        logger.exception(f"[FINAL] Failed to train final model: {e}")

    # finish wandb session created earlier (if any)
    try:
        run_session.finish()
    except Exception:
        pass
    import wandb as _wandb
    _wandb.finish()
    logger.info("Done.")
