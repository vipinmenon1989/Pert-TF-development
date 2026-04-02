#!/usr/bin/env python3
import os, json, logging, sys
from pathlib import Path
import numpy as np
import scanpy as sc
import torch

# --- PERTTF IMPORTS ---
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function_vipin import wrapper_train
from perttf.utils.safe_config import SafeConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("perttf_full")

# -------------------------
# FIXED BEST PARAMS
# -------------------------
BEST_PARAMS = dict(
    lr=0.0012273661723331349,
    perturbation_classifier_weight=50.0,
    dropout=0.15579754426081674,
    n_hvg=3000,
    KEEP_WT_FRAC=0.1,
    early_stop_patience=10,
    early_stop_min_delta=0.01,
    layer_size=64,
    nlayers=4,
    nhead=4,
    batch_size=64,
    epochs=100, 
)

def vocab_to_stoi(v):
    """Helper to convert torchtext vocab to standard python dict for JSON saving"""
    if hasattr(v, "get_stoi"): return dict(v.get_stoi())
    if hasattr(v, "stoi"): return dict(v.stoi)
    if hasattr(v, "token2id"): return dict(v.token2id)
    if hasattr(v, "get_itos"):
        itos = list(v.get_itos())
        return {tok: i for i, tok in enumerate(itos)}
    itos = list(v)
    return {tok: i for i, tok in enumerate(itos)}

def main():
    # ---------------- Setup ----------------
    preprocessed_h5ad = Path(os.environ["PREPROCESSED_H5AD"])
    outdir = Path(os.environ["OUTDIR"])
    seed = int(os.environ.get("SEED", "42"))
    max_epochs = int(os.environ.get("MAX_EPOCHS", str(BEST_PARAMS["epochs"])))
    wandb_mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()      

    if torch.cuda.is_available():
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Fixed output directory for full model
    fold_dir = outdir / "full_model"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training on COMPLETE DATASET | Output: {fold_dir}")

    # ---------------- Config ----------------
    base_defaults = dict(
        seed=seed, dataset_name="pancreatic", do_train=True, load_model=None,
        next_cell_pred_type="identity", GEPC=True, ecs_thres=0.7, ecs_weight=1.0, 
        dab_weight=0.0, this_weight=1.0, next_weight=10.0, n_rounds=1,
        cell_type_classifier=True, cell_type_classifier_weight=1.0,
        perturbation_classifier_weight=1.0, perturbation_input=True, CCE=True,
        mask_ratio=0.15, n_bins=51, schedule_ratio=0.99, schedule_interval=1,
        save_eval_interval=5, log_interval=60, do_sample_in_train=False,
        fast_transformer=True, pre_norm=False, amp=True, explicit_zero_prob=True,
        USE_HVG=True, n_hvg=3000, mask_value=-1, pad_value=-2, pad_token="<pad>",
        max_cells=None, max_cells_seed=1337, filter_missing_subcluster=True,
        KEEP_WT_FRAC=0.1, ps_weight=0.0, ADV=False, adv_weight=10000,
        DSBN=False, use_batch_label=False, per_seq_batch_sample=False,
    )
    base_defaults.update(BEST_PARAMS)
    base_defaults["epochs"] = int(max_epochs)
    
    cfg, _ = generate_config(base_defaults, wandb_mode=("disabled" if wandb_mode == "disabled" else "online"))
    cfg = SafeConfig(cfg)
    (fold_dir / "config.json").write_text(json.dumps(cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg.__dict__), indent=2, default=str))

    # ---------------- Data Load (NO SPLITS) ----------------
    ad = sc.read_h5ad(str(preprocessed_h5ad))
    
    # CRITICAL: Force every cell to be 'train'
    ad.obs["split"] = "train"
    
    # ---------------- Prepare Datasets ----------------
    fold_data = produce_training_datasets(ad, cfg, next_cell_pred="identity")
    
    # HACK: Map Validation -> Train so wrapper_train doesn't crash or error out
    # The 'validation loss' in logs will actually be the training loss.
    fold_data["valid_loader"] = fold_data["train_loader"]
    fold_data["adata_sorted"] = ad 

    vocab = fold_data["vocab"]
    
    # --- SAVE VOCAB (Both Formats) ---
    torch.save(vocab, fold_dir / "vocab.pt")
    
    try:
        vocab_dict = vocab_to_stoi(vocab)
        with open(fold_dir / "vocab.json", "w") as f:
            json.dump(vocab_dict, f, indent=2)
        logger.info(f"Saved vocab.json and vocab.pt to {fold_dir}")
    except Exception as e:
        logger.error(f"Failed to save vocab.json: {e}")

    # ---------------- Build Model ----------------
    ntokens = len(vocab)
    model = PerturbationTFModel(
        n_pert=fold_data["n_perturb"], nlayers_pert=3, n_ps=1, ntoken=ntokens,
        d_model=cfg.layer_size, nhead=cfg.nhead, d_hid=cfg.layer_size,
        nlayers=cfg.nlayers, nlayers_cls=3, n_cls=fold_data["n_cls"],
        vocab=vocab, dropout=cfg.dropout, pad_token=cfg.pad_token,
        pad_value=cfg.pad_value, do_mvc=cfg.GEPC, do_dab=(getattr(cfg, "dab_weight", 0.0) > 0),
        use_batch_labels=getattr(cfg, "use_batch_label", False),
        num_batch_labels=fold_data["num_batch_types"],
        domain_spec_batchnorm=getattr(cfg, "DSBN", False),
        n_input_bins=cfg.n_bins, ecs_threshold=cfg.ecs_thres,
        explicit_zero_prob=cfg.explicit_zero_prob,
        use_fast_transformer=getattr(cfg, "fast_transformer", False),
        pre_norm=getattr(cfg, "pre_norm", False),
    ).to(device)

    # ---------------- Train ----------------
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "PertTF_Full"), config=cfg.as_dict(), mode=wandb_mode)

    logger.info("Starting Full Dataset Training...")
    train_result = wrapper_train(
        model, cfg, fold_data,
        eval_adata_dict={"validation": fold_data["adata_sorted"]},
        save_dir=fold_dir, fold=1, run=run, use_early_stopping=True,
    )

    best_model = train_result["model"]
    torch.save(best_model.state_dict(), fold_dir / "best_model.pt")
    logger.info(f"[DONE] Full model saved to {fold_dir / 'best_model.pt'}")

if __name__ == "__main__":
    main()
