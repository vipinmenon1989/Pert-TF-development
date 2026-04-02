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
# HYBRID CLASS: Fixes the Mapping/Object Conflict
# -------------------------
class HybridConfig(dict):
    """Allows both config['key'] and config.key to satisfy inconsistent library calls."""
    def __init__(self, config_obj):
        data = config_obj.as_dict() if hasattr(config_obj, "as_dict") else dict(config_obj.__dict__)
        super().__init__(data)
        self.__dict__ = self 

# -------------------------
# FIXED BEST PARAMS
# -------------------------
BEST_PARAMS = dict(
    lr=0.0012273661723331349,
    perturbation_classifier_weight=1.0, # Will be overridden
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
    if hasattr(v, "get_stoi"): return dict(v.get_stoi())
    if hasattr(v, "stoi"): return dict(v.stoi)
    if hasattr(v, "token2id"): return dict(v.token2id)
    return {tok: i for i, tok in enumerate(v)}

def main():
    # 1. HARDWARE DIAGNOSTICS
    # Proves the GPU and LD_LIBRARY_PATH are working before loading data
    logger.info("Verifying CUDA and GPU health...")
    if not torch.cuda.is_available():
        logger.error("FATAL: CUDA not available. Check LD_LIBRARY_PATH.")
        sys.exit(1)
    
    try:
        test_tensor = torch.zeros(1).cuda()
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.error(f"FATAL: GPU communication failed: {e}")
        sys.exit(1)

    # 2. Setup Paths & Sweep Weight
    preprocessed_h5ad = Path(os.environ["PREPROCESSED_H5AD"])
    outdir = Path(os.environ["OUTDIR"])
    
    # Pull weight from Slurm Array
    target_weight = float(os.environ.get("PERT_LOSS_WEIGHT", 1.0))
    fold_dir = outdir / f"weight_{target_weight}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    seed = int(os.environ.get("SEED", "42"))
    max_epochs = int(os.environ.get("MAX_EPOCHS", str(BEST_PARAMS["epochs"])))
    wandb_mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()      

    # 3. Config Generation
    base_defaults = dict(
        seed=seed, dataset_name="pancreatic", do_train=True, load_model=None,
        next_cell_pred_type="identity", GEPC=True, ecs_thres=0.7, ecs_weight=1.0, 
        dab_weight=0.0, this_weight=1.0, next_weight=10.0, n_rounds=1,
        cell_type_classifier=True, cell_type_classifier_weight=1.0,
        perturbation_classifier_weight=target_weight, # Updated weight
        perturbation_input=True, CCE=True,
        mask_ratio=0.15, n_bins=51, schedule_ratio=0.99, schedule_interval=1,
        save_eval_interval=5, log_interval=60, do_sample_in_train=False,
        fast_transformer=True, pre_norm=False, amp=True, explicit_zero_prob=True,
        USE_HVG=True, n_hvg=3000, mask_value=-1, pad_value=-2, pad_token="<pad>",
        max_cells=None, max_cells_seed=1337, filter_missing_subcluster=True,
        KEEP_WT_FRAC=0.1, ps_weight=0.0, ADV=False, adv_weight=10000,
        DSBN=False, use_batch_label=False, per_seq_batch_sample=False,
    )
    base_defaults.update({k:v for k,v in BEST_PARAMS.items() if k != 'perturbation_classifier_weight'})
    base_defaults["epochs"] = int(max_epochs)
    
    cfg_raw, _ = generate_config(base_defaults, wandb_mode=("disabled" if wandb_mode == "disabled" else "online"))
    
    # WRAP IN HYBRID CLASS: This is the fix for the AttributeError/TypeError
    cfg = HybridConfig(SafeConfig(cfg_raw))
    
    (fold_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # 4. Data Loading
    logger.info(f"Loading data from {preprocessed_h5ad}...")
    ad = sc.read_h5ad(str(preprocessed_h5ad))
    ad.obs["split"] = "train"
    
    # HybridConfig satisfies the internal **config unpacking here
    fold_data = produce_training_datasets(ad, cfg, next_cell_pred="identity")
    fold_data["valid_loader"] = fold_data["train_loader"]
    fold_data["adata_sorted"] = ad 

    # 5. Vocab Handling
    vocab = fold_data["vocab"]
    torch.save(vocab, fold_dir / "vocab.pt")
    try:
        with open(fold_dir / "vocab.json", "w") as f:
            json.dump(vocab_to_stoi(vocab), f, indent=2)
    except Exception as e:
        logger.error(f"Vocab JSON export failed: {e}")

    # 6. Model Initialization
    # HybridConfig satisfies cfg.attribute calls here
    model = PerturbationTFModel(
        n_pert=fold_data["n_perturb"], nlayers_pert=3, n_ps=1, ntoken=len(vocab),
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
    ).to("cuda")

    # 7. Training
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "PertTF_Full"), config=cfg, mode=wandb_mode, name=f"weight_{target_weight}")

    logger.info("Starting training loop...")
    train_result = wrapper_train(
        model, cfg, fold_data,
        eval_adata_dict={"validation": fold_data["adata_sorted"]},
        save_dir=fold_dir, fold=1, run=run, use_early_stopping=True,
    )

    torch.save(train_result["model"].state_dict(), fold_dir / "best_model.pt")
    logger.info(f"Done. Results in {fold_dir}")

if __name__ == "__main__":
    main()