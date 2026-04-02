#!/usr/bin/env python3
import os, json, logging, subprocess
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
# CHANGED: Added StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold

from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function_vipin import wrapper_train
from perttf.utils.safe_config import SafeConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger("perttf_fold")


# -------------------------
# FIXED BEST PARAMS (your final values)
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
    epochs=100,  # early stopping controls actual stop
)


def vocab_to_stoi(v):
    if hasattr(v, "get_stoi"):
        return dict(v.get_stoi())
    if hasattr(v, "stoi"):
        return dict(v.stoi)
    if hasattr(v, "token2id"):
        return dict(v.token2id)
    if hasattr(v, "get_itos"):
        itos = list(v.get_itos())
        return {tok: i for i, tok in enumerate(itos)}
    itos = list(v)
    return {tok: i for i, tok in enumerate(itos)}


def _parse_int_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return None
    return int(float(s))


def _log_gpu_proof():
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    if not torch.cuda.is_available():
        logger.info("torch.cuda.is_available()=False (CPU mode)")
        return
    logger.info(f"torch.cuda.device_count()={torch.cuda.device_count()} (should be 1 if bound correctly)")
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU name={props.name} | total_mem_GB={props.total_memory/1e9:.2f}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
            text=True
        ).strip()
        logger.info("nvidia-smi visible GPUs:\n" + (out if out else "<empty>"))
    except Exception as e:
        logger.warning(f"nvidia-smi query failed (not fatal): {e}")


def main():
    # Required env inputs
    preprocessed_h5ad = Path(os.environ["PREPROCESSED_H5AD"])
    outdir = Path(os.environ["OUTDIR"])

    # Fold / CV
    fold_id = int(os.environ.get("FOLD_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "1")))
    n_splits = int(os.environ.get("N_SPLITS", "5"))
    seed = int(os.environ.get("SEED", "42"))

    # Runtime knobs
    max_epochs = int(os.environ.get("MAX_EPOCHS", str(BEST_PARAMS["epochs"])))
    max_cells = _parse_int_or_none(os.environ.get("MAX_CELLS", "none"))

    # Behavioral knobs
    next_cell_pred = os.environ.get("NEXT_CELL_PRED", "identity").strip().lower()  # "pert" or "identity"
    wandb_mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()      # "disabled" or "online"
    wandb_project = os.environ.get("WANDB_PROJECT", "PertTF")
    wandb_group = os.environ.get("WANDB_RUN_GROUP", f"cv_{os.environ.get('SLURM_JOB_ID','nogroup')}")

    if next_cell_pred not in {"pert", "identity"}:
        raise ValueError(f"NEXT_CELL_PRED must be 'pert' or 'identity' (got {next_cell_pred})")

    # ---------------- GPU binding ----------------
    if torch.cuda.is_available():
        # respect CUDA_VISIBLE_DEVICES if set by SLURM wrapper
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            # fallback: bind deterministically
            os.environ["CUDA_VISIBLE_DEVICES"] = str((fold_id - 1) % torch.cuda.device_count())
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    fold_dir = outdir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fold {fold_id}/{n_splits} | preprocessed={preprocessed_h5ad} | out={fold_dir}")
    logger.info(f"MAX_EPOCHS={max_epochs} | MAX_CELLS={max_cells} | NEXT_CELL_PRED={next_cell_pred} | WANDB_MODE={wandb_mode}")
    _log_gpu_proof()

    # ---------------- Build config ----------------
    base_defaults = dict(
        seed=seed,
        dataset_name="pancreatic",
        do_train=True,

        # Keep consistent with your train_function expectations
        load_model=None,

        # key behavior
        next_cell_pred_type=next_cell_pred,  # <-- IMPORTANT: matches produce_training_datasets below

        # losses / heads
        GEPC=True,
        ecs_thres=0.7,
        ecs_weight=1.0,
        dab_weight=0.0,

        this_weight=1.0,
        next_weight=10.0,
        n_rounds=1,

        cell_type_classifier=True,
        cell_type_classifier_weight=1.0,
        perturbation_classifier_weight=50.0,
        perturbation_input=False,

        CCE=True,

        # masking/binning
        mask_ratio=0.15,
        n_bins=51,

        # train schedule
        schedule_ratio=0.99,
        schedule_interval=1,
        save_eval_interval=5,
        log_interval=60,
        do_sample_in_train=False,

        # transformer settings
        fast_transformer=True,   # set False if you want to silence flash-attn warning
        pre_norm=False,
        amp=True,
        explicit_zero_prob=True,

        # vocab / padding
        USE_HVG=True,
        n_hvg=3000,
        mask_value=-1,
        pad_value=-2,
        pad_token="<pad>",

        # runtime controls
        max_cells=max_cells,
        max_cells_seed=1337,
        filter_missing_subcluster=True,
        KEEP_WT_FRAC=0.1,

        # ---- CRITICAL: this is what crashed you ----
        ps_weight=0.0,

        # adversarial knobs (kept for safety even if ADV=False)
        ADV=False,
        adv_weight=10000,
        adv_E_delay_epochs=2,
        adv_D_delay_epochs=2,
        lr_ADV=1e-3,

        # batch / dsbn flags some code may touch
        DSBN=False,
        use_batch_label=False,
        per_seq_batch_sample=False,
    )

    # apply fixed best params + runtime epochs
    base_defaults.update(BEST_PARAMS)
    base_defaults["epochs"] = int(max_epochs)
    base_defaults["early_stop_patience"] = int(BEST_PARAMS["early_stop_patience"])
    base_defaults["early_stop_min_delta"] = float(BEST_PARAMS["early_stop_min_delta"])

    cfg, _ = generate_config(base_defaults, wandb_mode=("disabled" if wandb_mode == "disabled" else "online"))
    cfg = SafeConfig(cfg)

    # Persist exact config used (for post-mortems)
    cfg_dict = cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg.__dict__)
    (fold_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    # ---------------- Fold split (Stratified) ----------------
    ad_backed = sc.read_h5ad(str(preprocessed_h5ad), backed="r")
    obs_names = np.array(ad_backed.obs_names)
    obs_df = ad_backed.obs.copy()
    
    # CRITICAL: We need the labels to stratify. 
    # Using 'celltype_2' as it is your primary label column.
    stratify_labels = obs_df["celltype_2"].astype(str).values
    del ad_backed

    # CHANGED: Use StratifiedKFold to ensure rare classes are in both Train and Val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # CRITICAL: Pass stratify_labels as the second argument (y)
    tr_idx, va_idx = list(skf.split(obs_names, stratify_labels))[fold_id - 1]

    obs_df["split"] = "unused"
    obs_df.loc[obs_names[tr_idx], "split"] = "train"
    obs_df.loc[obs_names[va_idx], "split"] = "validation"

    # ---------------- Load full matrix (in-memory) ----------------
    ad = sc.read_h5ad(str(preprocessed_h5ad))
    ad.obs = obs_df

    # ==============================================================================
    # BLUNT FIX: OVERSAMPLING RARE "ESC (D3)" CLASS
    # ==============================================================================
    target_rare_cls = "ESC (D3)"
    upsample_factor = 20  # Repeat these cells 20 times to force the model to learn them

    # 1. Identify training cells of this type
    is_train = ad.obs["split"] == "train"
    is_rare = ad.obs["celltype_2"] == target_rare_cls
    rare_train_idx = ad.obs_names[is_train & is_rare]

    if len(rare_train_idx) > 0:
        logger.info(f"Oversampling '{target_rare_cls}': Found {len(rare_train_idx)} training cells.")
        
        # 2. Extract them
        ad_rare = ad[rare_train_idx].copy()
        
        # 3. Create duplicates
        to_concat = [ad] # Start with original data
        
        for i in range(upsample_factor):
            ad_clone = ad_rare.copy()
            # CRITICAL: We must make indices unique or Scanpy/Pandas will crash
            ad_clone.obs_names = [f"{x}_dupe_{i}" for x in ad_clone.obs_names]
            to_concat.append(ad_clone)
            
        # 4. Concatenate back together
        ad = sc.concat(to_concat)
        
        # 5. Verify
        new_count = len(ad.obs_names[(ad.obs["split"] == "train") & (ad.obs["celltype_2"] == target_rare_cls)])
        logger.info(f"Oversampling Complete. '{target_rare_cls}' training count went from {len(rare_train_idx)} -> {new_count}")
    else:
        logger.warning(f"Oversampling failed: No '{target_rare_cls}' cells found in TRAINING split.")
    # ==============================================================================

    # Enforce max_cells WITHOUT touching validation
    cfg_max_cells = _parse_int_or_none(getattr(cfg, "max_cells", None))
    if cfg_max_cells is not None and ad.n_obs > cfg_max_cells:
        train_mask = (ad.obs["split"].values == "train")
        val_mask = (ad.obs["split"].values == "validation")

        n_val = int(val_mask.sum())
        budget_for_train = max(0, int(cfg_max_cells) - n_val)

        if budget_for_train < int(train_mask.sum()):
            rng = np.random.default_rng(seed)
            train_idx_all = np.where(train_mask)[0]
            keep_train = rng.choice(train_idx_all, size=budget_for_train, replace=False)
            keep_idx = np.concatenate([np.where(val_mask)[0], keep_train])
            keep_idx.sort()
            ad = ad[keep_idx].copy()
            logger.info(f"Applied max_cells={cfg_max_cells}: now n_obs={ad.n_obs} (val kept={n_val})")

    # ---------------- Dataset ----------------
    fold_data = produce_training_datasets(ad, cfg, next_cell_pred=next_cell_pred)

    # Minimal sanity checks (these are what your eval/test functions expect)
    ad_eval = fold_data.get("adata_sorted", None)
    if ad_eval is not None:
        missing = [k for k in ["celltype_2", "genotype"] if k not in ad_eval.obs.columns]
        if missing:
            raise RuntimeError(f"Missing required labels in adata_sorted.obs: {missing}")
        logger.info(f"[fold {fold_id}] validation label peek: "
                    f"celltype_2_unique={ad_eval.obs['celltype_2'].nunique()} "
                    f"genotype_unique={ad_eval.obs['genotype'].nunique()}")

    # Save vocab
    vocab = fold_data["vocab"]
    torch.save(vocab, fold_dir / "vocab.pt")
    (fold_dir / "vocab.json").write_text(json.dumps(vocab_to_stoi(vocab), indent=2))

    # ---------------- Model ----------------
    ntokens = len(vocab)

    use_batch_label = getattr(cfg, "use_batch_label", False)
    dsbn = getattr(cfg, "DSBN", False)
    fast_tx = getattr(cfg, "fast_transformer", False)
    pre_norm = getattr(cfg, "pre_norm", False)

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
        do_dab=(getattr(cfg, "dab_weight", 0.0) > 0),
        use_batch_labels=use_batch_label,
        num_batch_labels=fold_data["num_batch_types"],
        domain_spec_batchnorm=dsbn,
        n_input_bins=cfg.n_bins,
        ecs_threshold=cfg.ecs_thres,
        explicit_zero_prob=cfg.explicit_zero_prob,
        use_fast_transformer=fast_tx,
        pre_norm=pre_norm,
    ).to(device)

    # ---------------- W&B (optional) ----------------
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=wandb_project,
            name=f"cv_fold_{fold_id}",
            group=wandb_group,
            reinit=True,
            config=cfg_dict,
            mode=wandb_mode,
        )
        run.config.update({"fold": fold_id}, allow_val_change=True)

    # ---------------- Train ----------------
    train_result = wrapper_train(
        model, cfg, fold_data,
        eval_adata_dict={"validation": fold_data["adata_sorted"]},
        save_dir=fold_dir,
        fold=fold_id,
        run=run,
        trial=None,
        use_early_stopping=True,
    )

    best_model = train_result["model"]
    best_val_loss = float(train_result.get("best_val_loss", np.nan))
    best_epoch = int(train_result.get("best_model_epoch", -1))

    torch.save(best_model.state_dict(), fold_dir / "best_model.pt")

    metrics = {"fold": fold_id, "best_val_loss": best_val_loss, "best_epoch": best_epoch}
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info(f"[DONE] fold={fold_id} best_val_loss={best_val_loss} best_epoch={best_epoch}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
