#!/usr/bin/env python3
import os, json, logging, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Import your model modules
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.train_data_gen import produce_training_datasets

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("complete_umap")

# --- ROBUST CONFIG CLASS (FIXED: Supports both .attr and .get) ---
class LocalSafeConfig:
    def __init__(self, cfg_dict):
        self._cfg = cfg_dict
        
    def __getattr__(self, name):
        # Handles config.attribute access
        return self.get(name)

    def get(self, key, default=None):
        # Handles config.get('attribute', default) access
        val = self._cfg.get(key)
        if val is not None:
            return val
        
        # Hardcoded defaults for parameters missing from your older JSONs
        if key == "special_tokens":
            return ["<pad>", "<cls>", "<eoc>"]
        
        return default

def main():
    parser = argparse.ArgumentParser(description="Recover UMAP from saved model")
    parser.add_argument("--data", required=True, help="Path to preprocessed .h5ad")
    parser.add_argument("--outdir", required=True, help="Base results directory")
    parser.add_argument("--fold", type=int, required=True, help="Fold ID")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    # 1. SMART PATH DETECTION
    base_outdir = Path(args.outdir)
    fold_id = args.fold
    
    # Logic to find the model folder no matter how you pointed to it
    if (base_outdir / f"fold_{fold_id}" / "best_model.pt").exists():
        fold_dir = base_outdir / f"fold_{fold_id}"
    elif (base_outdir / "best_model.pt").exists():
        fold_dir = base_outdir
    elif (base_outdir.parent / f"fold_{fold_id}" / "best_model.pt").exists():
         fold_dir = base_outdir.parent / f"fold_{fold_id}"
    else:
        logger.error(f"CRITICAL: Could not find 'best_model.pt' in {base_outdir}")
        sys.exit(1)

    logger.info(f"Recovering FOLD {fold_id} from: {fold_dir}")
    
    # 2. LOAD CONFIG & VOCAB
    with open(fold_dir / "config.json", "r") as f:
        cfg_dict = json.load(f)
    # Use our FIXED robust class
    cfg = LocalSafeConfig(cfg_dict)
    
    # Allow custom classes (SimpleVocab)
    vocab = torch.load(fold_dir / "vocab.pt", weights_only=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. LOAD DATA
    logger.info("Loading AnnData...")
    preprocessed_h5ad = Path(args.data)
    ad_backed = sc.read_h5ad(str(preprocessed_h5ad), backed="r")
    obs_names = np.array(ad_backed.obs_names)
    obs_df = ad_backed.obs.copy()
    stratify_labels = obs_df["celltype_2"].astype(str).values
    del ad_backed

    # Recreate Split
    seed = cfg.get("seed", 42)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = list(skf.split(obs_names, stratify_labels))[fold_id - 1]

    # Load Full Data
    ad = sc.read_h5ad(str(preprocessed_h5ad))
    ad.obs = obs_df 
    
    # 4. PREPARE DATA LOADERS
    logger.info("Generating datasets...")
    next_cell_pred = cfg.get("next_cell_pred_type", "identity")
    
    # Pass 'vocab' explicitly
    data_info = produce_training_datasets(
        ad, cfg, 
        next_cell_pred=next_cell_pred, 
        vocab=vocab 
    )
    
    train_loader = data_info['train_loader']
    valid_loader = data_info['valid_loader']
    train_ids = data_info['cell_ids_train']
    valid_ids = data_info['adata_sorted'].obs.index

    # 5. BUILD & LOAD MODEL
    ntokens = len(vocab)
    model = PerturbationTFModel(
        n_pert=data_info["n_perturb"], nlayers_pert=3, n_ps=1, ntoken=ntokens,
        d_model=cfg.layer_size, nhead=cfg.nhead, d_hid=cfg.layer_size,
        nlayers=cfg.nlayers, nlayers_cls=3, n_cls=data_info["n_cls"],
        vocab=vocab, dropout=cfg.dropout, pad_token=cfg.pad_token,
        pad_value=cfg.pad_value, do_mvc=cfg.GEPC, do_dab=(cfg.get("dab_weight", 0.0) > 0),
        use_batch_labels=cfg.get("use_batch_label", False),
        num_batch_labels=data_info["num_batch_types"],
        domain_spec_batchnorm=cfg.get("DSBN", False),
        n_input_bins=cfg.n_bins, ecs_threshold=cfg.ecs_thres,
        explicit_zero_prob=cfg.explicit_zero_prob,
        use_fast_transformer=cfg.get("fast_transformer", False),
        pre_norm=cfg.get("pre_norm", False),
    ).to(device)

    model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device))
    model.eval()

    # 6. INFERENCE LOOP
    pad_id = vocab[cfg.pad_token]

    def run_inference(loader):
        embeddings = []
        def get_latent_hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            if output.shape[1] == cfg.batch_size: 
                 output = output.transpose(0, 1)
            embeddings.append(output.mean(dim=1).detach().cpu().numpy())

        try:
            handle = model.transformer_encoder.register_forward_hook(get_latent_hook)
        except AttributeError:
            handle = model.encoder.register_forward_hook(get_latent_hook)

        with torch.no_grad():
            for batch in loader:
                batch_data = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                src = batch_data.get("gene_ids")
                if src is None: src = batch_data.get("genes")
                values = batch_data.get("values")
                
                # Manual Padding Mask
                src_key_padding_mask = (src == pad_id)

                try:
                    _ = model(src, values, src_key_padding_mask)
                except Exception:
                    # Fallback
                    _ = model(src, values)
        
        handle.remove()
        return np.concatenate(embeddings, axis=0)

    logger.info("Extracting embeddings...")
    train_embs = run_inference(train_loader)
    valid_embs = run_inference(valid_loader)

    # 7. STITCH & PLOT
    df_train = pd.DataFrame(train_embs, index=train_ids)
    df_valid = pd.DataFrame(valid_embs, index=valid_ids)
    
    df_full = pd.concat([df_train, df_valid])
    df_aligned = df_full.reindex(ad.obs_names).fillna(0)

    ad.obsm["X_perttf"] = df_aligned.values
    
    logger.info("Computing UMAP...")
    sc.pp.neighbors(ad, use_rep="X_perttf")
    sc.tl.umap(ad)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc.pl.umap(ad, color='celltype_2', ax=axes[0], show=False, title=f"Complete: Celltype (Fold {fold_id})")
    sc.pl.umap(ad, color='genotype', ax=axes[1], show=False, title=f"Complete: Genotype (Fold {fold_id})")
    plt.tight_layout()
    
    out_fig = fold_dir / "UMAP_COMPLETE_CELLS_FINAL"
    plt.savefig(f"{out_fig}.png", dpi=300)
    plt.savefig(f"{out_fig}.pdf")
    ad.write_h5ad(fold_dir / "complete_data_with_latent.h5ad")
    
    logger.info(f"SUCCESS: Saved to {out_fig}")

if __name__ == "__main__":
    main()