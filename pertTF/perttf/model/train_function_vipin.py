import time
import torch
import random
import warnings
from pathlib import Path
import copy
import numpy as np

from typing import Dict, Mapping, Optional, Tuple, Any, Union
from typing import List, Tuple

from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from anndata import AnnData
import scanpy as sc

import wandb
from scipy.sparse import issparse

import scgpt as scg
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from perttf.utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel, AdversarialDiscriminator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perttf.model.train_data_gen import prepare_data,prepare_dataloader
from perttf.utils.set_optimizer import create_optimizer_dict
from perttf.custom_loss import cce_loss, criterion_neg_log_bernoulli, masked_mse_loss
from perttf.utils.plot import process_and_log_umaps

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, r2_score, average_precision_score
)
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import label_binarize
import pandas as pd
import os 
import gc
from perttf.utils.safe_config import SafeConfig
from pathlib import Path

def save_fig_both(fig, out_base, dpi=300, **savefig_kwargs):
    """
    Save matplotlib Figure as both PNG + PDF.
    out_base: path WITHOUT extension (Path or str).
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # sensible default, but allow caller override
    savefig_kwargs = dict(savefig_kwargs)
    savefig_kwargs.setdefault("bbox_inches", "tight")

    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, **savefig_kwargs)
    fig.savefig(pdf_path, **savefig_kwargs)

    # hard proof it saved
    if (not png_path.exists()) or png_path.stat().st_size == 0:
        raise RuntimeError(f"PNG not saved or empty: {png_path}")
    if (not pdf_path.exists()) or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"PDF not saved or empty: {pdf_path}")

    return png_path, pdf_path

def save_best_expression_csv(save_dir_best, fold, best_model_epoch, best_val_loss, pear, spear, r2, logger=None):
    """
    Write regression-style summary for best checkpoint:
    epoch,best_val_loss,pearson,spearman,r2
    """
    metrics_path = Path(save_dir_best) / f"best_epoch_expression_fold{fold}.csv"
    row = {
        "epoch": best_model_epoch,
        "best_val_loss": best_val_loss,
        "pearson": pear,
        "spearman": spear,
        "r2": r2,
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    if logger:
        logger.info(f"[best-metrics] Saved expression metrics CSV → {metrics_path}")


def compute_and_save_perclass_tables(
    model,
    valid_loader,
    data_gen,
    config,
    device,
    save_dir_best,
    fold,
    logger=None,
):
    """
    Build per-class classification tables for:
      - celltype head
      - genotype/pert head

    Writes TWO CSVs into save_dir_best:
      best_epoch_celltype_fold{fold}.csv
      best_epoch_genotype_fold{fold}.csv

    Each CSV row:
        label,AUC,AUPR,F1
        Alpha,0.82,0.77,0.70
        Beta,...
    """
    import inspect, os
    this_file = inspect.getfile(inspect.currentframe())
    print(f"[RUNTIME CHECK eval_testdata] running from {this_file}", flush=True)
    print("[RUNTIME CHECK eval_testdata] NEW guarded version loaded", flush=True)

    model.eval()

    # maps like: { "Alpha":0, "Beta":1, ... }
    celltype_to_index  = data_gen["cell_type_to_index"]
    genotype_to_index  = data_gen["genotype_to_index"]

    # invert to {0:"Alpha",1:"Beta",...}
    index_to_celltype  = {v: k for k, v in celltype_to_index.items()}
    index_to_genotype  = {v: k for k, v in genotype_to_index.items()}

    all_ct_true   = []
    all_ct_logits = []

    all_gt_true   = []
    all_gt_logits = []

    with torch.no_grad():
        for batch_data in valid_loader:
            gene_ids = batch_data["gene_ids"].to(device)
            inp      = batch_data["values"].to(device)

            # src_key_padding_mask
            skpm = None
            try:
                pad_id = data_gen["vocab"][config.pad_token]
                skpm   = gene_ids.eq(pad_id)
            except Exception:
                skpm = None

            # try common label field names for true labels
                        # try common label field names for true labels WITHOUT using `or`
            ct_true = None
            if "celltype_labels" in batch_data and batch_data["celltype_labels"] is not None:
                ct_true = batch_data["celltype_labels"]
            elif "celltypes_labels" in batch_data and batch_data["celltypes_labels"] is not None:
                ct_true = batch_data["celltypes_labels"]
            elif "celltype_label" in batch_data and batch_data["celltype_label"] is not None:
                ct_true = batch_data["celltype_label"]

            gt_true = None
            if "genotype_labels" in batch_data and batch_data["genotype_labels"] is not None:
                gt_true = batch_data["genotype_labels"]
            elif "perturbation_labels" in batch_data and batch_data["perturbation_labels"] is not None:
                gt_true = batch_data["perturbation_labels"]
            elif "genotype_label" in batch_data and batch_data["genotype_label"] is not None:
                gt_true = batch_data["genotype_label"]

            if ct_true is not None:
                ct_true = ct_true.to(device)
            if gt_true is not None:
                gt_true = gt_true.to(device)

            # forward
            with torch.cuda.amp.autocast(enabled=config.amp):
                out_dict = model(
                    gene_ids,
                    inp,
                    src_key_padding_mask=skpm,
                    batch_labels=(
                        batch_data["batch_labels"].to(device)
                        if getattr(config, "use_batch_label", False)
                        and "batch_labels" in batch_data
                        else None
                    ),
                    pert_labels=(
                        batch_data["perturbation_labels"].to(device)
                        if getattr(config, "perturbation_input", False)
                        and "perturbation_labels" in batch_data
                        else None
                    ),
                    MVC=getattr(config, "GEPC", False),
                    ECS=(getattr(config, "ecs_thres", 0) > 0),
                    CLS=getattr(config, "cell_type_classifier", False),
                    PERTPRED=(getattr(config, "perturbation_classifier_weight", 0) > 0),
                    PSPRED=(getattr(config, "ps_weight", 0) > 0),
                )

            # logits from heads
                        # helper: try multiple possible head names
            def _pick_first_key(d, keys):
                for k in keys:
                    if k in d:
                        return d[k]
                return None

            ct_logits = _pick_first_key(
                out_dict,
                ["cls_output", "cls_logits", "cls_preds", "celltype_logits", "celltype_head"]
            )
            gt_logits = _pick_first_key(
                out_dict,
                ["pert_output", "pert_logits", "pert_preds", "genotype_logits", "genotype_head"]
            )

            # helper: check shape is [N, C] not scalar/bool/etc.
            def _looks_like_logits(x, n_expected):
                if x is None:
                    return False
                if isinstance(x, bool):
                    return False
                if not isinstance(x, torch.Tensor):
                    return False
                if x.ndim != 2:
                    return False
                if x.shape[0] != n_expected:
                    return False
                return True

            if ct_true is not None and isinstance(ct_true, torch.Tensor) and _looks_like_logits(ct_logits, ct_true.shape[0]):
                all_ct_true.append(ct_true.detach().cpu())
                all_ct_logits.append(ct_logits.detach().cpu())

            if gt_true is not None and isinstance(gt_true, torch.Tensor) and _looks_like_logits(gt_logits, gt_true.shape[0]):
                all_gt_true.append(gt_true.detach().cpu())
                all_gt_logits.append(gt_logits.detach().cpu())

    def _cat_or_none(tlist):
        if len(tlist) == 0:
            return None
        return torch.cat(tlist, dim=0).numpy()

    all_ct_true   = _cat_or_none(all_ct_true)
    all_ct_logits = _cat_or_none(all_ct_logits)
    all_gt_true   = _cat_or_none(all_gt_true)
    all_gt_logits = _cat_or_none(all_gt_logits)

    def _perclass_df(y_true_int, logits, index_to_label, csv_path, head_name):
        """
        y_true_int: [N] int labels
        logits:     [N, C]
        """
        if y_true_int is None or logits is None:
            if logger:
                logger.warning(f"[per-class] {head_name}: missing data, skipping save.")
            return

        # ensure numpy
        if isinstance(y_true_int, torch.Tensor):
            y_true_int = y_true_int.detach().cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        # shape sanity
        if y_true_int.ndim != 1:
            if logger:
                logger.warning(f"[per-class] {head_name}: y_true_int shape {y_true_int.shape} not 1D, skipping.")
            return
        if logits.ndim != 2:
            if logger:
                logger.warning(f"[per-class] {head_name}: logits shape {logits.shape} not 2D, skipping.")
            return
        if logits.shape[0] != y_true_int.shape[0]:
            if logger:
                logger.warning(f"[per-class] {head_name}: mismatch N ({logits.shape[0]} vs {y_true_int.shape[0]}), skipping.")
            return

        probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
        y_pred_int = probs.argmax(axis=1)

        n_classes = probs.shape[1]
        rows = []
        for class_idx in range(n_classes):
            label_name = index_to_label.get(class_idx, f"class_{class_idx}")

            # binary view: "is this class or not"
            y_true_bin  = (y_true_int == class_idx).astype(int)          # shape [N], 0/1
            y_pred_bin  = (y_pred_int == class_idx).astype(int)          # shape [N], 0/1
            y_score_bin = probs[:, class_idx]                            # shape [N], prob for this class

            # F1 (one-vs-rest)
            f1_c = f1_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0,
            )

            # Precision, Recall, Accuracy (one-vs-rest)
            precision_c = precision_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0,
            )
            recall_c = recall_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0,
            )
            accuracy_c = accuracy_score(
                y_true_bin,
                y_pred_bin,
            )

            # AUC / AUPR (guard for classes with all 0s or all 1s)
            try:
                auc_c = roc_auc_score(y_true_bin, y_score_bin)
            except Exception:
                auc_c = np.nan

            try:
                aupr_c = average_precision_score(y_true_bin, y_score_bin)
            except Exception:
                aupr_c = np.nan

            rows.append({
                head_name: label_name,
                "AUC": auc_c,
                "AUPR": aupr_c,
                "F1": f1_c,
                "Precision": precision_c,
                "Recall": recall_c,
                "Accuracy": accuracy_c,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        if logger:
            logger.info(f"[best-metrics] Saved per-class table → {csv_path}")

    celltype_csv = Path(save_dir_best) / f"best_epoch_celltype_fold{fold}.csv"
    genotype_csv = Path(save_dir_best) / f"best_epoch_genotype_fold{fold}.csv"

    try:
        _perclass_df(
            y_true_int=all_ct_true,
            logits=all_ct_logits,
            index_to_label=index_to_celltype,
            csv_path=celltype_csv,
            head_name="Celltype",
        )
    except Exception as e_ct:
        if logger:
            logger.warning(f"[per-class] Failed to save celltype table: {e_ct}")

    try:
        _perclass_df(
            y_true_int=all_gt_true,
            logits=all_gt_logits,
            index_to_label=index_to_genotype,
            csv_path=genotype_csv,
            head_name="Genotype",
        )
    except Exception as e_gt:
        if logger:
            logger.warning(f"[per-class] Failed to save genotype table: {e_gt}")

def forward_pass(model, batch_data, device, config, vocab, has_lochness_next_pred = False):
    input_gene_ids = batch_data["gene_ids"].to(device)
    input_values = batch_data["values"].to(device)
    target_values = batch_data["target_values"].to(device)
    target_values_next = batch_data["target_values_next"].to(device)
    batch_labels = batch_data["batch_labels"].to(device)
    celltype_labels = batch_data["celltype_labels"].to(device) #added
    perturbation_labels = batch_data["perturbation_labels"].to(device) #added

    celltype_labels_next = batch_data["celltype_labels_next"].to(device) #added
    perturbation_labels_next = batch_data["perturbation_labels_next"].to(device) #added


    src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
    with torch.cuda.amp.autocast(enabled=config.amp):
        #import pdb; pdb.set_trace()

        output_dict = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
            pert_labels = perturbation_labels if config.perturbation_input else None,
            pert_labels_next = perturbation_labels_next if (config.next_weight >0 or has_lochness_next_pred )  else None,
            MVC=config.GEPC,
            ECS=config.ecs_thres > 0,
            CLS=config.cell_type_classifier,
            PERTPRED = config.perturbation_classifier_weight > 0,
            PSPRED = config.ps_weight >0,
        )

    return output_dict


_UMAP_BUDGET: Dict[str, Any] = {}
# ---- dynamic compute budget (one-time, future-proof) ----
def _auto_budget(config, n_obs_hint=None, logger=None):
    """
    Derive safe defaults from config + dataset size.
    MODIFIED: Target ~80k cells for a balance of speed and density.
    """
    # Base on sequence length
    L = int(getattr(config, "max_seq_len", 0))

    # --- batch size autoscale (idempotent) ---
    try:
        if L > 6144:
            new_bs = max(4, int(config.batch_size) // 4)
        elif L > 4096:
            new_bs = max(8, int(config.batch_size) // 2)
        else:
            new_bs = int(config.batch_size)
        if new_bs != int(config.batch_size):
            if logger: logger.warning(f"[budget] long seq {L} → batch_size {config.batch_size} -> {new_bs}")
            config.batch_size = new_bs
    except Exception:
        pass

    # --- snapshot cadence (epoch frequency) ---
    default_every = 10 if L <= 4096 else 10
    os_every = os.environ.get("PERTTF_EVAL_EVERY", "")
    try:
        eval_every = int(os_every) if os_every.strip() else default_every
    except Exception:
        eval_every = default_every

    # --- UMAP subsample budget (MIDDLE GROUND: 100K) ---
    TARGET_CAP = 100000
    
    try:
        n_obs = int(n_obs_hint) if n_obs_hint is not None else None
    except Exception:
        n_obs = None

    if n_obs is None:
        # Fallback if we don't know the size: use 50k (safe middle ground)
        cap = 50000 if L <= 4096 else 25000
    else:
        # Take up to 80k. 
        # If n_obs < 80k, it takes everything.
        # If n_obs > 80k, it samples 80k.
        cap = min(TARGET_CAP, n_obs)

        # Safety valve: if sequence length is huge (e.g. >4096 genes), 
        # reduce cap slightly to prevent OOM errors during neighbor calculation.
        if L > 4096:
            cap = int(cap * 0.6)

    # Respect env override if you set it manually in terminal
    os_cap = os.environ.get("PERTTF_MAX_UMAP_CELLS", "")
    try:
        cap = int(os_cap) if os_cap.strip() else cap
    except Exception:
        pass

    # --- neighbors params ---
    # With 80k cells, k=15 is too small and makes the plot look "shredded".
    # k=30 provides better global structure for this size.
    base_k = 30 if (cap > 50000) else 15
    min_dist = 0.5

    return {
        "EVAL_SNAPSHOT_EVERY": eval_every,
        "MAX_UMAP_CELLS": cap,
        "UMAP_N_NEIGHBORS": base_k,
        "UMAP_MIN_DIST": float(min_dist),
    }

def compute_expr_correlations(y_true, y_pred):
    """Compute Pearson, Spearman, and R² with safe nan handling."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    pear = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
    spear = spearmanr(y_true[mask], y_pred[mask], nan_policy="omit").correlation
    r2 = r2_score(y_true[mask], y_pred[mask])
    return pear, spear, r2

def train(model: nn.Module,
          loader: DataLoader,
          config,
          vocab,
          optim_dict: Dict,
          epoch = 0,
          logger = scg.logger,
          device = None,
          fold: int = 0) -> None:
    """
    Train the model for one epoch.
    """
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss() # this is the loss for predicting PS scores
    #criterion_ps = nn.CrossEntropyLoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    epoch_total_loss, epoch_total_mse, epoch_total_gepc = 0.0, 0.0, 0.0
    epoch_total_mse_next, epoch_total_gepc_next = 0.0, 0.0
    epoch_total_error, epoch_total_error_next = 0.0, 0.0
    epoch_total_dab, epoch_total_adv_E, epoch_total_adv_D = 0.0, 0.0, 0.0
    epoch_total_cls, epoch_total_pert, epoch_total_ps = 0.0, 0.0, 0.0

    interval_loss = interval_mse = interval_gepc = 0.0
    interval_mse_next = interval_gepc_next = 0.0
    interval_error = interval_error_next = 0.0
    interval_dab = interval_adv_E = interval_adv_D = 0.0
    interval_cls = interval_pert = interval_ps = 0.0

    log_interval = config.log_interval
    start_time = time.time()


    scaler=optim_dict["scaler"]
    discriminator=optim_dict["discriminator"]
    optimizer=optim_dict["optimizer"]
    scheduler=optim_dict["scheduler"]
    optimizer_dab=optim_dict["optimizer_dab"]
    scheduler_dab=optim_dict["scheduler_dab"]
    optimizer_E=optim_dict["optimizer_E"]
    scheduler_E=optim_dict["scheduler_E"]
    optimizer_D=optim_dict["optimizer_D"]
    scheduler_D=optim_dict["scheduler_D"]
    clipper = optim_dict['clipper']

    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        input_next_values = batch_data["values_next"].to(device)
        target_values = batch_data["target_values"].to(device)
        target_values_next = batch_data.get("target_values_next", batch_data["target_values"]).to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device) #added
        perturbation_labels = batch_data["perturbation_labels"].to(device) #added

        celltype_labels_next = batch_data["celltype_labels_next"].to(device) #added
        perturbation_labels_next = batch_data.get("perturbation_labels_next", batch_data["perturbation_labels"]).to(device)
        perturbation = batch_data["perturbation"].to(device)
        inv_perturbation = batch_data["inv_perturbation"].to(device) if batch_data["inv_perturbation"] is not None else None 
        pert_scale = batch_data['pert_scale'].to(device)
        inv_per_scale = batch_data['inv_pert_scale'].to(device)
        mvc_src = None if config.get('mvc_masked_train', True) else batch_data['full_gene_ids'].to(device)

        if config.ps_weight >0:
            ps_score = batch_data["ps"].to(device)
            ps_score_next = batch_data["ps_next"].to(device) #

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            #import pdb; pdb.set_trace()

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
                pert_labels = perturbation_labels if config.perturbation_input else None,
                pert_labels_next = perturbation_labels_next if config.perturbation_input else None,
                perturbation = perturbation if (config.next_weight >0 or has_lochness_next_pred ) else None,
                inv_perturbation = inv_perturbation if (config.get('reciprical_sampling', False)) else None,
                pert_scale = pert_scale if config.get('reciprical_sampling', False) else None,
                inv_pert_scale = inv_per_scale if config.get('reciprical_sampling', False) else None,
                values_next = input_next_values if config.get('CCE', False) and config.get('reciprical_sampling', False) else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.cell_type_classifier,
                CCE = config.CCE,
                PERTPRED = config.perturbation_classifier_weight > 0,
                PSPRED = config.ps_weight >0,
                mvc_src = mvc_src
            )

            masked_positions = input_values.eq(config.mask_value)  # the postions to predict
            loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            loss = config.this_weight * loss_mse
            metrics_to_log = {f"train/fold{fold}/mse": loss_mse.item()}
            if config.CCE and len(output_dict["contrastive_dict"]) > 0:
                cce_mode = config.get('cce_mode', 'cell+geno')
                logit_norm = config.get('logit_norm', False)
                if cce_mode == 'cell_geno': ## supervised contrastive loss plus a custom contrastive loss
                    cce_weight = max(config.perturbation_classifier_weight*config.cell_type_classifier_weight, 1)
                    input_labels = celltype_labels*1000+perturbation_labels # x1000 make labels unique combination of celltype and genotype
                    pert_labels = celltype_labels_next*1000+perturbation_labels_next
                    loss_cce = cce_loss(output_dict["contrastive_dict"], input_labels, pert_labels, logit_norm=logit_norm)
                    metrics_to_log["train/cce"] = loss_cce.item()
                    loss += loss_cce * cce_weight

                if cce_mode == 'celltype' or cce_mode == 'cell+geno':
                    loss_cce_celltype = cce_loss(output_dict["contrastive_dict"], celltype_labels, celltype_labels_next, logit_norm=logit_norm)
                    metrics_to_log["train/cce_celltype"] = loss_cce_celltype.item()
                    loss += loss_cce_celltype * max(config.cell_type_classifier_weight, 1)

                if cce_mode == 'genotype' or cce_mode == 'cell+geno':
                    loss_cce_genotype = cce_loss(output_dict["contrastive_dict"], perturbation_labels, perturbation_labels_next, logit_norm=logit_norm)
                    metrics_to_log["train/cce_genotype"] = loss_cce_genotype.item()
                    loss += loss_cce_genotype * max(config.perturbation_classifier_weight,1)
                
            
            
            # next value?
            loss_mse_next = criterion(
                output_dict["mlm_output"],
                target_values_next, masked_positions
            )
            # disable now
            #loss = loss + config.next_weight * loss_mse_next
            metrics_to_log.update({f"train/fold{fold}/mse_next": loss_mse_next.item()})
            if config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + config.this_weight *loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                # added
                loss_zero_log_prob_next = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values_next, masked_positions
                )
                #loss = loss + config.next_weight *loss_zero_log_prob_next
                metrics_to_log.update({"train/nzlp_next": loss_zero_log_prob_next.item()})


            if config.GEPC:
                mvc_target_values = target_values if config.get('mvc_masked_train', True) else batch_data["full_expr"].to(device)
                mvc_target_values_next = target_values_next if config.get('mvc_masked_train', True) else batch_data["full_expr_next"].to(device)
                mvc_masked_positions = masked_positions if config.get('mvc_masked_train', True) else None

                loss_gepc = criterion(output_dict["mvc_output"], mvc_target_values, mvc_masked_positions)
                loss = loss + config.this_weight * loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})

                loss_gepc_next = criterion(output_dict["mvc_output_next"], mvc_target_values_next, mvc_masked_positions)
                loss = loss + config.next_weight * loss_gepc_next
                metrics_to_log.update({"train/mvc_next": loss_gepc_next.item()})


            if config.explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], mvc_target_values, mvc_masked_positions
                    )
                    loss = loss + config.this_weight *loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                    # added
                    loss_gepc_zero_log_prob_next = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs_next"], mvc_target_values_next, mvc_masked_positions
                    )
                    loss = loss + config.next_weight * loss_gepc_zero_log_prob_next
                    metrics_to_log.update(
                        {"train/mvc_nzlp_next": loss_gepc_zero_log_prob_next.item()}
                    )

            if config.cell_type_classifier:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + config.cell_type_classifier_weight * loss_cls
                metrics_to_log.update({f"train/fold{fold}/cls": loss_cls.item()})
                # add for next cls prediction
                loss_cls_next = criterion_cls(output_dict["cls_output_next"], celltype_labels_next)
                loss = loss + config.cell_type_classifier_weight * config.next_weight *  loss_cls_next
                metrics_to_log.update({f"train/fold{fold}/cls_next": loss_cls_next.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

            if config.perturbation_classifier_weight > 0:
                loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels)
                loss = loss + config.perturbation_classifier_weight * loss_pert
                metrics_to_log.update({f"train/fold{fold}/pert": loss_pert.item()})
                # add for next pert prediction
                loss_pert_next = criterion_pert(output_dict["pert_output_next"], perturbation_labels_next)
                loss = loss + config.perturbation_classifier_weight * config.next_weight * loss_pert_next
                metrics_to_log.update({f"train/fold{fold}/pert_next": loss_pert_next.item()})

            if config.ps_weight >0:
                loss_ps = criterion_ps(output_dict["ps_output"], ps_score)
                #import pdb; pdb.set_trace()
                #print(f"loss_ps: {loss_ps}")
                loss = loss + config.ps_weight * loss_ps
                metrics_to_log.update({f"train/fold{fold}/ps": loss_ps.item()})
                loss_ps_next = criterion_ps(output_dict["ps_output_next"], ps_score_next)
                loss = loss + config.ps_weight * loss_ps_next * config.next_weight
                metrics_to_log.update({f"train/fold{fold}/ps_next": loss_ps_next.item()})

            if config.ecs_thres > 0:
                loss_ecs = config.ecs_weight  * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({f"train/fold{fold}/ecs": loss_ecs.item()})
            if config.dab_weight > 0:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({f"train/fold{fold}/dab": loss_dab.item()})
            interval_loss += loss.item()

        model.zero_grad()
        #print(f"loss: {loss}")
        #import pdb; pdb.set_trace()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if clipper is not None:
            clipper.step(model)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0 and logger is not None:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if config.ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
                pert_labels = perturbation_labels if config.perturbation_input else None,
                perturbation = perturbation if (config.next_weight >0 or has_lochness_next_pred )  else None,
                pert_scale = pert_scale if (config.get('reciprical_sampling', False)) else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.cell_type_classifier,
                #CCE=config.CCE,
                PERTPRED = config.perturbation_classifier_weight > 0,
                PSPRED = config.ps_weight >0,
                #do_sample=config.do_sample_in_train,
                #generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = config.adv_weight * criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > config.adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -1 * config.adv_weight * criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > config.adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        metrics_to_log["epoch"] = epoch
        metrics_to_log["batch"] = batch
        if getattr(config, "log_to_wandb", False):
            wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
            mre_next = masked_relative_error(
                output_dict["mlm_output"], target_values_next, masked_positions
            )

        epoch_total_mse += loss_mse.item()
        interval_mse+= loss_mse.item()
        epoch_total_mse_next += loss_mse_next.item()
        interval_mse_next += loss_mse_next.item()
        epoch_total_gepc += loss_gepc.item() if config.GEPC else 0.0
        interval_gepc += loss_gepc.item() if config.GEPC else 0.0
        epoch_total_gepc_next += loss_gepc_next.item() if config.GEPC else 0.0
        interval_gepc_next+= loss_gepc_next.item() if config.GEPC else 0.0
        epoch_total_error += mre.item()
        interval_error += mre.item()
        epoch_total_error_next += mre_next.item()
        interval_error_next += mre_next.item()
        epoch_total_dab += loss_dab.item() if config.dab_weight >0 else 0.0
        interval_dab += loss_dab.item() if config.dab_weight >0 else 0.0
        epoch_total_adv_E += loss_adv_E.item() if config.ADV else 0.0
        interval_adv_E  += loss_adv_E.item() if config.ADV else 0.0
        epoch_total_adv_D += loss_adv_D.item() if config.ADV else 0.0
        interval_adv_D += loss_adv_D.item() if config.ADV else 0.0
        epoch_total_cls += loss_cls.item() if config.cell_type_classifier else 0.0
        interval_cls += loss_cls.item() if config.cell_type_classifier else 0.0
        epoch_total_pert += loss_pert.item() if config.perturbation_classifier_weight > 0 else 0.0
        interval_pert += loss_pert.item() if config.perturbation_classifier_weight > 0 else 0.0
        epoch_total_ps += loss_ps.item() if config.ps_weight >0 else 0.0
        interval_ps += loss_ps.item() if config.ps_weight >0 else 0.0

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = interval_loss / log_interval
            cur_mse = interval_mse / log_interval
            cur_mse_next = interval_mse_next / log_interval
            cur_gepc = interval_gepc / log_interval if config.GEPC else 0.0
            cur_gepc_next = interval_gepc_next / log_interval if config.GEPC else 0.0
            cur_error = interval_error / log_interval
            cur_error_next = interval_error_next / log_interval
            cur_dab = interval_dab / log_interval if config.dab_weight >0 else 0.0
            cur_adv_E = interval_adv_E / log_interval if config.ADV else 0.0
            cur_adv_D = interval_adv_D / log_interval if config.ADV else 0.0
            cur_cls = interval_cls / log_interval if config.cell_type_classifier else 0.0
            cur_pert = interval_pert / log_interval if config.perturbation_classifier_weight > 0 else 0.0
            cur_ps = interval_ps / log_interval if config.ps_weight >0 else 0.0
            # ppl = math.exp(cur_loss)
            if logger is not None:
                logger.info(
                    f"[Fold {fold}] | epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.8f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    f"mse_next {cur_mse_next:5.2f} | mre_next {cur_error_next:5.2f} |"
                    f"cls {cur_cls:5.2f} | pert {cur_pert:5.2f} | ps {cur_ps:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                    + (f"gepc_next {cur_gepc_next:5.2f} |" if config.GEPC else "")
                    + (f"dab {cur_dab:5.2f} |" if config.dab_weight > 0 else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if config.ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if config.ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if config.ADV else "")
                )
            interval_loss = interval_mse = interval_gepc = 0.0
            interval_mse_next = interval_gepc_next = 0.0
            interval_error = interval_error_next = 0.0
            interval_dab = interval_adv_E = interval_adv_D = 0.0
            interval_cls = interval_pert = interval_ps = 0.0
            start_time = time.time()
    avg = lambda total: total / max(1, num_batches)
    return {
        "mse": avg(epoch_total_mse),
        "mre": avg(epoch_total_error),
        "mvc": avg(epoch_total_gepc) if config.GEPC else 0.0,
        "cls": avg(epoch_total_cls) if config.cell_type_classifier else 0.0,
        "pert": avg(epoch_total_pert) if config.perturbation_classifier_weight > 0 else 0.0,
        "ps": avg(epoch_total_ps) if config.ps_weight > 0 else 0.0,
        "dab": avg(epoch_total_dab) if config.dab_weight > 0 else 0.0,
    }  


def define_wandb_metrics(fold: int):
    # No-op if there isn’t an active run
    if not getattr(wandb, "run", None):
        return
    # VALID
    wandb.define_metric(f"valid/fold{fold}/mse",         summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/mre",         summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/mse_next",    summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/mre_next",    summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/dab",         summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/cls",         summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/pert",        summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/ps",          summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/val_loss",    summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/val_loss_next", summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/mvc",         summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/mvc_next",    summary="min", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_pearson",        summary="max", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_spearman",       summary="max", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_r2",             summary="max", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_pearson_next",   summary="max", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_spearman_next",  summary="max", step_metric="epoch")
    wandb.define_metric(f"valid/fold{fold}/expr_r2_next",        summary="max", step_metric="epoch")
    # TRAIN
    wandb.define_metric(f"train/fold{fold}/mse",         summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/mre",         summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/cls",         summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/pert",        summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/mvc",         summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/mvc_next",    summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/ps",          summary="min", step_metric="epoch")
    wandb.define_metric(f"train/fold{fold}/dab",         summary="min", step_metric="epoch")

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config,
    vocab,
    epoch: int = 0,
    fold: int = 0,
    device=None
) -> float:
    """
    Evaluate the model on the evaluation data.
    Returns:
      (mse, mse_next, mvc, mvc_next, mre, mre_next, dab, cls, pert, ps, ps_next)
    """
    import numpy as np
    import torch
    import torch.nn as nn

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_ps = nn.MSELoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    total_loss = 0.0
    total_loss_next = 0.0
    total_error = 0.0
    total_error_next = 0.0
    total_dab = 0.0
    total_cls = 0.0
    total_pert = 0.0
    total_ps = 0.0
    total_ps_next = 0.0
    total_num = 0.0
    total_mvc = 0.0
    total_mvc_next = 0.0

    # --- expression correlation accumulators (masked positions only) ---
    y_true_masked = []
    y_pred_masked = []
    y_true_masked_next = []
    y_pred_masked_next = []

    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)

            target_values = batch_data["target_values"].to(device)
            target_values_next = batch_data.get("target_values_next", batch_data["target_values"]).to(device)

            batch_labels = batch_data["batch_labels"].to(device)

            celltype_labels = batch_data["celltype_labels"].to(device)
            perturbation_labels = batch_data["perturbation_labels"].to(device)

            ps_score = batch_data["ps"].to(device)
            ps_score_next = batch_data["ps_next"].to(device)

            perturbation = batch_data["perturbation"].to(device)
            pert_scale = batch_data["pert_scale"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            mvc_src = None if config.get("mvc_masked_train", True) else batch_data["full_gene_ids"].to(device)

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.use_batch_label else None,
                    pert_labels=perturbation_labels if config.perturbation_input else None,
                    perturbation=perturbation if (config.next_weight > 0 or has_lochness_next_pred) else None,
                    pert_scale=pert_scale if config.get("reciprical_sampling", False) else None,
                    MVC=config.GEPC,
                    ECS=(config.ecs_thres > 0),
                    CLS=config.cell_type_classifier,
                    PERTPRED=(config.perturbation_classifier_weight > 0),
                    PSPRED=(config.ps_weight > 0),
                    mvc_src=mvc_src,
                )

                output_values = output_dict["mlm_output"]
                masked_positions = input_values.eq(config.mask_value)

                # primary losses
                loss_mse = criterion(output_values, target_values, masked_positions)
                loss_mse_next = criterion(output_values, target_values_next, masked_positions)

                # correlations on masked tokens (collect only)
                mp_bool = masked_positions.bool()
                yt = target_values[mp_bool].detach().float().view(-1).cpu()
                yp = output_values[mp_bool].detach().float().view(-1).cpu()
                if yt.numel() > 0 and yp.numel() > 0:
                    y_true_masked.append(yt.numpy())
                    y_pred_masked.append(yp.numpy())

                    ytn = target_values_next[mp_bool].detach().float().view(-1).cpu()
                    if ytn.numel() > 0:
                        y_true_masked_next.append(ytn.numpy())
                        y_pred_masked_next.append(yp.numpy())

                # build the "total loss" exactly like your code: start from MSE
                loss_total = loss_mse

                # MVC/GEPC
                if config.GEPC:
                    mvc_target_values = (
                        target_values if config.get("mvc_masked_train", True)
                        else batch_data["full_expr"].to(device)
                    )
                    mvc_target_values_next = (
                        target_values_next if config.get("mvc_masked_train", True)
                        else batch_data["full_expr_next"].to(device)
                    )
                    mvc_masked_positions = masked_positions if config.get("mvc_masked_train", True) else None

                    loss_gepc = criterion(output_dict["mvc_output"], mvc_target_values, mvc_masked_positions)
                    loss_gepc_next = criterion(output_dict["mvc_output_next"], mvc_target_values_next, mvc_masked_positions)

                    loss_total = loss_total + (config.this_weight * loss_gepc) + (config.next_weight * loss_gepc_next)

                    total_mvc += loss_gepc.item() * len(input_gene_ids)
                    total_mvc_next += loss_gepc_next.item() * len(input_gene_ids)

                # dab/cls/pert/ps heads (compute once)
                loss_dab = None
                loss_cls = None
                loss_pert = None
                loss_ps = None
                loss_ps_next = None

                if config.dab_weight > 0:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                if config.cell_type_classifier:
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)

                if config.perturbation_classifier_weight > 0:
                    loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels)

                if config.ps_weight > 0:
                    loss_ps = criterion_ps(output_dict["ps_output"], ps_score)

                if ps_next_training_weight > 0:
                    loss_ps_next = criterion_ps(output_dict["ps_output_next"], ps_score_next)
                    total_ps_next += loss_ps_next.item() * len(input_gene_ids)

            # aggregate totals (same behavior as your original code)
            total_loss += loss_total.item() * len(input_gene_ids)
            total_loss_next += loss_mse_next.item() * len(input_gene_ids)

            total_error += masked_relative_error(output_values, target_values, masked_positions).item() * len(input_gene_ids)
            total_error_next += masked_relative_error(output_values, target_values_next, masked_positions).item() * len(input_gene_ids)

            if config.dab_weight > 0 and loss_dab is not None:
                total_dab += loss_dab.item() * len(input_gene_ids)

            if config.cell_type_classifier and loss_cls is not None:
                total_cls += loss_cls.item() * len(input_gene_ids)

            if config.perturbation_classifier_weight > 0 and loss_pert is not None:
                total_pert += loss_pert.item() * len(input_gene_ids)

            if config.ps_weight > 0 and loss_ps is not None:
                total_ps += loss_ps.item() * len(input_gene_ids)

            total_num += len(input_gene_ids)

        # correlations: compute ONCE per evaluate()
        def _safe_cat(parts):
            return np.concatenate(parts) if parts else np.array([], dtype=float)

        xt, xp = _safe_cat(y_true_masked), _safe_cat(y_pred_masked)
        pear, spear, r2 = compute_expr_correlations(xt, xp)

        xtn, xpn = _safe_cat(y_true_masked_next), _safe_cat(y_pred_masked_next)
        pear_n, spear_n, r2_n = compute_expr_correlations(xtn, xpn)

        if getattr(config, "log_to_wandb", False):
            wandb.log({
                f"valid/fold{fold}/mse": total_loss / total_num,
                f"valid/fold{fold}/mse_next": total_loss_next / total_num,
                f"valid/fold{fold}/mvc": total_mvc / total_num,
                f"valid/fold{fold}/mvc_next": total_mvc_next / total_num,
                f"valid/fold{fold}/mre": total_error / total_num,
                f"valid/fold{fold}/mre_next": total_error_next / total_num,
                f"valid/fold{fold}/dab": total_dab / total_num,
                f"valid/fold{fold}/cls": total_cls / total_num,
                f"valid/fold{fold}/pert": total_pert / total_num,
                f"valid/fold{fold}/ps": total_ps / total_num,
                f"valid/fold{fold}/expr_pearson": pear,
                f"valid/fold{fold}/expr_spearman": spear,
                f"valid/fold{fold}/expr_r2": r2,
                f"valid/fold{fold}/expr_pearson_next": pear_n,
                f"valid/fold{fold}/expr_spearman_next": spear_n,
                f"valid/fold{fold}/expr_r2_next": r2_n,
                f"valid/fold{fold}/sum_mse_dab": (total_loss + config.dab_weight * total_dab) / total_num,
                "epoch": epoch,
            })

    return (
        total_loss / total_num,
        total_loss_next / total_num,
        total_mvc / total_num,
        total_mvc_next / total_num,
        total_error / total_num,
        total_error_next / total_num,
        total_dab / total_num,
        total_cls / total_num,
        total_pert / total_num,
        total_ps / total_num,
        total_ps_next / total_num,
    )

def _index_to_names(mapping_dict):
    inv = {v: k for k, v in mapping_dict.items()}
    return [inv[i] for i in range(len(inv))]

def _plot_confusion(y_true, y_pred, class_names, title):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    cmx = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cmx_norm = cmx.astype(float) / np.maximum(cmx.sum(axis=1, keepdims=True), 1)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cmx_norm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def _plot_multiclass_roc(y_true_idx, prob, title, class_names=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize
    C = prob.shape[1]
    y_true_bin = label_binarize(y_true_idx, classes=np.arange(C))
    fig = plt.figure()
    ax = plt.gca()
    for c in range(C):
        if y_true_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], prob[:, c])
        auc_c = roc_auc_score(y_true_bin[:, c], prob[:, c])
        label = (class_names[c] if class_names is not None else f"class {c}") + f" (AUC={auc_c:.3f})"
        ax.plot(fpr, tpr, label=label)
    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), prob.ravel())
    auc_micro = roc_auc_score(y_true_bin, prob, average="micro", multi_class="ovr")
    ax.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro (AUC={auc_micro:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    plt.tight_layout()
    return fig

def _plot_multiclass_pr(y_true_idx, prob, title, class_names=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    C = prob.shape[1]
    y_true_bin = label_binarize(y_true_idx, classes=np.arange(C))
    fig = plt.figure()
    ax = plt.gca()
    for c in range(C):
        if y_true_bin[:, c].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true_bin[:, c], prob[:, c])
        ap_c = average_precision_score(y_true_bin[:, c], prob[:, c])
        label = (class_names[c] if class_names is not None else f"class {c}") + f" (AP={ap_c:.3f})"
        ax.plot(rec, prec, label=label)
    # micro-average
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), prob.ravel())
    ap_micro = average_precision_score(y_true_bin, prob, average="micro")
    ax.plot(rec_micro, prec_micro, linestyle="--", label=f"micro (AP={ap_micro:.3f})")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    plt.tight_layout()
    return fig

def _compute_scalar_metrics(y_true_idx, y_pred_idx, prob):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
    )
    from sklearn.preprocessing import label_binarize
    import numpy as np
    C = prob.shape[1]
    y_true_bin = label_binarize(y_true_idx, classes=np.arange(C))

    metrics = {}
    metrics["acc"] = accuracy_score(y_true_idx, y_pred_idx)
    metrics["precision_macro"] = precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    metrics["precision_weighted"] = precision_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    # ROC-AUC / AUPR macro
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true_bin, prob, average="macro", multi_class="ovr")
    except Exception:
        metrics["roc_auc_ovr_macro"] = float("nan")
    try:
        metrics["roc_auc_ovo_macro"] = roc_auc_score(y_true_bin, prob, average="macro", multi_class="ovo")
    except Exception:
        metrics["roc_auc_ovo_macro"] = float("nan")
    try:
        metrics["aupr_macro"] = average_precision_score(y_true_bin, prob, average="macro")
    except Exception:
        metrics["aupr_macro"] = float("nan")
    return metrics

def _per_class_metrics(y_true_idx, y_pred_idx, prob, class_names):
    """
    Return a per-class DataFrame: class, support, precision, recall, f1, auc_roc, ap (AUPR)
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    )
    from sklearn.preprocessing import label_binarize

    C = prob.shape[1]
    y_true_bin = label_binarize(y_true_idx, classes=np.arange(C))

    prec = precision_score(y_true_idx, y_pred_idx, average=None, labels=np.arange(C), zero_division=0)
    rec  = recall_score(y_true_idx, y_pred_idx, average=None, labels=np.arange(C), zero_division=0)
    f1   = f1_score(y_true_idx, y_pred_idx, average=None, labels=np.arange(C), zero_division=0)

    # AUC/AP per class (NaN if class absent)
    aucs = []
    aps  = []
    for c in range(C):
        if y_true_bin[:, c].sum() == 0:
            aucs.append(float("nan"))
            aps.append(float("nan"))
        else:
            aucs.append(roc_auc_score(y_true_bin[:, c], prob[:, c]))
            aps.append(average_precision_score(y_true_bin[:, c], prob[:, c]))

    support = y_true_bin.sum(axis=0).astype(int)
    df = pd.DataFrame({
        "class": class_names,
        "support": support,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": aucs,
        "ap": aps,  # AUPR per class
    })
    return df

def _write_metric_csvs(eval_key, epoch, save_dir, per_class_df, summary_dict, prefix):
    """
    Write two CSVs:
      - <save_dir>/<eval_key>_<prefix>_per_class_e{epoch}.csv
      - <save_dir>/<eval_key>_<prefix>_summary_e{epoch}.csv
    """
    import pandas as pd
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    per_class_path = Path(save_dir) / f"{eval_key}_{prefix}_per_class_e{epoch}.csv"
    per_class_df.to_csv(per_class_path, index=False)

    summary_path = Path(save_dir) / f"{eval_key}_{prefix}_summary_e{epoch}.csv"
    pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)

    return str(per_class_path), str(summary_path)

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids: List[str],
    train_data_dict: Dict,
    config,
    include_types: List[str] = ["cls", "pert"],
    input_layer_key="X_binned",
    next_layer_key="X_binned_next",
    logger=scg.logger,
    epoch=0,
    eval_key="",  # e.g., "train" or "validation"
    make_plots=True,
    mask=False,
    predict_expr=True,
    no_pert_for_perturb=False,
    reciprical_sampling=False,
    mvc_full_expr=False,
    save_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Evaluate the model on adata_t + compute metrics.
    STRICT: uses obs['celltype_2'] only for cell type labels.
    """

    import os
    import numpy as np
    import pandas as pd
    import torch
    import random
    from pathlib import Path
    from scipy.sparse import issparse  # <-- important: avoid NameError

    # --- STRICT: use celltype_2 only ---
    ct_key = "celltype_2"
    if ct_key not in adata_t.obs.columns:
        raise ValueError("[eval_testdata] Required obs column 'celltype_2' not found.")
    adata_t = adata_t.copy()
    adata_t.obs[ct_key] = adata_t.obs[ct_key].astype(str)

    model.eval()

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _softmax_np(logits, axis=1):
        logits = np.asarray(logits)
        logits = logits - logits.max(axis=axis, keepdims=True)  # stability
        exps = np.exp(logits)
        return exps / np.sum(exps, axis=axis, keepdims=True)

    cell_type_to_index = train_data_dict["cell_type_to_index"]
    genotype_to_index = train_data_dict["genotype_to_index"]
    vocab = train_data_dict["vocab"]

    # keep only known cell types for metrics/plots
    adata_t = adata_t[adata_t.obs[ct_key].isin(cell_type_to_index)].copy()
    if adata_t.n_obs == 0:
        if logger is not None:
            logger.warning(f"[eval_testdata:{eval_key}] No cells after filtering; skipping metrics.")
        return {"adata": adata_t, "metrics": {}}

    if logger is not None:
        n_hit = int(adata_t.obs[ct_key].isin(cell_type_to_index).sum())
        logger.info(f"[eval_testdata:{eval_key}] Using ct_key=celltype_2 overlap={n_hit}/{adata_t.n_obs}")

    # counts
    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    if next_layer_key in adata_t.layers:
        all_counts_next = (
            adata_t.layers[next_layer_key].toarray()
            if issparse(adata_t.layers[next_layer_key])
            else adata_t.layers[next_layer_key]
        )
    else:
        all_counts_next = None

    # labels
    celltypes_labels = None
    if config.cell_type_classifier:
        celltypes_labels = np.array(
            [cell_type_to_index[c] for c in adata_t.obs[ct_key].astype(str).tolist()],
            dtype=int,
        )

    perturbation_labels = None
    if "genotype" in adata_t.obs.columns and (config.perturbation_classifier_weight > 0 or config.perturbation_input):
        perturbation_labels = np.array(
            [genotype_to_index[g] for g in adata_t.obs["genotype"].astype(str).tolist()],
            dtype=int,
        )

    # ----- NEXT prediction flags/setup -----
    next_cell_prediction = True
    perturbation_labels_next = None
    pert_scale = None

    # Skip next prediction entirely if next_weight=0 or next_cell_pred_type='identity'
    if getattr(config, "next_weight", 0) == 0 or getattr(config, "next_cell_pred_type", "") == "identity":
        next_cell_prediction = False
        if logger is not None:
            logger.info(
                f"[eval_testdata:{eval_key}] Skipping NEXT prediction (type={getattr(config,'next_cell_pred_type','')}, weight={getattr(config,'next_weight',0)})"
            )

    if next_cell_prediction and getattr(config, "next_cell_pred_type", "") == "pert":
        if "genotype_next" in adata_t.obs.columns:
            if config.perturbation_classifier_weight > 0:
                perturbation_labels_next = adata_t.obs["genotype_next"].tolist()
                if no_pert_for_perturb and not reciprical_sampling:
                    perturbation_labels_next = [
                        "WT" if perturbed else adata_t.obs.genotype_next.iloc[i]
                        for i, perturbed in enumerate(adata_t.obs.genotype == adata_t.obs.genotype_next)
                    ]
                if reciprical_sampling:
                    tmp = []
                    for curr, nxt in zip(adata_t.obs.genotype.tolist(), adata_t.obs.genotype_next.tolist()):
                        if curr == nxt:
                            tmp.append("WT")
                        else:
                            tmp.append(nxt if curr == "WT" else curr)
                    perturbation_labels_next = tmp
                    pert_scale = np.array(
                        [
                            np.array([-1.0]) if (adata_t.obs.genotype.iloc[i] != "WT" and adata_t.obs.genotype.iloc[i] != adata_t.obs.genotype_next.iloc[i]) else np.array([1.0])
                            for i in range(adata_t.n_obs)
                        ]
                    )
            else:
                perturbation_labels_next = None
                next_cell_prediction = False
        else:
            if logger is not None:
                logger.warning("[eval_testdata] next_cell_pred_type='pert' but adata.obs['genotype_next'] missing; disabling next prediction")
            next_cell_prediction = False

    if next_cell_prediction and getattr(config, "next_cell_pred_type", "") == "lochness":
        if hasattr(config, "pred_lochness_next") and config.pred_lochness_next > 0 and "genotype_next" in adata_t.obs.columns:
            perturbation_labels_next = adata_t.obs["genotype_next"].tolist()
        else:
            next_cell_prediction = False
            perturbation_labels_next = None

    if next_cell_prediction and perturbation_labels_next is not None:
        try:
            perturbation_labels_next = np.array(
                [genotype_to_index[g] for g in perturbation_labels_next], dtype=int
            )
        except Exception as e:
            if logger is not None:
                logger.warning(f"[eval_testdata:{eval_key}] Could not map genotype_next -> index; disabling next prediction. err={e}")
            next_cell_prediction = False
            perturbation_labels_next = None

    # batch ids
    if "batch_id" in adata_t.obs.columns:
        batch_ids = np.array(adata_t.obs["batch_id"].tolist())
    else:
        batch_ids = np.array(random.choices([0, 1], k=adata_t.n_obs))

    # mvc full expr src
    if mvc_full_expr:
        cls_gene_ids = np.insert(gene_ids, 0, vocab[config.cls_token])
        full_gene_ids = torch.stack(
            [torch.from_numpy(cls_gene_ids).long() for _ in range(adata_t.shape[0])], dim=0
        )
    else:
        full_gene_ids = None

    # ===== ALWAYS tokenize because we always encode (even include_types=["pert"]) =====
    if logger is not None and ("cls" in include_types):
        logger.info(f"[eval_testdata:{eval_key}] Tokenizing for CLS embeddings")

    tokenized_all, gene_idx_list = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=True,
        simple_sampling=config.get("simple_sampling", True),
        nonzero_prop=config.get("nonzero_prop", 0.7),
        fix_nonzero_prop=config.get("fix_nonzero_prop", False),
    )

    all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
    input_values = all_values.float()

    if mask:
        masked_values = random_mask_value(
            all_values,
            mask_ratio=config.mask_ratio,
            mask_value=config.mask_value,
            pad_value=config.pad_value,
            cls_value=config.cls_value,
        )
        input_values = masked_values.float()

    masked_positions = input_values.eq(config.mask_value).cpu().numpy().astype(np.int0)
    src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])

    # next tokenization (used for indexing consistency; might be used elsewhere)
    if all_counts_next is not None:
        tokenized_all_next, _ = tokenize_and_pad_batch(
            all_counts_next,
            gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,
            include_zero_gene=True,
            sample_indices=gene_idx_list,
            simple_sampling=config.get("simple_sampling", True),
            nonzero_prop=config.get("nonzero_prop", 0.7),
            fix_nonzero_prop=config.get("fix_nonzero_prop", False),
        )
        all_gene_ids_next, all_values_next = tokenized_all_next["genes"], tokenized_all_next["values"]  # noqa: F841

    # --- helper: normalize encode_batch_with_perturb return signature ---
    def _pad_encode_outputs(enc_out):
        """
        Ensure we always return 7 outputs:
        (cell_embeddings, cell_embeddings_next, pert_preds, cls_preds, ps_preds, ps_preds_next, expr_dict)
        """
        outs = list(enc_out)
        expr_guess = outs[-1] if (len(outs) > 0 and isinstance(outs[-1], dict)) else None
        while len(outs) < 7:
            outs.append(None)

        cell_embeddings = outs[0] if len(outs) > 0 else None
        cell_embeddings_next = outs[1] if len(outs) > 1 else None
        pert_preds = outs[2] if len(outs) > 2 else None
        cls_preds = outs[3] if len(outs) > 3 else None

        ps_preds = outs[4] if (len(outs) > 4 and not isinstance(outs[4], dict)) else None
        ps_preds_next = outs[5] if (len(outs) > 5 and not isinstance(outs[5], dict)) else None

        expr_dict_local = expr_guess if expr_guess is not None else (
            outs[6] if isinstance(outs[6], dict) else
            outs[4] if isinstance(outs[4], dict) else
            None
        )

        return (
            cell_embeddings,
            cell_embeddings_next,
            pert_preds,
            cls_preds,
            ps_preds,
            ps_preds_next,
            expr_dict_local,
        )

    def _run_encode_batch_with_fallback():
        common_kwargs = dict(
            src_key_padding_mask=src_key_padding_mask,
            batch_size=config.batch_size,
            batch_labels=(torch.from_numpy(batch_ids).long() if config.use_batch_label else None),
            pert_labels=(
                torch.from_numpy(perturbation_labels).long()
                if (perturbation_labels is not None and config.perturbation_input)
                else None
            ),
            pert_labels_next=(torch.from_numpy(perturbation_labels_next).long() if next_cell_prediction else None),
            time_step=0,
            return_np=True,
        )
        if full_gene_ids is not None:
            common_kwargs["mvc_src"] = full_gene_ids

        try:
            return model.encode_batch_with_perturb(
                all_gene_ids,
                all_values.float(),
                predict_expr=predict_expr,
                **common_kwargs,
            )
        except TypeError as e1:
            if "predict_expr" in str(e1):
                # retry without predict_expr
                try:
                    return model.encode_batch_with_perturb(
                        all_gene_ids,
                        all_values.float(),
                        **common_kwargs,
                    )
                except TypeError as e2:
                    if "mvc_src" in str(e2):
                        common_kwargs.pop("mvc_src", None)
                        return model.encode_batch_with_perturb(
                            all_gene_ids,
                            all_values.float(),
                            **common_kwargs,
                        )
                    raise
            if "mvc_src" in str(e1):
                common_kwargs.pop("mvc_src", None)
                try:
                    return model.encode_batch_with_perturb(
                        all_gene_ids,
                        all_values.float(),
                        predict_expr=predict_expr,
                        **common_kwargs,
                    )
                except TypeError as e3:
                    if "predict_expr" in str(e3):
                        return model.encode_batch_with_perturb(
                            all_gene_ids,
                            all_values.float(),
                            **common_kwargs,
                        )
                    raise
            raise

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        enc_out = _run_encode_batch_with_fallback()

    (
        cell_embeddings,
        cell_embeddings_next,
        pert_preds,
        cls_preds,
        ps_preds,
        ps_preds_next,
        expr_dict_raw,
    ) = _pad_encode_outputs(enc_out)

    expr_dict = expr_dict_raw if isinstance(expr_dict_raw, dict) else {}

    # normalize embeddings (guard)
    if cell_embeddings is not None:
        denom = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cell_embeddings = cell_embeddings / denom

    if cell_embeddings_next is not None:
        denom = np.linalg.norm(cell_embeddings_next, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cell_embeddings_next = cell_embeddings_next / denom

    # store embeddings
    adata_t.obsm["X_scGPT"] = cell_embeddings
    adata_t.obsm["X_scGPT_next"] = cell_embeddings_next

    # store ps preds only if present
    if getattr(config, "ps_weight", 0) > 0 and ps_preds is not None:
        adata_t.obsm["ps_pred"] = ps_preds
    if getattr(config, "next_cell_pred_type", "") == "lochness" and ps_preds_next is not None:
        adata_t.obsm["ps_pred_next"] = ps_preds_next

    # stash expr outputs safely
    for name, val in expr_dict.items():
        if val is None or isinstance(val, bool):
            continue
        if isinstance(val, dict):
            for subname, subval in val.items():
                if subval is None or isinstance(subval, bool):
                    continue
                try:
                    if isinstance(subval, (list, tuple)) and len(subval) >= 1:
                        adata_t.obsm[f"{name}_{subname}"] = _to_numpy(subval[0])
                        if len(subval) >= 2 and subval[1] is not None and not isinstance(subval[1], bool):
                            adata_t.obsm[f"{name}_{subname}_zero"] = _to_numpy(subval[1])
                    else:
                        adata_t.obsm[f"{name}_{subname}"] = _to_numpy(subval)
                except Exception as ee:
                    if logger is not None:
                        logger.warning(f"[eval_testdata:{eval_key}] could not stash nested expr output {name}.{subname}: {ee}")
            continue

        try:
            if isinstance(val, (list, tuple)):
                if len(val) >= 1 and val[0] is not None and not isinstance(val[0], bool):
                    adata_t.obsm[name] = _to_numpy(val[0])
                if len(val) >= 2 and val[1] is not None and not isinstance(val[1], bool):
                    adata_t.obsm[name + "_zero"] = _to_numpy(val[1])
            else:
                adata_t.obsm[name] = _to_numpy(val)
        except Exception as ee:
            if logger is not None:
                logger.warning(f"[eval_testdata:{eval_key}] could not stash expr output {name}: {ee}")

    # probs + predictions (safe)
    genotype_names = _index_to_names(genotype_to_index)
    celltype_names = _index_to_names(cell_type_to_index)

    pert_prob = None
    cls_prob = None
    pert_pred_idx = None
    cls_pred_idx = None

    if pert_preds is not None:
        pert_prob = _softmax_np(pert_preds, axis=1)
        pert_pred_idx = np.argmax(pert_prob, axis=1)

    if cls_preds is not None:
        cls_prob = _softmax_np(cls_preds, axis=1)
        cls_pred_idx = np.argmax(cls_prob, axis=1)

    if pert_prob is not None:
        adata_t.obsm["X_pert_pred_probs"] = pert_prob
    if cls_prob is not None:
        adata_t.obsm["X_cls_pred_probs"] = cls_prob

    if pert_pred_idx is not None:
        adata_t.obs["predicted_genotype"] = pd.Categorical(
            [genotype_names[i] for i in pert_pred_idx],
            categories=genotype_names,
        )
    else:
        adata_t.obs["predicted_genotype"] = pd.Categorical(
            ["NA"] * adata_t.n_obs,
            categories=genotype_names,
        )

    if cls_pred_idx is not None:
        adata_t.obs["predicted_celltype"] = pd.Categorical(
            [celltype_names[i] for i in cls_pred_idx],
            categories=celltype_names,
        )
    else:
        adata_t.obs["predicted_celltype"] = pd.Categorical(
            ["NA"] * adata_t.n_obs,
            categories=celltype_names,
        )

    adata_t.obsm["mask"] = masked_positions
    adata_t.obsm["gene_ids"] = np.array(gene_idx_list)

    results = {}
    metrics = {}

    # === UMAPs on learned embeddings (subsampled + reproducible) ===
    try:
        UMAP_CAP = int(os.environ.get("PERTTF_MAX_UMAP_CELLS", _UMAP_BUDGET.get("MAX_UMAP_CELLS", 15000)))
        UMAP_N_NEIGHB = int(os.environ.get("PERTTF_UMAP_NN", _UMAP_BUDGET.get("UMAP_N_NEIGHBORS", 12)))
        UMAP_MIN_DIST = float(os.environ.get("PERTTF_UMAP_MIN_DIST", _UMAP_BUDGET.get("UMAP_MIN_DIST", 0.5)))

        if adata_t.n_obs > UMAP_CAP:
            rng = np.random.default_rng(int(getattr(config, "seed", 0)))
            keep_idx = rng.choice(adata_t.n_obs, size=UMAP_CAP, replace=False)
            ad_plt = adata_t[keep_idx].copy()
            if logger is not None:
                logger.info(f"[eval_testdata:{eval_key}] UMAP subsample: {adata_t.n_obs} → {ad_plt.n_obs} cells")
        else:
            ad_plt = adata_t

        def _emb(ad, basis, colors, title):
            if not colors:
                return None
            return sc.pl.embedding(
                ad, basis=basis, color=colors,
                frameon=False, return_fig=True, show=False, title=title
            )

        sc.pp.neighbors(
            ad_plt, use_rep="X_scGPT", n_neighbors=UMAP_N_NEIGHB,
            key_added="scgpt", random_state=int(getattr(config, "seed", 0))
        )
        sc.tl.umap(
            ad_plt, neighbors_key="scgpt",
            random_state=int(getattr(config, "seed", 0)), min_dist=UMAP_MIN_DIST
        )
        ad_plt.obsm["X_umap_scgpt"] = ad_plt.obsm["X_umap"].copy()
        import numpy as np
        UM = np.full((adata_t.n_obs, 2), np.nan, dtype=np.float32)
        if ad_plt is adata_t:
            UM[:, :] = ad_plt.obsm["X_umap_scgpt"].astype(np.float32)
            used_mask = np.ones(adata_t.n_obs, dtype=bool)
        else:
            UM[keep_idx, :] = ad_plt.obsm["X_umap_scgpt"].astype(np.float32)
            used_mask = np.zeros(adata_t.n_obs, dtype=bool)
            used_mask[keep_idx] = True

        adata_t.obsm["X_umap_scgpt"] = UM
        adata_t.obs["umap_scgpt_used"] = used_mask
        have_next = (ad_plt.obsm.get("X_scGPT_next") is not None)
        if have_next:
            sc.pp.neighbors(
                ad_plt, use_rep="X_scGPT_next", n_neighbors=UMAP_N_NEIGHB,
                key_added="scgpt_next", random_state=int(getattr(config, "seed", 0))
            )
            sc.tl.umap(
                ad_plt, neighbors_key="scgpt_next",
                random_state=int(getattr(config, "seed", 0)), min_dist=UMAP_MIN_DIST
            )
            ad_plt.obsm["X_umap_scgpt_next"] = ad_plt.obsm["X_umap"].copy()
            UMN = np.full((adata_t.n_obs, 2), np.nan, dtype=np.float32)
            if ad_plt is adata_t:
                UMN[:, :] = ad_plt.obsm["X_umap_scgpt_next"].astype(np.float32)
            else:
                UMN[keep_idx, :] = ad_plt.obsm["X_umap_scgpt_next"].astype(np.float32)

        adata_t.obsm["X_umap_scgpt_next"] = UMN
        has_true_geno = "genotype" in ad_plt.obs.columns
        geno_color = ["genotype"] if has_true_geno else ["predicted_genotype"]

        results["batch_umap"] = _emb(
            ad_plt, "umap_scgpt",
            ["batch_id"] if "batch_id" in ad_plt.obs.columns else None,
            f"{eval_key} batch UMAP (e{epoch})"
        )
        results["celltype_umap"] = _emb(
            ad_plt, "umap_scgpt",
            [ct_key] if (ct_key in ad_plt.obs.columns) else None,
            f"{eval_key} celltype UMAP (e{epoch})"
        )
        results["genotype_umap"] = _emb(
            ad_plt, "umap_scgpt",
            geno_color,
            f"{eval_key} genotype UMAP (e{epoch})"
        )
        results["pred_celltype"] = _emb(
            ad_plt, "umap_scgpt",
            ["predicted_celltype"],
            f"{eval_key} predicted celltype (e{epoch})"
        )
        results["pred_genotype"] = _emb(
            ad_plt, "umap_scgpt",
            ["predicted_genotype"],
            f"{eval_key} predicted genotype (e{epoch})"
        )

        if have_next and getattr(config, "next_weight", 0) > 0 and getattr(config, "next_cell_pred_type", "") != "identity":
            results["next_umap_celltype"] = _emb(
                ad_plt, "umap_scgpt_next",
                [ct_key] if (ct_key in ad_plt.obs.columns) else None,
                f"{eval_key} NEXT celltype UMAP (e{epoch})"
            )
            results["next_umap_genotype"] = _emb(
                ad_plt, "umap_scgpt_next",
                geno_color,
                f"{eval_key} NEXT genotype UMAP (e{epoch})"
            )
            # --- ADDED: Predicted Genotype on Next Embedding ---
            results["next_umap_pred_genotype"] = _emb(
                ad_plt, "umap_scgpt_next",
                ["predicted_genotype"], 
                f"{eval_key} NEXT PREDICTED genotype UMAP (e{epoch})"
            )
            # ---------------------------------------------------
            if "genotype_next" in ad_plt.obs.columns:
                results["next_umap_genotype_next"] = _emb(
                    ad_plt, "umap_scgpt_next",
                    ["genotype_next"],
                    f"{eval_key} NEXT true-genotype UMAP (e{epoch})"
                )

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for key in [
                "batch_umap", "celltype_umap", "genotype_umap", "pred_celltype", "pred_genotype",
                "next_umap_celltype", "next_umap_genotype", "next_umap_genotype_next",
            ]:
                fig_obj = results.get(key)
                if fig_obj is None:
                    continue
                out_base = Path(save_dir) / f"{eval_key}_{key}_e{epoch}"
                try:
                    save_fig_both(fig_obj,out_base, dpi=300, bbox_inches="tight")
                except Exception as ee:
                    if logger is not None:
                        logger.warning(f"[eval_testdata:{eval_key}] could not save {key}: {ee}")

        if ad_plt is adata_t:
            adata_t.obsm["X_umap_scgpt"] = ad_plt.obsm["X_umap_scgpt"]
            if have_next:
                adata_t.obsm["X_umap_scgpt_next"] = ad_plt.obsm["X_umap_scgpt_next"]

    except Exception as e:
        if logger is not None:
            logger.warning(f"[eval_testdata:{eval_key}] UMAP plotting failed: {e}")

    # ====== METRICS: Genotype ======
    if ("pert" in include_types) and (perturbation_labels is not None) and (pert_pred_idx is not None) and (pert_prob is not None):
        gt_idx = perturbation_labels
        geno_scalars = _compute_scalar_metrics(gt_idx, pert_pred_idx, pert_prob)
        for k, v in geno_scalars.items():
            metrics[f"{eval_key}_genotype/{k}"] = v

        geno_per_class = _per_class_metrics(gt_idx, pert_pred_idx, pert_prob, genotype_names)

        if make_plots:
            fig_roc = _plot_multiclass_roc(gt_idx, pert_prob, f"{eval_key} genotype ROC (epoch {epoch})", class_names=genotype_names)
            fig_pr  = _plot_multiclass_pr(gt_idx, pert_prob, f"{eval_key} genotype PR  (epoch {epoch})", class_names=genotype_names)
            fig_cm  = _plot_confusion(gt_idx, pert_pred_idx, genotype_names, f"{eval_key} genotype Confusion (epoch {epoch})")
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_fig_both(fig_roc, Path(save_dir)/f"{eval_key}_genotype_roc_e{epoch}", dpi=300)
                save_fig_both(fig_pr,  Path(save_dir)/f"{eval_key}_genotype_pr_e{epoch}",  dpi=300)
                save_fig_both(fig_cm,  Path(save_dir)/f"{eval_key}_genotype_cm_e{epoch}",  dpi=300)
            results[f"{eval_key}_genotype_roc"] = fig_roc
            results[f"{eval_key}_genotype_pr"]  = fig_pr
            results[f"{eval_key}_genotype_cm"]  = fig_cm

        if save_dir is not None:
            per_path, sum_path = _write_metric_csvs(eval_key, epoch, save_dir, geno_per_class, geno_scalars, prefix="genotype")
            results[f"{eval_key}_genotype_per_class_csv"] = per_path
            results[f"{eval_key}_genotype_summary_csv"]   = sum_path

        results["genotype_per_class"] = geno_per_class
        results["genotype_summary"]   = geno_scalars

    # ====== METRICS: Genotype NEXT ======
    if (
        next_cell_prediction
        and ("genotype_next" in adata_t.obs.columns)
        and (perturbation_labels_next is not None)
        and (pert_pred_idx is not None)
        and (pert_prob is not None)
    ):
        try:
            geno_next_scalars = _compute_scalar_metrics(perturbation_labels_next, pert_pred_idx, pert_prob)
            for k, v in geno_next_scalars.items():
                metrics[f"{eval_key}_genotype_next/{k}"] = v

            geno_next_per_class = _per_class_metrics(perturbation_labels_next, pert_pred_idx, pert_prob, genotype_names)

            if make_plots:
                fig_roc_next = _plot_multiclass_roc(perturbation_labels_next, pert_prob, f"{eval_key} genotype NEXT ROC (epoch {epoch})", class_names=genotype_names)
                fig_pr_next  = _plot_multiclass_pr(perturbation_labels_next, pert_prob, f"{eval_key} genotype NEXT PR  (epoch {epoch})", class_names=genotype_names)
                fig_cm_next  = _plot_confusion(perturbation_labels_next, pert_pred_idx, genotype_names, f"{eval_key} genotype NEXT Confusion (epoch {epoch})")
                if save_dir is not None:
                    fig_roc_next.savefig(Path(save_dir)/f"{eval_key}_genotypeNEXT_roc_e{epoch}.png", dpi=300, bbox_inches="tight")
                    fig_pr_next.savefig(Path(save_dir)/f"{eval_key}_genotypeNEXT_pr_e{epoch}.png",   dpi=300, bbox_inches="tight")
                    fig_cm_next.savefig(Path(save_dir)/f"{eval_key}_genotypeNEXT_cm_e{epoch}.png",   dpi=300, bbox_inches="tight")
                results[f"{eval_key}_genotypeNEXT_roc"] = fig_roc_next
                results[f"{eval_key}_genotypeNEXT_pr"]  = fig_pr_next
                results[f"{eval_key}_genotypeNEXT_cm"]  = fig_cm_next

            if save_dir is not None:
                per_path_next, sum_path_next = _write_metric_csvs(eval_key, epoch, save_dir, geno_next_per_class, geno_next_scalars, prefix="genotypeNEXT")
                results[f"{eval_key}_genotypeNEXT_per_class_csv"] = per_path_next
                results[f"{eval_key}_genotypeNEXT_summary_csv"]   = sum_path_next

            results["genotypeNEXT_per_class"] = geno_next_per_class
            results["genotypeNEXT_summary"]   = geno_next_scalars

        except Exception as e:
            if logger is not None:
                logger.warning(f"[eval_testdata:{eval_key}] genotype NEXT metric computation failed: {e}")

    # ====== METRICS: Celltype ======
    if ("cls" in include_types) and (celltypes_labels is not None) and (cls_pred_idx is not None) and (cls_prob is not None):
        ct_idx = celltypes_labels
        ct_scalars = _compute_scalar_metrics(ct_idx, cls_pred_idx, cls_prob)
        for k, v in ct_scalars.items():
            metrics[f"{eval_key}_celltype/{k}"] = v

        ct_per_class = _per_class_metrics(ct_idx, cls_pred_idx, cls_prob, celltype_names)

        if make_plots:
            fig_roc_ct = _plot_multiclass_roc(ct_idx, cls_prob, f"{eval_key} celltype ROC (epoch {epoch})", class_names=celltype_names)
            fig_pr_ct  = _plot_multiclass_pr(ct_idx, cls_prob, f"{eval_key} celltype PR  (epoch {epoch})", class_names=celltype_names)
            fig_cm_ct  = _plot_confusion(ct_idx, cls_pred_idx, celltype_names, f"{eval_key} celltype Confusion (epoch {epoch})")
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig_roc_ct.savefig(Path(save_dir)/f"{eval_key}_celltype_roc_e{epoch}.png", dpi=300, bbox_inches="tight")
                fig_pr_ct.savefig(Path(save_dir)/f"{eval_key}_celltype_pr_e{epoch}.png",   dpi=300, bbox_inches="tight")
                fig_cm_ct.savefig(Path(save_dir)/f"{eval_key}_celltype_cm_e{epoch}.png",   dpi=300, bbox_inches="tight")
            results[f"{eval_key}_celltype_roc"] = fig_roc_ct
            results[f"{eval_key}_celltype_pr"]  = fig_pr_ct
            results[f"{eval_key}_celltype_cm"]  = fig_cm_ct

        if save_dir is not None:
            per_path, sum_path = _write_metric_csvs(eval_key, epoch, save_dir, ct_per_class, ct_scalars, prefix="celltype")
            results[f"{eval_key}_celltype_per_class_csv"] = per_path
            results[f"{eval_key}_celltype_summary_csv"]   = sum_path

        results["celltype_per_class"] = ct_per_class
        results["celltype_summary"]   = ct_scalars

    results["adata"] = adata_t
    results["metrics"] = metrics
    results.setdefault("genotype_next", None)
    return results

def save_metric_curve(metric_name, train_values, valid_values, epochs, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.plot(epochs, train_values, label=f"train/{metric_name}")
    plt.plot(epochs, valid_values, label=f"valid/{metric_name}")
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()

    out_base = save_dir / f"curve_{metric_name}"
    save_fig_both(fig, out_base, dpi=200)
    plt.close(fig)

def loader_classification_metrics(
    model,
    loader,
    config,
    train_data_dict,
    device,
):
    """
    Compute cls (celltype) and pert (genotype) metrics directly from the torch loader,
    no AnnData needed. Returns a dict of scalar metrics, e.g.:
    {
        "celltype/acc": ...,
        "celltype/roc_auc_ovr_macro": ...,
        "genotype/aupr_macro": ...,
        ...
    }
    """
    model.eval()
    all_cls_logits = []
    all_cls_true   = []
    all_pert_logits = []
    all_pert_true   = []

    with torch.no_grad():
        for batch_data in loader:
            gene_ids = batch_data["gene_ids"].to(device)
            vals     = batch_data["values"].to(device)

            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            perturbation_labels = batch_data["perturbation_labels"].to(device)

            skpm = gene_ids.eq(train_data_dict["vocab"][config.pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                out = model(
                    gene_ids,
                    vals,
                    src_key_padding_mask=skpm,
                    batch_labels=batch_labels if config.use_batch_label else None,
                    pert_labels=perturbation_labels if config.perturbation_input else None,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                    CLS=config.cell_type_classifier,
                    PERTPRED=(config.perturbation_classifier_weight > 0),
                    PSPRED=(config.ps_weight > 0),
                )

            if config.cell_type_classifier and "cls_output" in out:
                all_cls_logits.append(out["cls_output"].detach().cpu().numpy())
                all_cls_true.append(celltype_labels.detach().cpu().numpy())

            if config.perturbation_classifier_weight > 0 and "pert_output" in out:
                all_pert_logits.append(out["pert_output"].detach().cpu().numpy())
                all_pert_true.append(perturbation_labels.detach().cpu().numpy())

    import numpy as np
    metrics_dict = {}

    # --- celltype head ---
    if all_cls_logits:
        cls_logits = np.concatenate(all_cls_logits, axis=0)
        cls_true   = np.concatenate(all_cls_true,   axis=0)
        cls_prob   = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
        cls_prob   = cls_prob / cls_prob.sum(axis=1, keepdims=True)
        cls_pred   = np.argmax(cls_prob, axis=1)

        cls_scalars = _compute_scalar_metrics(cls_true, cls_pred, cls_prob)
        for k, v in cls_scalars.items():
            metrics_dict[f"celltype/{k}"] = v

    # --- genotype/pert head ---
    if all_pert_logits:
        pert_logits = np.concatenate(all_pert_logits, axis=0)
        pert_true   = np.concatenate(all_pert_true,   axis=0)
        pert_prob   = np.exp(pert_logits - pert_logits.max(axis=1, keepdims=True))
        pert_prob   = pert_prob / pert_prob.sum(axis=1, keepdims=True)
        pert_pred   = np.argmax(pert_prob, axis=1)

        pert_scalars = _compute_scalar_metrics(pert_true, pert_pred, pert_prob)
        for k, v in pert_scalars.items():
            metrics_dict[f"genotype/{k}"] = v

    return metrics_dict

def wrapper_train(
    model, config, data_gen,
    logger=scg.logger,
    save_dir=None,
    device=None,
    fold=0,
    eval_adata_dict: Dict = None,
    run=None,
    trial=None,
    use_early_stopping: bool = True
):
    config = SafeConfig(config)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- budget auto-tuning (decide snapshot cadence + UMAP limits) ----------
    n_obs_hint = None
    try:
        for _k, _ad in (eval_adata_dict or {}).items():
            if _ad is not None:
                n_obs_hint = int(_ad.n_obs)
                break
    except Exception:
        pass

    budget = _auto_budget(config, n_obs_hint=n_obs_hint, logger=logger)

    # Allow env var override ONCE (instead of clobbering every epoch)
    EVAL_SNAPSHOT_EVERY = int(os.environ.get("PERTTF_EVAL_EVERY", budget["EVAL_SNAPSHOT_EVERY"]))

    global _UMAP_BUDGET
    _UMAP_BUDGET = budget  # make visible to eval_testdata via closure

    if logger is not None:
        logger.info(
            f"[budget] snapshot_every={EVAL_SNAPSHOT_EVERY} "
            f"umap_cap={budget['MAX_UMAP_CELLS']} "
            f"k={budget['UMAP_N_NEIGHBORS']} min_dist={budget['UMAP_MIN_DIST']}"
        )

    # ---------- long sequence auto-tune warning ----------
    if int(getattr(config, "max_seq_len", 0)) > 4096:
        old_bs = int(config.batch_size)
        new_bs = max(8, old_bs // 2)
        if logger is not None and new_bs != old_bs:
            logger.warning(
                f"[AUTO-TUNE] Long seq detected ({getattr(config,'max_seq_len',None)}); "
                f"effective batch_size {new_bs} (was {old_bs}). "
                f"(Not mutating wandb config to avoid ConfigError.)"
            )

    num_batch_types = data_gen['num_batch_types']
    vocab = data_gen['vocab']

    optimizer_dict = create_optimizer_dict(model, device, config, num_batch_types)

    best_val_loss = float("inf")
    best_model = None
    best_model_epoch = -1

    if getattr(config, "log_to_wandb", False):
        define_wandb_metrics(fold)

    # ---- Early stopping trackers (only used if use_early_stopping=True) ----
    patience = int(getattr(config, "early_stop_patience", 20))
    min_delta = float(getattr(config, "early_stop_min_delta", 1e-4))
    patience_counter = 0

    if save_dir is None:
        save_dir = Path(f"./save/dev_{config.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # save the current configurations before epoch starts
    torch.save(vocab, save_dir / "vocab.pt")
    running_parameters = {
        'cell_type_to_index': data_gen["cell_type_to_index"],
        'genotype_to_index': data_gen["genotype_to_index"],
        'genes': data_gen["genes"],
        'gene_ids': data_gen["gene_ids"],
        'ps_names': data_gen["ps_names"],
        'config': config.as_dict(),
    }
    torch.save(running_parameters, save_dir / "running_parameters.pt")

    import json
    json.dump(config.as_dict(), open(save_dir / "config.json", "w"))

    train_loader, valid_loader = data_gen['train_loader'], data_gen['valid_loader']

    # Initialize trackers once per fold
    epochs_list = []
    train_history = {k: [] for k in ["mse", "mre", "cls", "pert", "ps", "dab"]}
    valid_history = {k: [] for k in ["mse", "mre", "cls", "pert", "ps", "dab"]}

    # Disable heavy visuals/logging (your choice — keeping your behavior)
    config.make_umaps = False
    config.log_to_wandb = False
    config.umap_on_best = True

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # ---------------- TRAIN ----------------
        train_metrics = {}
        if config.do_train:
            train_metrics = train(
                model,
                train_loader,
                config,
                vocab,
                optimizer_dict,
                epoch=epoch,
                logger=logger,
                device=device,
                fold=fold,
            )

            if logger is not None:
                logger.info("-" * 100)
                logger.info(f"[Fold {fold}] Epoch {epoch:03d} TRAIN losses")
                logger.info(f"    ├─ train_mse:   {train_metrics.get('mse', 0.0):10.6f}")
                logger.info(f"    ├─ train_mre:   {train_metrics.get('mre', 0.0):10.6f}")
                logger.info(f"    ├─ train_mvc:   {train_metrics.get('mvc', 0.0):10.6f}")
                logger.info(f"    ├─ train_cls:   {train_metrics.get('cls', 0.0):10.6f}")
                logger.info(f"    ├─ train_pert:  {train_metrics.get('pert', 0.0):10.6f}")
                logger.info(f"    ├─ train_ps:    {train_metrics.get('ps', 0.0):10.6f}")
                logger.info(f"    └─ train_dab:   {train_metrics.get('dab', 0.0):10.6f}")
                logger.info("-" * 100)

            if getattr(config, "log_to_wandb", False):
                wandb.log({
                    f"train/fold{fold}/mse": train_metrics["mse"],
                    f"train/fold{fold}/mre": train_metrics["mre"],
                    f"train/fold{fold}/mvc": train_metrics["mvc"],
                    f"train/fold{fold}/cls": train_metrics["cls"],
                    f"train/fold{fold}/pert": train_metrics["pert"],
                    f"train/fold{fold}/ps": train_metrics["ps"],
                    f"train/fold{fold}/dab": train_metrics["dab"],
                    "epoch": epoch,
                })

        # track curves
        epochs_list.append(epoch)
        train_history["mse"].append(train_metrics.get("mse", None))
        train_history["mre"].append(train_metrics.get("mre", None))
        train_history["cls"].append(train_metrics.get("cls", None))
        train_history["pert"].append(train_metrics.get("pert", None))
        train_history["ps"].append(train_metrics.get("ps", None))
        train_history["dab"].append(train_metrics.get("dab", None))

        # ---------------- EVAL ----------------
        (
            val_mse, val_mse_next, val_mvc, val_mvc_next, val_mre, val_mre_next,
            val_dab, val_cls, val_pert, val_ps, _val_ps_next
        ) = evaluate(
            model,
            loader=valid_loader,
            config=config,
            vocab=vocab,
            epoch=epoch,
            fold=fold,
            device=device
        )

        val_loss = val_mse
        val_loss_next = val_mse_next
        elapsed = time.time() - epoch_start_time

        if logger is not None:
            logger.info("=" * 100)
            logger.info(f"[Fold {fold}] End of epoch {epoch:03d} | elapsed {elapsed:6.2f}s")
            logger.info(f"    ├─ val_loss (MSE):         {val_loss:10.6f}")
            logger.info(f"    ├─ val_loss_next (MSE):    {val_loss_next:10.6f}")
            logger.info(f"    ├─ val_mre:                {val_mre:10.6f}")
            logger.info(f"    ├─ val_mvc:                {val_mvc:10.6f}")
            logger.info(f"    ├─ val_mvc_next:           {val_mvc_next:10.6f}")
            logger.info(f"    ├─ val_mre_next:           {val_mre_next:10.6f}")
            logger.info(f"    ├─ val_dab:                {val_dab:10.6f}")
            logger.info(f"    ├─ val_cls:                {val_cls:10.6f}")
            logger.info(f"    ├─ val_pert:               {val_pert:10.6f}")
            logger.info(f"    └─ val_ps:                 {val_ps:10.6f}")
            logger.info("=" * 100)

        classif_metrics = loader_classification_metrics(model, valid_loader, config, data_gen, device)
        if logger is not None:
            logger.info(f"[Fold {fold}] Classification metrics (epoch {epoch}):")
            for mk, mv in classif_metrics.items():
                logger.info(f"    {mk}: {mv}")

        if getattr(config, "log_to_wandb", False):
            wandb.log({
                f"valid/fold{fold}/val_loss": val_loss,
                f"valid/fold{fold}/mre": val_mre,
                f"valid/fold{fold}/val_loss_next": val_loss_next,
                f"valid/fold{fold}/mre_next": val_mre_next,
                f"valid/fold{fold}/dab": val_dab,
                f"valid/fold{fold}/cls": val_cls,
                f"valid/fold{fold}/pert": val_pert,
                f"valid/fold{fold}/ps": val_ps,
            })

        valid_history["mse"].append(val_loss)
        valid_history["mre"].append(val_mre)
        valid_history["cls"].append(val_cls)
        valid_history["pert"].append(val_pert)
        valid_history["ps"].append(val_ps)
        valid_history["dab"].append(val_dab)

        # ---------- Optuna reporting (optional) ----------
        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                try:
                    with open(Path(save_dir) / "PRUNED.txt", "a") as f:
                        f.write(f"Pruned at epoch {epoch} with val_loss={val_loss:.6f}\n")
                except Exception:
                    pass
                try:
                    import optuna
                    raise optuna.TrialPruned()
                except Exception:
                    raise Exception("OPTUNA_PRUNED")

        # save metric curves and raw history for this epoch
        for m in ["mse", "mre", "cls", "pert", "ps", "dab"]:
            save_metric_curve(m, train_history[m], valid_history[m], epochs_list, save_dir)

        import pandas as pd
        df_curves = pd.DataFrame({
            "epoch": epochs_list,
            **{f"train_{k}": train_history[k] for k in train_history},
            **{f"valid_{k}": valid_history[k] for k in valid_history},
        })
        df_curves.to_csv(save_dir / f"curve_metrics_epoch{epoch:03d}.csv", index=False)

        # ----- Early stopping / best model tracking -----
        current_score = float(val_loss)
        improvement = float(best_val_loss - current_score)

        if improvement > min_delta:
            best_val_loss = current_score
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            patience_counter = 0

            if logger is not None:
                logger.info(f"[early-stop] New best val_loss {best_val_loss:.6f} at epoch {epoch}.")

            torch.save(best_model.state_dict(), save_dir / "best_model.pt")

            # --- Compute correlations at best epoch (masked positions only) ---
            y_true_masked, y_pred_masked = [], []
            vocab_local = data_gen['vocab']
            pad_id = vocab_local[config.pad_token]

            model.eval()
            with torch.no_grad():
                for batch_data in valid_loader:
                    gene_ids = batch_data["gene_ids"].to(device)
                    inp = batch_data["values"].to(device)
                    tgt = batch_data["target_values"].to(device)

                    skpm = gene_ids.eq(pad_id)
                    with torch.cuda.amp.autocast(enabled=config.amp):
                        out = model(
                            gene_ids, inp,
                            src_key_padding_mask=skpm,
                            batch_labels=(batch_data["batch_labels"].to(device) if config.use_batch_label else None),
                            pert_labels=(batch_data["perturbation_labels"].to(device) if config.perturbation_input else None),
                            MVC=config.GEPC, ECS=(config.ecs_thres > 0),
                            CLS=config.cell_type_classifier,
                            PERTPRED=(config.perturbation_classifier_weight > 0),
                            PSPRED=(config.ps_weight > 0),
                        )["mlm_output"]

                    mp = inp.eq(config.mask_value)

                    yt = tgt[mp].detach().float().view(-1).cpu()
                    yp = out[mp].detach().float().view(-1).cpu()
                    if yt.numel() > 0 and yp.numel() > 0:
                        y_true_masked.append(yt.numpy())
                        y_pred_masked.append(yp.numpy())

            if y_true_masked and y_pred_masked:
                y_true = np.concatenate(y_true_masked)
                y_pred = np.concatenate(y_pred_masked)
                pear, spear, r2 = compute_expr_correlations(y_true, y_pred)
            else:
                pear = spear = r2 = float("nan")

            metrics_path = Path(save_dir) / f"best_epoch_metrics_fold{fold}.csv"
            pd.DataFrame([{
                "epoch": epoch,
                "val_loss": val_loss,
                "pearson": pear,
                "spearman": spear,
                "r2": r2,
            }]).to_csv(metrics_path, index=False)

            if logger:
                logger.info(f"[best-metrics] Saved correlation metrics → {metrics_path}")

        else:
            if use_early_stopping:
                patience_counter += 1
                if logger is not None:
                    logger.info(
                        f"[early-stop] No improvement ({improvement:.6f} < {min_delta}); "
                        f"patience {patience_counter}/{patience}."
                    )

        # ---------- Periodic artifacts / eval snapshots ----------
        if (not getattr(config, "umap_on_best", True)) and (epoch % EVAL_SNAPSHOT_EVERY == 0):
            if logger is not None:
                logger.info(f"Saving snapshot model at epoch {epoch} -> {save_dir}")

            # snapshot the *current* model (NOT best_model)
            torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")

            save_dir2 = save_dir / f"e{epoch}_imgs"
            save_dir2.mkdir(parents=True, exist_ok=True)

            metrics_to_log = {}

            for eval_dict_key, eval_adata in (eval_adata_dict or {}).items():
                dynamic_eval_key = f"{eval_dict_key}_fold{fold}_e{epoch}"
                try:
                    results = eval_testdata(
                        model=model,
                        adata_t=eval_adata,
                        gene_ids=data_gen['gene_ids'],
                        train_data_dict=data_gen,
                        config=config,
                        include_types=["cls", "pert"],
                        logger=logger,
                        epoch=epoch,
                        eval_key=dynamic_eval_key,
                        save_dir=save_dir2,
                        reciprical_sampling=config.get('reciprical_sampling', False),
                        no_pert_for_perturb=config.get('no_pert_for_perturb', config.get('reciprical_sampling', False)),
                    )
                except Exception as e_eval_snap:
                    if logger:
                        logger.warning(f"[snapshot:{eval_dict_key}] eval_testdata failed: {e_eval_snap}")
                    continue

                if not results or not isinstance(results, dict):
                    if logger:
                        logger.warning(f"[snapshot:{eval_dict_key}] eval_testdata returned invalid results; skipping.")
                    continue

                results.setdefault("metrics", {})
                results.setdefault("adata", None)

                save_image_types = [
                    "batch_umap", "celltype_umap", "genotype_umap",
                    "pred_genotype", "pred_celltype",
                    "next_umap_celltype", "next_umap_genotype", "next_umap_genotype_next",
                    "next_umap_pred_genotype",
                ]

                figure_logs = {}
                for key in save_image_types:
                    fig = results.get(key)
                    if fig is None:
                        continue
                    filename = f"{dynamic_eval_key}_{key}_fold{fold}_epoch{epoch}.png"
                    out_path = save_dir2 / filename
                    fig.savefig(out_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    if getattr(config, "log_to_wandb", False):
                        metrics_to_log[f"test/{dynamic_eval_key}_{key}"] = wandb.Image(str(out_path))
                        figure_logs[f"validation/fold{fold}/{key}_epoch{epoch}"] = wandb.Image(str(out_path))

                if getattr(config, "log_to_wandb", False) and figure_logs:
                    wandb.log(figure_logs)

                for k, v in results["metrics"].items():
                    metrics_to_log[f"metrics/{k}"] = v

                if results["adata"] is not None:
                    results["adata"].write_h5ad(save_dir / f'adata_last_validation_{eval_dict_key}.h5ad')

            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            if getattr(config, "log_to_wandb", False) and metrics_to_log:
                wandb.log(metrics_to_log)

        # ---------- step schedulers once per epoch (CORRECT PLACEMENT) ----------
        try:
            sch = optimizer_dict.get("scheduler", None)
            if sch is not None:
                if sch.__class__.__name__ == "ReduceLROnPlateau":
                    sch.step(val_loss)
                else:
                    sch.step()

            if optimizer_dict.get("DAB_separate_optim", False) and optimizer_dict.get("scheduler_dab", None) is not None:
                optimizer_dict["scheduler_dab"].step()

            if getattr(config, "ADV", False):
                if optimizer_dict.get("scheduler_D", None) is not None:
                    optimizer_dict["scheduler_D"].step()
                if optimizer_dict.get("scheduler_E", None) is not None:
                    optimizer_dict["scheduler_E"].step()

        except Exception as e_sched:
            if logger is not None:
                logger.warning(f"[sched] scheduler step failed: {e_sched}")

        # ---------- Early stop break (only if enabled) ----------
        if use_early_stopping and (patience_counter >= patience):
            if logger is not None:
                logger.info(f"[early-stop] Patience reached ({patience}). Stopping at epoch {epoch}.")
            try:
                with open(Path(save_dir) / "EARLY_STOP.txt", "a") as f:
                    f.write(
                        f"Stopped at epoch {epoch} | best_epoch={best_model_epoch} | "
                        f"best_val_loss={best_val_loss:.6f}\n"
                    )
            except Exception:
                pass
            break

    # ---------------- AFTER TRAINING LOOP ----------------
    if best_model is None:
        best_model = copy.deepcopy(model)
        if best_model_epoch < 0:
            best_model_epoch = config.epochs

    if logger:
        logger.info(
            f"[best-only] Summarizing best checkpoint from epoch {best_model_epoch} "
            f"(best_val_loss={best_val_loss:.6f})"
        )

    save_dir_best = save_dir / f"best_epoch_e{best_model_epoch}"
    save_dir_best.mkdir(parents=True, exist_ok=True)

    merged_scalar_metrics = {}

    # =====================================================
    # (1) BEST-EPOCH EVAL + UMAP EXPORT
    # =====================================================
    try:
        for eval_dict_key, eval_adata in (eval_adata_dict or {}).items():
            if eval_adata is None:
                continue

            dynamic_eval_key = f"{eval_dict_key}_best"

            try:
                results = eval_testdata(
                    model=best_model,
                    adata_t=eval_adata,
                    gene_ids=data_gen['gene_ids'],
                    train_data_dict=data_gen,
                    config=config,
                    include_types=["cls", "pert"],
                    logger=logger,
                    epoch=best_model_epoch,
                    eval_key=dynamic_eval_key,
                    save_dir=save_dir_best,
                    reciprical_sampling=config.get('reciprical_sampling', False),
                    no_pert_for_perturb=config.get('no_pert_for_perturb', config.get('reciprical_sampling', False)),
                )
            except Exception as e_eval_best:
                if logger:
                    logger.warning(f"[best-only:{eval_dict_key}] eval_testdata block failed: {e_eval_best}")
                continue

            if not results or not isinstance(results, dict):
                if logger:
                    logger.warning(f"[best-only:{eval_dict_key}] eval_testdata returned no results.")
                continue

            save_image_types = [
                "batch_umap", "celltype_umap", "genotype_umap",
                "pred_genotype", "pred_celltype",
                "next_umap_celltype", "next_umap_genotype", "next_umap_genotype_next",
                "next_umap_pred_genotype",
            ]
            for key in save_image_types:
                fig = results.get(key)
                if fig is None:
                    continue
                out_base = save_dir_best / f"{dynamic_eval_key}_{key}_fold{fold}_epoch{best_model_epoch}"
                try:
                    save_fig_both(fig, out_base, dpi=300, bbox_inches="tight")
                except Exception as e_fig:
                    if logger:
                        logger.warning(f"[best-only:{eval_dict_key}] Failed saving fig {key}: {e_fig}")
                plt.close(fig)

            extra_metric_figs = [
                f"{dynamic_eval_key}_genotype_roc",
                f"{dynamic_eval_key}_genotype_pr",
                f"{dynamic_eval_key}_genotype_cm",
                f"{dynamic_eval_key}_celltype_roc",
                f"{dynamic_eval_key}_celltype_pr",
                f"{dynamic_eval_key}_celltype_cm",
                f"{dynamic_eval_key}_genotypeNEXT_roc",
                f"{dynamic_eval_key}_genotypeNEXT_pr",
                f"{dynamic_eval_key}_genotypeNEXT_cm",
            ]
            for fig_key in extra_metric_figs:
                fig = results.get(fig_key)
                if fig is None:
                    continue
                out_base = save_dir_best / f"{fig_key}"
                try:
                    save_fig_both(fig, out_base,dpi=300, bbox_inches="tight")
                except Exception as e_fig2:
                    if logger:
                        logger.warning(f"[best-only:{eval_dict_key}] Failed saving fig {fig_key}: {e_fig2}")
                plt.close(fig)

            if isinstance(results.get("metrics"), dict):
                for k, v in results["metrics"].items():
                    merged_scalar_metrics[f"{eval_dict_key}:{k}"] = v

            if results.get("adata") is not None:
                try:
                    results["adata"].write_h5ad(save_dir_best / f"adata_best_{eval_dict_key}.h5ad")
                except Exception as e_ann:
                    if logger:
                        logger.warning(f"[best-only:{eval_dict_key}] Failed saving h5ad: {e_ann}")

        plt.close('all')
        gc.collect()

    except Exception as e_eval_outer:
        if logger:
            logger.warning(f"[best-only] Unexpected failure in best-epoch eval loop: {e_eval_outer}")

    # =====================================================
    # (2) HIGH-LEVEL CLASSIFICATION METRICS
    # =====================================================
    try:
        loader_best_metrics = loader_classification_metrics(
            best_model,
            data_gen['valid_loader'],
            config,
            data_gen,
            device,
        )
        merged_scalar_metrics.update(loader_best_metrics)
    except Exception as e_cls:
        if logger:
            logger.warning(f"[best-only] loader_classification_metrics failed: {e_cls}")

    # =====================================================
    # (3) PER-CLASS TABLES
    # =====================================================
    try:
        compute_and_save_perclass_tables(
            best_model,
            data_gen['valid_loader'],
            data_gen,
            config,
            device,
            save_dir_best,
            fold,
            logger=logger,
        )
    except Exception as e_pc:
        if logger:
            logger.warning(f"[best-only] per-class table generation failed: {e_pc}")

    # =====================================================
    # (4) EXPRESSION CORRELATIONS
    # =====================================================
    pear = spear = r2 = float("nan")
    try:
        y_true_masked, y_pred_masked = [], []
        vocab_local = data_gen['vocab']
        pad_id = vocab_local[config.pad_token]
        best_model.eval()

        with torch.no_grad():
            for batch_data in data_gen['valid_loader']:
                gene_ids = batch_data["gene_ids"].to(device)
                inp = batch_data["values"].to(device)
                tgt = batch_data["target_values"].to(device)

                skpm = gene_ids.eq(pad_id)
                with torch.cuda.amp.autocast(enabled=config.amp):
                    out_logits = best_model(
                        gene_ids, inp,
                        src_key_padding_mask=skpm,
                        batch_labels=(batch_data["batch_labels"].to(device) if config.use_batch_label else None),
                        pert_labels=(batch_data["perturbation_labels"].to(device) if config.perturbation_input else None),
                        MVC=config.GEPC,
                        ECS=(config.ecs_thres > 0),
                        CLS=config.cell_type_classifier,
                        PERTPRED=(config.perturbation_classifier_weight > 0),
                        PSPRED=(config.ps_weight > 0),
                    )["mlm_output"]

                mp = inp.eq(config.mask_value)
                yt = tgt[mp].detach().float().view(-1).cpu()
                yp = out_logits[mp].detach().float().view(-1).cpu()

                if yt.numel() > 0 and yp.numel() > 0:
                    y_true_masked.append(yt.numpy())
                    y_pred_masked.append(yp.numpy())

        if y_true_masked and y_pred_masked:
            yt_all = np.concatenate(y_true_masked)
            yp_all = np.concatenate(y_pred_masked)
            pear, spear, r2 = compute_expr_correlations(yt_all, yp_all)
        else:
            pear = spear = r2 = float("nan")

    except Exception as e_corr:
        if logger:
            logger.warning(f"[best-only] Correlation recompute failed: {e_corr}")

    # =====================================================
    # (5) ALWAYS WRITE regression summary CSV
    # =====================================================
    save_best_expression_csv(
        save_dir_best=save_dir_best,
        fold=fold,
        best_model_epoch=best_model_epoch,
        best_val_loss=best_val_loss,
        pear=pear,
        spear=spear,
        r2=r2,
        logger=logger,
    )

    # =====================================================
    # (6) DEBUG METRICS CSV (merged)
    # =====================================================
    try:
        import pandas as pd
        debug_metrics_path = save_dir_best / f"best_epoch_metrics_fold{fold}.csv"
        row = {
            "epoch": best_model_epoch,
            "best_val_loss": best_val_loss,
            "pearson": pear,
            "spearman": spear,
            "r2": r2,
        }
        row.update(merged_scalar_metrics)
        pd.DataFrame([row]).to_csv(debug_metrics_path, index=False)
        if logger:
            logger.info(f"[best-metrics] Saved debug metrics → {debug_metrics_path}")
    except Exception as e_save_debug:
        if logger:
            logger.warning(f"[best-only] Failed to save debug metrics CSV: {e_save_debug}")

    # final save of best model + config
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    torch.save(vocab, save_dir / "vocab.pt")

    running_parameters = {
        'cell_type_to_index': data_gen["cell_type_to_index"],
        'genotype_to_index': data_gen["genotype_to_index"],
        'genes': data_gen["genes"],
        'gene_ids': data_gen["gene_ids"],
        'ps_names': data_gen["ps_names"],
        'config': config.as_dict(),
    }
    torch.save(running_parameters, save_dir / "running_parameters.pt")
    json.dump(config.as_dict(), open(save_dir / "config.json", "w"))

    return {
        "model": best_model,
        "best_model_epoch": best_model_epoch,
        "best_val_loss": best_val_loss,
    }