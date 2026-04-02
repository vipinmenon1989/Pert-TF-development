# config_gen.py
import os
import sys
import copy
from typing import Literal

import wandb
from scgpt.utils import set_seed
import copy

# Check the Python interpreter being used
#print(sys.executable)
# ---- rank helpers (no hard torch import) ----
def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        return (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True

# Simple config wrapper when not using wandb.config
class _ConfigShim:
    def __init__(self, d: dict):
        # deep copy to avoid upstream mutation surprises
        self._d = copy.deepcopy(d)
    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        if k == "_d":
            return super().__setattr__(k, v)
        self._d[k] = v
    def as_dict(self):
        return copy.deepcopy(self._d)


def _safe_wandb_init(cfg: dict, project_name: str, requested_mode: str):
    """
    Initialize W&B robustly:
      - rank-0 honors requested_mode (online/offline); others are disabled
      - thread start method; long timeout; fail-open to disabled
    """
    # Prefer the interpreter we're actually running
    os.environ["WANDB__EXECUTABLE"] = sys.executable
    os.environ.setdefault("WANDB_START_METHOD", "thread")  # NOT 'fork'
    # Worker ranks never talk to WANDB
    if not _is_main_process():
        effective_mode = "disabled"
    else:
        effective_mode = requested_mode  # "online", "offline", or "disabled"

    try:
        run = wandb.init(
            config=cfg,
            project=project_name,
            reinit=True,
            settings=wandb.Settings(start_method="thread", init_timeout=300),
            mode=effective_mode,
        )
        return wandb.config, run
    except Exception as e:
        # Fail-open: keep training without WANDB
        print(f"[W&B] init failed → disabled mode. Reason: {e}", flush=True)
        try:
            run = wandb.init(mode="disabled")  # creates a dummy run
            # Keep a shim so the rest of the code can do config.attr
            return wandb.config, run
        except Exception:
            # Absolute fallback: no wandb object at all
            return _ConfigShim(cfg), None

def generate_config(parameter_dict,
        project_name = "scGPT",
        wandb_mode : Literal["disabled","online","offline"] = "disabled"):
    # If it's not the desired interpreter, set the WANDB__EXECUTABLE environment variable
    # For example, if you want to use Python 3.8:
    os.environ["WANDB__EXECUTABLE"] = "/usr/local/bin/python"  # Replace with the actual path


    # settings for input and preprocessing

    parameter_dict['special_tokens'] = [parameter_dict.get('pad_token', '<pad>'), 
                                        parameter_dict.get('cls_token', '<cls>'), 
                                        "<eoc>"]

    parameter_dict['simple_sampling']  = parameter_dict.get('simple_sampling', False)
    parameter_dict['fix_nonzero_prop'] = parameter_dict.get('fix_nonzero_prop', False)
    parameter_dict['nonzero_prop'] = parameter_dict.get('nonzero_prop', 0.9)

    parameter_dict['pert_exp_mode'] =  parameter_dict.get('pert_exp_mode', 'concat') 
    parameter_dict['reciprical_sampling'] = parameter_dict.get('reciprical_sampling', False)
    parameter_dict['no_pert_for_perturb'] = True if parameter_dict['reciprical_sampling'] else parameter_dict.get('no_pert_for_perturb', False)
    parameter_dict['reciprical_genotype'] =  parameter_dict.get('reciprical_genotype', False)
    
    if parameter_dict['next_cell_pred_type'] == 'identity':
      parameter_dict['next_weight'] = 0
    if parameter_dict['next_cell_pred_type'] in ['identity', 'lochness']:
      parameter_dict['reciprical_sampling'] = False
      parameter_dict['no_pert_for_perturb'] = False
      parameter_dict['reciprical_genotype'] =  False
    # --- HVG & sequence-length safety (does not assume fast_transformer exists) ---
    try:
        n_hvg_val = int(parameter_dict.get("n_hvg", 3000))
    except Exception:
        n_hvg_val = 3000
    parameter_dict["n_hvg"] = n_hvg_val
    parameter_dict['max_seq_len'] = parameter_dict['n_hvg'] + 1
    use_wandb=True
    if use_wandb:
        config, run = _safe_wandb_init(parameter_dict, project_name, wandb_mode)
    else:
        config = _ConfigShim(parameter_dict)
        run = None
    
        # ---------- PERTTF runtime knobs (safe defaults) ----------
    if not hasattr(config, "make_umaps"):            config.make_umaps = False
    if not hasattr(config, "umap_on_best"):          config.umap_on_best = True
    if not hasattr(config, "make_metric_plots"):     config.make_metric_plots = False
    if not hasattr(config, "log_to_wandb"):          config.log_to_wandb = True
    if not hasattr(config, "log_batch_metrics"):     config.log_batch_metrics = False
    if not hasattr(config, "eval_snapshot_every"):   config.eval_snapshot_every = 0
    if not hasattr(config, "mask_during_eval"):      config.mask_during_eval = False

    # Set seed (works for both shim and wandb.config)
    set_seed(config.seed)

    if config.ADV and config.dab_weight > 0:
        raise ValueError("ADV and DAB cannot be both True.")

    # For visibility in logs
    try:
        print(config)
    except Exception:
        # wandb.config prints fine; shim prints dict
        print(getattr(config, "as_dict", lambda: parameter_dict)())

    return (config, run)
