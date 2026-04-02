# config_gen.py
import os
import sys
import copy
from typing import Literal, Tuple, Optional

from scgpt.utils import set_seed

# ----------------------------
# rank helpers (no hard torch import)
# ----------------------------
def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        return (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


# ----------------------------
# Config shim (works like wandb.config but never KeyErrors)
# ----------------------------
class _ConfigShim:
    def __init__(self, d: dict):
        self._d = copy.deepcopy(d)

    def __getattr__(self, k):
        if k in self._d:
            return self._d[k]
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k == "_d":
            return super().__setattr__(k, v)
        self._d[k] = v

    def update(self, updates: dict, allow_val_change: bool = True):
        # allow_val_change kept for API compatibility
        for k, v in updates.items():
            self._d[k] = v

    def as_dict(self):
        return copy.deepcopy(self._d)

    def __repr__(self):
        return f"_ConfigShim({self._d})"


def _safe_wandb_init(cfg: dict, project_name: str, requested_mode: str):
    """
    Only used when wandb_mode != 'disabled'.
    Rank-0 honors requested_mode; others are disabled.
    """
    import wandb  # local import so disabled mode never touches wandb

    os.environ.setdefault("WANDB__EXECUTABLE", sys.executable)

    effective_mode = "disabled" if not _is_main_process() else requested_mode
    try:
        run = wandb.init(
            config=cfg,
            project=project_name,
            reinit=True,
            mode=effective_mode,
        )
        return wandb.config, run
    except Exception as e:
        print(f"[W&B] init failed → falling back to shim. Reason: {e}", flush=True)
        return _ConfigShim(cfg), None


def generate_config(
    parameter_dict: dict,
    project_name: str = "scGPT",
    wandb_mode: Literal["disabled", "online", "offline"] = "disabled",
) -> Tuple[object, Optional[object]]:
    # Prefer active interpreter
    os.environ.setdefault("WANDB__EXECUTABLE", sys.executable)

    # -----------------------
    # HARD DEFAULTS (keys your code expects to exist)
    # -----------------------
    REQUIRED_DEFAULTS = {
        # toggles / losses
        "ADV": False,
        "dab_weight": 0.0,
        "GEPC": True,

        # batch stuff (the ones crashing you)
        "DSBN": False,
        "use_batch_label": False,
        "per_seq_batch_sample": False,

        # padding + masking
        "pad_token": "<pad>",
        "pad_value": -2,
        "mask_value": -1,
        "mask_ratio": 0.15,

        # model
        "layer_size": 64,
        "nlayers": 4,
        "nhead": 4,
        "dropout": 0.4,
        "fast_transformer": False,
        "pre_norm": False,

        # training
        "epochs": 100,
        "lr": 1e-3,
        "batch_size": 64,
        "seed": 42,
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.01,

        # data
        "USE_HVG": True,
        "n_hvg": 3000,
        "explicit_zero_prob": True,
    }
    for k, v in REQUIRED_DEFAULTS.items():
        parameter_dict.setdefault(k, v)

    # -----------------------
    # derived fields
    # -----------------------
    parameter_dict["special_tokens"] = [
        parameter_dict.get("pad_token", "<pad>"),
        parameter_dict.get("cls_token", "<cls>"),
        "<eoc>",
    ]
    # seq len safety
    try:
        n_hvg_val = int(parameter_dict.get("n_hvg", 3000))
    except Exception:
        n_hvg_val = 3000
    parameter_dict["n_hvg"] = n_hvg_val
    parameter_dict["max_seq_len"] = n_hvg_val + 1

    # -----------------------
    # IMPORTANT: if disabled, DO NOT use wandb.config at all
    # -----------------------
    if wandb_mode == "disabled":
        cfg = _ConfigShim(parameter_dict)
        run = None
    else:
        cfg, run = _safe_wandb_init(parameter_dict, project_name, wandb_mode)

    # seed
    try:
        set_seed(int(getattr(cfg, "seed", 42)))
    except Exception:
        pass

    # mutual exclusion guard (safe)
    if getattr(cfg, "ADV", False) and float(getattr(cfg, "dab_weight", 0.0)) > 0:
        raise ValueError("ADV and DAB cannot be both enabled.")

    return cfg, run