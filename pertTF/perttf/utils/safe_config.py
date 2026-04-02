# perttf/utils/safe_config.py
class SafeConfig:
    """
    Drop-in wrapper for your config that makes *all* mutations safe with W&B.
    - config.foo = v     -> cfg.update({"foo": v}, allow_val_change=True) if possible
    - config["foo"] = v  -> same
    - config.update({...}) -> forces allow_val_change=True
    - getattr / dict-like access just proxy through
    """
    __slots__ = ("_cfg",)

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    # --- reads ---
    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def __getitem__(self, key):
        try:
            return self._cfg[key]
        except Exception:
            return getattr(self._cfg, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default

    def as_dict(self):
        try:
            return self._cfg.as_dict()
        except Exception:
            try:
                return dict(self._cfg)
            except Exception:
                return {k: getattr(self._cfg, k) for k in dir(self._cfg) if not k.startswith("_")}

    # --- writes (the magic) ---
    def __setattr__(self, name, value):
        self._safe_update({name: value})

    def __setitem__(self, key, value):
        self._safe_update({key: value})

    def update(self, d=None, **kwargs):
        d = dict(d or {})
        d.update(kwargs)
        self._safe_update(d)

    # --- internal ---
    def _safe_update(self, updates: dict):
        # Try wandb.Config route first
        try:
            if hasattr(self._cfg, "update"):
                # Force allow_val_change=True; ignore caller’s flags
                self._cfg.update(dict(updates), allow_val_change=True)
                return
        except Exception:
            pass
        # Fallback to plain attribute set
        for k, v in updates.items():
            try:
                setattr(self._cfg, k, v)
            except Exception:
                try:
                    self._cfg[k] = v
                except Exception:
                    # last resort: ignore
                    pass
