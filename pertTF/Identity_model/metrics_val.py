#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

# ===== EDIT THIS to your Identity root =====
BASE = Path("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model")

DEST = BASE / "metric_bovplot_validation_per_class"   # per your requested name
DEST.mkdir(parents=True, exist_ok=True)

FOLDS = [1, 2, 3, 4, 5]

# We will only copy these three families
WANTED_PREFIXES = (
    "validation_best_celltype_per_class_",
    "validation_best_genotype_per_class_",
)

def parse_epoch_from_dirname(p: Path) -> int:
    m = re.search(r"best_epoch_e(\d+)", p.name)
    return int(m.group(1)) if m else -1

def pick_latest_best_epoch_dir(fold_dir: Path) -> Path | None:
    cands = sorted(
        [d for d in fold_dir.glob("best_epoch_e*") if d.is_dir()],
        key=parse_epoch_from_dirname,
        reverse=True,
    )
    return cands[0] if cands else None

def add_fold_suffix(name: str, fold: int) -> str:
    return name[:-4] + f"_fold{fold}.csv" if name.lower().endswith(".csv") else f"{name}_fold{fold}"

def main():
    copied = 0
    for fold in FOLDS:
        fold_dir = BASE / f"fold_{fold}"
        if not fold_dir.exists():
            print(f"[WARN] missing {fold_dir}")
            continue

        best_dir = pick_latest_best_epoch_dir(fold_dir)
        if best_dir is None:
            print(f"[WARN] no best_epoch_eXX in {fold_dir}")
            continue

        # collect target CSVs in that best dir
        found_any = False
        for csv_path in best_dir.glob("*.csv"):
            name = csv_path.name
            if any(name.startswith(pref) for pref in WANTED_PREFIXES):
                out_name = add_fold_suffix(name, fold)
                dst = DEST / out_name
                shutil.copy2(csv_path, dst)
                print(f"[COPY] fold {fold}: {name} -> {dst.name}")
                copied += 1
                found_any = True

        if not found_any:
            print(f"[WARN] no matching summary CSVs in {best_dir}")

    print(f"\n[SUMMARY] Copied {copied} file(s) to: {DEST}")

if __name__ == "__main__":
    main()

