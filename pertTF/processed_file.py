#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def safe_series(x):
    s = pd.Series(x).astype(str)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return s


def summarize_col(obs: pd.DataFrame, key: str, topn: int = 20):
    if key not in obs.columns:
        print(f"\n[{key}] MISSING")
        return None
    s = safe_series(obs[key])
    nunique = s.nunique(dropna=True)
    nmissing = int(s.isna().sum())
    print(f"\n[{key}] dtype={obs[key].dtype}  nunique={nunique}  missing={nmissing}/{len(s)}")
    vc = s.value_counts(dropna=True).head(topn)
    print(f"[{key}] top-{topn}:\n{vc.to_string()}")
    return s


def main():
    ap = argparse.ArgumentParser(
        description="Inspect celltype vs celltype_2 in a .h5ad and optionally write a cleaned file without 'celltype'."
    )
    ap.add_argument("h5ad", type=Path, help="Input .h5ad file")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output .h5ad (default: <input>.celltype2_only.h5ad)")
    ap.add_argument("--force", action="store_true", help="Write cleaned output even if celltype == celltype_2")
    ap.add_argument("--topn", type=int, default=20, help="Top-N value counts to print per column (default: 20)")
    args = ap.parse_args()

    in_h5ad = args.h5ad.resolve()
    if not in_h5ad.exists():
        raise FileNotFoundError(f"Input not found: {in_h5ad}")

    out_h5ad = args.out
    if out_h5ad is None:
        out_h5ad = Path(str(in_h5ad).replace(".h5ad", ".celltype2_only.h5ad"))
    out_h5ad = out_h5ad.resolve()

    print(f"Reading (backed='r') for quick metadata… {in_h5ad}")
    ad_backed = sc.read_h5ad(str(in_h5ad), backed="r")

    obs_cols = list(ad_backed.obs.columns)
    print("\nobs columns:")
    for c in obs_cols:
        print(" -", c)

    obs = ad_backed.obs.copy()
    del ad_backed

    s1 = summarize_col(obs, "celltype", topn=args.topn)
    s2 = summarize_col(obs, "celltype_2", topn=args.topn)

    if s1 is None or s2 is None:
        print("\nOne of the columns is missing; not writing output.")
        return 1

    a = s1.to_numpy()
    b = s2.to_numpy()
    both_nan = pd.isna(a) & pd.isna(b)
    equal = (a == b) | both_nan
    frac_equal = float(np.mean(equal))

    diffs = obs.loc[~equal, ["celltype", "celltype_2"]].copy()
    print(f"\nComparison: frac(celltype == celltype_2) = {frac_equal:.6f}")
    print(f"n_different = {diffs.shape[0]}")

    if diffs.shape[0] > 0:
        print("\nExample differing rows (up to 25):")
        print(diffs.head(25).to_string())

        pairs = diffs.astype(str).value_counts().head(20)
        print("\nTop mismatch pairs (celltype, celltype_2) up to 20:")
        print(pairs.to_string())
    else:
        print("\nColumns appear identical (after string conversion).")

    if diffs.shape[0] == 0 and not args.force:
        print("\nNot writing new file (identical). Use --force to still write cleaned output.")
        return 0

    print(f"\nLoading full AnnData to write cleaned file… {in_h5ad}")
    ad = sc.read_h5ad(str(in_h5ad))

    if "celltype" in ad.obs.columns:
        ad.obs.drop(columns=["celltype"], inplace=True)
        print("Dropped obs['celltype']")

    if "celltype_2" in ad.obs.columns:
        ad.obs["celltype_2"] = ad.obs["celltype_2"].astype("category")
        print("Converted obs['celltype_2'] to category")

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing cleaned h5ad → {out_h5ad}")
    ad.write_h5ad(str(out_h5ad), compression="gzip")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
