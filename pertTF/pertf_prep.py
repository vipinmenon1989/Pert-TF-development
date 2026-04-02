#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scgpt.preprocess import Preprocessor


def parse_args():
    ap = argparse.ArgumentParser("Prepare & persist preprocessed AnnData for CV (celltype_2-safe)")
    ap.add_argument("--raw-h5ad", required=True, type=Path, help="Input .h5ad (must contain obs['celltype_2'])")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-wt-frac", type=float, default=0.1)
    ap.add_argument("--n-bins", type=int, default=51)
    ap.add_argument("--n-hvg", type=int, default=3000)

    ap.add_argument("--filter-missing-subcluster", action="store_true", default=True)
    ap.add_argument("--no-filter-missing-subcluster", dest="filter_missing_subcluster", action="store_false")

    # Make downstream code safe even if it still references obs["celltype"]
    ap.add_argument("--overwrite-celltype-with-celltype2", action="store_true", default=True)

    # Safety: crash if input celltype_2 looks overwritten to 20-class
    ap.add_argument("--guard-celltype2-cardinality", action="store_true", default=True)
    ap.add_argument("--no-guard-celltype2-cardinality", dest="guard_celltype2_cardinality", action="store_false")

    return ap.parse_args()


def nunique_obs(ad, key: str):
    if key not in ad.obs.columns:
        return None
    return ad.obs[key].astype(str).nunique()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    in_path = args.raw_h5ad.resolve()
    print("[prep] reading:", in_path)
    adata0 = sc.read_h5ad(str(in_path))
    print("[prep] loaded n_obs,n_vars:", adata0.n_obs, adata0.n_vars)

    print("[raw] nunique:",
          {k: nunique_obs(adata0, k) for k in ["celltype", "celltype_2", "sub.cluster", "celltype_2_old"]})

    if "celltype_2" not in adata0.obs.columns:
        raise ValueError("RAW input missing obs['celltype_2']. (Do NOT use celltype_2_old unless you map it.)")

    # Optional hard guard: if celltype_2 cardinality matches fine labels, it's likely overwritten upstream
    if args.guard_celltype2_cardinality:
        n_ct2 = nunique_obs(adata0, "celltype_2")
        n_cell = nunique_obs(adata0, "celltype")
        n_sub = nunique_obs(adata0, "sub.cluster")
        if (n_sub is not None and n_ct2 == n_sub) or (n_cell is not None and n_ct2 == n_cell):
            raise ValueError(
                f"obs['celltype_2'] looks overwritten (nunique={n_ct2}); "
                f"celltype={n_cell}, sub.cluster={n_sub}. "
                "You are not using the integrated file with the coarse 14-class labels."
            )

    # --- optional filter missing sub.cluster ---
    if args.filter_missing_subcluster and ("sub.cluster" in adata0.obs.columns):
        valid = adata0.obs["sub.cluster"].notna()
        adata0 = adata0[valid].copy()
        print("[prep] after sub.cluster filter n_obs:", adata0.n_obs)

    # --- tidy genotype labels ---
    if "gene" not in adata0.obs.columns:
        raise ValueError("adata.obs['gene'] missing")

    adata0.obs["gene"] = (
        adata0.obs["gene"].astype(str)
        .str.replace("124_NANOGe_het", "124_NANOGe-het", regex=False)
        .str.replace("123_NANOGe_het", "123_NANOGe-het", regex=False)
    )

    genotypes = (
        adata0.obs["gene"].str.split("_").str[-1]
        .replace({"WT111": "WT", "WT4": "WT", "NGN3": "NEUROG3"})
        .fillna("WT")
        .astype(str)
    )
    adata0.obs["genotype"] = genotypes

    # --- thinning WT ---
    rng = np.random.default_rng(int(args.seed))
    keep_mask = (genotypes != "WT") | ((genotypes == "WT") & (rng.random(adata0.n_obs) < float(args.keep_wt_frac)))
    adata = adata0[keep_mask, :].copy()
    print("[prep] after thinning n_obs,n_vars:", adata.n_obs, adata.n_vars)

    # Reattach (consistent) labels after subsetting
    adata.obs["genotype"] = adata0.obs["genotype"].loc[adata.obs_names].values

    # ==========================================================
    # IMPORTANT: preserve *existing* celltype_2 (coarse labels)
    # ==========================================================
    adata.obs["celltype_2"] = adata0.obs["celltype_2"].loc[adata.obs_names].astype("category")

    # Optional compatibility: force obs["celltype"] == obs["celltype_2"]
    if args.overwrite_celltype_with_celltype2:
        adata.obs["celltype"] = adata.obs["celltype_2"].copy()

    print("[prep] sanity nunique:",
          "celltype_2 =", adata.obs["celltype_2"].astype(str).nunique(),
          "| celltype =", adata.obs["celltype"].astype(str).nunique() if "celltype" in adata.obs.columns else None)

    # --- GPTin layer ---
    adata.layers["GPTin"] = adata.X

    # --- HVG scoring and subset ---
    ad_tmp = adata.copy()
    ad_tmp.X = ad_tmp.layers["GPTin"]
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)
    sc.pp.highly_variable_genes(
        ad_tmp, flavor="seurat_v3", batch_key=None, n_top_genes=int(args.n_hvg)
    )

    if "highly_variable_rank" not in ad_tmp.var:
        raise RuntimeError("Scanpy did not produce 'highly_variable_rank' with seurat_v3.")

    ranks0 = (ad_tmp.var["highly_variable_rank"].astype(float) - 1.0)
    adata.var["hvg_rank"] = ranks0.reindex(adata.var_names)

    keep_genes = adata.var.sort_values("hvg_rank").index[: int(args.n_hvg)]
    adata = adata[:, keep_genes].copy()
    print("[prep] after HVG subset n_obs,n_vars:", adata.n_obs, adata.n_vars)

    # --- binning ---
    pp = Preprocessor(
        use_key="GPTin",
        normalize_total=None,
        log1p=False,
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=int(args.n_bins),
        result_binned_key="X_binned",
    )
    pp(adata, batch_key=None)
    if "X_binned" not in adata.layers:
        raise RuntimeError("Binning did not produce X_binned.")

    # --- compact X_binned to uint8 CSR ---
    Xb = adata.layers["X_binned"]
    if sp.issparse(Xb):
        adata.layers["X_binned"] = Xb.astype(np.uint8).tocsr()
    else:
        adata.layers["X_binned"] = sp.csr_matrix(np.asarray(Xb, dtype=np.uint8))

    # --- slim ---
    adata.layers.pop("GPTin", None)
    adata.raw = None

    # Safer filename (prevents accidentally reading an older file)
    out_h5ad = (args.outdir / f"preprocessed_hvg{int(args.n_hvg)}.{in_path.stem}.h5ad").resolve()
    adata.write_h5ad(out_h5ad, compression="lzf")
    print("[prep] wrote:", out_h5ad)

    # Read-back sanity (catch “wrote somewhere else / read wrong file” mistakes)
    ad_chk = sc.read_h5ad(str(out_h5ad), backed="r")
    print("[wrote] nunique:",
          {k: nunique_obs(ad_chk, k) for k in ["celltype", "celltype_2", "sub.cluster", "celltype_2_old"]})


if __name__ == "__main__":
    main()