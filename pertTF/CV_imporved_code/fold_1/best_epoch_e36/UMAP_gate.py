#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def get_umap(adata, key):
    if key in adata.obsm:
        return np.asarray(adata.obsm[key])
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])
    raise KeyError(f"No UMAP found in obsm['{key}'] or obsm['X_umap'].")


def genotype_enrichment(obs_esc, tail_mask_esc):
    g_all = obs_esc["genotype"].astype(str)
    g_tail = g_all[tail_mask_esc]

    all_counts = g_all.value_counts()
    tail_counts = g_tail.value_counts()

    out = pd.DataFrame(
        {
            "count_in_tail": tail_counts,
            "count_in_ESC": all_counts,
        }
    ).fillna(0).astype(int)
    out["frac_in_tail"] = out["count_in_tail"] / max(1, int(tail_mask_esc.sum()))
    out["frac_in_ESC"] = out["count_in_ESC"] / max(1, len(g_all))
    out["enrichment_ratio"] = (out["frac_in_tail"] + 1e-12) / (out["frac_in_ESC"] + 1e-12)
    return out.sort_values(["count_in_tail", "enrichment_ratio"], ascending=[False, False])


def parse_args():
    ap = argparse.ArgumentParser(
        "Auto-gate ESC tail near DE using UMAP neighbor composition (NaN-safe + upper bias)"
    )
    ap.add_argument("--h5ad", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--celltype-key", default="celltype_2")
    ap.add_argument("--genotype-key", default="genotype")
    ap.add_argument("--umap-key", default="X_umap_scgpt")

    ap.add_argument("--esc-label", default="ESC")
    ap.add_argument("--de-label", default="DE")

    ap.add_argument("--k", type=int, default=50, help="kNN in UMAP space")
    ap.add_argument("--min-de-frac", type=float, default=0.20, help="min fraction of DE neighbors")
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.05,
        help="also include top X fraction of ESC_upper by score (0 disables)",
    )
    ap.add_argument("--seed", type=int, default=0)

    # “upper” controls
    ap.add_argument(
        "--min-umap2",
        type=float,
        default=None,
        help="Hard cutoff: require UMAP2 >= this value (finite subset coords).",
    )
    ap.add_argument(
        "--esc-umap2-quantile",
        type=float,
        default=None,
        help="Data-driven cutoff: keep ESC with UMAP2 >= quantile among ESC (e.g. 0.80).",
    )
    ap.add_argument(
        "--y-weight",
        type=float,
        default=0.0,
        help="Soft bias: multiply score by (normalized UMAP2 ** y_weight). 0 disables.",
    )

    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument(
        "--write-finite-subset-h5ad",
        action="store_true",
        default=False,
        help="Write the finite-UMAP subset h5ad used for gating (debugging).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(args.h5ad))
    obs = adata.obs.copy()

    if args.celltype_key not in obs.columns:
        raise ValueError(f"Missing obs['{args.celltype_key}']")
    if args.genotype_key not in obs.columns:
        raise ValueError(f"Missing obs['{args.genotype_key}']")

    xy_full = get_umap(adata, args.umap_key)

    # -----------------------------
    # NaN-safe UMAP subset
    # -----------------------------
    finite = np.isfinite(xy_full).all(axis=1)
    n_finite = int(finite.sum())
    n_total = int(adata.n_obs)

    (args.outdir / "umap_coord_key.txt").write_text(f"{args.umap_key}\n")
    (args.outdir / "umap_finite_summary.txt").write_text(
        f"umap_key={args.umap_key}\nfinite={n_finite}\ntotal={n_total}\nfinite_frac={n_finite/n_total:.6f}\n"
    )

    if n_finite == 0:
        raise RuntimeError(
            f"UMAP key '{args.umap_key}' exists but has 0 finite rows. "
            f"This h5ad does not contain usable UMAP coordinates for that key."
        )

    if n_finite < n_total:
        print(f"[warn] UMAP has NaNs: using only finite rows {n_finite}/{n_total} for gating.")
        (args.outdir / "umap_nonfinite_cell_ids.txt").write_text(
            "\n".join(map(str, adata.obs_names[~finite].tolist())) + "\n"
        )

    # Work only on finite subset
    ad_f = adata[finite].copy()
    obs_f = ad_f.obs
    xy = np.asarray(ad_f.obsm[args.umap_key] if args.umap_key in ad_f.obsm else ad_f.obsm["X_umap"])

    ct = obs_f[args.celltype_key].astype(str).to_numpy()
    gt = obs_f[args.genotype_key].astype(str).to_numpy()

    is_esc = (ct == args.esc_label)
    is_de = (ct == args.de_label)

    if is_esc.sum() == 0 or is_de.sum() == 0:
        raise ValueError(
            f"Need both ESC and DE present in {args.celltype_key} within the finite-UMAP subset. "
            f"Found ESC={int(is_esc.sum())}, DE={int(is_de.sum())}."
        )

    # -----------------------------
    # “Upper ESC” selection by UMAP2
    # -----------------------------
    y = xy[:, 1]
    umap2_cut = None
    if args.esc_umap2_quantile is not None:
        q = float(args.esc_umap2_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("--esc-umap2-quantile must be in (0,1), e.g. 0.80")
        umap2_cut = float(np.quantile(y[is_esc], q))
    elif args.min_umap2 is not None:
        umap2_cut = float(args.min_umap2)

    if umap2_cut is not None:
        is_esc_upper = is_esc & (y >= umap2_cut)
    else:
        is_esc_upper = is_esc

    # Stats for reproducible “no-eyeball”
    esc_y = y[is_esc]
    de_y = y[is_de]
    stats = {
        "ESC_n": int(is_esc.sum()),
        "DE_n": int(is_de.sum()),
        "ESC_umap2_min": float(np.min(esc_y)),
        "ESC_umap2_p10": float(np.quantile(esc_y, 0.10)),
        "ESC_umap2_p25": float(np.quantile(esc_y, 0.25)),
        "ESC_umap2_p50": float(np.quantile(esc_y, 0.50)),
        "ESC_umap2_p75": float(np.quantile(esc_y, 0.75)),
        "ESC_umap2_p90": float(np.quantile(esc_y, 0.90)),
        "ESC_umap2_p95": float(np.quantile(esc_y, 0.95)),
        "ESC_umap2_max": float(np.max(esc_y)),
        "DE_umap2_min": float(np.min(de_y)),
        "DE_umap2_p50": float(np.quantile(de_y, 0.50)),
        "DE_umap2_p90": float(np.quantile(de_y, 0.90)),
        "DE_umap2_max": float(np.max(de_y)),
        "used_umap2_cut": (None if umap2_cut is None else float(umap2_cut)),
        "ESC_upper_n": int(is_esc_upper.sum()),
    }
    pd.Series(stats).to_csv(args.outdir / "umap2_stats.tsv", sep="\t", header=False)

    # Also save ranked ESC list (helps you pick cutoffs)
    esc_rank = pd.DataFrame(
        {
            "cell_id": ad_f.obs_names[is_esc],
            "umap1": xy[is_esc, 0],
            "umap2": xy[is_esc, 1],
            "genotype": gt[is_esc],
        }
    ).sort_values("umap2", ascending=False)
    esc_rank.to_csv(args.outdir / "ESC_sorted_by_umap2.tsv", sep="\t", index=False)

    if is_esc_upper.sum() == 0:
        raise RuntimeError(
            "After applying the 'upper' constraint, ESC_upper_n = 0. "
            "Lower the cutoff (e.g. use --esc-umap2-quantile 0.7) or remove it."
        )

    # -----------------------------
    # HARD UMAP BOX GATE
    # UMAP1 in [-5, -0.5], UMAP2 in [0, 3]
    # -----------------------------
    umap1 = xy[:, 0]
    umap2 = xy[:, 1]

    in_umap_box = (
        (umap1 >= -5.0) & (umap1 <= -0.5) &
        (umap2 >= 0.0) & (umap2 <= 3.0)
    )
    in_umap_box_ESC = in_umap_box & is_esc

    # Save IDs: all cells in box (finite subset)
    box_all_ids = ad_f.obs_names[in_umap_box].tolist()
    (args.outdir / "umap_box_all_cells.txt").write_text(
        "\n".join(map(str, box_all_ids)) + "\n"
    )

    # Save IDs: ESC-only in box (finite subset)
    box_esc_ids = ad_f.obs_names[in_umap_box_ESC].tolist()
    (args.outdir / "umap_box_ESC_cells.txt").write_text(
        "\n".join(map(str, box_esc_ids)) + "\n"
    )

    # Save h5ad subsets from FULL adata for convenience
    if len(box_all_ids) > 0:
        adata[adata.obs_names.isin(box_all_ids)].copy().write_h5ad(
            args.outdir / "umap_box_all_cells_subset.h5ad", compression="gzip"
        )
    if len(box_esc_ids) > 0:
        adata[adata.obs_names.isin(box_esc_ids)].copy().write_h5ad(
            args.outdir / "umap_box_ESC_subset.h5ad", compression="gzip"
        )

    # -----------------------------
    # kNN in UMAP space
    # -----------------------------
    ad_knn = sc.AnnData(X=xy)
    sc.pp.neighbors(ad_knn, n_neighbors=args.k, use_rep="X", random_state=args.seed, key_added="umap_knn")

    A = ad_knn.obsp["umap_knn_connectivities"].tocsr()
    de_mask = is_de.astype(np.float32)

    denom = np.asarray(A.sum(axis=1)).reshape(-1)
    numer = np.asarray(A.dot(de_mask)).reshape(-1)
    de_frac = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    # distance-to-DE-centroid term
    de_centroid = xy[is_de].mean(axis=0)
    dist = np.linalg.norm(xy - de_centroid, axis=1)
    sigma = np.median(np.linalg.norm(xy[is_de] - de_centroid, axis=1)) + 1e-9

    score = de_frac * np.exp(-dist / sigma)

    # optional soft “upper” bias
    if args.y_weight and args.y_weight > 0:
        y_esc = y[is_esc]
        y_lo, y_hi = np.quantile(y_esc, 0.05), np.quantile(y_esc, 0.95)
        y_norm = (y - y_lo) / (y_hi - y_lo + 1e-9)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        score = score * (y_norm ** float(args.y_weight))

    # -----------------------------
    # Gate tail ESC (restricted to upper ESC if set)
    # -----------------------------
    tail_f = np.zeros(ad_f.n_obs, dtype=bool)

    base_gate = is_esc_upper & (de_frac >= args.min_de_frac)
    tail_f |= base_gate

    if args.top_frac and args.top_frac > 0:
        esc_upper_score = score[is_esc_upper]
        n_top = max(1, int(args.top_frac * is_esc_upper.sum()))
        top_idx = np.argsort(-esc_upper_score)[:n_top]
        esc_upper_indices = np.where(is_esc_upper)[0]
        tail_f[esc_upper_indices[top_idx]] = True

    # -----------------------------
    # Map tail back to FULL adata (IDs)
    # -----------------------------
    tail_ids = ad_f.obs_names[tail_f].tolist()
    (args.outdir / "tail_esc_cell_ids.txt").write_text("\n".join(map(str, tail_ids)) + "\n")

    # Write subsets
    adata[adata.obs_names.isin(tail_ids)].copy().write_h5ad(
        args.outdir / "tail_esc_subset.h5ad", compression="gzip"
    )

    if args.write_finite_subset_h5ad:
        ad_f.write_h5ad(args.outdir / "finite_umap_subset_used_for_gating.h5ad", compression="gzip")

    # Genotype enrichment (finite ESC only)
    obs_esc = obs_f.loc[is_esc].copy()
    obs_esc["genotype"] = obs_esc[args.genotype_key].astype(str)
    tail_mask_esc = tail_f[is_esc]
    enr = genotype_enrichment(obs_esc, tail_mask_esc)
    enr.to_csv(args.outdir / "genotype_enrichment_tailESC_vs_allESC.csv")

    # Debug table (finite subset)
    dbg = pd.DataFrame(
        {
            "cell_id": ad_f.obs_names,
            "celltype": ct,
            "genotype": gt,
            "umap1": xy[:, 0],
            "umap2": xy[:, 1],
            "de_neighbor_frac": de_frac,
            "dist_to_de_centroid": dist,
            "score": score,
            "is_esc": is_esc,
            "is_de": is_de,
            "is_esc_upper": is_esc_upper,
            "selected_tail": tail_f,
            "in_umap_box": in_umap_box,
            "in_umap_box_ESC": in_umap_box_ESC,
        }
    )
    dbg.to_csv(args.outdir / "finite_scoring_table_all_cells.csv", index=False)
    dbg.loc[is_esc].to_csv(args.outdir / "esc_scoring_table.csv", index=False)

    print(f"[OK] Finite-UMAP cells used: {ad_f.n_obs}/{adata.n_obs}")
    print(f"[OK] ESC_upper_n={int(is_esc_upper.sum())} (umap2_cut={stats['used_umap2_cut']})")
    print(f"[OK] Selected tail ESC cells: {int(tail_f.sum())}")
    print(f"[OK] UMAP box cells (all finite): {int(in_umap_box.sum())}")
    print(f"[OK] UMAP box ESC cells: {int(in_umap_box_ESC.sum())}")
    print(f"[OK] Wrote: {args.outdir.resolve()}")
    print("[Top genotypes in tail ESC]:")
    print(enr.head(15)[["count_in_tail", "frac_in_tail", "enrichment_ratio"]].to_string())

    # -----------------------------
    # Plot
    # -----------------------------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Consistent colors
        COL_ALL = "#d3d3d3"   # light grey
        COL_ESC = "#1f77b4"   # blue
        COL_DE  = "#ff7f0e"   # orange
        COL_BOX = "#2ca02c"   # green

        #===================================================
        # 1) Gating UMAP (args.umap_key) on finite subset
        #===================================================
        fig = plt.figure(figsize=(9, 7))
        ax = plt.gca()

        all_mask_f = np.ones(ad_f.n_obs, dtype=bool)

        # All finite cells (background)
        ax.scatter(
            xy[all_mask_f, 0],
            xy[all_mask_f, 1],
            s=2,
            alpha=0.05,
            color=COL_ALL,
            label="All (finite)",
        )

        # ESC (finite)
        ax.scatter(
            xy[is_esc, 0],
            xy[is_esc, 1],
            s=4,
            alpha=0.35,
            color=COL_ESC,
            label="ESC",
        )

        # DE (finite)
        ax.scatter(
            xy[is_de, 0],
            xy[is_de, 1],
            s=6,
            alpha=0.55,
            color=COL_DE,
            label="DE",
        )

        # ESC in hard UMAP box (cells of interest)
        if in_umap_box_ESC.any():
            ax.scatter(
                xy[in_umap_box_ESC, 0],
                xy[in_umap_box_ESC, 1],
                s=20,
                alpha=0.95,
                color=COL_BOX,
                label="ESC in UMAP box",
            )

        title = "Auto-gated ESC tail near DE (NaN-safe"
        if umap2_cut is not None:
            title += f"; UMAP2>= {umap2_cut:.3f}"
        if args.y_weight and args.y_weight > 0:
            title += f"; y_weight={args.y_weight:g}"
        title += ")"
        ax.set_title(title)

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(loc="best")

        fig.savefig(args.outdir / "tail_gate_debug.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        #===================================================
        # 2) Other UMAPs on FULL adata: 4-legend view
        #===================================================
        umap_keys = [k for k in adata.obsm.keys() if "umap" in k.lower()]
        if umap_keys:
            (args.outdir / "available_umap_keys.txt").write_text(
                "\n".join(umap_keys) + "\n"
            )

            ct_full = adata.obs[args.celltype_key].astype(str).to_numpy()
            is_esc_full = (ct_full == args.esc_label)
            is_de_full = (ct_full == args.de_label)
            box_esc_mask_full = adata.obs_names.isin(box_esc_ids)

            for key in umap_keys:
                xy2_full = np.asarray(adata.obsm[key])
                if xy2_full.shape[1] < 2:
                    print(f"[warn] UMAP '{key}' does not have 2 dims; skipping.")
                    continue

                finite2 = np.isfinite(xy2_full).all(axis=1)
                if not finite2.any():
                    print(f"[warn] UMAP '{key}' has no finite rows; skipping.")
                    continue

                all_mask_2 = finite2
                esc_mask_2 = is_esc_full & finite2
                de_mask_2 = is_de_full & finite2
                box_mask_2 = box_esc_mask_full & finite2

                fig2 = plt.figure(figsize=(9, 7))
                ax2 = plt.gca()

                # All finite cells (background)
                ax2.scatter(
                    xy2_full[all_mask_2, 0],
                    xy2_full[all_mask_2, 1],
                    s=2,
                    alpha=0.05,
                    color=COL_ALL,
                    label="All (finite)",
                )

                # ESC
                if esc_mask_2.any():
                    ax2.scatter(
                        xy2_full[esc_mask_2, 0],
                        xy2_full[esc_mask_2, 1],
                        s=4,
                        alpha=0.35,
                        color=COL_ESC,
                        label="ESC",
                    )

                # DE
                if de_mask_2.any():
                    ax2.scatter(
                        xy2_full[de_mask_2, 0],
                        xy2_full[de_mask_2, 1],
                        s=6,
                        alpha=0.55,
                        color=COL_DE,
                        label="DE",
                    )

                # ESC in UMAP box (green)
                if box_mask_2.any():
                    ax2.scatter(
                        xy2_full[box_mask_2, 0],
                        xy2_full[box_mask_2, 1],
                        s=20,
                        alpha=0.95,
                        color=COL_BOX,
                        label="ESC in UMAP box",
                    )

                ax2.set_title(f"UMAP '{key}': All / ESC / DE / ESC-box")
                ax2.set_xlabel("UMAP1")
                ax2.set_ylabel("UMAP2")
                ax2.legend(loc="best")

                fname = f"umap_box_ESC_on_{key}.png".replace(" ", "_")
                fig2.savefig(args.outdir / fname, dpi=300, bbox_inches="tight")
                plt.close(fig2)
        else:
            print("[warn] No 'umap'-like keys found in adata.obsm; only tail_gate_debug.png created.")


if __name__ == "__main__":
    main()