"""
spectral.analysis_c  --  Test C: Anisotropy Index Validation
============================================================

Operates on the AI-controls sweep (perturbation + noise_sign / noise_gauss /
noise_shuffle, RGB-median). Produces, per attack:
  * AI summary per epsilon and group,
  * image-wise Spearman monotonicity of AI vs epsilon,
  * perturbation-vs-control differential (bootstrap median gap, Mann-Whitney,
    Cliff's delta),
  * paired perturbation-vs-shuffle test (isolates spatial structure).

Output filenames match the ``AI_v2_*`` convention consumed by the plotting
notebook. Pure pandas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

from .stats import (finite_array, bootstrap_ci_median, bootstrap_ci_paired_delta,
                    bootstrap_ci_delta_median, cliffs_delta, interpret_cliffs_delta)

PRIMARY_GROUPS = ["perturbation", "noise_sign", "noise_gauss", "noise_shuffle"]
CONTROLS = ["noise_sign", "noise_gauss", "noise_shuffle"]


def _load(path, attack):
    df = pd.read_csv(path).copy()
    if "attack" not in df.columns:
        df["attack"] = attack
    df["eps_pix"] = pd.to_numeric(df["eps_pix"], errors="coerce")
    df["AI"] = pd.to_numeric(df["AI"], errors="coerce")
    df["type"] = df["type"].astype(str)
    return df[df["type"].isin(PRIMARY_GROUPS) & np.isfinite(df["eps_pix"]) & np.isfinite(df["AI"])].copy()


def _summary(df, attack, cfg):
    rows = []
    eps_levels = np.sort(df["eps_pix"].dropna().unique())
    for g_idx, group in enumerate(PRIMARY_GROUPS):
        dfg = df[df["type"] == group]
        for e_idx, eps in enumerate(eps_levels):
            vals = finite_array(dfg.loc[dfg["eps_pix"] == eps, "AI"].to_numpy())
            if len(vals) == 0:
                continue
            q25, q75 = np.quantile(vals, [0.25, 0.75])
            med, lo, hi = bootstrap_ci_median(
                vals, B=cfg.boot_b, ci=cfg.ci,
                rng=np.random.default_rng(cfg.seed + 1000 * g_idx + e_idx))
            rows.append({"attack": attack, "group": group, "eps_pix": float(eps),
                         "n_images": int(len(vals)), "median_AI": med, "q25_AI": float(q25),
                         "q75_AI": float(q75), "iqr_AI": float(q75 - q25),
                         "ci_low_AI": lo, "ci_high_AI": hi})
    return pd.DataFrame(rows)


def _monotonicity(df, attack, cfg):
    rho_tables, summary_rows = [], []
    for g_idx, group in enumerate(PRIMARY_GROUPS):
        dfg = df[df["type"] == group]
        rho_rows = []
        for image_id, g in dfg.groupby("image_id"):
            g = g.sort_values("eps_pix")
            eps = g["eps_pix"].to_numpy(float)
            vals = g["AI"].to_numpy(float)
            m = np.isfinite(eps) & np.isfinite(vals)
            eps, vals = eps[m], vals[m]
            if len(np.unique(eps)) < 3 or len(vals) < 2 or np.allclose(vals, vals[0]):
                continue
            rho, pval = spearmanr(eps, vals)
            if np.isfinite(rho):
                rho_rows.append({"attack": attack, "group": group, "image_id": image_id,
                                 "rho": float(rho), "pval": float(pval) if np.isfinite(pval) else np.nan,
                                 "n_eps": int(len(np.unique(eps)))})
        df_rho = pd.DataFrame(rho_rows)
        rho_tables.append(df_rho)
        if len(df_rho) == 0:
            summary_rows.append({"attack": attack, "group": group, "n_images": 0})
            continue
        rv = df_rho["rho"].to_numpy(float)
        q25, q75 = np.quantile(rv, [0.25, 0.75])
        med, lo, hi = bootstrap_ci_median(
            rv, B=cfg.boot_b, ci=cfg.ci, rng=np.random.default_rng(cfg.seed + 20000 + g_idx))
        if len(rv) >= 5 and not np.allclose(rv, 0):
            try:
                _, wp = wilcoxon(rv, alternative="two-sided", zero_method="wilcox")
                wp = float(wp)
            except ValueError:
                wp = np.nan
        else:
            wp = np.nan
        summary_rows.append({"attack": attack, "group": group, "n_images": int(len(rv)),
                             "rho_median": med, "rho_q25": float(q25), "rho_q75": float(q75),
                             "rho_iqr": float(q75 - q25), "rho_ci_low": lo, "rho_ci_high": hi,
                             "wilcoxon_p_vs_0": wp, "frac_positive": float(np.mean(rv > 0)),
                             "frac_negative": float(np.mean(rv < 0))})
    rho_all = pd.concat(rho_tables, ignore_index=True) if rho_tables else pd.DataFrame()
    return rho_all, pd.DataFrame(summary_rows)


def _differential(df, attack, cfg, focal="perturbation"):
    rows = []
    eps_levels = np.sort(df["eps_pix"].dropna().unique())
    for c_idx, ctrl in enumerate(CONTROLS):
        for e_idx, eps in enumerate(eps_levels):
            xf = finite_array(df[(df["type"] == focal) & (df["eps_pix"] == eps)]["AI"].to_numpy())
            xc = finite_array(df[(df["type"] == ctrl) & (df["eps_pix"] == eps)]["AI"].to_numpy())
            dmed, dlo, dhi = bootstrap_ci_delta_median(
                xc, xf, B=cfg.boot_b, ci=cfg.ci,
                rng=np.random.default_rng(cfg.seed + 40000 + 1000 * c_idx + e_idx))
            try:
                mw = mannwhitneyu(xf, xc, alternative="two-sided").pvalue
            except ValueError:
                mw = np.nan
            cd = cliffs_delta(xf, xc, rng=np.random.default_rng(cfg.seed + 60000 + 1000 * c_idx + e_idx))
            rows.append({"attack": attack, "focal_group": focal, "control_group": ctrl,
                         "eps_pix": float(eps), "n_focal": int(len(xf)), "n_control": int(len(xc)),
                         "median_focal": float(np.median(xf)) if len(xf) else np.nan,
                         "median_control": float(np.median(xc)) if len(xc) else np.nan,
                         "delta_median": float(dmed) if np.isfinite(dmed) else np.nan,
                         "delta_ci_low": float(dlo) if np.isfinite(dlo) else np.nan,
                         "delta_ci_high": float(dhi) if np.isfinite(dhi) else np.nan,
                         "mw_pvalue": float(mw) if np.isfinite(mw) else np.nan,
                         "cliffs_delta": float(cd) if np.isfinite(cd) else np.nan,
                         "cliffs_interpretation": interpret_cliffs_delta(cd)})
    return pd.DataFrame(rows)


def _paired_shuffle(df, attack, cfg):
    rows, pair_frames = [], []
    for e_idx, eps in enumerate(np.sort(df["eps_pix"].dropna().unique())):
        sub = df[df["eps_pix"] == eps]
        pivot = sub.pivot_table(index="image_id", columns="type", values="AI", aggfunc="first")
        for col in ("perturbation", "noise_shuffle"):
            if col not in pivot.columns:
                pivot[col] = np.nan
        paired = pivot[["perturbation", "noise_shuffle"]].dropna()
        if len(paired) == 0:
            rows.append({"attack": attack, "eps_pix": float(eps), "n_pairs": 0})
            continue
        diffs = paired["perturbation"].to_numpy(float) - paired["noise_shuffle"].to_numpy(float)
        dmed, dlo, dhi = bootstrap_ci_paired_delta(
            diffs, B=cfg.boot_b, ci=cfg.ci, rng=np.random.default_rng(cfg.seed + 80000 + e_idx))
        if len(diffs) >= 5 and not np.allclose(diffs, 0):
            try:
                _, wp = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                wp = float(wp)
            except ValueError:
                wp = np.nan
        else:
            wp = np.nan
        rows.append({"attack": attack, "eps_pix": float(eps), "n_pairs": int(len(paired)),
                     "median_perturbation": float(np.median(paired["perturbation"])),
                     "median_shuffle": float(np.median(paired["noise_shuffle"])),
                     "delta_median_paired": float(dmed), "delta_ci_low": float(dlo),
                     "delta_ci_high": float(dhi), "wilcoxon_pvalue": wp,
                     "frac_negative": float(np.mean(diffs < 0)), "frac_positive": float(np.mean(diffs > 0))})
        tmp = paired.reset_index().copy()
        tmp["attack"] = attack
        tmp["eps_pix"] = float(eps)
        tmp["diff_pert_minus_shuffle"] = diffs
        pair_frames.append(tmp)
    df_summary = pd.DataFrame(rows)
    df_pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    return df_summary, df_pairs


def run(cfg, ai_csv_map: dict):
    """ai_csv_map: {attack: path_to_C_AI_controls_csv}."""
    out = Path(cfg.results_c)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for attack, path in ai_csv_map.items():
        df = _load(path, attack)
        summ = _summary(df, attack, cfg)
        rho_all, mono = _monotonicity(df, attack, cfg)
        diff = _differential(df, attack, cfg)
        paired, pairs = _paired_shuffle(df, attack, cfg)
        for name, frame in [(f"AI_v2_summary_by_eps_group_{attack}", summ),
                            (f"AI_v2_monotonicity_imagewise_rho_{attack}", rho_all),
                            (f"AI_v2_monotonicity_summary_{attack}", mono),
                            (f"AI_v2_perturbation_vs_controls_{attack}", diff),
                            (f"AI_v2_paired_shuffle_summary_{attack}", paired),
                            (f"AI_v2_paired_shuffle_pairs_{attack}", pairs)]:
            p = out / f"{name}.csv"
            frame.to_csv(p, index=False)
            written.append(str(p))
    print(f"[C] wrote {len(written)} files to {out}")
    return written
