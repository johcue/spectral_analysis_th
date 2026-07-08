"""
spectral.analysis_b  --  Test B: Model Validation & Overfitting Control
======================================================================

Compares FD / MB / KWW via R2, AIC, BIC. For each attack + type + epsilon:
  * per-epsilon median R2 (+ bad-fit rate R2<0.2) per model,
  * pairwise (KWW-MB, FD-MB, FD-KWW) median deltas, win rates and AIC/BIC
    evidence-strength buckets,
  * a consistency / overfit classification per observation,
  * epsilon trend (Spearman) of every summary metric,
  * an attack-vs-control contrast against the sign/uniform noise baselines.

Pure pandas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stats import bootstrap_ci_median, bootstrap_ci_mean, q25_q75_iqr

PAIRS = [("kww", "mb"), ("fd", "mb"), ("fd", "kww")]
CORE_NUMERIC = ["eps_pix", "R2_fd", "R2_mb", "R2_kww",
                "AIC_fd", "AIC_mb", "AIC_kww", "BIC_fd", "BIC_mb", "BIC_kww"]


def load_csv_map(csv_map: dict, type_filters) -> pd.DataFrame:
    frames = []
    for attack_name, path in csv_map.items():
        df = pd.read_csv(path).copy()
        df["attack"] = attack_name
        if "type" in df.columns:
            df = df[df["type"].isin(type_filters)].copy()
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    for col in CORE_NUMERIC:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    return df_all


def add_pairwise_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m1, m2 in PAIRS:
        df[f"R2_diff_{m1}_{m2}"] = df[f"R2_{m1}"] - df[f"R2_{m2}"]
        df[f"dAIC_{m1}_{m2}"] = df[f"AIC_{m1}"] - df[f"AIC_{m2}"]
        df[f"dBIC_{m1}_{m2}"] = df[f"BIC_{m1}"] - df[f"BIC_{m2}"]
    return df


def summarize_pairwise_by_eps(df, attack_name, type_filters, cfg):
    df = df.copy()
    for col in CORE_NUMERIC:
        df = df[np.isfinite(df[col])].copy()
    rows = []
    for t_idx, type_name in enumerate(type_filters):
        dft = df[df["type"] == type_name]
        if len(dft) == 0:
            continue
        for e_idx, eps in enumerate(np.sort(dft["eps_pix"].dropna().unique())):
            sub = dft[dft["eps_pix"] == eps]
            row = {"attack": attack_name, "type": type_name, "eps_pix": float(eps), "n": int(len(sub))}
            for m_idx, model in enumerate(["fd", "mb", "kww"]):
                r2 = sub[f"R2_{model}"].to_numpy(float)
                med, lo, hi = bootstrap_ci_median(
                    r2, B=cfg.boot_b, ci=cfg.ci,
                    rng=np.random.default_rng(cfg.seed + 100000 * t_idx + 1000 * e_idx + 10 * m_idx + 1))
                bad, blo, bhi = bootstrap_ci_mean(
                    (r2 < 0.2).astype(float), B=cfg.boot_b, ci=cfg.ci,
                    rng=np.random.default_rng(cfg.seed + 100000 * t_idx + 1000 * e_idx + 10 * m_idx + 2))
                row.update({f"R2_{model}_median": med, f"R2_{model}_ci_low": lo, f"R2_{model}_ci_high": hi,
                            f"bad_{model}_lt_0p2": bad, f"bad_{model}_lt_0p2_ci_low": blo,
                            f"bad_{model}_lt_0p2_ci_high": bhi})
            for p_idx, (m1, m2) in enumerate(PAIRS):
                pfx = f"{m1}_{m2}"
                dR = sub[f"R2_diff_{pfx}"].to_numpy(float)
                dA = sub[f"dAIC_{pfx}"].to_numpy(float)
                dB = sub[f"dBIC_{pfx}"].to_numpy(float)
                for tag, arr, off in [("R2_diff", dR, 1), ("dAIC", dA, 2), ("dBIC", dB, 3)]:
                    med, lo, hi = bootstrap_ci_median(
                        arr, B=cfg.boot_b, ci=cfg.ci,
                        rng=np.random.default_rng(cfg.seed + 200000 * t_idx + 1000 * e_idx + 20 * p_idx + off))
                    q25, q75, iqr = q25_q75_iqr(arr)
                    row.update({f"{tag}_{pfx}_median": med, f"{tag}_{pfx}_ci_low": lo,
                                f"{tag}_{pfx}_ci_high": hi, f"{tag}_{pfx}_q25": q25,
                                f"{tag}_{pfx}_q75": q75, f"{tag}_{pfx}_iqr": iqr})
                for tag, cond, off in [("win_R2", dR > 0, 4), ("win_AIC", dA < 0, 5), ("win_BIC", dB < 0, 6)]:
                    w, wlo, whi = bootstrap_ci_mean(
                        cond.astype(float), B=cfg.boot_b, ci=cfg.ci,
                        rng=np.random.default_rng(cfg.seed + 200000 * t_idx + 1000 * e_idx + 20 * p_idx + off))
                    row.update({f"{tag}_{pfx}": w, f"{tag}_{pfx}_ci_low": wlo, f"{tag}_{pfx}_ci_high": whi})
                row[f"AIC_strong_{m1}_{pfx}"] = float(np.mean(dA < -10))
                row[f"AIC_moderate_{m1}_{pfx}"] = float(np.mean((dA < -4) & (dA >= -10)))
                row[f"AIC_tie_{pfx}"] = float(np.mean(np.abs(dA) < 2))
                row[f"AIC_strong_{m2}_{pfx}"] = float(np.mean(dA > 10))
                row[f"BIC_strong_{m1}_{pfx}"] = float(np.mean(dB < -10))
                row[f"BIC_moderate_{m1}_{pfx}"] = float(np.mean((dB < -4) & (dB >= -10)))
                row[f"BIC_tie_{pfx}"] = float(np.mean(np.abs(dB) < 2))
                row[f"BIC_strong_{m2}_{pfx}"] = float(np.mean(dB > 10))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["type", "eps_pix"]).reset_index(drop=True)


def build_trend_table(summary_df):
    protected = {"attack", "type", "eps_pix", "n"}
    metrics = [c for c in summary_df.columns if c not in protected]
    rows = []
    for (attack, type_name), g in summary_df.groupby(["attack", "type"]):
        g = g.sort_values("eps_pix")
        eps = g["eps_pix"].to_numpy(float)
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            mask = np.isfinite(eps) & np.isfinite(vals)
            if mask.sum() < 3 or len(np.unique(vals[mask])) < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(eps[mask], vals[mask])
            rows.append({"attack": attack, "type": type_name, "metric": metric,
                         "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                         "spearman_p": float(p) if np.isfinite(p) else np.nan})
    return pd.DataFrame(rows)


def _classify(dr, da, db, r2_tie, aic_tie, bic_tie):
    if abs(dr) < r2_tie and abs(da) < aic_tie and abs(db) < bic_tie:
        return "tie_or_ambiguous"
    if dr > 0 and da < 0 and db < 0:
        return "consistent_gain"
    if dr > 0 and (da >= 0 or db >= 0):
        return "possible_overfit"
    return "clear_loss"


def build_consistency_table(df, type_filters, cfg):
    rows = []
    for attack, dfa in df.groupby("attack"):
        for t_idx, type_name in enumerate(type_filters):
            dft = dfa[dfa["type"] == type_name]
            if len(dft) == 0:
                continue
            for e_idx, eps in enumerate(np.sort(dft["eps_pix"].dropna().unique())):
                sub = dft[dft["eps_pix"] == eps]
                base = {"attack": attack, "type": type_name, "eps_pix": float(eps), "n": int(len(sub))}
                for p_idx, (m1, m2) in enumerate(PAIRS):
                    pfx = f"{m1}_{m2}"
                    dR = sub[f"R2_diff_{pfx}"].to_numpy(float)
                    dA = sub[f"dAIC_{pfx}"].to_numpy(float)
                    dB = sub[f"dBIC_{pfx}"].to_numpy(float)
                    m = np.isfinite(dR) & np.isfinite(dA) & np.isfinite(dB)
                    labels = np.array([_classify(a, b, c, cfg.r2_tie, cfg.aic_tie, cfg.bic_tie)
                                       for a, b, c in zip(dR[m], dA[m], dB[m])], dtype=object)
                    row = dict(base, pair=pfx, n_valid=int(len(labels)))
                    for cat_idx, cat in enumerate(["consistent_gain", "possible_overfit",
                                                   "tie_or_ambiguous", "clear_loss"]):
                        ind = (labels == cat).astype(float)
                        mv, lo, hi = bootstrap_ci_mean(
                            ind, B=cfg.boot_b, ci=cfg.ci,
                            rng=np.random.default_rng(cfg.seed + 500000 * t_idx + 1000 * e_idx + 20 * p_idx + cat_idx + 1))
                        row.update({f"{cat}_rate": mv, f"{cat}_ci_low": lo, f"{cat}_ci_high": hi})
                    rows.append(row)
    return pd.DataFrame(rows)


def build_attack_vs_control(summary_df, attack_names, control_names):
    metrics = ["R2_fd_median", "R2_mb_median", "R2_kww_median",
               "dAIC_kww_mb_median", "dBIC_kww_mb_median", "win_AIC_kww_mb", "win_BIC_kww_mb",
               "dAIC_fd_mb_median", "dBIC_fd_mb_median", "win_AIC_fd_mb", "win_BIC_fd_mb",
               "dAIC_fd_kww_median", "dBIC_fd_kww_median", "win_AIC_fd_kww", "win_BIC_fd_kww"]
    rows = []
    for attack in attack_names:
        for control in control_names:
            merged = (summary_df[summary_df["attack"] == attack]
                      .merge(summary_df[summary_df["attack"] == control],
                             on=["type", "eps_pix"], suffixes=("_attack", "_control")))
            for _, r in merged.iterrows():
                row = {"attack": attack, "control": control, "type": r["type"], "eps_pix": float(r["eps_pix"])}
                for metric in metrics:
                    va, vc = r.get(f"{metric}_attack", np.nan), r.get(f"{metric}_control", np.nan)
                    row[f"{metric}_attack"] = va
                    row[f"{metric}_control"] = vc
                    row[f"{metric}_attack_minus_control"] = (float(va - vc)
                                                             if pd.notna(va) and pd.notna(vc) else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def run(cfg, csv_map: dict, attack_names, control_names):
    """csv_map: {label: path}. Must include the attack labels and control labels."""
    out = Path(cfg.results_b)
    out.mkdir(parents=True, exist_ok=True)
    df_all = add_pairwise_model_columns(load_csv_map(csv_map, cfg.test_b_type_filters))

    summaries = [summarize_pairwise_by_eps(df_all[df_all["attack"] == a], a, cfg.test_b_type_filters, cfg)
                 for a in df_all["attack"].unique()]
    df_summary_all = pd.concat(summaries, ignore_index=True)
    df_trends = build_trend_table(df_summary_all)
    df_consistency = build_consistency_table(df_all, cfg.test_b_type_filters, cfg)
    df_contrast = build_attack_vs_control(df_summary_all, attack_names, control_names)

    paths = {
        "per_eps_model_comparison_summary.csv": df_summary_all,
        "all_model_comparisons_trends.csv": df_trends,
        "consistency_overfit_table.csv": df_consistency,
        "attack_vs_control_summary_contrast.csv": df_contrast,
    }
    for name, frame in paths.items():
        frame.to_csv(out / name, index=False)
    print(f"[B] wrote {len(paths)} files to {out}")
    return df_summary_all
