"""
spectral.analysis_d  --  Test D: Saturation & Clipping Sanity Check
==================================================================

Consumes the per-channel saturation sweep (6 field types x {clip, raw_cap}
profile modes). Collapses channels to image level, then:
  * per-epsilon saturation-metric summary (median + bootstrap CI + IQR),
  * saturation-metric monotonicity vs epsilon (Spearman),
  * profile-mode regime comparison for model-selection metrics (does replacing
    the clipped profile by the unclipped high-quantile-capped one change model
    selection?).

Pure pandas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stats import bootstrap_ci_median, q25_q75_iqr

PAIRS = [("kww", "mb"), ("fd", "mb"), ("fd", "kww")]

CHANNEL_AGG_METRICS = [
    "AI", "occ_mean", "occ_std",
    "clip_frac", "clip_mean_excess", "clip_p95_excess",
    "occ_raw_mean", "occ_raw_p95", "occ_raw_p99", "occ_mean_2d", "occ_p95_2d",
    "R2_fd", "AIC_fd", "BIC_fd", "R2_mb", "AIC_mb", "BIC_mb", "R2_kww", "AIC_kww", "BIC_kww",
    "R2_diff_kww_mb", "dAIC_kww_mb", "dBIC_kww_mb",
    "R2_diff_fd_mb", "dAIC_fd_mb", "dBIC_fd_mb",
    "R2_diff_fd_kww", "dAIC_fd_kww", "dBIC_fd_kww",
]
IMAGE_GROUP_COLS = ["image_id", "attack", "type", "eps_pix", "alpha_pix", "steps", "profile_mode"]
SATURATION_METRICS = ["clip_frac", "clip_mean_excess", "clip_p95_excess",
                      "occ_raw_mean", "occ_raw_p95", "occ_raw_p99", "occ_mean", "occ_mean_2d"]
REGIME_METRICS = ["R2_diff_kww_mb", "dAIC_kww_mb", "dBIC_kww_mb",
                  "R2_diff_fd_mb", "dAIC_fd_mb", "dBIC_fd_mb",
                  "R2_diff_fd_kww", "dAIC_fd_kww", "dBIC_fd_kww", "AI", "occ_mean"]
CONTROL_TYPES = ["noise_sign", "noise_shuffle", "noise_gauss"]


def add_pairwise_model_columns(df):
    df = df.copy()
    for m1, m2 in PAIRS:
        df[f"R2_diff_{m1}_{m2}"] = df[f"R2_{m1}"] - df[f"R2_{m2}"]
        df[f"dAIC_{m1}_{m2}"] = df[f"AIC_{m1}"] - df[f"AIC_{m2}"]
        df[f"dBIC_{m1}_{m2}"] = df[f"BIC_{m1}"] - df[f"BIC_{m2}"]
    return df


def collapse_channels_per_image(df, agg="median"):
    df = df.copy()
    cols = [c for c in CHANNEL_AGG_METRICS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    collapsed = df.groupby(IMAGE_GROUP_COLS, as_index=False)[cols].agg(agg)
    collapsed["n_channels_collapsed"] = 3
    return collapsed


def make_subsets(df_img):
    return {
        "perturbation_only": df_img[df_img["type"].isin(["perturbation"] + CONTROL_TYPES)].copy(),
        "adv_and_perturbation": df_img[df_img["type"].isin(["perturbation"] + CONTROL_TYPES)].copy(),
    }


def saturation_summary_by_eps(df, cfg, metrics=SATURATION_METRICS):
    rows = []
    df = df.copy()
    for col in ["eps_pix", "steps"] + list(metrics):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for keys, sub in df.groupby(["attack", "type", "profile_mode", "steps", "eps_pix"]):
        attack, typ, pmode, steps, eps = keys
        for metric in metrics:
            vals = sub[metric].to_numpy()
            med, lo, hi = bootstrap_ci_median(vals, B=cfg.boot_b, ci=cfg.ci,
                                              rng=np.random.default_rng(cfg.seed))
            q25, q75, iqr = q25_q75_iqr(vals)
            rows.append({"attack": attack, "type": typ, "profile_mode": pmode, "steps": int(steps),
                         "eps_pix": float(eps), "metric": metric,
                         "n_images": int(np.isfinite(vals).sum()), "median": med,
                         "q25": q25, "q75": q75, "iqr": iqr, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def saturation_monotonicity(df, metrics=SATURATION_METRICS):
    rows = []
    df = df.copy()
    for col in ["eps_pix", "steps"] + list(metrics):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for keys, sub in df.groupby(["attack", "type", "profile_mode", "steps"]):
        attack, typ, pmode, steps = keys
        for metric in metrics:
            s = sub[np.isfinite(sub["eps_pix"]) & np.isfinite(sub[metric])]
            ev = s["eps_pix"].to_numpy()
            mv = s[metric].to_numpy()
            n = len(s)
            # Spearman is undefined when either input is constant (e.g. the
            # strict saturation indicators stay identically 0 across epsilon).
            if n < 2 or np.ptp(ev) == 0 or np.ptp(mv) == 0:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(ev, mv)
            rows.append({"attack": attack, "type": typ, "profile_mode": pmode, "steps": int(steps),
                         "metric": metric, "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                         "spearman_p": float(p) if np.isfinite(p) else np.nan, "n_images": int(n)})
    return pd.DataFrame(rows)


def compare_profile_modes(df, metric, cfg):
    rows = []
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp["eps_pix"] = pd.to_numeric(tmp["eps_pix"], errors="coerce")
    for keys, sub in tmp.groupby(["attack", "type", "steps", "profile_mode", "eps_pix"]):
        attack, typ, steps, pmode, eps = keys
        vals = sub[metric].to_numpy()
        med, lo, hi = bootstrap_ci_median(vals, B=cfg.boot_b, ci=cfg.ci,
                                          rng=np.random.default_rng(cfg.seed))
        q25, q75, iqr = q25_q75_iqr(vals)
        rows.append({"attack": attack, "type": typ, "steps": int(steps), "profile_mode": pmode,
                     "eps_pix": float(eps), "metric": metric,
                     "n_images": int(np.isfinite(vals).sum()), "median": med,
                     "q25": q25, "q75": q75, "iqr": iqr, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def run(cfg, sat_channel_csv: str):
    out = Path(cfg.results_d)
    out.mkdir(parents=True, exist_ok=True)
    df = add_pairwise_model_columns(pd.read_csv(sat_channel_csv))
    df_img = collapse_channels_per_image(df, agg="median")
    df_img.to_csv(out / "ALL_occupation_sat_sweep_image_level.csv", index=False)

    written = ["ALL_occupation_sat_sweep_image_level.csv"]
    for name, sub in make_subsets(df_img).items():
        sub.to_csv(out / f"{name}_agg_RGB.csv", index=False)
        saturation_summary_by_eps(sub, cfg).to_csv(out / f"{name}_summary_by_eps.csv", index=False)
        saturation_monotonicity(sub).to_csv(out / f"{name}_monotonicity.csv", index=False)
        for metric in REGIME_METRICS:
            (compare_profile_modes(sub, metric, cfg)
             .to_csv(out / f"{name}_profile_mode_{metric}.csv", index=False))
        written.append(name)
    print(f"[D] wrote summaries for {len(make_subsets(df_img))} subsets to {out}")
    return df_img
