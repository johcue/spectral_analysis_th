"""
spectral.analysis_a  --  Test A: Spectral Descriptor Evolution with Budget
=========================================================================

For each fitted parameter and each attack, on RGB-median (image-level) data:
  1. per-epsilon summary (median, IQR, bootstrap CI),
  2. image-wise Spearman monotonicity of the parameter vs epsilon
     (median rho, bootstrap CI, Wilcoxon vs 0, sign fractions),
  3. adjacent-epsilon median shifts + a normalized transition score.

Pure pandas/numpy/scipy: no torch, no GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .stats import finite_array, bootstrap_ci_median, bootstrap_ci_paired_delta

PARAMS = ["mu_fd", "T_fd", "A_fd", "C_fd",
          "T_mb", "A_mb", "C_mb",
          "T_kww", "beta", "A_kww", "C_kww"]

_FD = {"mu_fd", "T_fd", "A_fd", "C_fd"}
_MB = {"T_mb", "A_mb", "C_mb"}
_KWW = {"T_kww", "beta", "A_kww", "C_kww"}


def _r2_col(param: str) -> str:
    if param in _FD:
        return "R2_fd"
    if param in _MB:
        return "R2_mb"
    if param in _KWW:
        return "R2_kww"
    raise ValueError(f"Unknown parameter family: {param}")


def _build_param_df(df_raw, param, type_filter, min_r2):
    r2_col = _r2_col(param)
    d = df_raw[["image_id", "attack", "type", "eps_pix", param, r2_col]].copy()
    if type_filter is not None and "type" in d.columns:
        d = d[d["type"] == type_filter]
    for col in (param, r2_col, "eps_pix"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d[param]) & np.isfinite(d[r2_col])
          & np.isfinite(d["eps_pix"]) & (d[r2_col] >= min_r2)].copy()
    d = d.rename(columns={param: "value", r2_col: "median_r2"})
    return d[["image_id", "attack", "type", "eps_pix", "value", "median_r2"]]


def _summary(df_raw, attack, type_filter, cfg):
    rows = []
    for p_idx, param in enumerate(PARAMS):
        dfp = _build_param_df(df_raw, param, type_filter, cfg.test_a_min_r2)
        for e_idx, eps in enumerate(np.sort(dfp["eps_pix"].dropna().unique())):
            vals = finite_array(dfp.loc[dfp["eps_pix"] == eps, "value"].to_numpy())
            if len(vals) == 0:
                continue
            q25, q75 = np.quantile(vals, [0.25, 0.75])
            med, lo, hi = bootstrap_ci_median(
                vals, B=cfg.boot_b, ci=cfg.ci,
                rng=np.random.default_rng(cfg.seed + 1000 * p_idx + e_idx))
            rows.append({"attack": attack, "type": type_filter, "param": param,
                         "eps_pix": float(eps), "n_images": int(len(vals)),
                         "median": med, "q25": float(q25), "q75": float(q75),
                         "iqr": float(q75 - q25), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _monotonicity(df_raw, attack, type_filter, cfg):
    rho_tables, mono_rows = [], []
    for p_idx, param in enumerate(PARAMS):
        dfp = _build_param_df(df_raw, param, type_filter, cfg.test_a_min_r2)
        rho_rows = []
        for image_id, g in dfp.groupby("image_id"):
            g = g.sort_values("eps_pix")
            eps = finite_array(g["eps_pix"].to_numpy())
            vals = finite_array(g["value"].to_numpy())
            if len(np.unique(eps)) < 3 or len(vals) < 2 or np.allclose(vals, vals[0]):
                continue
            rho, pval = spearmanr(eps, vals)
            if np.isfinite(rho):
                rho_rows.append({"attack": attack, "type": type_filter, "param": param,
                                 "image_id": image_id, "rho": float(rho),
                                 "pval": float(pval) if np.isfinite(pval) else np.nan,
                                 "n_eps": int(len(np.unique(eps)))})
        df_rho = pd.DataFrame(rho_rows)
        rho_tables.append(df_rho)
        if len(df_rho) == 0:
            mono_rows.append({"attack": attack, "type": type_filter, "param": param,
                              "n_images": 0})
            continue
        rv = df_rho["rho"].to_numpy(dtype=float)
        q25, q75 = np.quantile(rv, [0.25, 0.75])
        med, lo, hi = bootstrap_ci_median(
            rv, B=cfg.boot_b, ci=cfg.ci,
            rng=np.random.default_rng(cfg.seed + 20000 + p_idx))
        if len(rv) >= 5 and not np.allclose(rv, 0):
            try:
                _, wp = wilcoxon(rv, alternative="two-sided", zero_method="wilcox")
                wp = float(wp)
            except ValueError:
                wp = np.nan
        else:
            wp = np.nan
        mono_rows.append({"attack": attack, "type": type_filter, "param": param,
                          "n_images": int(len(rv)), "rho_median": med,
                          "rho_q25": float(q25), "rho_q75": float(q75),
                          "rho_iqr": float(q75 - q25), "rho_ci_low": lo, "rho_ci_high": hi,
                          "wilcoxon_p_vs_0": wp,
                          "frac_positive": float(np.mean(rv > 0)),
                          "frac_negative": float(np.mean(rv < 0))})
    df_rho_all = pd.concat(rho_tables, ignore_index=True) if rho_tables else pd.DataFrame()
    return df_rho_all, pd.DataFrame(mono_rows)


def _transitions(df_raw, attack, type_filter, cfg):
    delta_rows, trans_rows = [], []
    for p_idx, param in enumerate(PARAMS):
        dfp = _build_param_df(df_raw, param, type_filter, cfg.test_a_min_r2)
        eps_levels = np.sort(dfp["eps_pix"].dropna().unique())
        if len(eps_levels) < 2:
            trans_rows.append({"attack": attack, "type": type_filter, "param": param,
                               "transition_score": np.nan})
            continue
        iqr_list = []
        for eps in eps_levels:
            vals = finite_array(dfp.loc[dfp["eps_pix"] == eps, "value"].to_numpy())
            if len(vals) >= 4:
                q25, q75 = np.quantile(vals, [0.25, 0.75])
                iqr_list.append(float(q75 - q25))
        scale = np.median(iqr_list) if iqr_list else np.nan
        pivot = dfp.pivot_table(index=["image_id", "attack"], columns="eps_pix",
                                values="value", aggfunc="first")
        best_score, best = -np.inf, None
        for e_idx in range(len(eps_levels) - 1):
            e0, e1 = eps_levels[e_idx], eps_levels[e_idx + 1]
            if e0 not in pivot.columns or e1 not in pivot.columns:
                continue
            paired = pivot[[e0, e1]].dropna()
            if len(paired) < cfg.test_a_min_pairs:
                continue
            diffs = paired[e1].to_numpy() - paired[e0].to_numpy()
            d_med, d_lo, d_hi = bootstrap_ci_paired_delta(
                diffs, B=cfg.boot_b, ci=cfg.ci,
                rng=np.random.default_rng(cfg.seed + 40000 + 1000 * p_idx + e_idx))
            deps = float(e1 - e0)
            score = abs(d_med) / (scale * deps) if (np.isfinite(scale) and scale > 0 and deps > 0) else np.nan
            delta_rows.append({"attack": attack, "type": type_filter, "param": param,
                               "eps_lo": float(e0), "eps_hi": float(e1),
                               "n_pairs": int(len(paired)), "delta_median": float(d_med),
                               "delta_ci_low": float(d_lo), "delta_ci_high": float(d_hi),
                               "scale_median_iqr": float(scale) if np.isfinite(scale) else np.nan,
                               "deps": deps,
                               "transition_score_segment": float(score) if np.isfinite(score) else np.nan})
            if np.isfinite(score) and score > best_score:
                best_score, best = score, (float(e0), float(e1), int(len(paired)),
                                           float(d_med), float(d_lo), float(d_hi))
        if best is None:
            trans_rows.append({"attack": attack, "type": type_filter, "param": param,
                               "transition_score": np.nan})
        else:
            e0, e1, npr, dm, dl, dh = best
            trans_rows.append({"attack": attack, "type": type_filter, "param": param,
                               "transition_score": float(best_score),
                               "best_jump_eps_lo": e0, "best_jump_eps_hi": e1,
                               "best_jump_delta_median": dm, "best_jump_delta_ci_low": dl,
                               "best_jump_delta_ci_high": dh, "n_pairs_best_jump": npr})
    return pd.DataFrame(delta_rows), pd.DataFrame(trans_rows)


def run(cfg, master_csv_map: dict):
    """master_csv_map: {attack: path_to_aggRGB_csv}. Writes CSVs to results_a."""
    out = Path(cfg.results_a)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for attack, path in master_csv_map.items():
        df_raw = pd.read_csv(path)
        for tf in cfg.test_a_type_filters:
            summ = _summary(df_raw, attack, tf, cfg)
            rho_all, mono = _monotonicity(df_raw, attack, tf, cfg)
            deltas, trans = _transitions(df_raw, attack, tf, cfg)
            prefix = f"A_{attack}_{tf}"
            for name, frame in [("summary_image_level", summ),
                                ("monotonicity_image_level", mono),
                                ("delta_medians_image_level", deltas),
                                ("transition_scores_image_level", trans),
                                ("imagewise_rho", rho_all)]:
                p = out / f"{prefix}_{name}.csv"
                frame.to_csv(p, index=False)
                written.append(str(p))
    print(f"[A] wrote {len(written)} files to {out}")
    return written
