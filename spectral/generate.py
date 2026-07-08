"""
spectral.generate
=================

All GPU/attack-dependent generation stages. Each writes tidy CSVs (and, for the
master stage, a compact ``.npz`` of aggregated radial profiles so the plotting
notebook can render radial-profile figures without ever re-running an attack).

Stages
------
generate_master        clean/adv/perturbation descriptors -> per-channel + RGB-median
generate_noise         test B matched noise controls (sign, uniform)
generate_ai_controls   test C AI sweep (perturbation, adv, sign/gauss/shuffle)
generate_saturation    test D saturation sweep (6 field types x 2 profile modes)

Requires torch. Import is deferred by the CLI so pure-analysis stages (A/B/C/D)
run without a GPU.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import attacks as A
from .core import SpectralAnalyzer
from .data import seed_everything


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def aggregate_rgb_median(df: pd.DataFrame, extra_group=("alpha_pix", "steps")) -> pd.DataFrame:
    """Median-collapse the 3 RGB channels into one image-level row."""
    group_cols = ["image_id", "eps_pix", "type", "attack"]
    group_cols += [c for c in extra_group if c in df.columns]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c not in ("image_id", "channel", "eps_pix")
                and c not in group_cols]
    df_agg = df.groupby(group_cols, as_index=False)[num_cols].median()
    ordered = group_cols + [c for c in df_agg.columns if c not in group_cols]
    df_agg = df_agg[ordered]
    try:
        df_agg["image_id"] = df_agg["image_id"].astype(int)
    except (ValueError, TypeError):
        pass
    return df_agg


def _fit_row(analyzer, radial_k, radial_occ):
    """Run the three fits and return a flat dict of parameters + gof."""
    mu_fd, T_fd, A_fd, C_fd, R2_fd, AIC_fd, BIC_fd = analyzer.fit_fd(radial_k, radial_occ)
    T_mb, A_mb, C_mb, R2_mb, AIC_mb, BIC_mb = analyzer.fit_mb(radial_k, radial_occ)
    T_kww, beta, A_kww, C_kww, R2_kww, AIC_kww, BIC_kww = analyzer.fit_kww(radial_k, radial_occ)
    return {
        "mu_fd": mu_fd, "T_fd": T_fd, "A_fd": A_fd, "C_fd": C_fd,
        "R2_fd": R2_fd, "AIC_fd": AIC_fd, "BIC_fd": BIC_fd,
        "T_mb": T_mb, "A_mb": A_mb, "C_mb": C_mb,
        "R2_mb": R2_mb, "AIC_mb": AIC_mb, "BIC_mb": BIC_mb,
        "T_kww": T_kww, "beta": beta, "A_kww": A_kww, "C_kww": C_kww,
        "R2_kww": R2_kww, "AIC_kww": AIC_kww, "BIC_kww": BIC_kww,
    }


class RadialProfileAccumulator:
    """Collects radial occupation curves keyed by (attack, type, eps) so the
    per-condition *median profile* (with a p25-p75 band) can be dumped once and
    replotted cheaply. Only curves matching the first-seen grid length are kept.
    """

    def __init__(self):
        self.k_ref = None
        self.store = defaultdict(list)

    def add(self, attack, kind, eps, radial_k, radial_occ):
        if len(radial_k) == 0:
            return
        if self.k_ref is None:
            self.k_ref = np.asarray(radial_k, dtype=np.float32)
        if len(radial_k) == len(self.k_ref):
            self.store[(attack, kind, float(eps))].append(
                np.asarray(radial_occ, dtype=np.float32)
            )

    def save(self, path):
        keys, meds, p25, p75, ns = [], [], [], [], []
        for key, curves in self.store.items():
            M = np.vstack(curves)
            keys.append("|".join(str(k) for k in key))
            meds.append(np.median(M, axis=0))
            p25.append(np.quantile(M, 0.25, axis=0))
            p75.append(np.quantile(M, 0.75, axis=0))
            ns.append(M.shape[0])
        # all curves share the k_ref grid, so these stack into clean 2D float
        # arrays (n_keys x n_bins). Do NOT use dtype=object: for equal-length
        # rows numpy would build a 2D object array whose rows load back as
        # object dtype and break downstream isfinite/plotting.
        np.savez_compressed(
            path,
            k_ref=self.k_ref if self.k_ref is not None else np.array([], dtype=np.float32),
            keys=np.array(keys),
            median=np.asarray(meds, dtype=np.float32),
            p25=np.asarray(p25, dtype=np.float32),
            p75=np.asarray(p75, dtype=np.float32),
            n=np.asarray(ns),
        )


# ---------------------------------------------------------------------------
# master descriptors  (clean / adv / perturbation)
# ---------------------------------------------------------------------------
def generate_master(cfg, model, criterion, mean, std, loader, device):
    out = Path(cfg.master_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = SpectralAnalyzer(224, 224, num_bins=cfg.num_bins)

    for attack_name in cfg.attacks:
        attack_fn = A.ATTACK_FNS[attack_name]
        acc = RadialProfileAccumulator()
        rows = []

        for eps in cfg.eps_values:
            eps_pix = eps / 255.0
            alpha_pix = eps_pix / 4.0
            seed_everything(cfg.seed)  # deterministic per (attack, eps)

            for idx, (x_raw, y) in tqdm(enumerate(loader), total=cfg.n_images,
                                        desc=f"{attack_name} master eps={eps}"):
                x_raw, y = x_raw.to(device), y.to(device)
                x_adv, delta = attack_fn(model, criterion, x_raw, y,
                                         eps_pix, alpha_pix, cfg.steps, mean, std)
                x_np = x_raw[0].cpu().numpy()
                a_np = x_adv[0].cpu().numpy()
                d_np = delta[0].cpu().numpy()

                for c in range(3):
                    for kind, field, eps_occ in [("clean", x_np[c], 1.0),
                                                 ("adv", a_np[c], 1.0),
                                                 ("perturbation", d_np[c], eps_pix)]:
                        rk, ro, AI = analyzer.radial_profile_E(field, epsilon=eps_occ)
                        acc.add(attack_name, kind, eps_pix, rk, ro)
                        row = {
                            "image_id": idx, "channel": c, "type": kind, "attack": attack_name,
                            "eps": int(eps), "eps_pix": float(eps_pix), "alpha_pix": float(alpha_pix),
                            "AI": AI, "num_bins": int(len(rk)),
                            "occ_mean": float(np.mean(ro)) if len(ro) else np.nan,
                            "occ_std": float(np.std(ro)) if len(ro) else np.nan,
                        }
                        row.update(_fit_row(analyzer, rk, ro))
                        rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out / f"{attack_name}_master_channel_{cfg.n_images}.csv", index=False)
        df_agg = aggregate_rgb_median(df)
        df_agg.to_csv(out / f"{attack_name}_master_aggRGB_{cfg.n_images}.csv", index=False)
        acc.save(out / f"{attack_name}_radial_profiles_{cfg.n_images}.npz")
        print(f"[master] {attack_name}: {len(df)} channel rows -> {len(df_agg)} image rows")


# ---------------------------------------------------------------------------
# test B: matched noise controls
# ---------------------------------------------------------------------------
def generate_noise(cfg, loader, device):
    out = Path(cfg.noise_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = SpectralAnalyzer(224, 224, num_bins=cfg.num_bins)
    eps_list = [e / 255.0 for e in cfg.eps_values]
    noise_makers = {"sign": A.make_linf_sign_noise, "uniform": A.make_linf_uniform_noise}

    for noise_name, maker in noise_makers.items():
        seed_everything(cfg.seed)
        rows = []
        for idx, (x_raw, y) in tqdm(enumerate(loader), total=cfg.n_images,
                                    desc=f"noise {noise_name}"):
            x_raw = x_raw.to(device)
            for eps_pix in eps_list:
                x_noisy, delta = maker(x_raw, eps_pix)
                a_np = x_noisy[0].cpu().numpy()
                d_np = delta[0].cpu().numpy()
                for c in range(3):
                    for kind, field, eps_occ in [("adv", a_np[c], 1.0),
                                                 ("perturbation", d_np[c], eps_pix)]:
                        rk, ro, AI = analyzer.radial_profile_E(field, epsilon=eps_occ)
                        row = {
                            "image_id": idx, "channel": c, "type": kind, "attack": noise_name,
                            "eps_pix": float(eps_pix), "AI": AI, "num_bins": int(len(rk)),
                            "occ_mean": float(np.mean(ro)) if len(ro) else np.nan,
                            "occ_std": float(np.std(ro)) if len(ro) else np.nan,
                        }
                        row.update(_fit_row(analyzer, rk, ro))
                        rows.append(row)
        df = pd.DataFrame(rows)
        df_agg = aggregate_rgb_median(df)
        df_agg.to_csv(out / f"B_{noise_name}_{cfg.n_images}.csv", index=False)
        print(f"[noise] {noise_name}: {len(df)} -> {len(df_agg)}")


# ---------------------------------------------------------------------------
# test C: AI sweep with matched controls
# ---------------------------------------------------------------------------
def generate_ai_controls(cfg, model, criterion, mean, std, loader, device):
    out = Path(cfg.ai_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = SpectralAnalyzer(224, 224, num_bins=cfg.num_bins)
    eps_list = [e / 255.0 for e in cfg.eps_values]

    for attack_name in cfg.attacks:
        attack_fn = A.ATTACK_FNS[attack_name]
        seed_everything(cfg.seed)
        rows = []
        for idx, (x_raw, y) in tqdm(enumerate(loader), total=cfg.n_images,
                                    desc=f"{attack_name} AI-controls"):
            x_raw, y = x_raw.to(device), y.to(device)
            for eps_pix in eps_list:
                alpha_pix = eps_pix / 4.0
                x_adv, delta = attack_fn(model, criterion, x_raw, y,
                                         eps_pix, alpha_pix, cfg.steps, mean, std)
                eps_key = int(round(eps_pix * 1e6))
                base = cfg.seed + 100000 * idx + 10 * eps_key + (0 if attack_name == "BIM" else 1)
                d_sign = A.noise_sign_tensor(delta, eps_pix, seed=base + 11)
                d_gauss = A.noise_gauss_clip_tensor(delta, eps_pix, sigma=eps_pix / 2.0, seed=base + 22)
                d_shuf = A.noise_shuffle_delta(delta, seed=base + 33)

                d_np = delta[0].cpu().numpy()
                a_np = x_adv[0].cpu().numpy()
                s_np = d_sign[0].cpu().numpy()
                g_np = d_gauss[0].cpu().numpy()
                h_np = d_shuf[0].cpu().numpy()

                for c in range(3):
                    for kind, field in [("perturbation", d_np[c]), ("adv", a_np[c]),
                                        ("noise_sign", s_np[c]), ("noise_gauss", g_np[c]),
                                        ("noise_shuffle", h_np[c])]:
                        rk, ro, AI = analyzer.radial_profile_E(field, epsilon=eps_pix)
                        rows.append({
                            "image_id": idx, "channel": c, "attack": attack_name, "type": kind,
                            "eps_pix": float(eps_pix), "alpha_pix": float(alpha_pix), "steps": int(cfg.steps),
                            "AI": float(AI), "num_bins": int(len(rk)),
                            "occ_mean": float(np.mean(ro)) if len(ro) else np.nan,
                            "occ_std": float(np.std(ro)) if len(ro) else np.nan,
                        })
        df = pd.DataFrame(rows)
        gcols = ["image_id", "attack", "type", "eps_pix", "alpha_pix", "steps"]
        vcols = ["AI", "num_bins", "occ_mean", "occ_std"]
        df_rgb = df.groupby(gcols)[vcols].median().reset_index()
        df_rgb["channel"] = "RGB_median"
        df_rgb.to_csv(out / f"C_AI_controls_{attack_name}.csv", index=False)
        print(f"[ai] {attack_name}: {len(df)} -> {len(df_rgb)}")


# ---------------------------------------------------------------------------
# test D: saturation / clipping sanity sweep
# ---------------------------------------------------------------------------
def generate_saturation(cfg, model, criterion, mean, std, loader, device):
    out = Path(cfg.sat_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = SpectralAnalyzer(224, 224, num_bins=cfg.num_bins)
    eps_list = [e / 255.0 for e in cfg.eps_values]

    frames = []
    for attack_name in cfg.attacks:
        attack_fn = A.ATTACK_FNS[attack_name]
        rng = np.random.default_rng(cfg.seed)
        rows = []
        for steps in cfg.sat_steps_list:
            for eps_pix in eps_list:
                alpha_pix = eps_pix * cfg.sat_alpha_ratio
                for idx, (x_raw, y) in tqdm(enumerate(loader), total=cfg.n_images,
                                            desc=f"{attack_name} sat eps={eps_pix:.4f} s={steps}"):
                    x_raw, y = x_raw.to(device), y.to(device)
                    x_adv, delta = attack_fn(model, criterion, x_raw, y,
                                             eps_pix, alpha_pix, steps, mean, std)
                    x_np = x_raw[0].cpu().numpy()
                    a_np = x_adv[0].cpu().numpy()
                    d_np = delta[0].cpu().numpy()

                    for c in range(3):
                        ns, nh, ng = A.make_noise_controls(d_np[c], eps_pix, rng)
                        fields = {
                            "clean": (x_np[c], 1.0), "adv": (a_np[c], 1.0),
                            "perturbation": (d_np[c], eps_pix),
                            "noise_sign": (ns, eps_pix), "noise_shuffle": (nh, eps_pix),
                            "noise_gauss": (ng, eps_pix),
                        }
                        for kind, (field, eps_occ) in fields.items():
                            sat = analyzer.occupation_stats(field, epsilon=eps_occ)
                            for pmode, clip, capq in [("clip", True, None),
                                                      ("raw_cap", False, cfg.sat_raw_cap_quantile)]:
                                rk, ro, AI = analyzer.radial_profile_E(
                                    field, epsilon=eps_occ, use_clipping=clip, cap_quantile=capq)
                                row = {
                                    "image_id": idx, "channel": c, "attack": attack_name, "type": kind,
                                    "eps_pix": float(eps_pix), "alpha_pix": float(alpha_pix),
                                    "steps": int(steps), "profile_mode": pmode,
                                    "AI": float(AI), "num_bins": int(len(rk)),
                                    "occ_mean": float(np.mean(ro)) if len(ro) else np.nan,
                                    "occ_std": float(np.std(ro)) if len(ro) else np.nan,
                                }
                                row.update(sat)
                                row.update(_fit_row(analyzer, rk, ro))
                                rows.append(row)
        df = pd.DataFrame(rows)
        frames.append(df)
        print(f"[sat] {attack_name}: {len(df)} rows")
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(out / f"ALL_saturation_channel_{cfg.n_images}.csv", index=False)
    return df_all
