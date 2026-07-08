#!/usr/bin/env python3
"""
selftest.py  --  offline end-to-end check of the ANALYSIS path (tests A-D).

Generates small, schema-correct synthetic CSVs (no ImageNet, no GPU, no torch)
that mimic the real generation output, then runs analysis stages A, B, C and D
against them and asserts the expected result files were written and are
non-empty.

Usage:
    python selftest.py                 # uses a temp dir under ./outputs_selftest
    python selftest.py --keep          # keep the synthetic data for inspection

This does NOT test the GPU generation stages (attacks / ResNet-50); those need
ImageNet + torch and are covered by the smoke run in the guide. It DOES verify
that every statistical test runs and writes its outputs correctly.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.config import Config
from spectral import analysis_a, analysis_b, analysis_c, analysis_d

FIT_COLS = ["mu_fd", "T_fd", "A_fd", "C_fd", "R2_fd", "AIC_fd", "BIC_fd",
            "T_mb", "A_mb", "C_mb", "R2_mb", "AIC_mb", "BIC_mb",
            "T_kww", "beta", "A_kww", "C_kww", "R2_kww", "AIC_kww", "BIC_kww"]


def _fake_fits(rng, n, kww_better=True):
    """Fabricate plausible fit params; KWW made slightly better than MB/FD."""
    r2_mb = rng.uniform(0.55, 0.9, n)
    r2_kww = np.clip(r2_mb + rng.uniform(0.0, 0.06, n), 0, 1) if kww_better else r2_mb
    r2_fd = np.clip(r2_mb - rng.uniform(0.0, 0.05, n), 0, 1)
    # AIC/BIC lower is better -> KWW lower
    aic_mb = rng.uniform(-40, -20, n)
    aic_kww = aic_mb - rng.uniform(0, 15, n)
    aic_fd = aic_mb + rng.uniform(0, 10, n)
    d = {
        "mu_fd": rng.uniform(0.1, 2, n), "T_fd": rng.uniform(0.1, 1, n),
        "A_fd": rng.uniform(0.3, 1.2, n), "C_fd": rng.uniform(0, 0.3, n),
        "R2_fd": r2_fd, "AIC_fd": aic_fd, "BIC_fd": aic_fd + 3,
        "T_mb": rng.uniform(0.1, 1, n), "A_mb": rng.uniform(0.3, 1.2, n),
        "C_mb": rng.uniform(0, 0.3, n), "R2_mb": r2_mb, "AIC_mb": aic_mb, "BIC_mb": aic_mb + 2,
        "T_kww": rng.uniform(0.1, 1, n), "beta": rng.uniform(0.3, 0.95, n),
        "A_kww": rng.uniform(0.3, 1.2, n), "C_kww": rng.uniform(0, 0.3, n),
        "R2_kww": r2_kww, "AIC_kww": aic_kww, "BIC_kww": aic_kww + 3,
    }
    return d


def make_master(cfg, attack, rng):
    eps_pix = [e / 255 for e in cfg.eps_values]
    rows = []
    for img in range(cfg.n_images):
        for e in eps_pix:
            for typ in ["clean", "adv", "perturbation"]:
                base = {"image_id": img, "eps_pix": e, "type": typ, "attack": attack,
                        "alpha_pix": e / 4, "AI": rng.uniform(0.2, 0.8),
                        "num_bins": 40, "occ_mean": rng.uniform(0.1, 0.5),
                        "occ_std": rng.uniform(0.05, 0.2)}
                f = _fake_fits(rng, 1)
                # inject a monotone trend into beta so test A has signal
                f["beta"] = np.array([0.9 - 3 * e + rng.normal(0, 0.02)])
                base.update({k: (v[0] if hasattr(v, "__len__") else v) for k, v in f.items()})
                rows.append(base)
    return pd.DataFrame(rows)


def make_noise(cfg, name, rng):
    eps_pix = [e / 255 for e in cfg.eps_values]
    rows = []
    for img in range(cfg.n_images):
        for e in eps_pix:
            for typ in ["adv", "perturbation"]:
                base = {"image_id": img, "eps_pix": e, "type": typ, "attack": name,
                        "AI": rng.uniform(0.2, 0.8), "num_bins": 40,
                        "occ_mean": rng.uniform(0.1, 0.5), "occ_std": rng.uniform(0.05, 0.2)}
                f = _fake_fits(rng, 1, kww_better=False)  # controls: no KWW edge
                base.update({k: (v[0] if hasattr(v, "__len__") else v) for k, v in f.items()})
                rows.append(base)
    return pd.DataFrame(rows)


def make_ai_controls(cfg, attack, rng):
    eps_pix = [e / 255 for e in cfg.eps_values]
    groups = ["perturbation", "adv", "noise_sign", "noise_gauss", "noise_shuffle"]
    rows = []
    for img in range(cfg.n_images):
        for e in eps_pix:
            for typ in groups:
                # perturbation AI rises with eps; controls flat/inverted
                if typ == "perturbation":
                    ai = 0.30 + 2.0 * e + rng.normal(0, 0.01)
                elif typ == "noise_shuffle":
                    ai = 0.45 - 0.5 * e + rng.normal(0, 0.01)
                else:
                    ai = 0.42 + rng.normal(0, 0.01)
                rows.append({"image_id": img, "attack": attack, "type": typ,
                             "eps_pix": e, "alpha_pix": e / 4, "steps": cfg.steps,
                             "AI": float(np.clip(ai, 0, 1)), "num_bins": 40,
                             "occ_mean": rng.uniform(0.1, 0.5), "occ_std": rng.uniform(0.05, 0.2),
                             "channel": "RGB_median"})
    return pd.DataFrame(rows)


def make_saturation(cfg, rng):
    eps_pix = [e / 255 for e in cfg.eps_values]
    types = ["clean", "adv", "perturbation", "noise_sign", "noise_shuffle", "noise_gauss"]
    rows = []
    for attack in cfg.attacks:
        for img in range(cfg.n_images):
            for e in eps_pix:
                for c in range(3):
                    for typ in types:
                        for pmode in ["clip", "raw_cap"]:
                            base = {"image_id": img, "channel": c, "attack": attack, "type": typ,
                                    "eps_pix": e, "alpha_pix": e * cfg.sat_alpha_ratio,
                                    "steps": cfg.steps, "profile_mode": pmode,
                                    "AI": rng.uniform(0.2, 0.8), "num_bins": 40,
                                    "occ_mean": rng.uniform(0.1, 0.5), "occ_std": rng.uniform(0.05, 0.2),
                                    "clip_frac": 0.0, "clip_mean_excess": 0.0, "clip_p95_excess": 0.0,
                                    "occ_raw_mean": rng.uniform(0.05, 0.3) + 2 * e,
                                    "occ_raw_p95": rng.uniform(0.3, 0.7) + 3 * e,
                                    "occ_raw_p99": rng.uniform(0.6, 0.95) + 4 * e,
                                    "occ_mean_2d": rng.uniform(0.05, 0.3)}
                            f = _fake_fits(rng, 1)
                            base.update({k: (v[0] if hasattr(v, "__len__") else v) for k, v in f.items()})
                            rows.append(base)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep synthetic data after run")
    args = ap.parse_args()

    root = Path("outputs_selftest")
    if root.exists():
        shutil.rmtree(root)

    # tiny + fast: 12 images, small bootstrap
    cfg = Config(n_images=12, eps_values=[2, 4, 8, 12, 16], attacks=["BIM", "PGD"],
                 boot_b=200, root=str(root))
    rng = np.random.default_rng(0)

    print("Synthesizing data ...")
    Path(cfg.master_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.noise_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.ai_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.sat_dir).mkdir(parents=True, exist_ok=True)

    for attack in cfg.attacks:
        make_master(cfg, attack, rng).to_csv(
            Path(cfg.master_dir) / f"{attack}_master_aggRGB_{cfg.n_images}.csv", index=False)
        make_ai_controls(cfg, attack, rng).to_csv(
            Path(cfg.ai_dir) / f"C_AI_controls_{attack}.csv", index=False)
    for name in ["sign", "uniform"]:
        make_noise(cfg, name, rng).to_csv(
            Path(cfg.noise_dir) / f"B_{name}_{cfg.n_images}.csv", index=False)
    make_saturation(cfg, rng).to_csv(
        Path(cfg.sat_dir) / f"ALL_saturation_channel_{cfg.n_images}.csv", index=False)

    print("\nRunning analysis A ...")
    master_map = {a: str(Path(cfg.master_dir) / f"{a}_master_aggRGB_{cfg.n_images}.csv") for a in cfg.attacks}
    analysis_a.run(cfg, master_map)

    print("\nRunning analysis B ...")
    b_map = dict(master_map)
    b_map["NOISE_sign"] = str(Path(cfg.noise_dir) / f"B_sign_{cfg.n_images}.csv")
    b_map["NOISE_uniform"] = str(Path(cfg.noise_dir) / f"B_uniform_{cfg.n_images}.csv")
    analysis_b.run(cfg, b_map, attack_names=cfg.attacks, control_names=["NOISE_sign", "NOISE_uniform"])

    print("\nRunning analysis C ...")
    ai_map = {a: str(Path(cfg.ai_dir) / f"C_AI_controls_{a}.csv") for a in cfg.attacks}
    analysis_c.run(cfg, ai_map)

    print("\nRunning analysis D ...")
    analysis_d.run(cfg, str(Path(cfg.sat_dir) / f"ALL_saturation_channel_{cfg.n_images}.csv"))

    # assertions
    checks = [
        Path(cfg.results_a) / "A_BIM_adv_summary_image_level.csv",
        Path(cfg.results_a) / "A_BIM_adv_monotonicity_image_level.csv",
        Path(cfg.results_b) / "per_eps_model_comparison_summary.csv",
        Path(cfg.results_b) / "consistency_overfit_table.csv",
        Path(cfg.results_c) / "AI_v2_summary_by_eps_group_BIM.csv",
        Path(cfg.results_c) / "AI_v2_perturbation_vs_controls_BIM.csv",
        Path(cfg.results_d) / "perturbation_only_summary_by_eps.csv",
    ]
    print("\nVerifying outputs ...")
    ok = True
    for p in checks:
        exists = p.exists() and p.stat().st_size > 0
        print(f"  {'OK ' if exists else 'MISS'}  {p}")
        ok = ok and exists

    # spot-check a couple of expected signals
    mono = pd.read_csv(Path(cfg.results_a) / "A_BIM_adv_monotonicity_image_level.csv")
    beta_rho = mono.loc[mono["param"] == "beta", "rho_median"]
    print(f"\n[A] beta monotonicity rho_median (expect < 0): "
          f"{beta_rho.iloc[0]:.3f}" if len(beta_rho) and pd.notna(beta_rho.iloc[0]) else "[A] beta rho: n/a")

    cliffs = pd.read_csv(Path(cfg.results_c) / "AI_v2_perturbation_vs_controls_BIM.csv")
    print(f"[C] Cliff's delta range vs controls: "
          f"[{cliffs['cliffs_delta'].min():.2f}, {cliffs['cliffs_delta'].max():.2f}]")

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    if not args.keep:
        shutil.rmtree(root)
        print("(cleaned up synthetic data; pass --keep to retain)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
