#!/usr/bin/env python3
"""
run_pipeline.py  --  command-line orchestrator for the spectral pipeline.

Stages
------
  generate    master descriptors (BIM/PGD, clean/adv/perturbation) [needs GPU]
  noise       test B matched noise controls (sign, uniform)         [needs GPU]
  gen-c       test C AI-controls sweep                              [needs GPU]
  gen-d       test D saturation sweep                               [needs GPU]
  A           test A analysis (parameter evolution)                 [CPU]
  B           test B analysis (model validation)                    [CPU]
  C           test C analysis (AI validation)                       [CPU]
  D           test D analysis (saturation sanity)                   [CPU]
  all         run everything in dependency order

Examples
--------
  python run_pipeline.py --config config.yaml --stage generate
  python run_pipeline.py --config config.yaml --stage A B C D     # analysis only
  python run_pipeline.py --config config.yaml --stage all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from spectral.config import Config

GEN_STAGES = {"generate", "noise", "gen-c", "gen-d"}
CPU_STAGES = {"A", "B", "C", "D"}
ALL_ORDER = ["generate", "noise", "gen-c", "gen-d", "A", "B", "C", "D"]


def _master_map(cfg):
    d = Path(cfg.master_dir)
    return {a: str(d / f"{a}_master_aggRGB_{cfg.n_images}.csv") for a in cfg.attacks}


def _ai_map(cfg):
    d = Path(cfg.ai_dir)
    return {a: str(d / f"C_AI_controls_{a}.csv") for a in cfg.attacks}


def _b_csv_map(cfg):
    m = _master_map(cfg)
    nd = Path(cfg.noise_dir)
    m["NOISE_sign"] = str(nd / f"B_sign_{cfg.n_images}.csv")
    m["NOISE_uniform"] = str(nd / f"B_uniform_{cfg.n_images}.csv")
    return m


# ---- GPU stages (lazy torch import) ---------------------------------------
def _run_generation(stage, cfg):
    from spectral import generate as G
    from spectral.data import get_device, build_loader, load_model, seed_everything

    device = get_device(cfg.device)
    print(f"[{stage}] device={device}")
    seed_everything(cfg.seed)
    loader = build_loader(cfg)

    if stage == "generate":
        model, crit, mean, std = load_model(cfg, device)
        G.generate_master(cfg, model, crit, mean, std, loader, device)
    elif stage == "noise":
        G.generate_noise(cfg, loader, device)
    elif stage == "gen-c":
        model, crit, mean, std = load_model(cfg, device)
        G.generate_ai_controls(cfg, model, crit, mean, std, loader, device)
    elif stage == "gen-d":
        model, crit, mean, std = load_model(cfg, device)
        G.generate_saturation(cfg, model, crit, mean, std, loader, device)


# ---- CPU analysis stages ---------------------------------------------------
def _run_analysis(stage, cfg):
    if stage == "A":
        from spectral import analysis_a
        analysis_a.run(cfg, _master_map(cfg))
    elif stage == "B":
        from spectral import analysis_b
        analysis_b.run(cfg, _b_csv_map(cfg),
                       attack_names=cfg.attacks,
                       control_names=["NOISE_sign", "NOISE_uniform"])
    elif stage == "C":
        from spectral import analysis_c
        analysis_c.run(cfg, _ai_map(cfg))
    elif stage == "D":
        from spectral import analysis_d
        sat_csv = Path(cfg.sat_dir) / f"ALL_saturation_channel_{cfg.n_images}.csv"
        analysis_d.run(cfg, str(sat_csv))


def run_stage(stage, cfg):
    if stage in GEN_STAGES:
        _run_generation(stage, cfg)
    elif stage in CPU_STAGES:
        _run_analysis(stage, cfg)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml (defaults if omitted)")
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=ALL_ORDER + ["all"], help="stage(s) to run")
    ap.add_argument("--n-images", type=int, default=None, help="override cfg.n_images (quick smoke test)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.n_images is not None:
        cfg.n_images = args.n_images

    stages = ALL_ORDER if args.stage == ["all"] or "all" in args.stage else args.stage
    print(f"Running stages: {stages}")
    for stage in stages:
        print(f"\n{'='*60}\n== stage: {stage}\n{'='*60}")
        run_stage(stage, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
