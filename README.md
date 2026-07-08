# Spectral Pipeline

Production refactor of the spectral-mechanics adversarial-detection workflow:
image/perturbation channels are projected via DCT, normalized (Hölder L∞→L1),
mapped to energy space `E = k²`, radially aggregated, and fitted with three
occupation laws (Fermi–Dirac, Maxwell–Boltzmann, KWW/stretched-exponential).
Data **generation** and the four statistical **tests (A–D)** are decoupled from
plotting and driven by a single config.

## Layout

```
spectral_pipeline/
├── config.yaml            # all paths + hyper-parameters (edit this)
├── run_pipeline.py        # CLI orchestrator (stage selection)
├── selftest.py            # offline end-to-end check (no GPU / no ImageNet)
├── requirements.txt
├── spectral/
│   ├── core.py            # model laws + unified SpectralAnalyzer (torch-free)
│   ├── stats.py           # shared bootstrap / Cliff's-delta helpers
│   ├── attacks.py         # BIM, PGD, matched noise controls (torch)
│   ├── data.py            # ImageNet loader + ResNet-50 + seeding (torch)
│   ├── generate.py        # generation stages -> CSV + radial-profile .npz
│   ├── analysis_a.py      # Test A  Spectral descriptor evolution vs budget
│   ├── analysis_b.py      # Test B  Model validation / overfitting control
│   ├── analysis_c.py      # Test C  Anisotropy index validation + controls
│   ├── analysis_d.py      # Test D  Saturation / clipping sanity check
│   └── config.py          # typed Config + YAML loader
└── plots/                 # (plotting notebook lives here)
```

## Stages

| stage      | needs GPU | reads                          | writes                                   |
|------------|:---------:|--------------------------------|------------------------------------------|
| `generate` |    yes    | ImageNet val                   | `master/{ATTACK}_master_*` + radial .npz |
| `noise`    |    yes    | ImageNet val                   | `B/control_noises/B_{sign,uniform}_*`    |
| `gen-c`    |    yes    | ImageNet val                   | `C/ai_controls/C_AI_controls_{ATTACK}`   |
| `gen-d`    |    yes    | ImageNet val                   | `D/saturation/ALL_saturation_channel_*`  |
| `A`        |    no     | master aggRGB                  | `A/results/…`                            |
| `B`        |    no     | master aggRGB + noise          | `B/results/…`                            |
| `C`        |    no     | AI-controls                    | `C/results/AI_v2_*`                      |
| `D`        |    no     | saturation channel-level       | `D/results/…`                            |

## Quick start

```bash
pip install -r requirements.txt
python selftest.py                       # verify analyses on synthetic data
# edit config.yaml -> imagenet_val_dir, root
python run_pipeline.py --config config.yaml --stage all
```

Generation stages are deterministic (`seed_everything` is called before each,
so a stage reproduces its CSV when re-run in isolation). See `GUIDE.md`
for a full step-by-step walkthrough.
