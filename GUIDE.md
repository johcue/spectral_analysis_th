# Testing Guide — Spectral Pipeline

A step-by-step protocol for reproducing and verifying the spectral
adversarial-detection pipeline (descriptor generation + statistical tests A–D).
It is written so that someone who has **never seen the code** can validate it,
first *without* a GPU or ImageNet, then on the real data.

There are two levels of testing:

- **Level 1 — Base self-test** (5 min, any laptop). Proves every statistical
  test runs correctly on synthetic data. No GPU, no ImageNet, no torch.
- **Level 2 — Real run** (GPU machine). Runs the actual attacks on ImageNet and
  the full analysis on the resulting descriptors.

Do Level 1 first. If it passes, the analysis logic is sound and only the data
generation depends on your hardware.

---

## 0. What you received

```
spectral_pipeline/
├── config.yaml          <- the only file you normally edit
├── run_pipeline.py      <- command-line entry point
├── selftest.py          <- Level-1 Base check
├── requirements.txt
├── README.md
├── TUTOR_GUIDE.md       <- this file
└── spectral/            <- the package (core, stats, attacks, data, analyses)
```

The pipeline is split into **generation** stages (need a GPU + ImageNet) and
**analysis** stages A/B/C/D (pure pandas, run anywhere). This split is what lets
you test the science Base.

---

## 1. Set up the environment

Requires **Python 3.10 or newer**. From inside `spectral_pipeline/`:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

For **Level 1** (Base test) you only need the scientific stack:

```bash
pip install numpy scipy pandas scikit-learn matplotlib pyyaml tqdm
```

For **Level 2** you also need PyTorch built for your CUDA driver. Do **not**
blindly `pip install torch`; pick the right build from
https://pytorch.org/get-started/locally/ , e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(`pip install -r requirements.txt` installs everything at once, but the torch
line there is a generic `torch>=2.0`; prefer the official selector on a GPU box.)

Verify:

```bash
python -c "import numpy, scipy, pandas, sklearn, matplotlib, yaml, tqdm; print('core OK')"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"   # Level 2 only
```

---

## 2. Level 1 — Base self-test (do this first)

This synthesizes tiny, schema-correct CSVs (12 fake images, the five ε budgets)
and runs analyses A, B, C and D against them, then checks the outputs.

```bash
python selftest.py
```

Expected tail of the output:

```
Running analysis A ...
[A] wrote 20 files to outputs_selftest/A/results
Running analysis B ...
[B] wrote 4 files to outputs_selftest/B/results
Running analysis C ...
[C] wrote 12 files to outputs_selftest/C/results
Running analysis D ...
[D] wrote summaries for 2 subsets to outputs_selftest/D/results

Verifying outputs ...
  OK   .../A_BIM_adv_summary_image_level.csv
  ...
[A] beta monotonicity rho_median (expect < 0): -1.000
[C] Cliff's delta range vs controls: [-1.00, 0.65]

SELF-TEST PASSED
```

What this proves:

- All four tests execute without error and write their result CSVs.
- The **A** check recovers an injected decreasing β-vs-ε trend
  (`rho_median ≈ -1`), i.e. the monotonicity machinery works.
- The **C** check shows the perturbation group separates from the random
  controls (Cliff's delta reaches ≈ ±1), i.e. the effect-size machinery works.

Add `--keep` to retain the synthetic CSVs under `outputs_selftest/` and open
them to inspect the exact columns each test produces:

```bash
python selftest.py --keep
ls -R outputs_selftest
```

If this step fails, stop here and send us the traceback — it is independent of
your data and hardware.

---

## 3. Level 2 — point the config at ImageNet

Open `config.yaml` and set the two starred fields:

```yaml
imagenet_val_dir: "/data/imagenet/val"   # ImageFolder layout: val/<class>/<img>.JPEG
n_images: 500                            # keep 500 for the stats; raise later if desired
root: "outputs"                          # everything is written under here
```

`imagenet_val_dir` must be a standard `torchvision.datasets.ImageFolder` root
(one sub-directory per class). Everything else in `config.yaml` already matches
the thesis settings (ε = {2,4,8,12,16}/255, 10 steps, ResNet-50, seed 42).

---

## 4. Level 2 — a fast smoke run (recommended before the full run)

Before committing to the full sweep, confirm the GPU path works on a handful of
images by overriding `n_images` from the command line:

```bash
python run_pipeline.py --config config.yaml --stage generate --n-images 4
```

This runs BIM+PGD on 4 images across all five ε and writes
`outputs/master/{BIM,PGD}_master_aggRGB_4.csv` plus the per-channel CSV and a
radial-profile `.npz`. It should finish in well under a minute on a GPU. Inspect:

```bash
python -c "import pandas as pd; d=pd.read_csv('outputs/master/BIM_master_aggRGB_4.csv'); print(d.shape); print(d['type'].value_counts()); print(d.columns.tolist()[:12])"
```

You should see `clean/adv/perturbation` rows and the fit columns
(`R2_kww`, `beta`, `dAIC…` after analysis, etc.).

---

## 5. Level 2 — the full run

Run the stages in dependency order. You can do them one at a time or all at once.

**One at a time (clearest for a first run):**

```bash
# generation (GPU) — writes the descriptor CSVs the tests consume
python run_pipeline.py --config config.yaml --stage generate   # master: clean/adv/perturbation
python run_pipeline.py --config config.yaml --stage noise      # test-B sign/uniform controls
python run_pipeline.py --config config.yaml --stage gen-c      # test-C AI controls sweep
python run_pipeline.py --config config.yaml --stage gen-d      # test-D saturation sweep

# analysis (CPU) — reads the CSVs above, writes the result tables
python run_pipeline.py --config config.yaml --stage A B C D
```

**Or everything in order:**

```bash
python run_pipeline.py --config config.yaml --stage all
```

Order matters: `A` needs `generate`; `B` needs `generate` + `noise`; `C` needs
`gen-c`; `D` needs `gen-d`. The `all` target already sequences them correctly.

---

## 6. Where the results land

Under `outputs/` (i.e. `root`):

```
outputs/
├── master/       BIM/PGD descriptors (channel-level, RGB-median, radial .npz)
├── B/control_noises/   sign & uniform noise descriptors
├── C/ai_controls/      AI-controls sweep
├── D/saturation/       saturation channel-level sweep
├── A/results/    parameter evolution: summary / monotonicity / deltas / transition
├── B/results/    model comparison: per-eps summary / trends / consistency / contrast
├── C/results/    AI_v2_* : summary / monotonicity / perturbation-vs-controls / paired-shuffle
└── D/results/    saturation summaries / monotonicity / profile-mode comparison
```

Quick sanity checks against the thesis:

- **Test C** — open `C/results/AI_v2_perturbation_vs_controls_BIM.csv`. The
  `cliffs_delta` column vs `noise_shuffle` should be strongly negative
  (thesis reports Cliff's delta ≈ −1.00), confirming perturbations are far more
  oriented than shuffled controls.
- **Test B** — open `B/results/per_eps_model_comparison_summary.csv`. Median
  `win_AIC_kww_mb` should be high (KWW preferred over MB), most strongly in the
  perturbation domain.
- **Test A** — open `A/results/A_BIM_adv_monotonicity_image_level.csv`. β should
  show a negative `rho_median` (decreasing with budget).

---

## 7. Reproducibility

- The seed is fixed (`seed: 42`) and `seed_everything()` is called at the start
  of every generation stage, so re-running a stage reproduces its CSVs exactly.
- The bootstrap seed *arithmetic* inside each test is preserved from the
  original notebooks, so the summary statistics reproduce the reported numbers
  (bit-for-bit for the analysis stages given identical input CSVs).
- To re-run a single stage cleanly, delete its output folder and re-invoke that
  stage; downstream stages will pick up the fresh CSVs.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: spectral` | Run commands from inside `spectral_pipeline/` (or `pip install -e .`). |
| `selftest.py` fails | Analysis bug independent of data — send us the traceback. |
| `FileNotFoundError: imagenet` | `imagenet_val_dir` wrong, or not an ImageFolder (needs class sub-dirs). |
| `torch.cuda.is_available()` is `False` | CPU-only torch or driver mismatch — reinstall the correct CUDA build. |
| A/B/C/D error “file not found” | The matching generation stage hasn't run yet (see the order in §5). |
| Very slow generation | Expected: cost ≈ `n_images × 5 ε × attacks × fits`. Use `--n-images` to scale down while testing. |
| `ConstantInputWarning` (suppressed) | The strict saturation indicators stay 0 (no clipping) — this is the *expected* result, not an error. |
| Test-A `*_transition_scores` plot is empty | Transition scores need `test_a_min_pairs` (default 5) images sharing each adjacent-ε pair. On a tiny test (e.g. `--n-images 4`) that can't be met, so every score is `NaN` and the plot is skipped. Use the real `n_images`, or set `test_a_min_pairs: 2` in `config.yaml` for smoke tests. Not a bug. |

---

## 9. Generating the figures

Plots are produced by `plots/figures.ipynb`, which reads **only** the result
CSVs / `.npz` written above — it never re-runs an attack or a fit, which is the
point of the decoupling. After the pipeline has run:

```bash
pip install jupyter            # if not already available
jupyter notebook plots/figures.ipynb
```

At the top of the notebook set three variables to match your run, then
Restart-and-Run-All:

```python
RESULTS_DIR = REPO / "outputs"     # your config `root`
N_IMAGES    = 500                  # your config n_images
ATTACKS     = ["BIM", "PGD"]
```

It renders, inline and to `plots/figures/` as **both `.pdf` and `.svg`** (vector,
LaTeX-styled): the descriptor distributions (β, R², ΔAIC), the radial occupation
profiles with fit overlays, and the summary figures for tests A–D. Set
`USE_TEX = True` only if a LaTeX toolchain is installed; the default mathtext
rendering needs nothing extra. Each figure function skips gracefully with a
`[skip] missing:` message if its input CSV isn't present, so you can run the
notebook after a partial pipeline run.

## 10. What is intentionally out of scope

The cross-attack ML section (BIM→PGD Random-Forest detection) lives in the
leak-fixed notebooks and consumes `outputs/master/…_master_channel_*.csv`; it is
not part of tests A–D.
