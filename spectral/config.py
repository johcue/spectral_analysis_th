"""
spectral.config
===============

Single typed configuration object for the whole pipeline. Load from YAML with
``Config.load('config.yaml')``; every path and hyper-parameter lives here so no
stage hard-codes anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class Config:
    # -- data / model ---------------------------------------------------------
    imagenet_val_dir: str = "/path/to/imagenet/val"
    n_images: int = 500
    num_workers: int = 2
    device: str = "auto"                       # auto | cuda | cpu
    seed: int = 42

    # -- attack sweep ---------------------------------------------------------
    attacks: List[str] = field(default_factory=lambda: ["BIM", "PGD"])
    eps_values: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 16])   # in /255
    steps: int = 10
    num_bins: int = 61

    # -- saturation (test D) sweep -------------------------------------------
    sat_steps_list: List[int] = field(default_factory=lambda: [10])
    sat_alpha_ratio: float = 0.25
    sat_raw_cap_quantile: float = 0.999

    # -- statistics -----------------------------------------------------------
    boot_b: int = 5000
    ci: float = 0.95
    test_a_type_filters: List[str] = field(default_factory=lambda: ["adv", "perturbation"])
    test_a_min_r2: float = 0.35
    test_a_min_pairs: int = 5      # min images sharing an adjacent-eps pair for a transition score
    test_b_type_filters: List[str] = field(default_factory=lambda: ["adv", "perturbation"])
    r2_tie: float = 0.01
    aic_tie: float = 2.0
    bic_tie: float = 2.0

    # -- output layout --------------------------------------------------------
    root: str = "outputs"

    # derived directories -----------------------------------------------------
    @property
    def master_dir(self) -> str: return str(Path(self.root) / "master")
    @property
    def noise_dir(self) -> str: return str(Path(self.root) / "B" / "control_noises")
    @property
    def ai_dir(self) -> str: return str(Path(self.root) / "C" / "ai_controls")
    @property
    def sat_dir(self) -> str: return str(Path(self.root) / "D" / "saturation")
    @property
    def results_a(self) -> str: return str(Path(self.root) / "A" / "results")
    @property
    def results_b(self) -> str: return str(Path(self.root) / "B" / "results")
    @property
    def results_c(self) -> str: return str(Path(self.root) / "C" / "results")
    @property
    def results_d(self) -> str: return str(Path(self.root) / "D" / "results")

    # -- io -------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | None) -> "Config":
        if path is None:
            return cls()
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def dump(self) -> dict:
        return asdict(self)
