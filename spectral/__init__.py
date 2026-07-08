"""Spectral-mechanics adversarial-detection pipeline.

Importing ``spectral`` pulls in only torch-free surfaces (core, stats, config,
analysis_*). The generation stages (``spectral.generate``, ``spectral.attacks``,
``spectral.data``) import torch and are loaded lazily by the CLI so the analysis
tests can run on a machine without a GPU.
"""
from .config import Config
from .core import SpectralAnalyzer

__all__ = ["Config", "SpectralAnalyzer"]
__version__ = "1.0.0"
