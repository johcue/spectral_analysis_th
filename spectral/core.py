"""
spectral.core
=============

Torch-free numerical core of the spectral pipeline.

Contains:
  * the three candidate occupation laws in energy space E = k**2
    (Fermi-Dirac, Maxwell-Boltzmann, Kohlrausch-Williams-Watts / stretched exp.),
  * the anisotropy index,
  * a single ``SpectralAnalyzer`` that unifies the "standard" analyzer used for
    descriptor generation and the "sanity" analyzer used for the saturation
    check (test D). The saturation-specific behaviour is exposed through the
    optional ``use_clipping`` / ``cap_quantile`` arguments and the
    ``occupation_maps`` / ``occupation_stats`` methods; with the defaults the
    class reproduces the original generation analyzer exactly.

This module depends only on numpy + scipy, so it can be imported (and unit
tested) without a GPU or torch installed.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# DCT helper
# ---------------------------------------------------------------------------
def dct2(x: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II."""
    return dct(dct(x.T, type=2, norm="ortho").T, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# model curves in E = k**2
# ---------------------------------------------------------------------------
def fermi_dirac_E(E: np.ndarray, mu: float, T: float, A: float, C: float) -> np.ndarray:
    """n(E) = A / (exp((E - mu)/T) + 1) + C."""
    return A / (np.exp((E - mu) / (T + 1e-8)) + 1.0) + C


def maxwell_boltzmann_E(E: np.ndarray, T: float, A: float, C: float) -> np.ndarray:
    """n(E) = A * exp(-E / T) + C."""
    return A * np.exp(-E / (T + 1e-8)) + C


def kww_glass_E(E: np.ndarray, T_kww: float, beta: float, A: float, C: float) -> np.ndarray:
    """Stretched exponential (KWW / glassy): n(E) = A*exp(-(E/T)**beta) + C.

    beta == 1 -> Maxwell-Boltzmann (KWW nests MB); beta < 1 -> stretched decay.
    """
    Epos = np.maximum(E, 0.0)
    return A * np.exp(-(Epos / (T_kww + 1e-8)) ** (beta + 1e-8)) + C


# backwards-compatible alias (the older notebooks called this "spin_glass")
spin_glass_E = kww_glass_E


def compute_anisotropy_index(occupation: np.ndarray, theta: np.ndarray) -> float:
    """Angular order magnitude |S| of the occupation over the DCT lattice.

    NOTE: this returns the raw orientation order parameter |S| (high == oriented).
    The thesis' canonical Anisotropy Index is AI = 1 - |S| (high == isotropic);
    keep the convention consistent between tables, text and plots.
    """
    weights = occupation.flatten()
    angles = theta.flatten()
    if np.all(weights == 0):
        return 0.0
    complex_sum = np.sum(weights * np.exp(1j * 2.0 * angles))
    order_mag = np.abs(complex_sum) / np.sum(weights)
    return float(order_mag)


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------
class SpectralAnalyzer:
    """Transforms an image/perturbation channel into a radial occupation
    profile in energy space and fits the FD / MB / KWW laws.

    Parameters
    ----------
    H, W : int
        Spatial resolution (224 x 224 in the thesis).
    num_bins : int
        Number of radial bin edges (61 -> 60 candidate annuli).
    """

    def __init__(self, H: int, W: int, num_bins: int = 61):
        self.H = H
        self.W = W
        self.num_bins = num_bins
        self.l1_norms = self._precompute_dct_l1norms()
        self.k_grid, self.theta_grid = self._precompute_k_theta()

    # -- precomputation -----------------------------------------------------
    def _dct1d_l1(self, N: int) -> np.ndarray:
        n = np.arange(N)
        k = np.arange(N)[:, None]
        alpha = np.sqrt(2.0 / N) * np.ones(N)
        alpha[0] = 1.0 / np.sqrt(N)
        phi = alpha[:, None] * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
        return np.sum(np.abs(phi), axis=1)

    def _precompute_dct_l1norms(self) -> np.ndarray:
        l1_y = self._dct1d_l1(self.H)
        l1_x = self._dct1d_l1(self.W)
        return np.outer(l1_y, l1_x)

    def _precompute_k_theta(self):
        y, x = np.indices((self.H, self.W))
        omega_u = np.pi * x / self.W
        omega_v = np.pi * y / self.H
        k = np.sqrt(omega_u ** 2 + omega_v ** 2)
        theta = np.arctan2(omega_v, omega_u)
        return k, theta

    # -- occupation ---------------------------------------------------------
    def occupation_maps(self, field_2d: np.ndarray, epsilon: float):
        """Return (occ_raw, occ_clipped). Hoelder L-inf -> L1 normalization."""
        coeffs = dct2(field_2d)
        denom = epsilon * self.l1_norms + 1e-8
        occ_raw = np.abs(coeffs) / denom
        occ_clip = np.clip(occ_raw, 0.0, 1.0)
        return occ_raw, occ_clip

    def occupation(self, field_2d: np.ndarray, epsilon: float) -> np.ndarray:
        """Clipped occupation map (the standard descriptor)."""
        return self.occupation_maps(field_2d, epsilon)[1]

    def occupation_stats(self, field_2d: np.ndarray, epsilon: float) -> dict:
        """Full-map saturation / occupation statistics (test D)."""
        occ_raw, occ_clip = self.occupation_maps(field_2d, epsilon)
        clipped_mask = occ_raw >= 1.0
        clip_frac = float(np.mean(clipped_mask))
        if np.any(clipped_mask):
            excess = occ_raw[clipped_mask] - 1.0
            clip_mean_excess = float(np.mean(excess))
            clip_p95_excess = float(np.quantile(excess, 0.95))
        else:
            clip_mean_excess = 0.0
            clip_p95_excess = 0.0
        return {
            "clip_frac": clip_frac,
            "clip_mean_excess": clip_mean_excess,
            "clip_p95_excess": clip_p95_excess,
            "occ_raw_mean": float(np.mean(occ_raw)),
            "occ_raw_p95": float(np.quantile(occ_raw, 0.95)),
            "occ_raw_p99": float(np.quantile(occ_raw, 0.99)),
            "occ_mean_2d": float(np.mean(occ_clip)),
            "occ_p95_2d": float(np.quantile(occ_clip, 0.95)),
        }

    # -- radial profile -----------------------------------------------------
    def radial_profile_E(
        self,
        field_2d: np.ndarray,
        epsilon: float,
        use_clipping: bool = True,
        cap_quantile: float | None = None,
    ):
        """Radial occupation profile + anisotropy index.

        With ``use_clipping=True`` and ``cap_quantile=None`` this reproduces the
        standard generation behaviour. ``use_clipping=False`` + ``cap_quantile``
        gives the unclipped high-quantile-capped variant used by the saturation
        sanity check.
        """
        occ_raw, occ_clip = self.occupation_maps(field_2d, epsilon)
        if use_clipping:
            occ = occ_clip
        else:
            occ = occ_raw.copy()
            if cap_quantile is not None:
                cap_val = np.quantile(occ, cap_quantile)
                occ = np.minimum(occ, cap_val)

        AI = compute_anisotropy_index(occ, self.theta_grid)

        k_flat = self.k_grid.flatten()
        occ_flat = occ.flatten()
        bins = np.linspace(0.0, float(np.max(k_flat)), self.num_bins)
        bin_ids = np.digitize(k_flat, bins)

        radial_k, radial_occ = [], []
        for b in range(1, len(bins)):
            mask = bin_ids == b
            if mask.sum() > 20:
                radial_k.append(np.mean(k_flat[mask]))
                radial_occ.append(np.median(occ_flat[mask]))
        return np.array(radial_k), np.array(radial_occ), AI

    # -- goodness of fit ----------------------------------------------------
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k_params: int):
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2) + 1e-12
        aic = n * np.log(rss / n) + 2 * k_params
        bic = n * np.log(rss / n) + k_params * np.log(n)
        return float(aic), float(bic)

    # -- fits ---------------------------------------------------------------
    def fit_fd(self, radial_k: np.ndarray, radial_occ: np.ndarray):
        nan7 = (np.nan,) * 7
        if len(radial_k) <= 4:
            return nan7
        E = radial_k ** 2
        E_min, E_max = float(E.min()), float(E.max())
        if E_max <= E_min:
            return nan7
        E_range = E_max - E_min
        mu0 = 0.5 * (E_min + E_max)
        T0 = 0.3 * E_range
        C0 = max(0.0, float(np.min(radial_occ)) * 0.5)
        try:
            popt, _ = curve_fit(
                fermi_dirac_E, E, radial_occ,
                p0=[mu0, T0, 1.0, C0],
                bounds=([E_min, 0.01 * E_range, 0.0, 0.0],
                        [1.5 * E_max, 10.0 * E_range, 2.0, 0.5]),
                maxfev=20000,
            )
            mu_fd, T_fd, A_fd, C_fd = popt
            pred = fermi_dirac_E(E, mu_fd, T_fd, A_fd, C_fd)
            R2 = self.r2(radial_occ, pred)
            aic, bic = self.aic_bic(radial_occ, pred, k_params=4)
        except Exception:
            mu_fd = T_fd = A_fd = C_fd = R2 = aic = bic = np.nan
        return mu_fd, T_fd, A_fd, C_fd, R2, aic, bic

    def fit_mb(self, radial_k: np.ndarray, radial_occ: np.ndarray):
        nan6 = (np.nan,) * 6
        if len(radial_k) <= 4:
            return nan6
        E = radial_k ** 2
        E_min, E_max = float(E.min()), float(E.max())
        if E_max <= E_min:
            return nan6
        try:
            popt, _ = curve_fit(
                maxwell_boltzmann_E, E, radial_occ,
                p0=[0.3 * (E_max - E_min), 1.0, 0.0],
                bounds=([0.01 * (E_max - E_min), 0.0, 0.0],
                        [10.0 * (E_max - E_min), 2.0, 0.5]),
                maxfev=10000,
            )
            T_mb, A_mb, C_mb = popt
            pred = maxwell_boltzmann_E(E, T_mb, A_mb, C_mb)
            R2 = self.r2(radial_occ, pred)
            aic, bic = self.aic_bic(radial_occ, pred, k_params=3)
        except Exception:
            T_mb = A_mb = C_mb = R2 = aic = bic = np.nan
        return T_mb, A_mb, C_mb, R2, aic, bic

    def fit_kww(self, radial_k: np.ndarray, radial_occ: np.ndarray):
        nan7 = (np.nan,) * 7
        if len(radial_k) <= 4:
            return nan7
        E = radial_k ** 2
        E_min, E_max = float(E.min()), float(E.max())
        if E_max <= E_min:
            return nan7
        C0 = max(0.0, float(np.min(radial_occ)) * 0.5)
        try:
            popt, _ = curve_fit(
                kww_glass_E, E, radial_occ,
                p0=[0.3 * (E_max - E_min), 0.7, 1.0, C0],
                bounds=([0.01 * (E_max - E_min), 0.10, 0.0, 0.0],
                        [10.0 * (E_max - E_min), 1.00, 2.0, 0.5]),
                maxfev=20000,
            )
            T_kww, beta, A_kww, C_kww = popt
            pred = kww_glass_E(E, T_kww, beta, A_kww, C_kww)
            R2 = self.r2(radial_occ, pred)
            aic, bic = self.aic_bic(radial_occ, pred, k_params=4)
        except Exception:
            T_kww = beta = A_kww = C_kww = R2 = aic = bic = np.nan
        return T_kww, beta, A_kww, C_kww, R2, aic, bic

    # legacy alias
    fit_sg = fit_kww

    # -- prediction helpers -------------------------------------------------
    def predict_fd(self, radial_k, params):
        mu, T, A, C = params
        return fermi_dirac_E(radial_k ** 2, mu, T, A, C)

    def predict_mb(self, radial_k, params):
        T, A, C = params
        return maxwell_boltzmann_E(radial_k ** 2, T, A, C)

    def predict_kww(self, radial_k, params):
        T_kww, beta, A, C = params
        return kww_glass_E(radial_k ** 2, T_kww, beta, A, C)

    predict_sg = predict_kww
