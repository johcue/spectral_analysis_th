"""
spectral.attacks
================

Adversarial attacks (BIM, PGD) in raw pixel space plus the L-inf-bounded noise
controls used by tests B, C and D. Requires torch.

Every attack takes and returns tensors in [0, 1] (raw pixel space); the model
forward is done in normalized space internally so the perturbation budget is
physically consistent.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# gradient attacks
# ---------------------------------------------------------------------------
def bim_attack_raw_physics(model, criterion, x_raw, y, eps_pix, alpha_pix, steps, mean, std):
    """Basic Iterative Method, L-inf, raw pixel space."""
    device = x_raw.device
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    x0 = x_raw.detach()
    x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model((x_adv - mean_t) / std_t)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha_pix * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x0, -eps_pix, eps_pix)
            x_adv = torch.clamp(x0 + delta, 0.0, 1.0)
        x_adv = x_adv.detach()

    delta = (x_adv - x0).detach()
    return x_adv, delta


def pgd_attack_raw_physics(model, criterion, x_raw, y, eps_pix, alpha_pix, steps, mean, std):
    """Projected Gradient Descent, L-inf, raw pixel space, random start."""
    device = x_raw.device
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    x0 = x_raw.detach()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model((x_adv - mean_t) / std_t)
        loss = criterion(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha_pix * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x0, -eps_pix, eps_pix)
            x_adv = torch.clamp(x0 + delta, 0.0, 1.0)
        x_adv = x_adv.detach()

    delta = (x_adv - x0).detach()
    return x_adv, delta


ATTACK_FNS = {
    "BIM": bim_attack_raw_physics,
    "PGD": pgd_attack_raw_physics,
}


# ---------------------------------------------------------------------------
# tensor-level noise (test B: matched L-inf noise fields)
# ---------------------------------------------------------------------------
def make_linf_uniform_noise(x_raw: torch.Tensor, eps_pix: float):
    delta = (torch.rand_like(x_raw) * 2.0 - 1.0) * eps_pix
    x_noisy = torch.clamp(x_raw + delta, 0.0, 1.0)
    return x_noisy.detach(), (x_noisy - x_raw).detach()


def make_linf_sign_noise(x_raw: torch.Tensor, eps_pix: float):
    u = torch.rand_like(x_raw) * 2.0 - 1.0
    delta = eps_pix * torch.sign(u)
    x_noisy = torch.clamp(x_raw + delta, 0.0, 1.0)
    return x_noisy.detach(), (x_noisy - x_raw).detach()


# ---------------------------------------------------------------------------
# perturbation-matched controls (test C: applied to the *delta* tensor)
# ---------------------------------------------------------------------------
def unwrap_delta(delta_like):
    """Recover the torch.Tensor delta from common attack return formats."""
    if torch.is_tensor(delta_like):
        return delta_like
    if isinstance(delta_like, (list, tuple)):
        for item in delta_like:
            d = unwrap_delta(item)
            if torch.is_tensor(d):
                return d
    raise TypeError(f"Could not find a torch.Tensor delta inside {type(delta_like)}")


def noise_sign_tensor(delta_raw: torch.Tensor, eps_pix: float, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(delta_raw.shape, dtype=np.float32)
    return torch.from_numpy(eps_pix * np.sign(z)).to(delta_raw.device)


def noise_gauss_clip_tensor(delta_raw: torch.Tensor, eps_pix: float, sigma: float | None = None, seed: int = 0) -> torch.Tensor:
    if sigma is None:
        sigma = eps_pix / 2.0
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(delta_raw.shape, dtype=np.float32) * np.float32(sigma)
    out = np.clip(z, -eps_pix, eps_pix).astype(np.float32)
    return torch.from_numpy(out).to(delta_raw.device)


def noise_shuffle_delta(delta_raw: torch.Tensor, seed: int = 0) -> torch.Tensor:
    delta_np = delta_raw.detach().cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(delta_np)
    B, C, H, W = out.shape
    for b in range(B):
        for c in range(C):
            flat = delta_np[b, c].reshape(-1).copy()
            rng.shuffle(flat)
            out[b, c] = flat.reshape(H, W)
    return torch.from_numpy(out).to(delta_raw.device)


# ---------------------------------------------------------------------------
# numpy per-channel controls (test D: built from a single delta channel)
# ---------------------------------------------------------------------------
def make_noise_controls(delta_channel: np.ndarray, eps_pix: float, rng: np.random.Generator):
    """Return (noise_sign, noise_shuffle, noise_gauss) numpy channels."""
    H, W = delta_channel.shape
    noise_sign = eps_pix * np.sign(rng.standard_normal((H, W))).astype(np.float32)

    noise_shuffle = delta_channel.copy().reshape(-1)
    rng.shuffle(noise_shuffle)
    noise_shuffle = noise_shuffle.reshape(H, W).astype(np.float32)

    noise_gauss = rng.normal(loc=0.0, scale=eps_pix / 3.0, size=(H, W)).astype(np.float32)
    noise_gauss = np.clip(noise_gauss, -eps_pix, eps_pix)
    return noise_sign, noise_shuffle, noise_gauss
