"""
ZPhoton schedules: analytic sigma-schedule family for Z-Image (flow matching).

All segments are built in t-space and mapped through the flow shift:
    sigma(t) = shift * t / (1 + (shift - 1) * t)
    t(sigma) = sigma / (shift - (shift - 1) * sigma)

A "restart" segment is encoded directly into the SIGMAS tensor as an ascending
jump right after a zero (e.g. ... 0.0, 0.66, 0.44, 0.22, 0.0).  The ZPhoton
sampler detects the ascending jump and performs a proper flow re-noise:
    x <- (1 - sigma) * x + sigma * fresh_noise
"""
import torch


def sigma_from_t(t: float, shift: float) -> float:
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def t_from_sigma(s: float, shift: float) -> float:
    if shift == 1.0:
        return s
    return s / (shift - (shift - 1.0) * s)


def _refine_segment(sigma_start: float, n_steps: int, curve: float = 1.0) -> list[float]:
    """Descending refine segment from sigma_start to 0 (n_steps model calls)."""
    sigs = []
    for j in range(n_steps):
        u = j / n_steps
        sigs.append(sigma_start * (1.0 - u) ** curve)
    sigs.append(0.0)
    return sigs


def build_turbo_sigmas(steps: int,
                       shift: float = 3.0,
                       sigma_max: float = 0.992,
                       sigma_floor: float = 0.75,
                       curve: float = 1.15,
                       restart: bool = True,
                       restart_sigma: float = 0.66,
                       restart_fraction: float = 0.30) -> torch.Tensor:
    """
    Schedule for distilled Z-Image-Turbo.

    Structure: dense steps at high sigma (composition), then a single "plunge"
    to 0 (the distilled model can one-shot x0 from mid sigma), then an optional
    restart/refine segment that re-noises to `restart_sigma` and descends
    linearly to 0.  Reproduces the shape of hand-tuned Power Nodes presets at
    8-9 steps and scales correctly to any step count.
    """
    steps = max(3, int(steps))
    n_r = 0
    if restart:
        n_r = max(1, min(int(round(steps * restart_fraction)), steps - 2))
    n_m = steps - n_r  # main pass: (n_m - 1) structure steps + 1 plunge step

    t_hi = t_from_sigma(min(sigma_max, 0.9999), shift)
    t_lo = t_from_sigma(min(sigma_floor, sigma_max - 1e-3), shift)

    sigs = []
    if n_m <= 1:
        sigs = [sigma_max]
    else:
        for i in range(n_m):
            u = i / (n_m - 1)
            t = t_hi - (t_hi - t_lo) * (u ** curve)
            sigs.append(sigma_from_t(t, shift))
    sigs.append(0.0)  # plunge

    if n_r > 0:
        sigs += _refine_segment(restart_sigma, n_r)
    return torch.FloatTensor(sigs)


def build_base_sigmas(steps: int,
                      shift: float = 3.0,
                      sigma_max: float = 0.997,
                      curve: float = 1.0,
                      restart: bool = True,
                      restart_sigma: float = 0.45,
                      restart_fraction: float = 0.22) -> torch.Tensor:
    """
    Schedule for non-distilled Z-Image Base.

    Full smooth descent (no plunge - the base model cannot one-shot x0),
    plus an optional restart segment at mid-low sigma (Restart Sampling:
    re-traversing mid sigmas reduces accumulated error and adds detail).
    """
    steps = max(4, int(steps))
    n_r = 0
    if restart:
        n_r = int(round(steps * restart_fraction))
        n_r = max(0, min(n_r, steps - 4))
        if n_r < 2:
            n_r = 0
    n_m = steps - n_r

    t_hi = t_from_sigma(min(sigma_max, 0.9999), shift)
    sigs = []
    for i in range(n_m):
        u = i / n_m
        t = t_hi * (1.0 - u) ** curve
        sigs.append(sigma_from_t(t, shift))
    sigs.append(0.0)

    if n_r > 0:
        sigs += _refine_segment(restart_sigma, n_r)
    return torch.FloatTensor(sigs)


def build_refiner_sigmas(steps: int,
                         denoise: float,
                         shift: float = 3.0,
                         curve: float = 1.1) -> torch.Tensor:
    """
    Second-pass (hires-fix) schedule: descend from sigma(t=denoise) to 0.
    `denoise` is interpreted in t-space, so the shift maps it to a sensible
    sigma (e.g. denoise 0.35 @ shift 3.0 -> sigma_start ~= 0.62).
    """
    steps = max(1, int(steps))
    sigma_start = sigma_from_t(max(1e-3, min(denoise, 1.0)), shift)
    sigs = []
    for j in range(steps):
        u = j / steps
        sigs.append(sigma_start * (1.0 - u) ** curve)
    sigs.append(0.0)
    return torch.FloatTensor(sigs)
