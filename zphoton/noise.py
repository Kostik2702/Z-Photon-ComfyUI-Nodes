"""
ZPhoton initial-noise shaping.

- contrast ("noise overdose"): scales initial noise amplitude -> contrast/saturation.
- bias: small constant offset -> brightness-ish control.
- variation: variance-preserving blend of the LOW-FREQUENCY component of a
  second seed's noise.  Composition (low frequencies) changes while the high
  frequency statistics (texture/detail) of the main seed are preserved.
- lf_boost: amplifies the LF band slightly (>1.0) for extra compositional
  spread across seeds (mild, controlled out-of-distribution push - the safe
  analog of Power Nodes' latent scrambling).

  Implementation notes (important):
  * The LF component must be extracted with true area averaging
    (avg_pool2d).  F.interpolate(bilinear) WITHOUT antialias subsamples
    instead of averaging, producing an "LF" field ~8x too strong that
    drowns the init noise in colored blobs.
  * The blend is variance-preserving: lf' = sqrt(1-v^2)*lf1 + v*lf2.
    For independent gaussians this keeps the LF band variance exactly,
    so the model still sees statistically white noise.
"""
import math
import torch
import torch.nn.functional as F

_LF_FACTOR = 8  # low-pass cutoff: latent / 8


def _low_freq(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    f = _LF_FACTOR
    if h < f * 2 or w < f * 2:
        return torch.zeros_like(x)
    down = F.avg_pool2d(x, kernel_size=f, stride=f, ceil_mode=True)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def prepare_noise(shape,
                  seed: int,
                  contrast: float = 0.0,
                  bias: float = 0.0,
                  variation_seed: int = 0,
                  variation_strength: float = 0.0,
                  lf_boost: float = 1.0,
                  dtype=torch.float32) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xffffffffffffffff)
    noise = torch.randn(shape, generator=gen, device="cpu", dtype=dtype)

    v = max(0.0, min(1.0, float(variation_strength)))
    boost = max(0.5, min(1.6, float(lf_boost)))
    if v > 0.0 or boost != 1.0:
        lf1 = _low_freq(noise)
        if v > 0.0:
            gen_v = torch.Generator(device="cpu").manual_seed(int(variation_seed) & 0xffffffffffffffff)
            noise_v = torch.randn(shape, generator=gen_v, device="cpu", dtype=dtype)
            lf_mix = math.sqrt(1.0 - v * v) * lf1 + v * _low_freq(noise_v)
        else:
            lf_mix = lf1
        # variance-preserving blend in the LF band, optionally boosted
        noise = noise - lf1 + lf_mix * boost

    if contrast != 0.0:
        noise = noise * (1.0 + contrast)
    if bias != 0.0:
        noise = noise + bias
    return noise
