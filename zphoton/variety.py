"""
ZPhoton mid-trajectory variety operator.

The problem with input-noise variety (see research/01_RESEARCH_variety.md):
perturbing the *initial* noise barely moves composition on the distilled
short-schedule Turbo trajectory, and the old `lf_boost` inflated the LF band
variance (broke the white-noise assumption -> colored-blob artifacts).

This operator instead perturbs the LATENT on the mid trajectory (after stage-1
composition is set), the same point Power Nodes uses for its scramble - but
instead of cutting/resampling fragments (which destroys high-frequency
texture) it performs a smooth, spectrally-selective, variance-preserving
rotation of ONLY the low-frequency (composition) band, leaving the
high-frequency (texture/detail) band exactly intact and restoring per-channel
mean/std (palette/style).

Guarantees (proven in research/02_*.py, research/03_*.py):
  * LF-band variance preserved exactly:  lf' = sqrt(1-a^2)*lf + a*g_hat
  * a is an interpretable knob: corr(lf', lf) = sqrt(1-a^2)
  * HF band carried over identically (hard idempotent band mask)
  * per-channel mean/std restored to the input (style preserved)
"""
import torch

# cache of hard radial low-pass masks, keyed by (H, W, cutoff_x1000, device)
_MASK_CACHE = {}


def _lf_mask(h: int, w: int, cutoff: float, device, dtype=torch.float32):
    """Hard radial low-pass mask in rFFT layout (h, w//2+1).  Idempotent."""
    key = (h, w, int(round(cutoff * 1000)), str(device))
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m.to(device=device, dtype=dtype)
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
    r = torch.sqrt(fy * fy + fx * fx)
    m = (r <= cutoff * 0.5).to(dtype)
    _MASK_CACHE[key] = m
    return m


def _split_bands(x: torch.Tensor, mask: torch.Tensor):
    """Return (lf, hf) with x = lf + hf exactly (hard-mask band split)."""
    X = torch.fft.rfft2(x, norm="backward")
    lf = torch.fft.irfft2(X * mask, s=x.shape[-2:], norm="backward")
    return lf, x - lf


@torch.no_grad()
def lf_recompose(x: torch.Tensor, *, seed_v: int, a: float,
                 cutoff: float = 0.25) -> torch.Tensor:
    """
    Variance-preserving rotation of the latent's low-frequency band.

    x      : (B, C, H, W) latent on the mid trajectory.
    seed_v : seed for the new LF pattern (derive deterministically from the
             user seed; offset per batch element for in-batch diversity).
    a      : strength in [0, 1]; corr with the original composition is
             sqrt(1 - a^2).  a = 0 returns x unchanged (zero cost).
    cutoff : fraction of Nyquist separating the LF (composition) band from
             the HF (texture/detail) band.

    Returns a tensor with x's per-channel mean/std, with only the LF band
    rotated toward the new seed and the HF band left intact.
    """
    a = float(max(0.0, min(1.0, a)))
    if a <= 0.0:
        return x

    b, c, h, w = x.shape
    # degenerate latent: band too small to separate -> no-op (mirrors _low_freq)
    if h < int(round(1.0 / max(cutoff, 1e-3))) or w < int(round(1.0 / max(cutoff, 1e-3))):
        return x

    work_dtype = torch.float32
    xf = x.to(work_dtype)
    mask = _lf_mask(h, w, cutoff, x.device, work_dtype)

    lf, hf = _split_bands(xf, mask)

    # new LF pattern from a fresh seed, per batch element for diversity
    g = torch.empty_like(xf)
    for i in range(b):
        gen = torch.Generator(device="cpu").manual_seed(
            (int(seed_v) + 0x9E3779B9 * i) & 0xffffffffffffffff)
        ni = torch.randn((c, h, w), generator=gen, device="cpu", dtype=work_dtype)
        g[i] = ni.to(x.device)
    g, _ = _split_bands(g, mask)

    # per-channel band-energy equalization: ||g_c|| -> ||lf_c||
    flat = (b, c, -1)
    lf_norm = lf.reshape(flat).norm(dim=2).view(b, c, 1, 1)
    g_norm = g.reshape(flat).norm(dim=2).view(b, c, 1, 1).clamp(min=1e-6)
    g_hat = g * (lf_norm / g_norm)

    # variance-preserving rotation of the LF band
    lf_new = (1.0 - a * a) ** 0.5 * lf + a * g_hat
    y = lf_new + hf

    # restore per-channel mean/std (palette/style)
    y_mean = y.mean(dim=(2, 3), keepdim=True)
    y_std = y.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    x_mean = xf.mean(dim=(2, 3), keepdim=True)
    x_std = xf.std(dim=(2, 3), keepdim=True)
    y = (y - y_mean) / y_std * x_std + x_mean

    return y.to(x.dtype)
