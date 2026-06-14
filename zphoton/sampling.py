"""
ZPhoton sampler core.

A single custom KSAMPLER loop that handles, in one pass:
  * euler / euler_2m (2nd-order Adams-Bashforth on the flow derivative,
    with automatic fallback to plain euler on large steps and after restarts)
  * restart segments encoded as ascending sigma jumps (proper flow re-noise)
  * detail boost: step-relative sigma nudge (bounded by the next state,
    cannot accumulate into speckles; skipped on plunge/final steps)
  * optional fine grain injection at low sigmas

LoRA phase scheduling (run_sampling with composition_model):
  the high-sigma "composition" phase runs on a clean model, the low-sigma
  phase on the LoRA-patched model.  Distilled-turbo composition stays intact
  (no mutations), identity/style still lands.  The split is exact: both
  segments share the boundary sigma, so ComfyUI's noise_scaling /
  inverse_noise_scaling rescale cancels out identically.

Everything goes through comfy.sample.sample_custom, so CFG, LoRA patches,
previews and interrupts work exactly like with built-in samplers.
"""
import torch
import comfy.sample
import comfy.utils
from comfy.samplers import KSAMPLER


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _detail_envelope(p: float, start: float, end: float, peak: float) -> float:
    """Smooth window over normalized progress p in [0,1], peaking at `peak`."""
    if end <= start:
        return 0.0
    u = (p - start) / (end - start)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    peak = max(0.05, min(0.95, peak))
    w = u / peak if u < peak else (1.0 - u) / (1.0 - peak)
    return _smoothstep(w)


@torch.no_grad()
def zphoton_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                         detail_amount=0.0, detail_start=0.15, detail_end=0.95,
                         detail_peak=0.6, order=1, restart_seed=0, grain=0.0):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None

    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # --- restart segment: ascending jump -> proper flow re-noise ---
        if s_next > s_cur + 1e-6:
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            old_d = None
            continue

        if s_cur <= 1e-6:
            continue

        # --- detail boost: step-relative sigma nudge ---
        p = i / max(n - 1, 1)
        is_final = s_next <= 1e-6
        is_plunge = (s_cur - s_next) > 0.25
        if is_final or is_plunge:
            a = 0.0
        else:
            a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        denoised = model(x, sigma_model * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        d = (x - denoised) / s_cur
        dt = s_next - s_cur
        if order >= 2 and old_d is not None and abs(dt) <= 0.25:
            d_use = 1.5 * d - 0.5 * old_d   # Adams-Bashforth 2
        else:
            d_use = d
        x = x + d_use * dt
        old_d = d

        # --- optional fine grain at low sigma (anti-plastic-skin) ---
        if grain > 0.0 and 1e-6 < s_next < 0.35:
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = x + eps * (grain * 0.02 * s_next)

    return x


def make_zphoton_ksampler(detail_amount=0.0, detail_start=0.15, detail_end=0.95,
                          detail_peak=0.6, order=1, restart_seed=0, grain=0.0) -> KSAMPLER:
    return KSAMPLER(zphoton_sampler_loop, extra_options={
        "detail_amount": float(detail_amount),
        "detail_start": float(detail_start),
        "detail_end": float(detail_end),
        "detail_peak": float(detail_peak),
        "order": int(order),
        "restart_seed": int(restart_seed),
        "grain": float(grain),
    })


def zero_conditioning(cond):
    """Zeroed-out copy of a conditioning (honest unconditional for cfg > 1)."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


def _run_one(model, positive, negative, latent_dict, sigmas, *,
             seed, cfg, noise, add_noise,
             detail_amount, detail_start, detail_end, detail_peak, order, grain):
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])

    if negative is None:
        negative = zero_conditioning(positive)

    if not add_noise:
        noise = torch.zeros(latent.shape, dtype=latent.dtype, device="cpu")
    elif noise is None:
        noise = torch.randn(latent.shape, device="cpu", dtype=torch.float32,
                            generator=torch.Generator(device="cpu").manual_seed(int(seed) & 0xffffffffffffffff))

    sampler = make_zphoton_ksampler(detail_amount, detail_start, detail_end,
                                    detail_peak, order, seed, grain)

    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas,
                                         positive, negative, latent,
                                         noise_mask=latent_dict.get("noise_mask"),
                                         callback=callback,
                                         disable_pbar=disable_pbar, seed=seed)
    out = latent_dict.copy()
    out["samples"] = samples
    return out


def run_sampling(model, positive, negative, latent_dict, sigmas, *,
                 seed, cfg=1.0, noise=None, add_noise=True,
                 detail_amount=0.0, detail_start=0.15, detail_end=0.95,
                 detail_peak=0.6, order=1, grain=0.0,
                 composition_model=None, composition_end=0.85,
                 neg_cfg=1.0, neg_cfg_end=0.80):
    """Shared entry point for all ZPhoton sampler nodes.

    Multi-phase execution (segments share boundary sigmas, so ComfyUI's
    noise_scaling rescale cancels exactly between segments):

    * composition_model: sigmas above `composition_end` run on the clean
      model (stable composition), the rest on `model` (with LoRAs).
    * neg_cfg > 1: sigmas above `neg_cfg_end` run with true CFG and the
      provided negative - this gives a WORKING negative prompt on the
      distilled Turbo model (cfg 1 normally ignores it) at the cost of a
      few extra model calls on the early steps only.
    """
    kw = dict(detail_amount=detail_amount, detail_start=detail_start,
              detail_end=detail_end, detail_peak=detail_peak,
              order=order, grain=grain)

    sig_list = [float(s) for s in sigmas]

    def first_idx_below(th):
        for i, s in enumerate(sig_list):
            if s <= th:
                return i
        return None

    use_comp = composition_model is not None
    use_neg = (neg_cfg > 1.0 + 1e-3) and (negative is not None) and (cfg <= 1.0 + 1e-3)

    split_idxs = set()
    for th, flag in ((composition_end, use_comp), (neg_cfg_end, use_neg)):
        if flag:
            i = first_idx_below(th)
            if i is not None and 0 < i < len(sig_list) - 1:
                split_idxs.add(i)

    if not split_idxs and not (use_neg and first_idx_below(neg_cfg_end) is None):
        return _run_one(model, positive, negative, latent_dict, sigmas,
                        seed=seed, cfg=cfg, noise=noise, add_noise=add_noise, **kw)

    bounds = [0] + sorted(split_idxs) + [len(sig_list) - 1]
    cur = latent_dict
    cur_noise, cur_add = noise, add_noise
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        if a == b:
            continue
        seg = sigmas[a:b + 1]
        sig_start = sig_list[a]
        m = composition_model if (use_comp and sig_start > composition_end) else model
        c = neg_cfg if (use_neg and sig_start > neg_cfg_end) else cfg
        cur = _run_one(m, positive, negative, cur, seg,
                       seed=seed, cfg=c, noise=cur_noise, add_noise=cur_add, **kw)
        cur_noise, cur_add = None, False
    return cur
