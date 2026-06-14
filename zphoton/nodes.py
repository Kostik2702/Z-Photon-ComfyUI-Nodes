"""ZPhoton node definitions."""
import math
import torch
import comfy.utils
import comfy.sample

from .schedules import (build_turbo_sigmas, build_base_sigmas,
                        build_refiner_sigmas, sigma_from_t)
from .noise import prepare_noise
from .sampling import run_sampling
from .styles import PHOTO_STYLES

CATEGORY = "ZPhoton"
MAX_SEED = 0xffffffffffffffff

_SAMPLERS = ["euler", "euler_2m"]
_MODES = ["turbo", "base"]

# Native-friendly resolution buckets (all dims divisible by 32).
_BUCKETS = {
    "S (~0.6 MP)":  {"1:1": (800, 800),   "4:3": (896, 672),   "3:2": (960, 640),   "16:9": (1024, 576),  "21:9": (1152, 480)},
    "M (~1.0 MP)":  {"1:1": (1024, 1024), "4:3": (1152, 864),  "3:2": (1216, 832),  "16:9": (1344, 768),  "21:9": (1536, 640)},
    "L (~1.7 MP)":  {"1:1": (1312, 1312), "4:3": (1504, 1120), "3:2": (1600, 1088), "16:9": (1728, 960),  "21:9": (1984, 832)},
    "XL (~2.1 MP)": {"1:1": (1440, 1440), "4:3": (1664, 1248), "3:2": (1792, 1184), "16:9": (1920, 1088), "21:9": (2176, 928)},
}
_ASPECTS = list(_BUCKETS["M (~1.0 MP)"].keys())

# quality/speed presets: every internal knob is baked in
_PRESETS = {
    "turbo / fast (6 steps)":      dict(mode="turbo", steps=6,  cfg=1.0, detail=0.30, contrast=0.05),
    "turbo / balanced (9 steps)":  dict(mode="turbo", steps=9,  cfg=1.0, detail=0.35, contrast=0.08),
    "turbo / quality (14 steps)":  dict(mode="turbo", steps=14, cfg=1.0, detail=0.40, contrast=0.10),
    "turbo / max (20 steps)":      dict(mode="turbo", steps=20, cfg=1.0, detail=0.40, contrast=0.10),
    "base / quality (28 steps)":   dict(mode="base",  steps=28, cfg=4.0, detail=0.50, contrast=0.0),
    "base / max (36 steps)":       dict(mode="base",  steps=36, cfg=4.0, detail=0.50, contrast=0.0),
}

# variety level -> (LF blend strength, LF boost)
_VARIETY = {
    "off":    (0.00, 1.00),
    "low":    (0.45, 1.00),
    "medium": (0.75, 1.12),
    "high":   (1.00, 1.25),
}

# look -> initial-noise contrast offset (true-to-life vs punchy)
_LOOKS = {
    "natural (true-to-life)": -0.10,
    "standard": 0.0,
    "vivid": +0.05,
}

# system prompts for the Qwen3 text encoder (Lumina2 template)
_SYSTEM_PROMPTS = {
    "photo (recommended)": (
        "You are an assistant designed to generate high-quality images with the highest "
        "degree of image-text alignment based on textual prompts. The image must look like "
        "a real photograph captured with a real camera: natural lighting, authentic skin "
        "texture, physically plausible anatomy and composition, true-to-life colors, "
        "no CGI, no illustration, no over-smoothing."
    ),
    "alignment (stock)": (
        "You are an assistant designed to generate high-quality images with the highest "
        "degree of image-text alignment based on textual prompts."
    ),
    "superior (stock)": (
        "You are an assistant designed to generate superior images with the superior "
        "degree of image-text alignment based on textual prompts or user prompts."
    ),
    "none": "",
}


def _build_sigmas(mode, steps, shift, restart):
    if mode == "base":
        return build_base_sigmas(steps, shift=shift, restart=restart)
    return build_turbo_sigmas(steps, shift=shift, restart=restart)


class ZPhotonSampler:
    """Photorealistic Z-Image sampler: pick a quality preset and a variety
    level - everything else is baked into the preset.  Connect `clean_model`
    (the checkpoint BEFORE LoRA loaders) to lock composition on the clean
    model and apply LoRAs only in the identity/detail phase - prevents LoRA
    mutations.  Use ZPhoton Sampler (Advanced) for full control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
                "preset": (list(_PRESETS.keys()), {"default": "turbo / quality (14 steps)", "tooltip": "turbo presets for Z-Image-Turbo (cfg 1, negative ignored); base presets for Z-Image Base (cfg 4, connect negative)."}),
                "variety": (list(_VARIETY.keys()), {"default": "off", "tooltip": "Composition variety without changing style/texture (low-frequency seed blend)."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Used by base presets (cfg > 1). Ignored by turbo presets."}),
                "clean_model": ("MODEL", {"tooltip": "Model WITHOUT LoRAs (checkpoint output). If connected, the composition phase (high sigmas) runs on it and LoRAs apply only in the identity/detail phase - prevents LoRA mutations."}),
                "look": (list(_LOOKS.keys()), {"default": "standard", "tooltip": "natural = true-to-life colors (negative noise overdose, photographic wash); vivid = extra punch. Pair 'natural' with the ZPhoton Tone node for full anti-oversaturation."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent, seed, preset, variety,
               negative=None, clean_model=None, look="standard"):
        p = _PRESETS[preset]
        v_strength, lf_boost = _VARIETY[variety]
        sigmas = _build_sigmas(p["mode"], p["steps"], 3.0, True)
        lat = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        vseed = (int(seed) ^ 0x9E3779B97F4A7C15) & MAX_SEED
        noise = prepare_noise(lat.shape, seed,
                              contrast=p["contrast"] + _LOOKS.get(look, 0.0),
                              variation_seed=vseed,
                              variation_strength=v_strength,
                              lf_boost=lf_boost)
        # working negative on Turbo: true CFG on the early (structure) phase only
        neg_cfg = 2.5 if (p["mode"] == "turbo" and negative is not None) else 1.0
        out = run_sampling(model, positive, negative, latent, sigmas,
                           seed=seed, cfg=p["cfg"], noise=noise,
                           detail_amount=p["detail"], order=1,
                           composition_model=clean_model,
                           composition_end=0.85,
                           neg_cfg=neg_cfg, neg_cfg_end=0.70)
        return (out,)


class ZPhotonSamplerAdvanced:
    """ZPhoton sampler with explicit SIGMAS input and all knobs exposed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler": (_SAMPLERS, {"default": "euler"}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "detail_amount": ("FLOAT", {"default": 0.4, "min": -1.0, "max": 1.0, "step": 0.01}),
                "detail_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_end": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_peak": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.95, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.01}),
                "noise_bias": ("FLOAT", {"default": 0.0, "min": -0.05, "max": 0.05, "step": 0.001}),
                "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "lf_boost": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.6, "step": 0.01}),
                "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "composition_end": ("FLOAT", {"default": 0.85, "min": 0.3, "max": 0.99, "step": 0.01, "tooltip": "Sigma where clean_model hands over to model (when clean_model is connected)."}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "clean_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent, sigmas, seed, cfg, sampler, add_noise,
               detail_amount, detail_start, detail_end, detail_peak, contrast,
               noise_bias, variation_strength, variation_seed, lf_boost, grain,
               composition_end, negative=None, clean_model=None):
        lat = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        noise = None
        if add_noise:
            noise = prepare_noise(lat.shape, seed,
                                  contrast=contrast, bias=noise_bias,
                                  variation_seed=variation_seed,
                                  variation_strength=variation_strength,
                                  lf_boost=lf_boost)
        out = run_sampling(model, positive, negative, latent, sigmas,
                           seed=seed, cfg=cfg, noise=noise, add_noise=add_noise,
                           detail_amount=detail_amount, detail_start=detail_start,
                           detail_end=detail_end, detail_peak=detail_peak,
                           order=2 if sampler == "euler_2m" else 1, grain=grain,
                           composition_model=clean_model,
                           composition_end=composition_end)
        return (out,)


class ZPhotonEncode:
    """Prompt encoder for Z-Image (Qwen3 / Lumina2 template) with a curated
    photographic system prompt and optional style injection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "system": (list(_SYSTEM_PROMPTS.keys()), {"default": "photo (recommended)"}),
                "style": (list(PHOTO_STYLES.keys()), {"default": "none"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "final_text")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, text, system, style):
        if clip is None:
            raise RuntimeError("clip input is invalid (None).")
        user = text.strip()
        snippet = PHOTO_STYLES.get(style, "")
        if snippet:
            user = (user.rstrip().rstrip(".,") + ". " + snippet) if user else snippet
        sys_text = _SYSTEM_PROMPTS.get(system, "")
        prompt = f"{sys_text} <Prompt Start> {user}" if sys_text else user
        tokens = clip.tokenize(prompt)
        return (clip.encode_from_tokens_scheduled(tokens), user)


class ZPhotonScheduler:
    """Analytic sigma schedule family for Z-Image (with restart segment)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_MODES, {"default": "turbo"}),
                "steps": ("INT", {"default": 9, "min": 3, "max": 100}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.05}),
                "restart": ("BOOLEAN", {"default": True}),
                "restart_sigma": ("FLOAT", {"default": 0.66, "min": 0.05, "max": 0.95, "step": 0.01}),
                "restart_fraction": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.6, "step": 0.01}),
                "sigma_floor": ("FLOAT", {"default": 0.75, "min": 0.2, "max": 0.95, "step": 0.01, "tooltip": "turbo only: sigma where the plunge to 0 happens."}),
                "curve": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = CATEGORY

    def get_sigmas(self, mode, steps, shift, restart, restart_sigma,
                   restart_fraction, sigma_floor, curve):
        if mode == "base":
            sigmas = build_base_sigmas(steps, shift=shift, curve=min(curve, 1.5),
                                       restart=restart, restart_sigma=min(restart_sigma, 0.6),
                                       restart_fraction=restart_fraction)
        else:
            sigmas = build_turbo_sigmas(steps, shift=shift, sigma_floor=sigma_floor,
                                        curve=curve, restart=restart,
                                        restart_sigma=restart_sigma,
                                        restart_fraction=restart_fraction)
        return (sigmas,)


class ZPhotonEmptyLatent:
    """Empty latent snapped to Z-Image-native resolution buckets
    (reduces edge mutations on non-square aspect ratios)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect": (_ASPECTS, {"default": "3:2"}),
                "orientation": (["landscape", "portrait"], {"default": "portrait"}),
                "size": (list(_BUCKETS.keys()), {"default": "M (~1.0 MP)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, aspect, orientation, size, batch_size):
        w, h = _BUCKETS[size][aspect]
        if orientation == "portrait":
            w, h = h, w
        latent = torch.zeros([batch_size, 16, h // 8, w // 8], device="cpu")
        return ({"samples": latent}, w, h)


class ZPhotonRefiner:
    """Second pass (img2img refine): latent upscale + flow re-noise + detail
    descent.  NOTE: for the standard t2i pipeline prefer SeedVR2 for the
    upscale - this node softens fine detail at scale_by > 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.5, "step": 0.05}),
                "denoise": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.01}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 40}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.05}),
                "sampler": (_SAMPLERS, {"default": "euler"}),
                "detail": ("FLOAT", {"default": 0.2, "min": -1.0, "max": 1.0, "step": 0.01}),
                "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "refine"
    CATEGORY = CATEGORY

    def refine(self, model, positive, latent, seed, scale_by, denoise, steps,
               cfg, shift, sampler, detail, grain, negative=None):
        samples = latent["samples"]
        if scale_by > 1.0:
            h = int(round(samples.shape[-2] * scale_by / 2) * 2)
            w = int(round(samples.shape[-1] * scale_by / 2) * 2)
            samples = comfy.utils.common_upscale(samples, w, h, "bicubic", "disabled")
        lat = dict(latent)
        lat["samples"] = samples
        lat.pop("noise_mask", None)

        sigmas = build_refiner_sigmas(steps, denoise, shift=shift)
        noise = prepare_noise(samples.shape, seed)
        out = run_sampling(model, positive, negative, lat, sigmas,
                           seed=seed, cfg=cfg, noise=noise,
                           detail_amount=detail, detail_start=0.0,
                           detail_end=0.85, detail_peak=0.45,
                           order=2 if sampler == "euler_2m" else 1, grain=grain)
        return (out,)


class ZPhotonPhotoStyle:
    """Inject a curated photographic style into a prompt string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "style": (list(PHOTO_STYLES.keys()), {"default": "natural portrait (window light)"}),
                "position": (["append", "prepend"], {"default": "append"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, text, style, position):
        snippet = PHOTO_STYLES.get(style, "")
        if not snippet:
            return (text,)
        if not text.strip():
            return (snippet,)
        if position == "prepend":
            return (snippet + ". " + text,)
        return (text.rstrip().rstrip(".,") + ". " + snippet,)


class ZPhotonTone:
    """Filmic anti-oversaturation: soft-compresses only the TOP of the
    saturation range (neon/HDR look) while leaving low and mid saturation
    untouched - true-to-life / human-eye color response.  Place after
    VAEDecode (and after ColorMatch when using SeedVR2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How strongly the over-saturated range is compressed."}),
                "knee": ("FLOAT", {"default": 0.6, "min": 0.2, "max": 0.95, "step": 0.05, "tooltip": "Saturation level where compression starts. Below the knee colors are untouched."}),
                "highlight_desat": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Extra desaturation of bright highlights (film-like highlight rolloff)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tone"
    CATEGORY = CATEGORY

    def tone(self, images, strength, knee, highlight_desat):
        x = images.float().clamp(0, 1)
        rgb = x[..., :3]
        luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).unsqueeze(-1)
        v, _ = rgb.max(dim=-1, keepdim=True)
        m, _ = rgb.min(dim=-1, keepdim=True)
        sat = (v - m) / (v + 1e-5)

        # soft-knee compression of saturation above `knee`
        over = (sat - knee).clamp(min=0.0)
        target_sat = knee + over / (1.0 + 2.0 * over)        # smooth rolloff
        f = torch.where(sat > 1e-5, target_sat / sat.clamp(min=1e-5),
                        torch.ones_like(sat))
        f = 1.0 + strength * (f - 1.0)

        # filmic highlight desaturation
        if highlight_desat > 0:
            hi = ((v - 0.85) / 0.15).clamp(0, 1)
            f = f * (1.0 - highlight_desat * 0.5 * hi)

        out_rgb = (luma + (rgb - luma) * f).clamp(0, 1)
        out = x.clone()
        out[..., :3] = out_rgb
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ZPhotonSampler": ZPhotonSampler,
    "ZPhotonTone": ZPhotonTone,
    "ZPhotonSamplerAdvanced": ZPhotonSamplerAdvanced,
    "ZPhotonEncode": ZPhotonEncode,
    "ZPhotonScheduler": ZPhotonScheduler,
    "ZPhotonEmptyLatent": ZPhotonEmptyLatent,
    "ZPhotonRefiner": ZPhotonRefiner,
    "ZPhotonPhotoStyle": ZPhotonPhotoStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZPhotonSampler": "ZPhoton Sampler",
    "ZPhotonTone": "ZPhoton Tone (true-to-life)",
    "ZPhotonSamplerAdvanced": "ZPhoton Sampler (Advanced)",
    "ZPhotonEncode": "ZPhoton Encode",
    "ZPhotonScheduler": "ZPhoton Scheduler",
    "ZPhotonEmptyLatent": "ZPhoton Empty Latent",
    "ZPhotonRefiner": "ZPhoton Refiner",
    "ZPhotonPhotoStyle": "ZPhoton Photo Style",
}
