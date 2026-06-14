"""
ZPhoton LoRA Stack (Auto) - a model-aware LoRA loader.

What it does beyond a chain of LoraLoaders:

1. Tensor analysis of every LoRA: which transformer layers it touches, how
   hard (size-normalized RMS of ||up||*||down||*alpha/rank per key), rank,
   adaLN involvement, trigger metadata.

2. Auto-balance: LoRAs are trained with wildly different learning rates and
   scales, so "0.8 + 0.8" of two LoRAs rarely means equal influence.  In
   `auto` mode every LoRA's effective magnitude is normalized to the stack
   median, so strength 1.0 means "one comparable unit of effect" for every
   slot, and the total stack effect is softly capped.

3. Composition protection: the early transformer layers and adaLN modulation
   drive global composition - exactly where stacked LoRAs cause mutations on
   the distilled Z-Image-Turbo trajectory.  `normal`/`strong` attenuate LoRA
   deltas on the early ~30% of layers (and adaLN in `strong`), keeping
   identity/style from mid-late layers intact.  Combine with the sampler's
   `clean_model` input for maximum robustness.

Outputs the patched MODEL/CLIP plus a text report of what was detected and
applied.
"""
import math
import re
import logging

import folder_paths
import comfy.utils
import comfy.lora
import comfy.lora_convert

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _key_str(k):
    """Patch dict keys can be strings or tuples (e.g. fused qkv mappings)."""
    if isinstance(k, str):
        return k
    if isinstance(k, (tuple, list)) and k and isinstance(k[0], str):
        return k[0]
    return str(k)

_PROTECT = {
    "off":    dict(early=1.00, adaln=1.00),
    "normal": dict(early=0.40, adaln=1.00),
    "strong": dict(early=0.15, adaln=0.60),
}

_EARLY_FRACTION = 0.30
_TOTAL_CAP = 3.0
# soft normalization: mult = (median/rms)^0.5, clamped - nudges the stack
# toward balance without flipping intended conventions (e.g. sliders are
# deliberately quiet per unit strength, style loras deliberately loud)
_MULT_RANGE = (0.5, 2.0)
_MULT_GAMMA = 0.5


def _lora_list():
    return ["None"] + folder_paths.get_filename_list("loras")


def _load_loaded_patches(lora_path, model, clip):
    """Load a lora file and map it onto model/clip keys (comfy standard path)."""
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
    return loaded


def _analyze(loaded):
    """Per-key effect estimate + layer index; returns (info, rms, max_layer, rank)."""
    info = {}
    effs = []
    max_layer = 0
    rank = None
    for k, v in loaded.items():
        eff = None
        try:
            tensors = None
            if isinstance(v, tuple) and len(v) >= 2 and v[0] == "lora":
                tensors = v[1]                      # legacy tuple format
            elif hasattr(v, "weights") and getattr(v, "name", "") == "lora":
                tensors = v.weights                 # comfy weight_adapter.LoRAAdapter
            if tensors is not None:
                up, down = tensors[0], tensors[1]
                alpha = tensors[2] if len(tensors) > 2 else None
                r = down.shape[0] if hasattr(down, "shape") and len(down.shape) > 0 else 1
                rank = rank or int(r)
                scale = (float(alpha) / r) if alpha is not None else 1.0
                eff = float(up.float().norm().item() * down.float().norm().item()) * abs(scale)
                eff = eff / math.sqrt(max(1, up.numel() + down.numel()))
        except Exception:
            eff = None
        m = _LAYER_RE.search(_key_str(k))
        idx = int(m.group(1)) if m else None
        if idx is not None:
            max_layer = max(max_layer, idx)
        info[k] = (eff, idx)
        if eff is not None:
            effs.append(eff)
    rms = math.sqrt(sum(e * e for e in effs) / len(effs)) if effs else 0.0
    return info, rms, max_layer, rank


class ZPhotonLoraStack:
    """Model-aware LoRA stack with tensor analysis, auto-balancing and
    composition protection."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = _lora_list()
        req = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "balance": (["auto (recommended)", "off"], {"default": "auto (recommended)", "tooltip": "auto: normalize every LoRA's effective magnitude to the stack median, so strength 1.0 = one comparable unit of effect; total softly capped."}),
            "composition_protect": (list(_PROTECT.keys()), {"default": "normal", "tooltip": "Attenuate LoRA deltas on early layers (and adaLN in strong) - the layers that cause composition mutations. Combine with the sampler's clean_model input."}),
        }
        for i in range(1, 5):
            req[f"lora_{i}"] = (loras, {"default": "None"})
            req[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    FUNCTION = "load"
    CATEGORY = "ZPhoton"

    def load(self, model, clip, balance, composition_protect, **kw):
        entries = []
        for i in range(1, 5):
            name = kw.get(f"lora_{i}", "None")
            strength = float(kw.get(f"strength_{i}", 0.0))
            if name and name != "None" and abs(strength) > 1e-4:
                entries.append((name, strength))

        if not entries:
            return (model, clip, "ZPhoton LoRA Stack: empty (no LoRAs selected)")

        prot = _PROTECT[composition_protect]
        report = [f"ZPhoton LoRA Stack | balance={balance.split(' ')[0]} | protect={composition_protect}"]

        analyzed = []
        for name, strength in entries:
            path = folder_paths.get_full_path("loras", name)
            loaded = _load_loaded_patches(path, model, clip)
            info, rms, max_layer, rank = _analyze(loaded)
            analyzed.append(dict(name=name, strength=strength, loaded=loaded,
                                 info=info, rms=rms, max_layer=max_layer, rank=rank))

        # --- auto-balance: normalize each lora's magnitude to the stack median
        mults = [1.0] * len(analyzed)
        if balance.startswith("auto"):
            rms_vals = sorted(a["rms"] for a in analyzed if a["rms"] > 0)
            if rms_vals:
                median = rms_vals[len(rms_vals) // 2]
                for j, a in enumerate(analyzed):
                    if a["rms"] > 0:
                        m = (median / a["rms"]) ** _MULT_GAMMA
                        mults[j] = max(_MULT_RANGE[0], min(_MULT_RANGE[1], m))
            total = sum(abs(a["strength"]) * m for a, m in zip(analyzed, mults))
            if total > _TOTAL_CAP:
                scale = _TOTAL_CAP / total
                mults = [m * scale for m in mults]
                report.append(f"  total stack effect {total:.2f} > cap {_TOTAL_CAP}, all scaled x{scale:.2f}")

        new_model = model.clone()
        new_clip = clip.clone()

        for a, mult in zip(analyzed, mults):
            s = a["strength"] * mult
            max_layer = max(1, a["max_layer"])
            early_cut = _EARLY_FRACTION * max_layer

            base_keys, early_keys, adaln_keys, clip_keys = {}, {}, {}, {}
            for k, v in a["loaded"].items():
                _, idx = a["info"].get(k, (None, None))
                ks = _key_str(k)
                if not ks.startswith("diffusion_model."):
                    clip_keys[k] = v
                elif "adaLN" in ks and prot["adaln"] < 1.0:
                    adaln_keys[k] = v
                elif idx is not None and idx <= early_cut:
                    early_keys[k] = v
                else:
                    base_keys[k] = v

            patched = []
            if base_keys:
                patched += new_model.add_patches(base_keys, s)
            if early_keys:
                patched += new_model.add_patches(early_keys, s * prot["early"])
            if adaln_keys:
                patched += new_model.add_patches(adaln_keys, s * prot["early"] * prot["adaln"])
            cl = new_clip.add_patches(clip_keys, s) if clip_keys else []

            n_unet = len(base_keys) + len(early_keys) + len(adaln_keys)
            report.append(
                f"  {a['name'].split(chr(92))[-1]}: rank={a['rank']}, unet_keys={n_unet}, "
                f"clip_keys={len(clip_keys)}, rms={a['rms']:.4g}, mult=x{mult:.2f}, "
                f"applied={s:.2f} (early x{prot['early']:.2f})"
            )
            if n_unet and not patched:
                report.append("    WARNING: no unet keys matched the connected model!")

        text = "\n".join(report)
        logging.info(text)
        return (new_model, new_clip, text)


NODE_CLASS_MAPPINGS = {"ZPhotonLoraStack": ZPhotonLoraStack}
NODE_DISPLAY_NAME_MAPPINGS = {"ZPhotonLoraStack": "ZPhoton LoRA Stack (Auto)"}
