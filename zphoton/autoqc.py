"""
ZPhoton AutoQC - automatic quality control with optional auto-fix.

Metrics (calibrated on real good/bad Z-Image outputs):
  * residual noise: mean 3x3 high-pass residual inside flat regions
    (flattest 30% of 8x8 blocks).  Good images: 0.5-0.9; speckled/noisy
    failures: > 1.5 (default threshold).
  * sharpness: Laplacian variance on ~1MP luma.  Good: 70-450;
    degenerate/blurry failures: < 40 (default threshold).
  * face confidence (optional): best face score from
    models/ultralytics/bbox/face_yolov8m.pt if ultralytics is available.

Auto-fix: when model/positive/latent/vae are connected and the check fails,
the latent is re-refined (flow re-noise to a mid sigma with a fresh seed and
descended again), re-decoded and re-checked, up to `max_retries` times.
The best-scoring attempt is returned - "never ship a bad frame".
"""
import logging

import torch
import torch.nn.functional as F

import folder_paths
from .schedules import build_refiner_sigmas
from .sampling import run_sampling

MAX_SEED = 0xffffffffffffffff
_face_model = None


def _luma_1mp(img):
    """img: (H,W,C) float 0..1 -> (1,1,h,w) luma 0..255 at <= ~1MP."""
    x = img[..., :3].permute(2, 0, 1).unsqueeze(0)
    lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255.0
    h, w = lum.shape[-2:]
    scale = (1_000_000 / max(1, h * w)) ** 0.5
    if scale < 1.0:
        lum = F.interpolate(lum, size=(int(h * scale), int(w * scale)),
                            mode="area")
    return lum


def _metrics(img):
    lum = _luma_1mp(img.float().cpu())
    # sharpness: Laplacian variance
    k = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).view(1, 1, 3, 3)
    lap = F.conv2d(lum, k)
    sharpness = float(lap.var().item())
    # noise: 3x3 high-pass residual in flat 8x8 blocks
    blur = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    resid = (lum - blur).abs()
    bs = 8
    h, w = lum.shape[-2:]
    hc, wc = h // bs * bs, w // bs * bs
    blocks_std = lum[..., :hc, :wc].reshape(1, 1, hc // bs, bs, wc // bs, bs).std(dim=(3, 5))
    blocks_res = resid[..., :hc, :wc].reshape(1, 1, hc // bs, bs, wc // bs, bs).mean(dim=(3, 5))
    thr = torch.quantile(blocks_std.flatten(), 0.30)
    flat = blocks_std <= thr
    noise = float(blocks_res[flat].median().item()) if flat.any() else float(blocks_res.mean().item())
    return sharpness, noise


def _face_conf(img):
    """Best face confidence via ultralytics, or None if unavailable."""
    global _face_model
    try:
        if _face_model is None:
            import os
            from ultralytics import YOLO
            path = None
            for getter in (lambda: folder_paths.get_full_path("ultralytics_bbox", "face_yolov8m.pt"),
                           lambda: folder_paths.get_full_path("ultralytics", "bbox/face_yolov8m.pt")):
                try:
                    path = getter()
                except Exception:
                    path = None
                if path:
                    break
            if not path:
                cand = os.path.join(folder_paths.models_dir, "ultralytics", "bbox", "face_yolov8m.pt")
                path = cand if os.path.exists(cand) else None
            if path is None:
                return None
            _face_model = YOLO(path)
        import numpy as np
        a = (img[..., :3].float().cpu().numpy() * 255).astype("uint8")
        res = _face_model(a, verbose=False, imgsz=960)
        confs = [float(c) for r in res for c in r.boxes.conf]
        return max(confs) if confs else 0.0
    except Exception as e:
        logging.info(f"[ZPhoton AutoQC] face check unavailable: {e}")
        return None


class ZPhotonAutoQC:
    """Quality gate: noise / sharpness / face checks with optional automatic
    re-refine retries ("never ship a bad frame")."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_noise": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 10.0, "step": 0.05, "tooltip": "Residual noise threshold. Good Z-Image frames: 0.5-0.9, speckled failures: >1.5."}),
                "min_sharpness": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 500.0, "step": 1.0, "tooltip": "Laplacian-variance threshold. Good frames: 70-450, degenerate blurs: <40."}),
                "check_face": ("BOOLEAN", {"default": True, "tooltip": "Require a confident face detection (face_yolov8m). Disable for faceless scenes."}),
                "min_face_conf": ("FLOAT", {"default": 0.55, "min": 0.1, "max": 0.95, "step": 0.01}),
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5, "tooltip": "Auto-fix attempts (needs model/positive/latent/vae connected)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("images", "latent", "passed", "report")
    FUNCTION = "check"
    CATEGORY = "ZPhoton"

    def _evaluate(self, img, max_noise, min_sharpness, check_face, min_face_conf):
        sharp, noise = _metrics(img)
        face = _face_conf(img) if check_face else None
        fails = []
        if noise > max_noise:
            fails.append(f"noise {noise:.2f} > {max_noise}")
        if sharp < min_sharpness:
            fails.append(f"sharpness {sharp:.0f} < {min_sharpness:.0f}")
        if check_face and face is not None and face < min_face_conf:
            fails.append(f"face_conf {face:.2f} < {min_face_conf}")
        # composite score: higher is better
        score = min(sharp / max(min_sharpness, 1e-3), 5.0) - 2.0 * max(0.0, noise - max_noise)
        if face is not None:
            score += face
        return dict(sharp=sharp, noise=noise, face=face, fails=fails, score=score)

    def check(self, images, max_noise, min_sharpness, check_face, min_face_conf,
              max_retries, seed, model=None, positive=None, negative=None,
              latent=None, vae=None):
        report = ["ZPhoton AutoQC"]
        cur_images, cur_latent = images, latent
        best = None

        can_fix = all(x is not None for x in (model, positive, latent, vae)) and max_retries > 0

        for attempt in range(max_retries + 1):
            ev = self._evaluate(cur_images[0], max_noise, min_sharpness,
                                check_face, min_face_conf)
            tag = "initial" if attempt == 0 else f"retry {attempt}"
            face_s = "n/a" if ev["face"] is None else f"{ev['face']:.2f}"
            report.append(f"  {tag}: sharp={ev['sharp']:.0f} noise={ev['noise']:.2f} "
                          f"face={face_s} -> {'PASS' if not ev['fails'] else 'FAIL: ' + '; '.join(ev['fails'])}")
            if best is None or ev["score"] > best[0]:
                best = (ev["score"], cur_images, cur_latent, not ev["fails"])
            if not ev["fails"] or not can_fix or attempt == max_retries:
                break

            # auto-fix: re-refine the latent with a fresh restart seed
            denoise = 0.22 + 0.05 * attempt
            sigmas = build_refiner_sigmas(5, denoise, shift=3.0)
            retry_seed = (int(seed) + 0x9E37 * (attempt + 1)) & MAX_SEED
            noise_t = torch.randn(cur_latent["samples"].shape, device="cpu", dtype=torch.float32,
                                  generator=torch.Generator(device="cpu").manual_seed(retry_seed))
            cur_latent = run_sampling(model, positive, negative, cur_latent, sigmas,
                                      seed=retry_seed, cfg=1.0, noise=noise_t,
                                      detail_amount=0.25, detail_start=0.0,
                                      detail_end=0.85, detail_peak=0.45, order=1)
            cur_images = vae.decode(cur_latent["samples"])

        _, out_images, out_latent, passed = best
        if out_latent is None:
            out_latent = {"samples": torch.zeros(1, 16, 8, 8)}
        text = "\n".join(report)
        logging.info(text)
        return (out_images, out_latent, passed, text)


NODE_CLASS_MAPPINGS = {"ZPhotonAutoQC": ZPhotonAutoQC}
NODE_DISPLAY_NAME_MAPPINGS = {"ZPhotonAutoQC": "ZPhoton AutoQC"}
