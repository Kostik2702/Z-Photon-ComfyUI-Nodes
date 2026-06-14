# ComfyUI-ZPhoton

Photorealism-focused sampling nodes for **Z-Image / Z-Image-Turbo** (Tongyi-MAI).
Inspired by martin-rizzo's Z-Image Power Nodes, rebuilt on analytic math instead of
hand-tuned sigma tables, with first-class support for the **Z-Image Base** model.

## Nodes

- **ZPhoton Sampler** — all-in-one: analytic schedule (turbo/base), restart refine pass,
  detail boost (sigma nudge), noise contrast, low-frequency seed variation.
- **ZPhoton Sampler (Advanced)** — same engine, SIGMAS input, every knob exposed.
- **ZPhoton Scheduler** — the schedule family as a SIGMAS node (restart encoded as
  ascending jump; built-in SamplerCustom will NOT understand it — use ZPhoton samplers).
- **ZPhoton Empty Latent** — photo aspect ratios / sizes for Z-Image (16ch).
- **ZPhoton Refiner** — hires-fix second pass: latent upscale + flow re-noise + detail descent.
- **ZPhoton Photo Style** — curated photographic style snippets (string in/out).

## Usage (v2: presets)

**ZPhoton Sampler** now has just three controls: `seed`, `preset`, `variety`.

Presets: `turbo / fast (6)`, `turbo / balanced (9)`, `turbo / quality (14)`, `turbo / max (20)` —
for Z-Image-Turbo and turbo merges (cfg 1, negative ignored); `base / quality (28)`,
`base / max (36)` — for Z-Image Base (cfg 4, connect a negative).
Variety: `off / low / medium / high` — composition diversity via variance-preserving
low-frequency seed blend (style and texture stay intact).

**LoRA phase scheduling (anti-mutation):** connect the checkpoint output (before LoRA
loaders) to the sampler's `clean_model` input, and the LoRA-patched model to `model`.
Composition (sigma > 0.85) runs on the clean model, LoRAs apply only in the
identity/detail phase. Validated: prompt adherence and framing survive a 3-LoRA stack
that otherwise overrides them; style/identity still lands.

**ZPhoton Encode:** Qwen3 encoder with a curated photographic system prompt
("photo (recommended)") + optional style. Outputs conditioning and the final text.

**Negative on Turbo:** connect a negative to a turbo preset and the sampler runs
true CFG 2.5 on the structure phase only (sigma > 0.7), then cfg 1. Validated limits:
good for suppressing artifacts/styles and nudging away from defaults; it can NOT
flip dominant attributes (e.g. hair color) - guidance distillation leaves almost no
cond/uncond contrast in Turbo. Roadmap: attention-level NAG for the Lumina arch.

**ZPhoton AutoQC:** quality gate between VAEDecode and the upscaler. Calibrated
thresholds (validated on real outputs): residual noise > 1.5 = speckle failure,
sharpness < 40 = degenerate frame, face confidence via face_yolov8m. Connect
model/positive/latent/vae to enable auto re-refine retries on failure.

Recommended pipeline (validated against [PHOTOREAL]-class workflows):
checkpoint (e.g. fascium turbo merge) → CLIPTextEncodeLumina2 (system prompt) →
ZPhoton Empty Latent → **ZPhoton Sampler** → VAEDecode → SeedVR2 (7b sharp) → ColorMatch.
Skip the latent Refiner in this pipeline — SeedVR2 handles sharpness better; the Refiner
remains for img2img/special cases.

## How it works (short)

- sigma(t) = shift·t/(1+(shift−1)·t); turbo schedule = dense high-sigma structure steps +
  one plunge to 0 + restart refine segment (re-noise to ~0.66, linear descent) — the shape
  of the best hand-tuned community presets, generalized to any step count.
- detail = step-relative sigma nudge: the model is queried at sigma − a·env(p)·(sigma − sigma_next).
  Bounded by the next state, so the bias can never accumulate into speckle artifacts;
  it is skipped on plunge steps and on the final step of each segment.
- variation = blending the low-frequency component of a second seed's noise: composition
  changes, texture statistics stay — no scrambling hallucinations.
- restart jumps are honest flow re-noise: x ← (1−σ)x + σ·ε.
