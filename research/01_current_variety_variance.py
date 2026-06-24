"""
Численная валидация математики variety для research-документа.
Чистый numpy (без torch) — проверяем формулы, не реальную модель.
Сиды фиксированы для воспроизводимости.
"""
import numpy as np

rng = lambda s: np.random.default_rng(s)

# ----------------------------------------------------------------------
# 1. ТЕКУЩИЙ ZPhoton: variance-preserving LF-blend и эффект lf_boost
# ----------------------------------------------------------------------
def low_freq(x, f=8):
    """area-average downsample by f, then bilinear-ish upsample (nearest-block).
       Воспроизводит avg_pool2d(kernel=f) + interpolate."""
    C, H, W = x.shape
    Hc, Wc = H // f * f, W // f * f
    xc = x[:, :Hc, :Wc]
    down = xc.reshape(C, Hc//f, f, Wc//f, f).mean(axis=(2,4))   # area average
    up = np.repeat(np.repeat(down, f, axis=1), f, axis=2)        # block upsample
    out = np.zeros_like(x)
    out[:, :Hc, :Wc] = up
    return out

def zphoton_variety_noise(shape, seed, vseed, v, boost):
    n = rng(seed).standard_normal(shape)
    lf1 = low_freq(n)
    nv = rng(vseed).standard_normal(shape)
    lf_mix = np.sqrt(1 - v*v) * lf1 + v * low_freq(nv)
    out = n - lf1 + lf_mix * boost      # текущая формула из noise.py
    return out, n

shape = (16, 128, 128)
print("="*70)
print("БЛОК 1: текущий ZPhoton variety — дисперсия входного шума")
print("="*70)
print(f"{'variety':>8} {'boost':>6} {'var(noise)':>12} {'отклон.от 1.0':>14}")
for name, v, boost in [("off",0.0,1.00),("low",0.45,1.00),("medium",0.75,1.12),("high",1.0,1.25)]:
    out, base = zphoton_variety_noise(shape, 42, 1234, v, boost)
    var = out.var()
    print(f"{name:>8} {boost:>6.2f} {var:>12.4f} {(var-1.0):>+14.4f}")
print("\nВывод: при boost=1 дисперсия ~1.0 (бел.шум сохранён).")
print("При boost>1 (medium/high) дисперсия раздута -> вход НЕ N(0,1) -> OOD для модели.")

# Спектральная локализация раздутия: считаем энергию в LF-полосе
print("\nЭнергия по полосам (LF = первые 1/8 частот, доля):")
def lf_energy_fraction(x, f=8):
    C,H,W = x.shape
    E = np.abs(np.fft.fft2(x, axes=(1,2)))**2
    Etot = E.sum()
    cutH, cutW = H//f, W//f
    # низкие частоты — углы спектра (без fftshift): берём блок [0:cut] по каждой оси и зеркала
    mask = np.zeros((H,W), bool)
    mask[:cutH,:cutW]=True; mask[-cutH:,:cutW]=True; mask[:cutH,-cutW:]=True; mask[-cutH:,-cutW:]=True
    Elf = (E * mask[None]).sum()
    return Elf/Etot
for name, v, boost in [("off",0.0,1.00),("high",1.0,1.25)]:
    out,_ = zphoton_variety_noise(shape, 42, 1234, v, boost)
    print(f"  {name:>6}: LF-доля энергии = {lf_energy_fraction(out):.4f}")
