import numpy as np
rng = lambda s: np.random.default_rng(s)

print("="*70)
print("БЛОК 2: variance-preserving смешивание (теорема корректности)")
print("="*70)
# Утверждение: для независимых X,Y ~ N(0,sigma^2),
#   Z = sqrt(1-a^2) X + a Y  =>  Var(Z) = sigma^2 (точно, при любом a в [0,1])
# Проверим эмпирически на полном поле (не только LF), и корреляцию Z с X.
N = 2_000_000
for a in [0.0, 0.3, 0.45, 0.707, 0.85, 1.0]:
    X = rng(1).standard_normal(N)
    Y = rng(2).standard_normal(N)
    Z = np.sqrt(1-a*a)*X + a*Y
    corr = np.corrcoef(Z, X)[0,1]
    print(f"  a={a:>5.3f}: Var(Z)={Z.var():.4f}  Corr(Z,X)={corr:.4f}  (ожид corr=sqrt(1-a^2)={np.sqrt(1-a*a):.4f})")
print("\nВывод: смешивание sqrt(1-a^2)X + aY сохраняет дисперсию ТОЧНО при любом a.")
print("Это корректная основа. Проблема НЕ в нём, а в множителе boost ПОСЛЕ него.")

print()
print("="*70)
print("БЛОК 3: PN scramble_tensor — сохранение поканальной статистики")
print("="*70)
def interp_bilinear(frag, H, W):
    # простой numpy bilinear resize фрагмента (C,h,w)->(C,H,W)
    C,h,w = frag.shape
    ys = (np.linspace(0, h-1, H))
    xs = (np.linspace(0, w-1, W))
    y0 = np.floor(ys).astype(int); y1=np.minimum(y0+1,h-1); wy=(ys-y0)
    x0 = np.floor(xs).astype(int); x1=np.minimum(x0+1,w-1); wx=(xs-x0)
    a = frag[:,y0][:,:,x0]; b=frag[:,y0][:,:,x1]; c=frag[:,y1][:,:,x0]; d=frag[:,y1][:,:,x1]
    top = a*(1-wx)[None,None,:]+b*wx[None,None,:]
    bot = c*(1-wx)[None,None,:]+d*wx[None,None,:]
    return top*(1-wy)[None,:,None]+bot*wy[None,:,None]

def random_fragment(x, g, size=(0.5,0.75), anchor='left', flip=False):
    C,H,W = x.shape
    ratio = g.uniform(size[0], size[1])
    fw, fh = int(W*ratio), int(H*ratio)
    if anchor in ('left','right'):
        top = g.integers(0, H-fh+1); left = 0 if anchor=='left' else W-fw
    else:
        left = g.integers(0, W-fw+1); top = 0 if anchor=='top' else H-fh
    frag = x[:, top:top+fh, left:left+fw]
    if flip and g.uniform()>0.5: frag=frag[:,:,::-1]
    if flip and g.uniform()>0.5: frag=frag[:,::-1,:]
    return interp_bilinear(np.ascontiguousarray(frag), H, W)

def scramble(x, counts, seed):
    g = rng(seed)
    names=('left','top','right','bottom')
    if not any(counts): return x
    x_scale = x.std(axis=(1,2), keepdims=True)
    x_bias  = x.mean(axis=(1,2), keepdims=True)
    result = np.zeros_like(x)
    for ai in range(4):
        for _ in range(abs(counts[ai])):
            result += random_fragment(x, g, anchor=names[ai], flip=counts[ai]<0)
    r_scale=result.std(axis=(1,2),keepdims=True); r_bias=result.mean(axis=(1,2),keepdims=True)
    sf = x_scale/np.clip(r_scale,1e-6,None)
    return result*sf + (x_bias - r_bias*sf)

# латент-подобное поле: коррелированный (НЕ белый), как реальный латент на mid-sigma
def correlated_field(seed, C=16, H=64, W=64, f=6):
    n = rng(seed).standard_normal((C,H,W))
    # сглаживание для пространственной корреляции
    from numpy.fft import fft2, ifft2, fftfreq
    fy=fftfreq(H)[:,None]; fx=fftfreq(W)[None,:]
    flt = np.exp(-(fy**2+fx**2)*(f**2)*20)
    out = np.real(ifft2(fft2(n,axes=(1,2))*flt[None],axes=(1,2)))
    return (out-out.mean(axis=(1,2),keepdims=True))/out.std(axis=(1,2),keepdims=True)*0.8 + rng(seed+9).uniform(-2,2,(C,1,1))

x = correlated_field(7)
print(f"{'counts':>16} {'max|dμ|':>10} {'max|dσ|':>10} {'структ.сдвиг':>14}")
for counts in [(1,0,1,0),(2,-1,2,-1),(-2,-2,-2,-2)]:
    xs = scramble(x, counts, seed=123)
    dmu = np.abs(xs.mean(axis=(1,2)) - x.mean(axis=(1,2))).max()
    dsd = np.abs(xs.std(axis=(1,2)) - x.std(axis=(1,2))).max()
    # структурный сдвиг: нормированная L2-разница полей
    struct = np.linalg.norm(xs-x)/np.linalg.norm(x)
    print(f"{str(counts):>16} {dmu:>10.2e} {dsd:>10.2e} {struct:>14.3f}")
print("\nВывод: scramble сохраняет поканальные mean/std (dμ,dσ ~ 1e-7),")
print("то есть глобальную цвето-яркостную статистику (=стиль/палитру),")
print("но СИЛЬНО меняет структуру поля (struct ~ 1.0+) = композицию.")
