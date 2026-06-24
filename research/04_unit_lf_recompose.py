"""
Unit-тест МАТЕМАТИКИ lf_recompose (numpy-эквивалент torch-модуля variety.py).
Проверяет 5 гарантий из плана, Шаг 2. Сиды фиксированы.
torch в песочнице нет -> тестируем эквивалентную numpy-реализацию тех же формул;
сам torch-модуль отдельно проходит py_compile в среде ComfyUI.
"""
import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq

def lf_mask(h,w,cutoff):
    fy=fftfreq(h)[:,None]; fx=rfftfreq(w)[None,:]; r=np.sqrt(fy*fy+fx*fx)
    return (r<=cutoff*0.5).astype(np.float64)

def split(x,m):
    X=rfft2(x,norm="backward"); lf=irfft2(X*m,s=x.shape[-2:],norm="backward"); return lf, x-lf

def lf_recompose(x, seed_v, a, cutoff=0.25):
    a=float(max(0.0,min(1.0,a)))
    if a<=0.0: return x.copy()
    b,c,h,w=x.shape
    m=lf_mask(h,w,cutoff)
    lf,hf=split(x,m)
    g=np.empty_like(x)
    for i in range(b):
        rng=np.random.default_rng((seed_v + 0x9E3779B9*i)&0xffffffffffffffff)
        g[i]=rng.standard_normal((c,h,w))
    g,_=split(g,m)
    lfn=np.linalg.norm(lf.reshape(b,c,-1),axis=2).reshape(b,c,1,1)
    gn=np.clip(np.linalg.norm(g.reshape(b,c,-1),axis=2).reshape(b,c,1,1),1e-6,None)
    ghat=g*(lfn/gn)
    lf_new=np.sqrt(1-a*a)*lf+a*ghat
    y=lf_new+hf
    ym=y.mean(axis=(2,3),keepdims=True); ys=np.clip(y.std(axis=(2,3),keepdims=True),1e-6,None)
    xm=x.mean(axis=(2,3),keepdims=True); xs=x.std(axis=(2,3),keepdims=True)
    return (y-ym)/ys*xs+xm

# натуральное латент-подобное поле
def field(seed,B=2,C=16,H=64,W=64,alpha=1.6):
    fy=fftfreq(H)[:,None]; fx=fftfreq(W)[None,:]; r=np.sqrt(fy**2+fx**2); r[0,0]=1e-3
    amp=1/r**alpha; rng=np.random.default_rng(seed)
    x=np.empty((B,C,H,W))
    for bi in range(B):
        ph=rng.uniform(0,2*np.pi,(C,H,W)); xi=np.real(np.fft.ifft2(amp[None]*np.exp(1j*ph),axes=(1,2)))
        xi=(xi-xi.mean(axis=(1,2),keepdims=True))/xi.std(axis=(1,2),keepdims=True)
        x[bi]=xi*0.9+rng.uniform(-1.5,1.5,(C,1,1))
    return x

def bandcorr(a_,b_,m):
    _,ha=split(a_,m); _,hb=split(b_,m); return np.corrcoef(ha.ravel(),hb.ravel())[0,1]

PASS=True
def check(name, cond, detail=""):
    global PASS; ok="PASS" if cond else "FAIL"; PASS = PASS and cond
    print(f"  [{ok}] {name} {detail}")

x=field(7); m=lf_mask(64,64,0.25)
print("Тест 1: a=0 => выход идентичен входу")
y0=lf_recompose(x,seed_v=11,a=0.0)
check("bitwise equal", np.allclose(y0,x,atol=0,rtol=0))

print("Тест 2: сохранение поканальных mean/std")
for a in (0.3,0.55,0.8):
    y=lf_recompose(x,seed_v=11,a=a)
    dmu=np.abs(y.mean(axis=(2,3))-x.mean(axis=(2,3))).max()
    dsd=np.abs(y.std(axis=(2,3))-x.std(axis=(2,3))).max()
    check(f"a={a}: mean/std", dmu<1e-5 and dsd<1e-5, f"(dμ={dmu:.1e}, dσ={dsd:.1e})")

print("Тест 3: ВЧ-текстура сохранена (corr>0.98 после style-renorm; =1.000 без renorm)")
# renorm — часть оператора (сохраняет стиль) — слегка масштабирует поле целиком,
# поэтому HF corr чуть ниже 1.0. Это ожидаемо; критерий — сохранность текстуры >0.98.
for a in (0.3,0.55,0.8):
    y=lf_recompose(x,seed_v=11,a=a)
    hc=bandcorr(y,x,m)
    check(f"a={a}: HF corr", hc>0.98, f"(corr={hc:.4f})")

print("Тест 4: эффективная decorrelation НЧ монотонна, raw=sqrt(1-a²)")
prev=2.0
for a in (0.3,0.55,0.8):
    y=lf_recompose(x,seed_v=11,a=a)
    lfx,_=split(x,m); lfy,_=split(y,m)
    cc=np.corrcoef(lfy.ravel(),lfx.ravel())[0,1]; exp=np.sqrt(1-a*a)
    check(f"a={a}: LF corr монотонна, ≥sqrt(1-a²)-0.02", cc<prev and cc>=exp-0.02,
          f"(eff.corr={cc:.3f}, raw=sqrt(1-a²)={exp:.3f})")
    prev=cc

print("Тест 5: в батче кадры различаются (сид сдвигается по индексу)")
y=lf_recompose(x,seed_v=11,a=0.6)
lf0,_=split(y[0:1],m); lf1,_=split(y[1:2],m)
diff=np.linalg.norm(lf0-lf1)/np.linalg.norm(lf0)
check("batch diversity", diff>0.1, f"(отн.разн.НЧ={diff:.2f})")

print("\nИТОГ:", "ВСЕ ТЕСТЫ ПРОЙДЕНЫ" if PASS else "ЕСТЬ ПРОВАЛЫ")
import sys; sys.exit(0 if PASS else 1)
