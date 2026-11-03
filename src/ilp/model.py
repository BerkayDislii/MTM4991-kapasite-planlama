"""Çok dönemli kapasite planlama ILP'si (MIMARI.md Bölüm 6, denklemler (1)–(6)).

    min  Σ_t Σ_k c_k x_kt + s_k y_kt + d_k z_kt                                   (1)
    s.t. Σ_k ρ^cpu C_k x_kt ≥ (1+α) D̂^cpu_t                       ∀t              (2)
         Σ_k ρ^mem M_k x_kt ≥ (1+α) D̂^mem_t                       ∀t              (3)
         x_kt − x_k,t−1 = y_kt − z_kt,  x_k0 = x0_k                ∀k,t            (4)
         Σ_{τ=max(1,t−L_k+1)}^{t} y_kτ + Σ_{τ=t−L_k+1}^{0} ŷ_kτ ≤ x_kt   ∀k,t      (5)
         x, y, z ∈ Z≥0                                                             (6)

Dizinler kodda 0 tabanlıdır: t = 0..H−1. Önceki günden devralınan başlatmalar ŷ, önceki planın y dizisidir
(son sütunu τ = 0, yani önceki günün son dilimi).
"""
import math
from dataclasses import dataclass, field

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from src.common.config import yaml_oku


@dataclass(frozen=True)
class SunucuTipleri:
    """K sunucu tipinin vektörleri: ad, çekirdek (vCPU) C, bellek M (GiB), saatlik maliyet c (USD)."""
    ad: tuple[str, ...]
    C: np.ndarray
    M: np.ndarray
    c: np.ndarray

    @property
    def K(self) -> int:
        return len(self.ad)


@dataclass(frozen=True)
class IlpParam:
    """Model parametreleri (ilp.yaml model bölümü). L ve s, d tip başına vektördür."""
    alpha: float
    s: np.ndarray                  # açma maliyeti s_k
    d: np.ndarray                  # kapatma maliyeti d_k
    L: np.ndarray                  # minimum çalışma süresi L_k (≥ 1)
    rho_cpu: float = 1.0
    rho_mem: float = 1.0
    N: np.ndarray | None = None    # isteğe bağlı filo üst sınırı


@dataclass
class Plan:
    """Bir ufkun çözümü: (K, H) tamsayı dizileri ve amaç değeri."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    amac: float
    durum: str = "OPTIMAL"
    bilgi: dict = field(default_factory=dict)


def tipleri_oku() -> SunucuTipleri:
    """configs/server_types.yaml'dan sunucu tipleri (ilp.yaml'daki sırayla)."""
    st, sec = yaml_oku("server_types"), yaml_oku("ilp")["sunucu"]["tipler"]
    tip = {t["ad"]: t for t in st["tipler"]}
    assert list(tip) == sec, f"server_types.yaml {list(tip)} ≠ ilp.yaml {sec}: fiyat_cek.py'yi yeniden çalıştır"
    return SunucuTipleri(tuple(sec), np.array([tip[a]["vcpu"] for a in sec], float),
                         np.array([tip[a]["bellek_gib"] for a in sec], float),
                         np.array([tip[a]["saatlik_usd"] for a in sec], float))


def param_oku(tipler: SunucuTipleri, **degistir) -> IlpParam:
    """ilp.yaml model bölümünden IlpParam (s_k = c_k·w, d_k = c_k·w_kapatma). degistir: üzerine yazılacak anahtarlar."""
    m = {**yaml_oku("ilp")["model"], **degistir}
    assert m["commit_h"] == m["horizon_h"], "commit_h ≠ horizon_h henüz desteklenmiyor"
    N = m["filo_ust_siniri"]
    return IlpParam(alpha=float(m["alpha"]), s=tipler.c * m["w"], d=tipler.c * m["w_kapatma"],
                    L=np.full(tipler.K, int(m["L"])), rho_cpu=float(m["rho_cpu"]), rho_mem=float(m["rho_mem"]),
                    N=None if N is None else np.broadcast_to(np.asarray(N, float), (tipler.K,)).copy())


def devralinan(y_onceki: np.ndarray | None, L: np.ndarray, H: int) -> np.ndarray:
    """(5)'in ikinci toplamı: (K, H) dizisi; [k, t] = önceki ufukta τ ∈ [t−L_k+1, 0] diliminde açılan sayı."""
    K = len(L)
    out = np.zeros((K, H))
    if y_onceki is None:
        return out
    Hp = y_onceki.shape[1]
    for k in range(K):
        for t in range(min(H, int(L[k]) - 1)):
            # 1 tabanlı t' = t+1; τ ∈ [t'−L+1, 0] → önceki y sütunları Hp−1+τ
            bas = Hp - 1 + (t + 1 - int(L[k]) + 1)
            out[k, t] = y_onceki[k, max(0, bas):Hp].sum()
    return out


def coz(D_cpu: np.ndarray, D_mem: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
        x0: np.ndarray | None = None, y_onceki: np.ndarray | None = None,
        cozucu: str = "SCIP", zaman_siniri_s: float = 60.0, lp: bool = False) -> Plan:
    """(1)–(6)'yı pywraplp ile çözer. lp=True: LP gevşetmesi (GLOP). OPTIMAL değilse hata verir."""
    H, K = len(D_cpu), tipler.K
    x0 = np.zeros(K) if x0 is None else np.asarray(x0, float)
    yd = devralinan(y_onceki, p.L, H)
    s = pywraplp.Solver.CreateSolver("GLOP" if lp else cozucu)
    assert s is not None, f"{cozucu} çözücüsü yok"
    s.SetTimeLimit(int(zaman_siniri_s * 1000))
    yeni = s.NumVar if lp else s.IntVar
    ust = [s.infinity() if p.N is None else float(p.N[k]) for k in range(K)]
    x = [[yeni(0, ust[k], f"x{k}_{t}") for t in range(H)] for k in range(K)]
    y = [[yeni(0, s.infinity(), f"y{k}_{t}") for t in range(H)] for k in range(K)]
    z = [[yeni(0, s.infinity(), f"z{k}_{t}") for t in range(H)] for k in range(K)]
    for t in range(H):
        s.Add(sum(p.rho_cpu * tipler.C[k] * x[k][t] for k in range(K)) >= (1 + p.alpha) * D_cpu[t])   # (2)
        s.Add(sum(p.rho_mem * tipler.M[k] * x[k][t] for k in range(K)) >= (1 + p.alpha) * D_mem[t])   # (3)
        for k in range(K):
            onceki = x0[k] if t == 0 else x[k][t - 1]
            s.Add(x[k][t] - onceki == y[k][t] - z[k][t])                                              # (4)
            L = int(p.L[k])
            s.Add(sum(y[k][tau] for tau in range(max(0, t - L + 1), t + 1)) + yd[k, t] <= x[k][t])    # (5)
    s.Minimize(sum(tipler.c[k] * x[k][t] + p.s[k] * y[k][t] + p.d[k] * z[k][t]
                   for k in range(K) for t in range(H)))                                              # (1)
    durum = s.Solve()
    if durum != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Çözüm durumu OPTIMAL değil: {durum}")
    al = (lambda v: v.solution_value()) if lp else (lambda v: round(v.solution_value()))
    X, Y, Z = (np.array([[al(v) for v in satir] for satir in a], float) for a in (x, y, z))
    if not lp:
        X, Y, Z = acma_kapama_duzelt(X, Y, Z, x0, p)
    return Plan(X, Y, Z, amac=float(s.Objective().Value()), bilgi={"sure_ms": s.wall_time(),
                                                                    "n_dugum": s.nodes() if not lp else 0})


def acma_kapama_duzelt(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, x0: np.ndarray, p: IlpParam):
    """Bölüm 6.3: s_k = d_k = 0 olan tiplerde y, z belirsizdir → y = max(0, Δx), z = max(0, −Δx)."""
    fark = np.diff(np.concatenate([x0[:, None], X], axis=1), axis=1)
    sifir = (p.s + p.d) == 0
    Y, Z = Y.copy(), Z.copy()
    Y[sifir], Z[sifir] = np.maximum(fark[sifir], 0), np.maximum(-fark[sifir], 0)
    return X, Y, Z


def amac_hesapla(plan: Plan, tipler: SunucuTipleri, p: IlpParam) -> float:
    """(1) amacını plan dizilerinden yeniden hesaplar."""
    return float((tipler.c[:, None] * plan.x + p.s[:, None] * plan.y + p.d[:, None] * plan.z).sum())


def dogrula(plan: Plan, D_cpu: np.ndarray, D_mem: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
            x0: np.ndarray | None = None, y_onceki: np.ndarray | None = None, tol: float = 1e-6) -> list[str]:
    """(2)–(6)'yı modelden bağımsız denetler; ihlal açıklamalarının listesini döndürür (boş = geçerli)."""
    K, H = plan.x.shape
    x0 = np.zeros(K) if x0 is None else np.asarray(x0, float)
    ihlal = []
    for ad, a in (("x", plan.x), ("y", plan.y), ("z", plan.z)):
        if (a < -tol).any() or not np.allclose(a, np.round(a)):
            ihlal.append(f"(6) {ad} negatif veya tamsayı değil")
    kap_cpu = (p.rho_cpu * tipler.C[:, None] * plan.x).sum(0)
    kap_mem = (p.rho_mem * tipler.M[:, None] * plan.x).sum(0)
    for t in np.where(kap_cpu < (1 + p.alpha) * D_cpu - tol)[0]:
        ihlal.append(f"(2) t={t}: {kap_cpu[t]:.3f} < {(1 + p.alpha) * D_cpu[t]:.3f}")
    for t in np.where(kap_mem < (1 + p.alpha) * D_mem - tol)[0]:
        ihlal.append(f"(3) t={t}: {kap_mem[t]:.3f} < {(1 + p.alpha) * D_mem[t]:.3f}")
    onceki = np.concatenate([x0[:, None], plan.x[:, :-1]], axis=1)
    if not np.allclose(plan.x - onceki, plan.y - plan.z, atol=tol):
        ihlal.append("(4) durum geçişi sağlanmıyor")
    # (5): son L_k dilimdeki açılışlar (önceki ufuktan devralınanlar dahil) hâlâ çalışıyor olmalı
    if y_onceki is None:
        y_onceki = np.zeros((K, 0))
    y_tum = np.concatenate([y_onceki, plan.y], axis=1)
    o = y_onceki.shape[1]
    for k in range(K):
        L = int(p.L[k])
        for t in range(H):
            acik = y_tum[k, max(0, o + t - L + 1): o + t + 1].sum()
            if acik > plan.x[k, t] + tol:
                ihlal.append(f"(5) k={k} t={t}: son {L} dilimde açılan {acik:.0f} > x={plan.x[k, t]:.0f}")
    if p.N is not None and (plan.x > p.N[:, None] + tol).any():
        ihlal.append("filo üst sınırı aşıldı")
    return ihlal


def coz_cpsat(D_cpu: np.ndarray, D_mem: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
              olcek: int = 10_000_000, ust: int = 10_000, zaman_siniri_s: float = 60.0) -> Plan:
    """Doğrulama amaçlı CP-SAT çözümü (Bölüm 1): maliyetler olcek ile tamsayıya ölçeklenir.

    Kapasiteler tamsayı ve ρ = 1 iken sol taraf tamsayıdır, bu yüzden sağ taraf ceil((1+α)D) ile tam eşdeğerdir.
    Yalnızca x0 = 0 ve devralınan başlatma yokken (küçük test örnekleri) kullanılır.
    """
    assert p.rho_cpu == p.rho_mem == 1.0 and np.allclose(tipler.C, np.round(tipler.C)) \
        and np.allclose(tipler.M, np.round(tipler.M)), "CP-SAT için tamsayı kapasite ve ρ = 1 gerekir"
    H, K = len(D_cpu), tipler.K
    m = cp_model.CpModel()
    x = [[m.NewIntVar(0, ust, "") for _ in range(H)] for _ in range(K)]
    y = [[m.NewIntVar(0, ust, "") for _ in range(H)] for _ in range(K)]
    z = [[m.NewIntVar(0, ust, "") for _ in range(H)] for _ in range(K)]
    C, M = tipler.C.astype(int), tipler.M.astype(int)
    for t in range(H):
        m.Add(sum(int(C[k]) * x[k][t] for k in range(K)) >= math.ceil((1 + p.alpha) * D_cpu[t] - 1e-9))
        m.Add(sum(int(M[k]) * x[k][t] for k in range(K)) >= math.ceil((1 + p.alpha) * D_mem[t] - 1e-9))
        for k in range(K):
            m.Add(x[k][t] - (x[k][t - 1] if t else 0) == y[k][t] - z[k][t])
            L = int(p.L[k])
            m.Add(sum(y[k][tau] for tau in range(max(0, t - L + 1), t + 1)) <= x[k][t])
    tam = lambda v: int(round(v * olcek))
    assert all(abs(v * olcek - round(v * olcek)) < 1e-6 for v in np.concatenate([tipler.c, p.s, p.d])), \
        "Maliyetler olcek ile tam ölçeklenmiyor"
    m.Minimize(sum(tam(tipler.c[k]) * x[k][t] + tam(p.s[k]) * y[k][t] + tam(p.d[k]) * z[k][t]
                   for k in range(K) for t in range(H)))
    cs = cp_model.CpSolver()
    cs.parameters.max_time_in_seconds = zaman_siniri_s
    cs.parameters.num_workers = 8
    durum = cs.Solve(m)
    if durum != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT durumu OPTIMAL değil: {cs.StatusName(durum)}")
    X, Y, Z = (np.array([[cs.Value(v) for v in satir] for satir in a], float) for a in (x, y, z))
    return Plan(X, Y, Z, amac=cs.ObjectiveValue() / olcek)


def lp_yuvarla(D_cpu: np.ndarray, D_mem: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
               x0: np.ndarray | None = None, y_onceki: np.ndarray | None = None) -> tuple[Plan, Plan, bool]:
    """Bölüm 11(g): LP gevşetmesi + x'i yukarı yuvarlama sezgiseli.

    Yuvarlanmış x'ten y = max(0, Δx), z = max(0, −Δx) türetilir. Döndürür: (LP planı, yuvarlanmış plan, (5) uygun mu).
    (2)–(4) yukarı yuvarlamada korunur; (5) korunmayabilir (bu yüzden ayrıca denetlenir).
    """
    K = tipler.K
    x0 = np.zeros(K) if x0 is None else np.asarray(x0, float)
    lp = coz(D_cpu, D_mem, tipler, p, x0, y_onceki, lp=True)
    X = np.ceil(lp.x - 1e-7)
    fark = np.diff(np.concatenate([x0[:, None], X], axis=1), axis=1)
    yuv = Plan(X, np.maximum(fark, 0), np.maximum(-fark, 0), amac=0.0)
    yuv.amac = amac_hesapla(yuv, tipler, p)
    uygun = not dogrula(yuv, D_cpu, D_mem, tipler, p, x0, y_onceki)
    return lp, yuv, uygun
