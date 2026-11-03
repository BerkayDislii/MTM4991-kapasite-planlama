"""İki aşamalı stokastik kapasite planlama ILP'si (senaryo tabanlı; deterministik eşdeğeri yine bir MILP).

Neden? Önceki deneyler güvenilirliğin asıl kaynağının tepkisel tampon olduğunu gösterdi; ama raporun ILP'si
planı tampondan habersiz kurar (yalnız sabit güvenlik payı α ile). Burada plan, gün içinde bir tamponun
devreye gireceğini ve onun gecikmeli ve pahalı olduğunu bilerek kurulur.

1. aşama (gece, talep bilinmeden; her senaryoda aynı):  x_kt, y_kt, z_kt ∈ Z≥0   — raporun (4)–(6) kısıtları
2. aşama (senaryo s, saat t; talep gerçekleştikçe):
   b_st ≥ 0    tampon (ölçek tipi m8i) sunucusu; raporun R2 kuralı gibi BİR ÖNCEKİ saatin talebine tepki verir:
               ρ·q^r·b_st ≥ D^r_{s,t−1} / hedef − Σ_k ρ·C^r_k x_kt                 ∀ r ∈ {cpu, mem}
   u^r_st ≥ 0  karşılanamayan talep (eksik kapasite):
               u^r_st ≥ D^r_st − Σ_k ρ·C^r_k x_kt − ρ·q^r·b_st                     ∀ r
Amaç:  min  Σ_t Σ_k (c_k x_kt + s_k y_kt + d_k z_kt)
          + (1/S) Σ_s Σ_t [ (c_b + s_b) b_st + λ_cpu u^cpu_st + λ_mem u^mem_st ]
   c_b + s_b: tampon sunucusunun saatlik fiyatı + açma maliyeti (tampon her saat yeniden açılıyormuş gibi; temkinli)
   λ_r = κ · (r kaynağının en ucuz birim fiyatı); κ ceza katsayısı = yöntemin tek ayarı (raporun α'sının yerine)

Senaryolar: köken O'dan önceki son W gündeki tam gözlenmiş 24 saatlik tahmin hatası yolları (oran artıkları,
src/forecast/kantil.py ile aynı mantık) nokta tahmine çarpılır: D_s[h] = ŷ[O, h] · r_{o_s}[h]. Yol bütün
olarak örneklendiği için saatler arası ve kaynaklar arası ilişki korunur. Yalnız geçmiş kullanılır.

Tampon bir karar değil, kuraldır: b_st = max(0, g^cpu_st, g^mem_st) (ikili değişkenler + büyük-M ile tam eşitlik);
aksi hâlde model tamponu o saatin talebini görerek büyütür (geleceği görme). Basitleştirmeler (rapora yazılacak):
b sürekli (tamsayı yuvarlaması yok); tamponun eşik ve soğuma süresi davranışı modelde yok. Değerlendirmede plan,
raporun gerçek R2 tampon kuralıyla birlikte çalıştırılır (src/baselines/tepkisel.esik_simule); model yalnız
1. aşama planını seçer.
"""
from __future__ import annotations

import numpy as np
from ortools.linear_solver import pywraplp

from src.baselines.tepkisel import EsikParam
import dataclasses

from src.ilp.model import IlpParam, Plan, SunucuTipleri, acma_kapama_duzelt, coz, devralinan

UFUK = 24


def senaryo_uret(nokta: dict[int, np.ndarray], y: np.ndarray, O: int, pencere_gun: int, S: int,
                 tohum: int) -> np.ndarray:
    """Köken O için (S, 24, 2) talep senaryoları: geçmiş tam hata yolları × nokta tahmin."""
    kokenler = [o for o in nokta if O - 24 * pencere_gun <= o and o + UFUK <= O]
    assert len(kokenler) >= S, f"köken {O}: {len(kokenler)} tam hata yolu var, {S} senaryo istendi"
    rng = np.random.default_rng([tohum, O])
    secilen = rng.choice(kokenler, size=S, replace=False)
    oran = np.stack([y[o: o + UFUK] / nokta[o] for o in secilen])           # (S, 24, 2)
    return nokta[O][None] * oran


def coz_stokastik(senaryo: np.ndarray, D_once: np.ndarray, tipler: SunucuTipleri, p: IlpParam, e: EsikParam,
                  kappa: float, x0: np.ndarray | None = None, y_onceki: np.ndarray | None = None,
                  cozucu: str = "SCIP", zaman_siniri_s: float = 120.0, bosluk: float = 1e-2,
                  ipucu: np.ndarray | None = None) -> Plan:
    """Deterministik eşdeğeri çözer; 1. aşama planını (K, H) döndürür. bilgi: beklenen tampon ve eksik."""
    S, H, _ = senaryo.shape
    K = tipler.K
    x0 = np.zeros(K) if x0 is None else np.asarray(x0, float)
    yd = devralinan(y_onceki, p.L, H)
    cap = np.stack([p.rho_cpu * tipler.C, p.rho_mem * tipler.M])          # (2, K)
    q = cap[:, e.tip]                                                     # tampon sunucusunun kapasitesi (2,)
    birim_fiyat = np.array([(tipler.c / cap[0]).min(), (tipler.c / cap[1]).min()])
    lam = kappa * birim_fiyat
    c_b = tipler.c[e.tip] + p.s[e.tip]

    s = pywraplp.Solver.CreateSolver(cozucu)
    assert s is not None
    s.SetTimeLimit(int(zaman_siniri_s * 1000))
    inf = s.infinity()
    # Filo üst sınırı (büyük-M'leri sıkılaştırmak için): tek başına en büyük senaryo talebini 1,5 kat karşılayacak
    # sayı; başlangıç filosundan küçük olamaz. Pratikte bağlayıcı değildir (çözümde denetlenir).
    dmax = senaryo.reshape(-1, 2).max(0)
    N = np.maximum(np.ceil(1.5 * (dmax[:, None] / cap).max(0)), x0)
    x = [[s.IntVar(0, float(N[k]), f"x{k}_{t}") for t in range(H)] for k in range(K)]
    y = [[s.IntVar(0, inf, f"y{k}_{t}") for t in range(H)] for k in range(K)]
    z = [[s.IntVar(0, inf, f"z{k}_{t}") for t in range(H)] for k in range(K)]
    for t in range(H):
        for k in range(K):
            onceki = x0[k] if t == 0 else x[k][t - 1]
            s.Add(x[k][t] - onceki == y[k][t] - z[k][t])                                       # (4)
            L = int(p.L[k])
            s.Add(sum(y[k][tau] for tau in range(max(0, t - L + 1), t + 1)) + yd[k, t] <= x[k][t])  # (5)
    kap = [[sum(cap[r, k] * x[k][t] for k in range(K)) for t in range(H)] for r in range(2)]
    # Tampon bir kuraldır, serbest karar değil: b = max(0, g_cpu, g_mem), g_r = (D^r_{t−1}/hedef − kap^r_t)/q^r.
    # Yalnız "b ≥ g_r" yazılsaydı model b'yi o saatin talebini görecek kadar büyütebilirdi (geleceği görme).
    # Eşitlik ikili değişkenlerle (hangi terimin en büyük olduğu) ve büyük-M ile kurulur.
    B = float((np.maximum(dmax, D_once) / e.hedef / q).max())
    capmax = cap @ N
    M = [B + capmax[r] / q[r] for r in range(2)]
    ikinci = []
    for si in range(S):
        for t in range(H):
            b = s.NumVar(0, inf, "")
            d_once = D_once if t == 0 else senaryo[si, t - 1]
            d = [s.BoolVar(""), s.BoolVar("")]                      # hangi kaynak tamponu belirliyor
            s.Add(d[0] + d[1] <= 1)
            s.Add(b <= B * (d[0] + d[1]))                            # ikisi de 0 → b = 0
            for r in range(2):
                g = (d_once[r] / e.hedef - kap[r][t]) * (1.0 / q[r])
                s.Add(b >= g)
                s.Add(b <= g + M[r] * (1 - d[r]))
                u = s.NumVar(0, inf, "")
                s.Add(u >= senaryo[si, t, r] - kap[r][t] - q[r] * b)
                ikinci.append(lam[r] * u)
            ikinci.append(c_b * b)
    birinci = sum(tipler.c[k] * x[k][t] + p.s[k] * y[k][t] + p.d[k] * z[k][t] for k in range(K) for t in range(H))
    s.Minimize(birinci + (1.0 / S) * sum(ikinci))
    if ipucu is not None:                                            # deterministik planla sıcak başlangıç
        degisken = [x[k][t] for k in range(K) for t in range(H)]
        s.SetHint(degisken, [float(ipucu[k, t]) for k in range(K) for t in range(H)])
    prm = pywraplp.MPSolverParameters()
    prm.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, bosluk)
    durum = s.Solve(prm)
    if durum not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Stokastik model çözülemedi: {durum}")
    X, Y, Z = (np.array([[round(v.solution_value()) for v in sat] for sat in a], float) for a in (x, y, z))
    X, Y, Z = acma_kapama_duzelt(X, Y, Z, x0, p)
    assert (X < N[:, None]).all() or (X <= x0[:, None]).all(), "filo üst sınırı bağlayıcı oldu: N'yi büyüt"
    return Plan(X, Y, Z, amac=float(s.Objective().Value()),
                bilgi={"sure_ms": s.wall_time(), "optimal": durum == pywraplp.Solver.OPTIMAL})


def gunluk_stokastik_plan(nokta: dict[int, np.ndarray], y: np.ndarray, bas: int, son: int, tipler: SunucuTipleri,
                          p: IlpParam, e: EsikParam, kappa: float, S: int, pencere_gun: int, tohum: int,
                          cozucu: str = "SCIP", zaman_siniri_s: float = 120.0) -> tuple[Plan, list[float]]:
    """[bas, son) için gün gün stokastik plan (raporun kayan penceresiyle aynı durum devri)."""
    assert bas % 24 == 0 and (son - bas) % 24 == 0
    K = tipler.K
    planlar, sureler = [], []
    x0, y_onceki = np.zeros(K), None
    for O in range(bas, son, 24):
        sen = senaryo_uret(nokta, y, O, pencere_gun, S, tohum)
        # Sıcak başlangıç: raporun deterministik planı (nokta tahmin, α = 0,05) — her zaman uygun bir 1. aşama
        det = coz(nokta[O][:, 0], nokta[O][:, 1], tipler, dataclasses.replace(p, alpha=0.05), x0, y_onceki, cozucu)
        plan = coz_stokastik(sen, y[O - 1], tipler, p, e, kappa, x0, y_onceki, cozucu, zaman_siniri_s, ipucu=det.x)
        planlar.append(plan)
        sureler.append(plan.bilgi["sure_ms"])
        x0, y_onceki = plan.x[:, -1].copy(), plan.y.copy()
    return Plan(*(np.concatenate([getattr(q, v) for q in planlar], axis=1) for v in ("x", "y", "z")),
                amac=float("nan")), sureler
