"""Kayan ufuklu ILP (model öngörülü kontrol, MPC).

Raporun günlük ILP'si gece yarısı bir kez çözülür ve plan 24 saat boyunca değişmez. Kayan ufukta her R saatte
bir, o ana kadar gözlenen talep ve köken t'deki 24 saat ileri tahminle (1)–(6) yeniden çözülür:
    x0      = o anki filo (bir önceki saatin sunucu sayıları)
    ŷ       = son L−1 saatin gerçekleşen açılışları (en az çalışma süresi kısıtı (5) saatler boyunca sürer)
ve planın yalnız ilk R saati uygulanır. R = 24 ve gün başına hizalı iken yöntem raporun günlük ILP'siyle aynıdır
(denetim: kayan_plan(..., R=24) ≡ outputs/ilp/plans).

Isınma: doğrulama döneminin başından soğuk başlangıçla (x0 = 0) başlanır (src/ilp/solve.py ile aynı);
ölçütler yalnız test saatlerinde hesaplanır.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.ilp.model import IlpParam, Plan, SunucuTipleri, coz, dogrula


@dataclass
class KayanSonuc:
    plan: Plan                       # (K, T) gerçekleşen x, y, z
    n_cozum: int
    sure_ms: list[float]             # çözüm başına süre


def kayan_plan(tahmin: dict[int, np.ndarray], bas: int, son: int, R: int, tipler: SunucuTipleri, p: IlpParam,
               cozucu: str = "SCIP", zaman_siniri_s: float = 60.0) -> KayanSonuc:
    """[bas, son) saatleri için kayan ufuklu plan. tahmin: köken → (24, 2) [cpu_hat, mem_hat].

    Her yeniden planlamada ufuk H = 24 saattir (gerekirse tahmin ufkunun sonuna kadar); uygulanan R saattir.
    """
    K, T = tipler.K, son - bas
    Lmax = int(p.L.max())
    X, Y, Z = np.zeros((K, T)), np.zeros((K, T)), np.zeros((K, T))
    x0 = np.zeros(K)
    sureler: list[float] = []
    plan, i_plan = None, 0
    for i in range(T):
        t = bas + i
        if i % R == 0:
            Dh = tahmin[t]
            y_onceki = Y[:, max(0, i - Lmax + 1): i] if i > 0 else None
            t0 = time.perf_counter()
            plan = coz(Dh[:, 0], Dh[:, 1], tipler, p, x0, y_onceki, cozucu, zaman_siniri_s)
            sureler.append(1000 * (time.perf_counter() - t0))
            ihlal = dogrula(plan, Dh[:, 0], Dh[:, 1], tipler, p, x0, y_onceki)
            assert not ihlal, f"t={t}: doğrulayıcı ihlali {ihlal[:3]}"
            i_plan = 0
        X[:, i], Y[:, i], Z[:, i] = plan.x[:, i_plan], plan.y[:, i_plan], plan.z[:, i_plan]
        x0 = X[:, i].copy()
        i_plan += 1
    return KayanSonuc(Plan(X, Y, Z, amac=float("nan")), len(sureler), sureler)
