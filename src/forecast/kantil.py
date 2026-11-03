"""Kantil (olasılıksal) gün-ileri tahminler: "talep %p olasılıkla bu değerin altında kalır".

Raporun ILP'si nokta tahminin üzerine sabit bir güvenlik payı α ekler: kapasite ≥ (1 + α)·ŷ. Bu pay her saat,
her havuz ve her belirsizlik düzeyi için aynıdır. Burada pay veriden öğrenilir: ILP'ye ŷ yerine p-kantili Q_p
verilir ve α = 0 alınır. İki yol:

1. Kayan pencereli konformal kalibrasyon (model bağımsız; Vovk vd., 2005; Angelopoulos ve Bates, 2023):
   Gün başı kökeni O için, O'dan önceki son W gündeki saatlik kökenlerin gözlenmiş oran artıkları
   r = y[o + h] / ŷ[o, h]  (o + h < O, yani yalnız geçmiş) ufuk h ve kaynak başına toplanır;
   Q_p[O, h] = ŷ[O, h] · q_p(r_h), q_p sonlu örnek konformal kantili: sıralı artıkların ⌈(n + 1)p⌉. elemanı.
   Oran (çarpımsal) artık, talep düzeyi değiştikçe payın da orantılı ölçeklenmesini sağlar; pencere kaydıkça
   kalibrasyon son davranışa uyar (dağılım kaymasına karşı).
2. SARIMA'nın kendi tahmin aralığı (yalnız SARIMA): Q_p = ŷ + z_p · se (normal varsayımı).

Çıktılar sözlük olarak döner: gün başı kökeni → (24, 2) [cpu, mem] kantil tahmini.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

UFUK = 24


def konformal_kantil(nokta: dict[int, np.ndarray], y: np.ndarray, gun_kokenleri: list[int], p: float,
                     pencere_gun: int, en_az: int = 24) -> dict[int, np.ndarray]:
    """Kayan pencereli konformal kantil tahmin.

    nokta: saatlik köken → (24, 2) nokta tahmini (src/forecast/saatlik.py); pencere için yeterince geriye gitmeli.
    Pencerede ufuk h için en_az artıktan azı varsa hata verir (yetersiz geçmiş).
    """
    kokenler = np.array(sorted(nokta))
    T = np.stack([nokta[o] for o in kokenler])                               # (n, 24, 2)
    hedef_idx = kokenler[:, None] + np.arange(UFUK)[None, :]                  # (n, 24)
    gercek = np.where(hedef_idx[..., None] < len(y), y[np.minimum(hedef_idx, len(y) - 1)], np.nan)
    oran = gercek / T                                                         # (n, 24, 2)
    out = {}
    for O in gun_kokenleri:
        q = np.zeros((UFUK, 2))
        for h in range(UFUK):
            secili = (kokenler >= O - 24 * pencere_gun) & (kokenler + h < O)  # yalnız gözlenmiş geçmiş
            r = oran[secili, h, :]
            n = r.shape[0]
            assert n >= en_az, f"köken {O}, ufuk {h}: pencerede {n} artık var (en az {en_az})"
            k = math.ceil((n + 1) * p)
            s = np.sort(r, axis=0)
            q[h] = s[min(k, n) - 1]                                           # k > n: en büyük artık (temkinli)
        out[O] = nokta[O] * q
    return out


def sarima_aralik_kantil(ortalama: dict[int, np.ndarray], se: dict[int, np.ndarray], gun_kokenleri: list[int],
                         p: float) -> dict[int, np.ndarray]:
    """SARIMA normal tahmin dağılımının p-kantili."""
    z = float(norm.ppf(p))
    return {O: ortalama[O] + z * se[O] for O in gun_kokenleri}


def kapsama(kantil: dict[int, np.ndarray], y: np.ndarray, gun_kokenleri: list[int]) -> np.ndarray:
    """Kaynak başına gözlenen kapsama oranı P(y ≤ Q_p): (2,) — kalibrasyon denetimi."""
    v = np.stack([y[O: O + UFUK] <= kantil[O] for O in gun_kokenleri if O + UFUK <= len(y)])
    return v.reshape(-1, 2).mean(0)
