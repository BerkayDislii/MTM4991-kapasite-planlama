"""İki aşamalı stokastik ILP denetimleri (küçük örnek, hızlı)."""
import numpy as np

from src.baselines.tepkisel import esik_param_oku
from src.ilp.model import param_oku, tipleri_oku
from src.ilp.stokastik import coz_stokastik, senaryo_uret


def test_kahin_senaryo_buyuk_ceza_talebi_karsilar():
    """Tek senaryo gerçek talepse ve ceza büyükse plan (tampon dahil) talebi neredeyse her saat karşılar."""
    tipler = tipleri_oku()
    p, e = param_oku(tipler, alpha=0.0), esik_param_oku(tipler, uyumlu=True, alpha=0.0)
    rng = np.random.default_rng(0)
    D = np.stack([400 + 150 * np.sin(np.arange(24) / 24 * 2 * np.pi), np.full(24, 600.0)], 1) * rng.uniform(0.9, 1.1, (24, 2))
    plan = coz_stokastik(D[None], D[0], tipler, p, e, kappa=1000.0, zaman_siniri_s=60)
    kap = np.stack([(tipler.C[:, None] * plan.x).sum(0), (tipler.M[:, None] * plan.x).sum(0)], 1)
    # Modeldeki tampon kuralı: bir önceki saatin talebini (hedef = 1) karşılayacak kadar ölçek tipi sunucu
    q = np.array([tipler.C[e.tip], tipler.M[e.tip]])
    d_once = np.vstack([D[:1], D[:-1]])
    b = np.maximum(0, ((d_once / e.hedef - kap) / q).max(1))
    assert (kap + b[:, None] * q >= D - 1e-6).all(1).mean() >= 0.95


def test_kucuk_ceza_daha_kucuk_plan():
    """Ceza katsayısı κ büyüdükçe 1. aşama planı küçülmez (monotonluk)."""
    tipler = tipleri_oku()
    p, e = param_oku(tipler, alpha=0.0), esik_param_oku(tipler, uyumlu=True, alpha=0.0)
    rng = np.random.default_rng(1)
    sen = 500 * rng.lognormal(0, 0.15, size=(5, 24, 2))
    x = [coz_stokastik(sen, sen[0, 0], tipler, p, e, kappa=k, zaman_siniri_s=60).x.sum() for k in (1.0, 100.0)]
    assert x[0] <= x[1]


def test_senaryolar_yalniz_gecmisi_kullanir():
    y = 100 * np.random.default_rng(2).lognormal(0, 0.2, size=(24 * 20, 2))
    nokta = {o: np.full((24, 2), 100.0) for o in range(24, 24 * 19)}
    O = 24 * 15
    a = senaryo_uret(nokta, y, O, 7, 10, 0)
    y2 = y.copy()
    y2[O:] *= 5
    assert np.array_equal(a, senaryo_uret(nokta, y2, O, 7, 10, 0))
