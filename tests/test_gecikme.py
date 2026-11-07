"""Açılış gecikmesi değerlendirmesi: δ = 0'da raporun saatlik ölçütleriyle aynı sonuç."""
import numpy as np

from src.evaluate import plan_olcutleri
from src.forecast.ortak import saatlik_seri
from src.forecast.veri import bes_dk_seri
from src.gecikme_deney import degerlendir
from src.ilp.model import Plan, param_oku, tipleri_oku
from src.kayan_deney import havuz_bul


def test_gecikme_sifir_saatlik_tanimla_ayni():
    h = havuz_bul("bitbrains_fastStorage")
    y, D5 = saatlik_seri(h), bes_dk_seri(h)
    tipler = tipleri_oku()
    p = param_oku(tipler)
    rng = np.random.default_rng(0)
    T = 48
    bas = h.sinir["test"][0]
    x = rng.integers(100, 200, size=(tipler.K, T)).astype(float)
    x[1:] = rng.integers(0, 30, size=(tipler.K - 1, T))
    sifir = np.zeros_like(x)
    yy = np.maximum(np.diff(np.concatenate([np.zeros((tipler.K, 1)), x], 1), axis=1), 0)
    D = y[bas: bas + T]
    o = degerlendir(x, sifir, yy, sifir, D5[12 * bas: 12 * (bas + T)], np.ones(T, bool), tipler, p, 0)
    ref = plan_olcutleri(Plan(x, yy, sifir, 0.0), D[:, 0], D[:, 1], tipler, p)
    assert abs(o["eksik_saat_orani"] - ref["eksik_saat_orani"]) < 1e-12
    assert abs(o["maliyet_gunluk"] - ref["maliyet"] / (T / 24)) < 1e-6


def test_gecikme_tepkisel_kapasiteyi_azaltir():
    """Gecikme yalnız tepkisel açılışları etkiler; eksik saat azalmaz."""
    h = havuz_bul("bitbrains_fastStorage")
    D5 = bes_dk_seri(h)
    tipler = tipleri_oku()
    p = param_oku(tipler)
    T, bas = 48, h.sinir["test"][0]
    tep = np.zeros((tipler.K, T))
    tep[0] = np.tile([100.0, 200.0], T // 2)
    sifir = np.zeros_like(tep)
    args = (sifir, tep, sifir, tep, D5[12 * bas: 12 * (bas + T)], np.ones(T, bool), tipler, p)
    assert degerlendir(*args, 10)["eksik_saat_orani"] >= degerlendir(*args, 0)["eksik_saat_orani"]
