"""M4 tahmin birim testleri (MIMARI.md Bölüm 11): taban modeller, ölçütler ve sızıntı denetimleri."""
import numpy as np
import pytest

from src.forecast.lstm import Olcekleyici, ornekler
from src.forecast.naif import mevsimsel_naif, son_deger
from src.forecast.ortak import mase_olcegi, olcutler
from src.forecast.sarima import gun_ileri, uydur

TOHUM = 42


def _seri(n_gun: int = 6) -> np.ndarray:
    """Günlük mevsimsel + gürültülü, pozitif iki kanallı saatlik seri."""
    rng = np.random.default_rng(TOHUM)
    t = np.arange(24 * n_gun)
    x = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
    return np.column_stack([x, 2 * x])


def test_taban_modeller_elle() -> None:
    """Mevsimsel naif = önceki günün aynı saatleri; son değer = köken öncesi son gözlem."""
    y = np.column_stack([np.arange(72.0), np.arange(72.0) * 10])
    mn = mevsimsel_naif(y, [2])
    sd = son_deger(y, [2])
    assert np.array_equal(mn[0], y[24:48])
    assert np.all(sd[0, :, 0] == 47) and np.all(sd[0, :, 1] == 470)


def test_olcutler_elle() -> None:
    """MAE, RMSE, MAPE, MASE ve eksik tahmin oranı küçük örnekte."""
    g = np.array([[10.0, 1.0], [20.0, 2.0]])
    t = np.array([[12.0, 1.0], [15.0, 3.0]])
    cpu, mem = olcutler(g, t, np.array([2.0, 1.0]))
    assert cpu["mae"] == pytest.approx(3.5) and cpu["rmse"] == pytest.approx(np.sqrt((4 + 25) / 2))
    assert cpu["mape"] == pytest.approx(100 * (0.2 + 0.25) / 2) and cpu["mase"] == pytest.approx(1.75)
    assert cpu["eksik_tahmin_orani"] == 0.5 and cpu["ortalama_eksik"] == pytest.approx(5.0)
    assert mem["eksik_tahmin_orani"] == 0.0


def test_mase_olcegi_yalniz_egitim() -> None:
    """MASE paydası eğitim sonrası veriden etkilenmez."""
    y = _seri()
    y2 = y.copy()
    y2[96:] *= 100
    assert np.array_equal(mase_olcegi(y, 96), mase_olcegi(y2, 96))


def test_taban_gelecek_veri_kullanmiyor() -> None:
    """Köken sonrası değerleri bozmak taban tahminlerini değiştirmez."""
    y = _seri()
    y2 = y.copy()
    y2[72:] = -1.0
    for f in (mevsimsel_naif, son_deger):
        assert np.array_equal(f(y, [3]), f(y2, [3]))


def test_sarima_walk_forward_sizinti_yok() -> None:
    """SARIMA gün-ileri tahmini yalnız y[:o]'yu kullanır ve apply(y[:o]).forecast ile aynıdır."""
    y = _seri()[:, 0]
    res, _ = uydur(y[:96], (1, 0, 0), (0, 0, 0, 0), 200)
    a = gun_ileri(res, y, [4, 5])
    b = np.stack([res.apply(y[:24 * g]).forecast(24) for g in (4, 5)])
    np.testing.assert_allclose(a, b, rtol=1e-8)
    y2 = y.copy()
    y2[96:] = 1e6
    np.testing.assert_allclose(gun_ileri(res, y2, [4]), a[:1], rtol=1e-8)


def test_lstm_olcekleyici_yalniz_egitim() -> None:
    """Ölçekleyici istatistikleri yalnız verilen eğitim diliminden hesaplanır."""
    x = _seri()
    sc = Olcekleyici(x[:96])
    np.testing.assert_allclose(sc.ort, x[:96].mean(axis=0))
    np.testing.assert_allclose(sc.std, x[:96].std(axis=0))
    np.testing.assert_allclose(sc.geri(sc.donustur(x)), x)


def test_lstm_ornek_girdisi_koken_oncesi() -> None:
    """LSTM örneğinde girdi köken öncesi L saatlik 5 dk adımlar, hedef köken sonrası 24 tepe."""
    k, L = 12, 2
    x5 = np.arange(24 * 4 * k, dtype=np.float64).reshape(-1, 1).repeat(2, axis=1)
    y = x5.reshape(-1, k, 2).max(axis=1)
    X, Y = ornekler(x5, y, [48], L, k)
    assert X.shape == (1, L * k, 2) and Y.shape == (1, 24, 2)
    assert X[0, -1, 0] == 48 * k - 1          # son girdi adımı kökenden hemen önce
    assert Y[0, 0, 0] == y[48, 0]
