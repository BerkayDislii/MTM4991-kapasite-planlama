"""Konformal kantil tahmin denetimleri (yapay veriyle, hızlı)."""
import numpy as np

from src.forecast.kantil import UFUK, kapsama, konformal_kantil


def _yapay(n_saat=24 * 40, tohum=0):
    rng = np.random.default_rng(tohum)
    y = 100 * rng.lognormal(0, 0.2, size=(n_saat, 2))                 # gerçek talep
    nokta = {o: np.full((UFUK, 2), 100.0) for o in range(24, n_saat - UFUK)}   # sabit nokta tahmini
    return y, nokta


def test_kapsama_hedefe_yakin():
    """Değiştirilebilir (i.i.d.) veride konformal p-kantili yaklaşık p oranında kapsar."""
    y, nokta = _yapay()
    gun = list(range(24 * 10, 24 * 38, 24))
    for p in (0.5, 0.8, 0.9):
        c = kapsama(konformal_kantil(nokta, y, gun, p, pencere_gun=7), y, gun)
        assert np.all(np.abs(c - p) < 0.06), (p, c)


def test_yalniz_gecmis_kullanilir():
    """Köken O'daki kantil, O ve sonrasındaki gerçek değerlere bağlı değildir (sızıntı yok)."""
    y, nokta = _yapay()
    O = 24 * 20
    a = konformal_kantil(nokta, y, [O], 0.9, pencere_gun=7)[O]
    y2 = y.copy()
    y2[O:] *= 10
    b = konformal_kantil(nokta, y2, [O], 0.9, pencere_gun=7)[O]
    assert np.array_equal(a, b)
