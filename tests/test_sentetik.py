"""M7b sentetik senaryo üretimi testleri (MIMARI.md Bölüm 8)."""
import numpy as np

from src.sentetik import baslangic_sec, bozulmus_seri

K = 12


def seri(n_dilim: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(1, 2, size=(n_dilim * K, 2))


def test_carpan_yalniz_pencerede_ve_testte() -> None:
    """Çarpan yalnız [baş, baş+süre) dilimlerine uygulanır; saatlik tepe 5 dk maksimumundan yeniden hesaplanır."""
    x5 = seri()
    x, y = bozulmus_seri(x5, {"tur": "carpan", "carpan": 2.0, "sure_saat": 2}, (6, 10), 7, K)
    y0 = x5.reshape(-1, K, 2).max(1)
    assert np.allclose(y[[7, 8]], 2 * y0[[7, 8]])
    assert np.array_equal(y[:7], y0[:7]) and np.array_equal(y[9:], y0[9:])
    # Süre null → test sonuna dek
    _, y2 = bozulmus_seri(x5, {"tur": "carpan", "carpan": 0.5, "sure_saat": None}, (6, 10), 8, K)
    assert np.allclose(y2[8:], 0.5 * y0[8:]) and np.array_equal(y2[:8], y0[:8])


def test_kayma_gecmis_degerleri_kullanir() -> None:
    """Kaymada test dilimi t, t − s diliminin değerini alır; test öncesi değişmez."""
    x5 = seri()
    _, y = bozulmus_seri(x5, {"tur": "kayma", "kayma_saat": 3}, (6, 10), None, K)
    y0 = x5.reshape(-1, K, 2).max(1)
    assert np.array_equal(y[6:10], y0[3:7]) and np.array_equal(y[:6], y0[:6])


def test_kontrol_ve_baslangic() -> None:
    """Kontrol seriyi değiştirmez; başlangıç etkinin sığacağı aralıkta ve tohumla tekrarlanabilir."""
    x5 = seri()
    x, y = bozulmus_seri(x5, {"tur": "yok"}, (6, 10), None, K)
    assert np.array_equal(x, x5)
    sc = {"tur": "carpan", "carpan": 1.5, "sure_saat": None, "min_etki_saat": 3}
    b = [baslangic_sec(sc, (6, 10), np.random.default_rng(s)) for s in range(50)]
    assert min(b) >= 6 and max(b) <= 7
    assert baslangic_sec(sc, (6, 10), np.random.default_rng(1)) == baslangic_sec(sc, (6, 10), np.random.default_rng(1))
