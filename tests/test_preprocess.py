"""Ön işleme birim testleri (MIMARI.md Bölüm 11): elle hesaplanmış küçük örnekler."""
import numpy as np
import polars as pl
import pytest

from src.ingest import (GEREKLI_SUTUNLAR, VeriDogrulamaHatasi, birim_donustur, kisa_bosluklari_doldur,
                        vm_izgaraya_hizala, zaman_birimi_tespit)
from src.preprocess import havuz_bes_dk_toplam, havuz_uyeleri, saatlik_tepe

ADIM = 300


def test_toplamin_tepesi_tepelerin_toplamindan_farkli() -> None:
    """Eş zamanlı olmayan tepeler: toplamın tepesi (5) < tepelerin toplamı (8)."""
    # İki VM, tek saat (12 adım). VM1 adım 0'da 4, VM2 adım 6'da 4 tepe yapar, diğer adımlar 1.
    cell = np.arange(12)
    vm1 = np.ones(12); vm1[0] = 4
    vm2 = np.ones(12); vm2[6] = 4
    df = pl.DataFrame({"vm_id": np.repeat([1, 2], 12).astype(np.int32), "cell": np.tile(cell, 2),
                       "cpu_cores": np.concatenate([vm1, vm2]), "mem_gb": np.concatenate([vm1, vm2])})
    d5 = havuz_bes_dk_toplam(df, [1, 2], 0, 11)
    saat, eksik = saatlik_tepe(d5, 0, 12)
    assert eksik == 0
    assert saat["cpu_peak"][0] == pytest.approx(5.0)          # max(4+1, 1+4, 1+1)
    assert vm1.max() + vm2.max() == pytest.approx(8.0)        # yasak yöntem
    assert saat["cpu_mean"][0] == pytest.approx((24 + 6) / 12)
    assert saat["n_active"][0] == 2


def test_birim_donusumu() -> None:
    """cpu = usage × cores / capacity, mem = KB / 1024²; kapasite 0 -> null; kırpma yok."""
    df = pl.DataFrame({
        GEREKLI_SUTUNLAR[0]: [0.0, 300.0, 600.0],
        "CPU cores": [4.0, 0.0, 2.0],
        "CPU capacity provisioned [MHZ]": [8000.0, 0.0, 4000.0],
        "CPU usage [MHZ]": [2000.0, 0.0, 6000.0],          # 3. satır kapasitenin üstünde
        "Memory usage [KB]": [2.0 * 1024**2, 1024.0, 1.0 * 1024**2],
    })
    b = birim_donustur(df)
    assert b["cpu_cores"].to_list() == [pytest.approx(1.0), None, pytest.approx(3.0)]
    assert b["mem_gb"].to_list() == [pytest.approx(2.0), None, pytest.approx(1.0)]


def test_zaman_birimi_tespiti() -> None:
    """Fark medyanı 300 -> saniye, 300000 -> milisaniye, diğer -> DUR."""
    assert zaman_birimi_tespit(300.0, ADIM, 0.1) == 1
    assert zaman_birimi_tespit(300_000.0, ADIM, 0.1) == 1000
    with pytest.raises(VeriDogrulamaHatasi):
        zaman_birimi_tespit(60.0, ADIM, 0.1)


def test_enterpolasyon_limiti() -> None:
    """<= 3 adımlık iç boşluk doldurulur, 4 adımlık ve uçtaki boşluklar NaN kalır (kısmen bile değil)."""
    x = np.array([0.0, np.nan, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0, np.nan])
    y, doldur = kisa_bosluklari_doldur(x, 3)
    np.testing.assert_allclose(y[:5], [0, 1, 2, 3, 4])
    assert np.isnan(y[5:9]).all() and y[9] == 9.0 and np.isnan(y[10])
    assert doldur.tolist() == [False, True, True, True] + [False] * 7


def test_hucre_ortalamasi_ve_yasam_suresi() -> None:
    """Aynı hücredeki ölçümler ortalanır; uçlardaki geçersiz satırlar yaşam süresinden atılır."""
    # Geçersiz (NaN) satır, sonra 2 geçerli ölçüm aynı hücrede, sonra geçerli, sonra geçersiz
    ts = np.array([0, 600, 700, 900, 1500], dtype=float)
    cpu = np.array([np.nan, 1.0, 3.0, 5.0, np.nan])
    g = vm_izgaraya_hizala(ts, cpu, cpu.copy(), ADIM, 3)
    assert g is not None
    assert g["cell"].tolist() == [2, 3]                       # yaşam = ilk/son geçerli hücre
    np.testing.assert_allclose(g["cpu_cores"], [2.0, 5.0])
    assert g["inside_rows"].tolist() == [False, True, True, True, False]
    assert vm_izgaraya_hizala(ts[:1], cpu[:1], cpu[:1], ADIM, 3) is None


def test_yasam_suresi_disi_ve_uzun_bosluk_sifir() -> None:
    """Yaşam süresi dışı ve uzun boşluk (null) hücreleri havuz toplamında 0 sayılır."""
    df = pl.DataFrame({"vm_id": np.array([1, 1, 2], dtype=np.int32), "cell": [0, 1, 1],
                       "cpu_cores": [2.0, None, 3.0], "mem_gb": [1.0, None, 1.0]})
    d5 = havuz_bes_dk_toplam(df, [1, 2], 0, 2)
    assert d5["cpu_cores"].to_list() == [2.0, 3.0, 0.0]       # hücre 2'de VM yok -> 0
    assert d5["n_active"].to_list() == [1, 2, 0]


def test_eksik_son_dilim_atilir() -> None:
    """12 adımdan kısa son saat atılır."""
    d5 = pl.DataFrame({"cell": np.arange(30, dtype=np.int64), "cpu_cores": np.ones(30),
                       "mem_gb": np.ones(30), "n_active": np.ones(30, dtype=np.int64)})
    saat, eksik = saatlik_tepe(d5, 0, 12)
    assert saat.height == 2 and eksik == 1


def test_alt_havuzlar_ayrik_ve_tohumlu() -> None:
    """Alt havuzlar ayrık, kaynağı tam kapsar ve aynı tohumla aynıdır."""
    pcfg = {"natural": ["a"], "subpools": {"source": "a", "seed": 1, "n_pools": 3}}
    ids = {"a": list(range(10))}
    h1, h2 = havuz_uyeleri(ids, pcfg), havuz_uyeleri(ids, pcfg)
    assert h1 == h2
    altlar = [set(u) for ad, (_, u) in h1.items() if "_s" in ad]
    assert len(altlar) == 3 and set().union(*altlar) == set(range(10)) and sum(map(len, altlar)) == 10
