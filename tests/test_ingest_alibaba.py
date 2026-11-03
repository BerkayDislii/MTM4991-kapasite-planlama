"""Alibaba ingest birim testleri (MIMARI.md Bölüm 4.2–4.3, 11): elle hesaplanmış küçük örnekler."""
import numpy as np
import polars as pl
import pytest

from src.ingest import VeriDogrulamaHatasi
from src.ingest_alibaba import bes_dk_ortalama, makineleri_hizala, meta_dogrula

CFG = {"grid": {"step_seconds": 300}, "gaps": {"max_gap_steps": 3}, "alibaba": {"util_valid_range": [0, 100]}}


def test_birim_donusumu_ve_gecersiz_yuzde() -> None:
    """cpu = util/100 × cpu_num; [0,100] dışı (-1, 101) değer hücre ortalamasına girmez."""
    lf = pl.LazyFrame({"machine_id": np.array([1, 1, 1, 1], dtype=np.int32),
                       "time_stamp": [0, 10, 20, 300],
                       "cpu_util_percent": [50.0, 101.0, 30.0, -1.0],
                       "mem_util_percent": [80.0, 60.0, -1.0, 50.0]})
    cpu_num = pl.DataFrame({"machine_id": np.array([1], dtype=np.int32), "cpu_num": [96]})
    hucre, sayac = bes_dk_ortalama(lf, cpu_num, CFG)
    assert hucre["cell"].to_list() == [0, 1]
    assert hucre["cpu_cores"][0] == pytest.approx(0.40 * 96)          # (50+30)/2 = 40 %
    assert hucre["mem_gb_r1"][0] == pytest.approx(0.70 * 96)          # (80+60)/2 = 70 %
    assert hucre["cpu_cores"][1] is None                              # tek ölçüm geçersiz -> null
    assert sayac["cpu_aralik_disi"] == 2 and sayac["mem_aralik_disi"] == 1


def test_makine_yasam_suresi_ve_kisa_bosluk() -> None:
    """Uçtaki geçersiz hücre yaşam dışı; 2 adımlık iç boşluk doldurulur, 4 adımlık uzun boşluk null kalır."""
    cell = np.array([0, 1, 4, 5, 10, 11])
    cpu = [None, 1.0, 4.0, 5.0, 10.0, 11.0]
    hucre = pl.DataFrame({"machine_id": np.full(6, 7, dtype=np.int32), "cell": cell,
                          "cpu_cores": cpu, "mem_gb_r1": cpu}, schema_overrides={"cpu_cores": pl.Float64, "mem_gb_r1": pl.Float64})
    out, sayac = makineleri_hizala(hucre, CFG)
    assert out["cell"].to_list() == list(range(1, 12))
    np.testing.assert_allclose(out["cpu_cores"].to_numpy()[:5], [1, 2, 3, 4, 5])
    assert out["cpu_cores"][5:9].null_count() == 4
    assert sayac["yasam_disi_hucre"] == 1 and sayac["doldurulan_hucre"] == 2 and sayac["uzun_bosluk_hucre"] == 4


def test_meta_cpu_num_degisirse_dur() -> None:
    """machine_meta'da cpu_num zamanla değişen makine varsa DUR."""
    meta = pl.DataFrame({"machine_id": ["m_1", "m_1"], "time_stamp": [0, 10], "cpu_num": [96, 64],
                         "mem_size": [100, 100], "status": ["USING", "USING"]})
    with pytest.raises(VeriDogrulamaHatasi):
        meta_dogrula(meta, pl.Series([1], dtype=pl.Int32))


def test_alibaba_havuzlari_ayrik_tohumlu_artan_disarida() -> None:
    """Havuzlar ayrık, aynı tohumla aynı; boyuta tamamlanamayan artan makineler hiçbir havuza girmez."""
    from src.preprocess_alibaba import alibaba_havuz_uyeleri
    h1, h2 = alibaba_havuz_uyeleri(list(range(23)), 42, 5), alibaba_havuz_uyeleri(list(range(23)), 42, 5)
    assert h1 == h2 and len(h1) == 4
    tum = [m for u in h1.values() for m in u]
    assert len(tum) == len(set(tum)) == 20


def test_r_olcekleme_toplamin_tepesiyle_ayni() -> None:
    """Belleği r ile sonradan ölçeklemek, r ile çarpılmış seriden tepe/ortalama hesaplamakla aynıdır."""
    from src.preprocess import havuz_bes_dk_toplam, saatlik_tepe
    from src.preprocess_alibaba import r_ile_olcekle
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"vm_id": np.repeat([1, 2], 24).astype(np.int32), "cell": np.tile(np.arange(24), 2),
                       "cpu_cores": rng.random(48), "mem_gb": rng.random(48)})
    saat_r1, _ = saatlik_tepe(havuz_bes_dk_toplam(df, [1, 2], 0, 23), 0, 12)
    dogrudan, _ = saatlik_tepe(havuz_bes_dk_toplam(df.with_columns(pl.col("mem_gb") * 4), [1, 2], 0, 23), 0, 12)
    np.testing.assert_allclose(r_ile_olcekle(saat_r1, 4)["mem_peak"], dogrudan["mem_peak"])
    np.testing.assert_allclose(r_ile_olcekle(saat_r1, 4)["mem_mean"], dogrudan["mem_mean"])
