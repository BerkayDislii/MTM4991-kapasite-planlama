"""İstatistik testlerinin akıl sağlığı denetimleri (sentetik veri, saniyeler içinde)."""
import numpy as np
import polars as pl

from src.istatistik import bootstrap, diebold_mariano


def test_dm_ayni_hatalar_fark_yok():
    rng = np.random.default_rng(0)
    e = rng.normal(size=500)
    _, p, fark = diebold_mariano(e, e.copy())
    assert np.isnan(p) or p > 0.9
    assert fark == 0.0


def test_dm_belirgin_iyi_model_anlamli():
    rng = np.random.default_rng(1)
    e = rng.normal(size=24 * 30)
    _, p, fark = diebold_mariano(0.5 * e, e)                # A'nın hatası her saat yarı yarıya
    assert p < 0.01 and fark < 0


def test_bootstrap_kendisiyle_fark_sifir():
    gun = np.arange(20)
    satir = [{"havuz": "h", "aile": a, "hedef": 0.1, "ayar": 0.0, "R": 24, "gun": g,
              "maliyet": 100.0 + g, "kahin_maliyet": 80.0, "eksik_saat": 1.0}
             for a in ("sabit_alfa_r2u", "kopya") for g in gun]
    b = bootstrap(pl.DataFrame(satir)).filter(pl.col("aile") == "kopya").row(0, named=True)
    assert abs(b["fark_regret"]) < 1e-9 and abs(b["fark_regret_alt"]) < 1e-9 and abs(b["fark_regret_ust"]) < 1e-9
