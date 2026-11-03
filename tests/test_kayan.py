"""Kayan ufuklu ILP ve saatlik kökenli tahmin denetimleri."""
import numpy as np
import pytest

from src.forecast.ortak import saatlik_seri
from src.forecast.saatlik import gun_basi_farki, kokenli_tahmin
from src.ilp.kayan import kayan_plan
from src.ilp.model import param_oku, tipleri_oku
from src.kayan_deney import havuz_bul


@pytest.fixture(scope="module")
def ortam():
    h = havuz_bul("bitbrains_fastStorage")
    return h, saatlik_seri(h), tipleri_oku()


@pytest.mark.parametrize("model", ["son_deger", "mevsimsel_naif", "sarima"])
def test_gun_basi_tahminleri_raporla_ayni(ortam, model):
    h, y, _ = ortam
    kok = list(range(h.sinir["val"][0], h.sinir["test"][1], 24))
    assert gun_basi_farki(h, model, kok, kokenli_tahmin(h, model, y, kok)) < 1e-6


def test_R24_gunluk_ilp_ile_ayni(ortam):
    """R = 24 ve gün başına hizalı kayan ufuk = raporun günlük ILP'si (outputs/ilp/plans)."""
    import polars as pl
    from src.common.config import yaml_oku, yol
    h, y, tipler = ortam
    bas, son = h.sinir["val"][0], h.sinir["test"][1]
    kok = list(range(bas, son, 24))
    tahmin = dict(zip(kok, kokenli_tahmin(h, "sarima", y, kok)))
    plan = kayan_plan(tahmin, bas, son, 24, tipler, param_oku(tipler)).plan
    d = yol(yaml_oku("ilp")["outputs"]["dizin"]) / "plans" / h.veri_seti / h.ad / "sarima"
    ref = np.concatenate([np.stack([pl.read_parquet(d / f"plan_{g}.parquet").sort("slot")[f"x_{a}"].to_numpy()
                                    for a in tipler.ad]) for g in range(bas // 24, son // 24)], axis=1)
    assert np.array_equal(plan.x, ref)

