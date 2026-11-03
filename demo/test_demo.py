"""Demo çekirdeğinin raporun benzetimini birebir ürettiğini denetler (Kubernetes gerekmez).

Çalıştırma: .venv/bin/python -m pytest demo/test_demo.py -q
"""
import numpy as np
import polars as pl
import pytest

from demo.calistir import kosu
from demo.cekirdek import ayar_oku, hazirla
from demo.kume import SahteKume
from src.common.config import yaml_oku, yol

SIM = {"ilp": "ilp_sarima", "r2_uyumlu": "r2_hibrit_uyumlu_sarima"}


@pytest.mark.parametrize("strateji", ["ilp", "r2_uyumlu"])
def test_kuru_kosu_benzetimle_ayni(strateji, tmp_path):
    ayar = ayar_oku() | {"strateji": strateji, "bas_gun": 24, "gun_sayisi": 6, "canli_ilp": True}
    hz = hazirla(ayar)
    kosu(hz, SahteKume(hz.tipler, hz.birim), hizli=True, saat_saniye=0, zaman_asimi_s=0, cikti=tmp_path)
    demo = pl.read_csv(tmp_path / "saat.csv")
    sim = pl.read_parquet(yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "sim" / "bitbrains"
                          / ayar["havuz"] / f"{SIM[strateji]}.parquet").filter(pl.col("slot") >= 24 * 24)
    for a in hz.tipler.ad:
        assert np.array_equal(demo[f"x_{a.split('.')[0]}"].to_numpy(), sim[f"x_{a}"].to_numpy()), a
    kap = np.stack([sim["kap_cpu"].to_numpy(), sim["kap_mem"].to_numpy()], 1)
    eksik_sim = ((sim.select("cpu_peak", "mem_peak").to_numpy() - kap) > 1e-9).any(1)
    assert np.array_equal(demo["eksik"].to_numpy().astype(bool), eksik_sim)


def test_birim_pod_esdegerligi():
    """Yer bulamayan pod var ⇔ talep > kapasite (birim pod kapasitelere tam bölündüğü için)."""
    ayar = ayar_oku()
    hz = hazirla(ayar | {"bas_gun": 20, "gun_sayisi": 1})
    pod_kap = hz.birim.dugum_pod_kapasitesi(hz.tipler)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = rng.integers(0, 50, hz.tipler.K)
        kap = np.array([(hz.tipler.C * x).sum(), (hz.tipler.M * x).sum()])
        talep = rng.uniform(0, 1.2, 2) * np.maximum(kap, 1)
        n = hz.birim.pod_sayisi(talep)
        bekleyen = np.maximum(np.array(n) - (pod_kap * x[:, None]).sum(0), 0)
        assert np.array_equal(bekleyen > 0, talep > kap + 1e-9)
