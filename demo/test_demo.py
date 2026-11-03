"""Demo çekirdeğinin raporun benzetimini birebir ürettiğini denetler (Kubernetes gerekmez).

Çalıştırma: .venv/bin/python -m pytest demo/test_demo.py -q
"""
import numpy as np
import polars as pl
import pytest

from demo.calistir import GRUP, Kayit, Kol, kosu
from demo.cekirdek import ayar_oku, hazirla
from demo.kume import SahteKume
from src.common.config import yaml_oku, yol

SIM = {"ilp": "ilp_sarima", "r2_uyumlu": "r2_hibrit_uyumlu_sarima"}


@pytest.mark.parametrize("strateji", ["ilp", "r2_uyumlu"])
def test_kuru_kosu_benzetimle_ayni(strateji, tmp_path):
    ayar = ayar_oku() | {"strateji": strateji, "bas_gun": 24, "gun_sayisi": 6, "canli_ilp": True}
    hz = hazirla(ayar)
    grup = GRUP[strateji]
    kol = Kol(grup, hz, SahteKume(hz.tipler, hz.birim), Kayit(tmp_path / grup, hz.tipler.ad))
    kosu([kol], hizli=True, saat_saniye=0, zaman_asimi_s=0, cikti=tmp_path)
    demo = pl.read_csv(tmp_path / grup / "saat.csv")
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
    hz = hazirla(ayar | {"strateji": "ilp", "bas_gun": 20, "gun_sayisi": 1})
    pod_kap = hz.birim.dugum_pod_kapasitesi(hz.tipler)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = rng.integers(0, 50, hz.tipler.K)
        kap = np.array([(hz.tipler.C * x).sum(), (hz.tipler.M * x).sum()])
        talep = rng.uniform(0, 1.2, 2) * np.maximum(kap, 1)
        n = hz.birim.pod_sayisi(talep)
        bekleyen = np.maximum(np.array(n) - (pod_kap * x[:, None]).sum(0), 0)
        assert np.array_equal(bekleyen > 0, talep > kap + 1e-9)


def test_senaryo_komutlari(tmp_path):
    """Panel komutu: darbe 3 saat (36 adım) ×2, sonra kendiliğinden biter; "normal" etkin senaryoyu bitirir."""
    import json

    from demo.calistir import Senaryo
    sc = Senaryo(tmp_path)
    assert sc.guncelle(0) == ("", 1.0)
    (tmp_path / "komut.json").write_text(json.dumps({"senaryo": "darbe", "id": 1}))
    assert sc.guncelle(10) == ("darbe", 2.0)
    assert sc.guncelle(10 + 35) == ("darbe", 2.0)
    assert sc.guncelle(10 + 36) == ("", 1.0)
    (tmp_path / "komut.json").write_text(json.dumps({"senaryo": "basamak", "id": 2}))
    assert sc.guncelle(100) == ("basamak", 1.5)
    assert sc.guncelle(500) == ("basamak", 1.5)
    (tmp_path / "komut.json").write_text(json.dumps({"senaryo": "normal", "id": 3}))
    assert sc.guncelle(501) == ("", 1.0)
    assert [(g["ad"], g["bas"], g["bit"]) for g in sc.durum()["gecmis"]] == [("darbe", 10, 46), ("basamak", 100, 501)]
