"""Açılış gecikmesi altında sağlamlık testi: "plan + tampon" bulgusu, sunucular hemen hazır olmadığında da geçerli mi?

Şimdiye kadarki tüm deneylerde açılan bir sunucu açıldığı saatin başından itibaren hazır sayıldı. Gerçekte yeni bir
düğümün hizmete girmesi δ dakika sürer (sanal makine açılışı, düğümün kümeye katılması, pod'ların yerleşmesi).

Değerlendirme 5 dakikalık adımlarla yapılır (rapordaki saatlik tepe = saatin 12 adımının en büyüğü):
  - Planlı (proaktif) sunucular: plan önceden bilindiği için δ dakika ÖNCEDEN açılır ve saat başında hazırdır;
    bedeli her açılış için δ/60 saatlik ek ücrettir.
  - Tepkisel sunucular (tampon, R0, R1): karar saat başında verilir; o saat yeni açılanlar ilk δ dakika hazır
    değildir. Kapatma anında gerçekleşir.
  Bir saat, herhangi bir 5 dk adımında talep kullanılabilir kapasiteyi aşarsa "eksik kapasiteli saat" sayılır.
  δ = 0'da bu tanım raporun saatlik tanımıyla birebir aynıdır (denetlenir).

Aileler (SARIMA, günlük ILP): sabit_alfa (yalnız plan), sabit_alfa_r2u (plan + uyumlu tampon), r0 (tepkisel ILP),
r1 (HPA benzeri eşik kuralı; ayarı yok). Ayar seçimi önceki deneylerle aynı: τ başına doğrulamada (aynı δ ile) en
ucuz uygun ayar, testte bir kez ölçüm; yalnız gerçek veri (kontrol).

Çıktılar (outputs/gecikme/): izgara.csv, secim.csv, tablo_ozet.csv
Çalıştırma: .venv/bin/python -m src.gecikme_deney [--havuzlar ...]
"""
from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config

from src.baselines.tepkisel import esik_param_oku, esik_simule, tepkisel_ilp
from src.common.config import yaml_oku, yol
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.forecast.saatlik import sarima_kokenler
from src.forecast.veri import bes_dk_seri
from src.ilp.kayan import kayan_plan
from src.ilp.model import IlpParam, Plan, SunucuTipleri, param_oku, tipleri_oku
from src.kayan_karsilastirma import ISINMA_GUN, grup_adi

log = logging.getLogger(__name__)
ALFA = (0.0, 0.05, 0.10, 0.20)
GECIKME_DK = (0, 2, 5, 10)
HEDEFLER = (0.05, 0.10, 0.15, 0.20)
ADIM_DK = 5


def kapasite(x: np.ndarray, tipler: SunucuTipleri, p: IlpParam) -> np.ndarray:
    """(K, T) → (T, 2) [çekirdek, GiB]."""
    return np.stack([(p.rho_cpu * tipler.C[:, None] * x).sum(0), (p.rho_mem * tipler.M[:, None] * x).sum(0)], 1)


def degerlendir(planli: np.ndarray, tepkisel: np.ndarray, y_planli: np.ndarray, y_tepkisel: np.ndarray,
                D5: np.ndarray, maske: np.ndarray, tipler: SunucuTipleri, p: IlpParam, delta: int) -> dict:
    """5 dk adımlarla gecikmeli değerlendirme.

    planli, tepkisel: (K, T) sunucu sayıları (toplam = planli + tepkisel); y_*: açılışlar (maliyet için).
    D5: (T·12, 2) 5 dk talep. maske: ölçülen saatler (T,).
    """
    K, T = planli.shape
    k_adim = 60 // ADIM_DK
    gec_adim = math.ceil(delta / ADIM_DK)
    onceki = np.concatenate([np.zeros((K, 1)), tepkisel[:, :-1]], axis=1)
    hazir_erken = np.minimum(tepkisel, onceki)                  # saatin ilk δ dakikasında hazır olan tepkisel
    kap_tam = kapasite(planli + tepkisel, tipler, p)             # (T, 2)
    kap_erken = kapasite(planli + hazir_erken, tipler, p)
    D = D5.reshape(T, k_adim, 2)
    kap_adim = np.repeat(kap_tam[:, None, :], k_adim, axis=1)
    kap_adim[:, :gec_adim, :] = kap_erken[:, None, :]
    acik = D - kap_adim                                         # (T, 12, 2)
    eksik = (acik > 1e-9).any(axis=(1, 2))
    calisma = (tipler.c[:, None] * (planli + tepkisel)).sum(0)
    acma = (p.s[:, None] * (y_planli + y_tepkisel)).sum(0)
    on_isinma = (tipler.c[:, None] * y_planli).sum(0) * delta / 60  # planlı açılışların δ dk erken açılma ücreti
    maliyet = (calisma + acma + on_isinma)[maske].sum() / (maske.sum() / 24)
    e = np.concatenate([[False], eksik[maske], [False]]).astype(np.int8)
    olay = np.flatnonzero(np.diff(e) == -1) - np.flatnonzero(np.diff(e) == 1)
    return {"maliyet_gunluk": float(maliyet), "eksik_saat_orani": float(eksik[maske].mean()),
            "olay_sure_max": float(olay.max()) if olay.size else 0.0}


def ayrik(toplam: Plan, baz: Plan) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """esik_simule çıktısını (baz + tampon) planlı ve tepkisel parçalara ayırır."""
    return baz.x, toplam.x - baz.x, baz.y, toplam.y - baz.y


def kosu(h: Havuz, y: np.ndarray, D5: np.ndarray, nokta: dict, bas: int, son: int, olcum_bas: int) -> list[dict]:
    cfg = yaml_oku("ilp")
    cozucu, zs = cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"]
    tipler = tipleri_oku()
    D, D_once = y[bas:son], y[bas - 1]
    D5p = D5[12 * bas: 12 * son]
    maske = np.arange(bas, son) >= olcum_bas
    sifir = np.zeros((tipler.K, son - bas))
    satir = []
    p_ref = param_oku(tipler)
    y_genis = np.vstack([y, np.repeat(y[-1:], 24, axis=0)])
    kahin = kayan_plan({t: y_genis[t: t + 24] for t in range(bas, son)}, bas, son, 24, tipler, p_ref, cozucu, zs).plan
    for delta in GECIKME_DK:
        k = degerlendir(kahin.x, sifir, kahin.y, sifir, D5p, maske, tipler, p_ref, delta)
        satir.append({"aile": "kahin", "ayar": 0.05, "gecikme_dk": delta} | k)
    gun = {O: nokta[O] for O in range(bas, son, 24)}
    e_hpa = esik_param_oku(tipler, uyumlu=False)
    for a in ALFA:
        p = param_oku(tipler, alpha=a)
        e = esik_param_oku(tipler, uyumlu=True, alpha=a)
        plan = kayan_plan(gun, bas, son, 24, tipler, p, cozucu, zs).plan
        r2 = ayrik(esik_simule(D, D_once, plan, tipler, p, e), plan)
        r0, _ = tepkisel_ilp(D, D_once, tipler, p, cozucu, zs)
        for delta in GECIKME_DK:
            satir.append({"aile": "sabit_alfa", "ayar": a, "gecikme_dk": delta}
                         | degerlendir(plan.x, sifir, plan.y, sifir, D5p, maske, tipler, p, delta))
            satir.append({"aile": "sabit_alfa_r2u", "ayar": a, "gecikme_dk": delta}
                         | degerlendir(*r2[:2], *r2[2:], D5p, maske, tipler, p, delta))
            satir.append({"aile": "r0", "ayar": a, "gecikme_dk": delta}
                         | degerlendir(sifir, r0.x, sifir, r0.y, D5p, maske, tipler, p, delta))
    p = param_oku(tipler)
    r1 = esik_simule(D, D_once, Plan(sifir, sifir, sifir, 0.0), tipler, p, e_hpa,
                     int(cfg["tepkisel"]["r1_min_sunucu"]))
    for delta in GECIKME_DK:
        satir.append({"aile": "r1", "ayar": 0.0, "gecikme_dk": delta}
                     | degerlendir(sifir, r1.x, sifir, r1.y, D5p, maske, tipler, p, delta))
    kahin_m = {r["gecikme_dk"]: r["maliyet_gunluk"] for r in satir if r["aile"] == "kahin"}
    for r in satir:
        r["regret_yuzde"] = 100 * (r["maliyet_gunluk"] - kahin_m[r["gecikme_dk"]]) / kahin_m[r["gecikme_dk"]]
    return satir


def gorev(h: Havuz) -> list[dict]:
    t0 = time.time()
    y, D5 = saatlik_seri(h), bes_dk_seri(h)
    n = len(y)
    assert np.allclose(D5[: 12 * n].reshape(n, 12, 2).max(1), y), f"{h.ad}: 5 dk ve saatlik tepe tutarsız"
    v0, v1, t1 = h.sinir["val"][0], h.sinir["val"][1], h.sinir["test"][1]
    sec_bas = v0 - 24 * ISINMA_GUN[h.veri_seti]
    gunler = list(range(sec_bas, t1, 24))
    nokta = dict(zip(gunler, sarima_kokenler(h, y, gunler)))
    satir = []
    for donem, bas, son, ob in (("secim", sec_bas, v1, v0), ("test", v0, t1, h.sinir["test"][0])):
        for r in kosu(h, y, D5, nokta, bas, son, ob):
            satir.append({"veri_seti": h.veri_seti, "havuz": h.ad, "donem": donem} | r)
    log.info("%s: %.0f s", h.ad, time.time() - t0)
    return satir


def secim_tablosu(izgara: pl.DataFrame) -> pl.DataFrame:
    sec, test = izgara.filter(pl.col("donem") == "secim"), izgara.filter(pl.col("donem") == "test")
    satirlar = []
    for (havuz, aile, delta), g in sec.filter(pl.col("aile") != "kahin").group_by(
            ["havuz", "aile", "gecikme_dk"], maintain_order=True):
        for tau in HEDEFLER:
            uygun = g.filter(pl.col("eksik_saat_orani") <= tau + 1e-9)
            s = (uygun.sort("maliyet_gunluk") if uygun.height
                 else g.sort(["eksik_saat_orani", "maliyet_gunluk"])).row(0, named=True)
            r = test.filter((pl.col("havuz") == havuz) & (pl.col("aile") == aile) & (pl.col("gecikme_dk") == delta)
                            & ((pl.col("ayar") - s["ayar"]).abs() < 1e-9)).row(0, named=True)
            satirlar.append({"veri_seti": r["veri_seti"], "havuz": havuz, "grup": grup_adi(havuz), "aile": aile,
                             "gecikme_dk": delta, "hedef": tau, "ayar": s["ayar"],
                             "secim_eksik": s["eksik_saat_orani"], "test_maliyet": r["maliyet_gunluk"],
                             "test_eksik": r["eksik_saat_orani"], "test_regret": r["regret_yuzde"],
                             "test_olay_max": r["olay_sure_max"],
                             "testte_saglandi": r["eksik_saat_orani"] <= tau + 1e-9})
    return pl.DataFrame(satirlar)


def main() -> None:
    ap = argparse.ArgumentParser(description="Açılış gecikmesi altında sağlamlık testi")
    ap.add_argument("--havuzlar", nargs="*")
    ap.add_argument("--n-is", type=int, default=yaml_oku("ilp")["cozucu"]["n_jobs"])
    a = ap.parse_args()
    havuzlar = [h for h in havuzlari_listele() if not a.havuzlar or h.ad in a.havuzlar]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        sonuc = Parallel(n_jobs=a.n_is, verbose=5)(delayed(gorev)(h) for h in havuzlar)
    izgara = pl.DataFrame([r for s in sonuc for r in s], infer_schema_length=None)
    # Denetim: δ = 0'da 5 dk tanımı saatlik tanımla aynı olmalı → sabit_alfa(α=0,05) raporun ILP sonucunu verir
    d = yol("outputs/gecikme")
    d.mkdir(parents=True, exist_ok=True)
    izgara.write_csv(d / "izgara.csv")
    secim = secim_tablosu(izgara)
    secim.write_csv(d / "secim.csv")
    (secim.group_by(["grup", "aile", "gecikme_dk", "hedef"], maintain_order=True)
     .agg(pl.col("test_maliyet").mean().alias("maliyet_gunluk"), (100 * pl.col("test_eksik")).mean().alias("eksik_saat_yuzde"),
          pl.col("test_regret").mean().alias("regret_yuzde"), pl.col("test_olay_max").mean().alias("olay_sure_max"),
          pl.col("testte_saglandi").mean().alias("hedef_saglama_orani"), pl.len().alias("n_havuz"))
     .write_csv(d / "tablo_ozet.csv"))
    log.info("Bitti. outputs/gecikme/{izgara,secim,tablo_ozet}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
