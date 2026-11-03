"""İki aşamalı stokastik ILP'nin raporun yöntemiyle karşılaştırması (doğrulamada seçim, testte ölçüm).

Aileler (hepsi günlük planlama, gün başında; "_r2u" ekli olanlar raporun uyumlu tamponuyla çalıştırılır):
  sabit_alfa(_r2u)   rapor: deterministik ILP, kapasite ≥ (1 + α)·ŷ                 ayar: α ∈ ALFA
  stokastik(_r2u)    src/ilp/stokastik.py: tamponu ve eksik kapasite cezasını bilen   ayar: κ ∈ KAPPA
                     iki aşamalı model (S senaryo, kayan pencereli hata yolları)
Stokastik model tamponla birlikte çalışılacağını varsayarak kurulur; asıl karşılaştırılan sabit_alfa_r2u ↔
stokastik_r2u'dur. Tamponsuz satırlar "model tampona ne kadar yaslanıyor?" sorusu için raporlanır.

Seçim ve ölçüm src/kayan_karsilastirma.py ile aynıdır (hedef τ başına doğrulamada en ucuz uygun ayar, testte
ölçüm, sentetik senaryolar yalnız testi bozar). Hesap yükü nedeniyle yalnız SARIMA modeli ve (varsayılan) yalnız
Bitbrains havuzları kullanılır; Alibaba'da test dönemi tek gün ve çözümler sık sık zaman sınırına takılıyor.

Çıktılar (outputs/stokastik/): izgara.csv, secim.csv, tablo_ozet.csv
Çalıştırma: .venv/bin/python -m src.stokastik_karsilastirma [--havuzlar ...] [--n-is 14]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config

from src.baselines.tepkisel import esik_param_oku, esik_simule
from src.common.config import yaml_oku, yol
from src.forecast.ortak import Havuz, havuzlari_listele
from src.forecast.saatlik import gun_basi_farki, sarima_kokenler
from src.ilp.kayan import kayan_plan
from src.ilp.model import param_oku
from src.ilp.stokastik import gunluk_stokastik_plan
from src.kantil_karsilastirma import ozet_tablosu
from src.kayan_karsilastirma import ISINMA_GUN, Koşu, olcut, senaryo_serisi

log = logging.getLogger(__name__)
ALFA = (0.0, 0.05, 0.10, 0.20)
KAPPA = (1.0, 3.0, 10.0, 30.0, 100.0)
HEDEFLER = (0.05, 0.10, 0.15, 0.20)
SENARYOLAR = ("kontrol", "basamak", "darbe", "dusus")
S_SENARYO = 10                 # senaryo sayısı (çözüm süresi ↔ örneklem; pilotta S = 10 ile çözüm ≤ ~11 s)
PENCERE_GUN = 7
TOHUM = 2026


def kosu_izgarasi(k: Koşu, nokta: dict) -> list[dict]:
    tipler, satir = k.tipler, []
    gun = {O: nokta[O] for O in range(k.bas, k.son, 24)}
    for a in ALFA:
        p = param_oku(tipler, alpha=a)
        e = esik_param_oku(tipler, uyumlu=True, alpha=a)
        plan = kayan_plan(gun, k.bas, k.son, 24, tipler, p, k.cozucu, k.zs).plan
        satir.append({"aile": "sabit_alfa", "ayar": a} | olcut(plan, k.D, k.maske, tipler, p))
        satir.append({"aile": "sabit_alfa_r2u", "ayar": a}
                     | olcut(esik_simule(k.D, k.D_once, plan, tipler, p, e), k.D, k.maske, tipler, p))
    p0 = param_oku(tipler, alpha=0.0)
    e0 = esik_param_oku(tipler, uyumlu=True, alpha=0.0)
    for kappa in KAPPA:
        plan, sure = gunluk_stokastik_plan(nokta, k.y, k.bas, k.son, tipler, p0, e0, kappa, S_SENARYO,
                                           PENCERE_GUN, TOHUM, k.cozucu)
        ek = {"cozum_s_ort": float(np.mean(sure)) / 1000, "cozum_s_max": float(np.max(sure)) / 1000}
        satir.append({"aile": "stokastik", "ayar": kappa} | olcut(plan, k.D, k.maske, tipler, p0) | ek)
        satir.append({"aile": "stokastik_r2u", "ayar": kappa}
                     | olcut(esik_simule(k.D, k.D_once, plan, tipler, p0, e0), k.D, k.maske, tipler, p0) | ek)
    return satir


def gorev(h: Havuz, senaryolar: tuple[str, ...], denetim: bool) -> list[dict]:
    t0 = time.time()
    v0, v1, t1 = h.sinir["val"][0], h.sinir["val"][1], h.sinir["test"][1]
    sec_bas = v0 - 24 * ISINMA_GUN[h.veri_seti]
    satirlar = []
    for sc in senaryolar:
        y, sc_bas = senaryo_serisi(h, sc)
        ilk = sec_bas if sc == "kontrol" else v0
        kokenler = list(range(max(24, ilk - 24 * PENCERE_GUN), t1))
        nokta = dict(zip(kokenler, sarima_kokenler(h, y, kokenler)))
        if denetim and sc == "kontrol":
            gb = [o for o in kokenler if o >= v0]
            fark = gun_basi_farki(h, "sarima", gb, np.stack([nokta[o] for o in gb]))
            assert fark < 1e-6, f"{h.ad}: gün başı tahminleri raporla aynı değil ({fark:.2e})"
        donemler = [("test", Koşu(h, y, None, v0, t1, h.sinir["test"][0]))]
        if sc == "kontrol":
            donemler.insert(0, ("secim", Koşu(h, y, None, sec_bas, v1, v0)))
        for donem, k in donemler:
            ortak = {"veri_seti": h.veri_seti, "havuz": h.ad, "model": "sarima", "senaryo": sc,
                     "senaryo_bas": sc_bas, "donem": donem}
            kahin = k.kahin()
            for r in kosu_izgarasi(k, nokta):
                r["regret_yuzde"] = 100 * (r["maliyet_gunluk"] - kahin["maliyet_gunluk"]) / kahin["maliyet_gunluk"]
                satirlar.append(ortak | r)
        log.info("%s / %s tamam (%.0f s)", h.ad, sc, time.time() - t0)
    return satirlar


def secim_tablosu(izgara: pl.DataFrame) -> pl.DataFrame:
    sec, test = izgara.filter(pl.col("donem") == "secim"), izgara.filter(pl.col("donem") == "test")
    satirlar = []
    for (havuz, aile), g in sec.group_by(["havuz", "aile"], maintain_order=True):
        for tau in HEDEFLER:
            uygun = g.filter(pl.col("eksik_saat_orani") <= tau + 1e-9)
            s = (uygun.sort("maliyet_gunluk") if uygun.height
                 else g.sort(["eksik_saat_orani", "maliyet_gunluk"])).row(0, named=True)
            t = test.filter((pl.col("havuz") == havuz) & (pl.col("aile") == aile)
                            & ((pl.col("ayar") - s["ayar"]).abs() < 1e-9))
            for r in t.iter_rows(named=True):
                satirlar.append({"veri_seti": r["veri_seti"], "havuz": havuz, "model": "sarima", "aile": aile,
                                 "secim_yolu": "dogrulama", "hedef": tau, "ayar": s["ayar"],
                                 "secim_eksik": s["eksik_saat_orani"], "secimde_saglandi": bool(uygun.height),
                                 "senaryo": r["senaryo"], "test_maliyet": r["maliyet_gunluk"],
                                 "test_eksik": r["eksik_saat_orani"], "test_regret": r["regret_yuzde"],
                                 "test_olay_max": r["olay_sure_max"],
                                 "testte_saglandi": r["eksik_saat_orani"] <= tau + 1e-9})
    return pl.DataFrame(satirlar)


def main() -> None:
    ap = argparse.ArgumentParser(description="İki aşamalı stokastik ILP karşılaştırması")
    ap.add_argument("--havuzlar", nargs="*")
    ap.add_argument("--veri-seti", default="bitbrains", choices=["bitbrains", "alibaba", "ikisi"],
                    help="varsayılan yalnız Bitbrains: Alibaba'da test 1 gün ve çözümler çok ağır")
    ap.add_argument("--senaryolar", nargs="+", default=list(SENARYOLAR))
    ap.add_argument("--n-is", type=int, default=yaml_oku("ilp")["cozucu"]["n_jobs"])
    ap.add_argument("--denetim-yok", action="store_true")
    a = ap.parse_args()
    havuzlar = [h for h in havuzlari_listele() if (not a.havuzlar or h.ad in a.havuzlar)
                and (a.veri_seti == "ikisi" or h.veri_seti == a.veri_seti)]
    # Uzun süren Bitbrains görevleri önce başlasın (paralel yük dengesi)
    havuzlar.sort(key=lambda h: h.veri_seti != "bitbrains")
    log.info("%d havuz, %d paralel (SARIMA, S = %d, κ ∈ %s)", len(havuzlar), a.n_is, S_SENARYO, KAPPA)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        sonuc = Parallel(n_jobs=a.n_is, verbose=5)(
            delayed(gorev)(h, tuple(a.senaryolar), not a.denetim_yok) for h in havuzlar)
    d = yol("outputs/stokastik")
    d.mkdir(parents=True, exist_ok=True)
    izgara = pl.DataFrame([r for s in sonuc for r in s], infer_schema_length=None)
    izgara.write_csv(d / "izgara.csv")
    secim = secim_tablosu(izgara)
    secim.write_csv(d / "secim.csv")
    ozet_tablosu(secim).write_csv(d / "tablo_ozet.csv")
    log.info("Bitti: %.1f dk. outputs/stokastik/{izgara,secim,tablo_ozet}.csv", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
