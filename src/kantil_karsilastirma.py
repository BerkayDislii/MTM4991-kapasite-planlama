"""Kantil tahminle planlama: sabit güvenlik payı α yerine veriden öğrenilen pay.

Aileler (hepsi raporun günlük ILP'si, R = 24; "_r2u" ekli olanlara plana uyumlu tepkisel tampon eklenir):
  sabit_alfa       rapor: ILP girdisi nokta tahmin, kapasite ≥ (1 + α)·ŷ           ayar: α ∈ ALFA
  konformal        ILP girdisi kayan pencereli konformal p-kantili, α = 0          ayar: p ∈ P
  sarima_aralik    ILP girdisi SARIMA normal aralığının p-kantili, α = 0 (yalnız SARIMA)

Değerlendirme kayan ufuk karşılaştırmasıyla aynıdır (src/kayan_karsilastirma.py): her hizmet hedefi τ için
ayar doğrulama döneminde seçilir (τ'yu sağlayan en ucuz ayar), testte bir kez ölçülür. Kantil aileleri için ek
olarak seçimsiz bir kural da raporlanır: p = 1 − τ/2 (iki kaynağın her biri için τ/2 aşım olasılığı; birleşik
aşım olasılığı ≤ τ, Bonferroni). Kural doğrulama verisine hiç bakmaz.

Çıktılar (outputs/kantil/): izgara.csv, secim.csv, tablo_ozet.csv, kalibrasyon.csv (kantillerin gözlenen kapsaması)
Çalıştırma: .venv/bin/python -m src.kantil_karsilastirma [--havuzlar ...] [--n-is 14]
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
from src.forecast.kantil import kapsama, konformal_kantil, sarima_aralik_kantil
from src.forecast.ortak import Havuz, havuzlari_listele
from src.forecast.saatlik import gun_basi_farki, kokenli_tahmin, sarima_kokenler
from src.ilp.kayan import kayan_plan
from src.ilp.model import param_oku, tipleri_oku
from src.kayan_karsilastirma import ISINMA_GUN, Koşu, grup_adi, olcut, senaryo_serisi

log = logging.getLogger(__name__)
ALFA = (0.0, 0.05, 0.10, 0.20)
P = (0.5, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975)
HEDEFLER = (0.05, 0.10, 0.15, 0.20)
MODELLER = ("sarima", "son_deger", "mevsimsel_naif")
SENARYOLAR = ("kontrol", "basamak", "darbe", "dusus")
PENCERE_GUN = 7


def kural_p(tau: float) -> float:
    return round(1 - tau / 2, 3)


def nokta_tahminleri(h: Havuz, model: str, y: np.ndarray, kokenler: list[int]):
    """Saatlik kökenli nokta tahminleri (ve SARIMA için standart hata)."""
    if model == "sarima":
        T, se = sarima_kokenler(h, y, kokenler, se_dondur=True)
        return dict(zip(kokenler, T)), dict(zip(kokenler, se))
    return dict(zip(kokenler, kokenli_tahmin(h, model, y, kokenler))), None


def kosu_izgarasi(k: Koşu, nokta: dict, se: dict | None) -> tuple[list[dict], dict]:
    """Bir pencere için tüm ailelerin ayarları. Döndürür: (satırlar, {p: konformal kantiller})."""
    tipler = k.tipler
    gun = list(range(k.bas, k.son, 24))
    satir, konf = [], {}

    def planla(aile: str, ayar: float, tahmin: dict, alfa: float) -> None:
        p_ = param_oku(tipler, alpha=alfa)
        e = esik_param_oku(tipler, uyumlu=True, alpha=alfa)
        plan = kayan_plan(tahmin, k.bas, k.son, 24, tipler, p_, k.cozucu, k.zs).plan
        satir.append({"aile": aile, "ayar": ayar} | olcut(plan, k.D, k.maske, tipler, p_))
        tam = esik_simule(k.D, k.D_once, plan, tipler, p_, e)
        satir.append({"aile": f"{aile}_r2u", "ayar": ayar} | olcut(tam, k.D, k.maske, tipler, p_))

    nokta_gun = {O: nokta[O] for O in gun}
    for a in ALFA:
        planla("sabit_alfa", a, nokta_gun, a)
    for p in P:
        konf[p] = konformal_kantil(nokta, k.y, gun, p, PENCERE_GUN)
        planla("konformal", p, konf[p], 0.0)
        if se is not None:
            planla("sarima_aralik", p, sarima_aralik_kantil(nokta, se, gun, p), 0.0)
    return satir, konf


def gorev(h: Havuz, model: str, senaryolar: tuple[str, ...], denetim: bool) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    v0, v1, t1 = h.sinir["val"][0], h.sinir["val"][1], h.sinir["test"][1]
    sec_bas = v0 - 24 * ISINMA_GUN[h.veri_seti]
    satirlar, kal = [], []
    for sc in senaryolar:
        y, sc_bas = senaryo_serisi(h, sc)
        ilk = sec_bas if sc == "kontrol" else v0
        kal_bas = max(24, ilk - 24 * PENCERE_GUN)                    # kalibrasyon penceresi için geçmiş kökenler
        kokenler = list(range(kal_bas, t1))
        nokta, se = nokta_tahminleri(h, model, y, kokenler)
        if denetim and sc == "kontrol":
            gb = [o for o in kokenler if o >= v0]
            fark = gun_basi_farki(h, model, gb, np.stack([nokta[o] for o in gb]))
            assert fark < 1e-6, f"{h.ad}/{model}: gün başı tahminleri raporla aynı değil ({fark:.2e})"
        donemler = [("test", Koşu(h, y, None, v0, t1, h.sinir["test"][0]))]
        if sc == "kontrol":
            donemler.insert(0, ("secim", Koşu(h, y, None, sec_bas, v1, v0)))
        for donem, k in donemler:
            ortak = {"veri_seti": h.veri_seti, "havuz": h.ad, "model": model, "senaryo": sc,
                     "senaryo_bas": sc_bas, "donem": donem}
            kahin = k.kahin()
            izgara, konf = kosu_izgarasi(k, nokta, se)
            for r in izgara:
                r["regret_yuzde"] = 100 * (r["maliyet_gunluk"] - kahin["maliyet_gunluk"]) / kahin["maliyet_gunluk"]
                satirlar.append(ortak | r)
            if donem == "test":
                olcum_gun = [O for O in range(k.bas, k.son, 24) if O >= h.sinir["test"][0]]
                for p, q in konf.items():
                    c = kapsama(q, y, olcum_gun)
                    kal.append(ortak | {"p": p, "kapsama_cpu": float(c[0]), "kapsama_mem": float(c[1])})
    log.info("%s / %s: %.0f s", h.ad, model, time.time() - t0)
    return satirlar, kal


def secim_tablosu(izgara: pl.DataFrame) -> pl.DataFrame:
    """τ başına: doğrulamada seçilen ayar (tüm aileler) ve seçimsiz kural (kantil aileleri), test sonuçlarıyla."""
    sec = izgara.filter(pl.col("donem") == "secim")
    test = izgara.filter(pl.col("donem") == "test")
    satirlar = []
    for (havuz, model, aile), g in sec.group_by(["havuz", "model", "aile"], maintain_order=True):
        for tau in HEDEFLER:
            uygun = g.filter(pl.col("eksik_saat_orani") <= tau + 1e-9)
            secilen = (uygun.sort("maliyet_gunluk") if uygun.height
                       else g.sort(["eksik_saat_orani", "maliyet_gunluk"])).row(0, named=True)
            yollar = [("dogrulama", secilen["ayar"], secilen["eksik_saat_orani"], bool(uygun.height))]
            if not aile.startswith("sabit_alfa"):
                kural = g.filter((pl.col("ayar") - kural_p(tau)).abs() < 1e-9).row(0, named=True)
                yollar.append(("kural", kural["ayar"], kural["eksik_saat_orani"], kural["eksik_saat_orani"] <= tau))
            for yol_, ayar, sec_eksik, sec_ok in yollar:
                t = test.filter((pl.col("havuz") == havuz) & (pl.col("model") == model) & (pl.col("aile") == aile)
                                & ((pl.col("ayar") - ayar).abs() < 1e-9))
                for r in t.iter_rows(named=True):
                    satirlar.append({"veri_seti": r["veri_seti"], "havuz": havuz, "model": model, "aile": aile,
                                     "secim_yolu": yol_, "hedef": tau, "ayar": ayar, "secim_eksik": sec_eksik,
                                     "secimde_saglandi": sec_ok, "senaryo": r["senaryo"],
                                     "test_maliyet": r["maliyet_gunluk"], "test_eksik": r["eksik_saat_orani"],
                                     "test_regret": r["regret_yuzde"], "test_olay_max": r["olay_sure_max"],
                                     "testte_saglandi": r["eksik_saat_orani"] <= tau + 1e-9})
    return pl.DataFrame(satirlar)


def ozet_tablosu(secim: pl.DataFrame) -> pl.DataFrame:
    return (secim.with_columns(pl.col("havuz").map_elements(grup_adi, return_dtype=pl.String).alias("grup"))
            .group_by(["grup", "senaryo", "model", "aile", "secim_yolu", "hedef"], maintain_order=True)
            .agg(pl.col("test_maliyet").mean().alias("maliyet_gunluk"),
                 (100 * pl.col("test_eksik")).mean().alias("eksik_saat_yuzde"),
                 pl.col("test_regret").mean().alias("regret_yuzde"),
                 pl.col("test_olay_max").mean().alias("olay_sure_max"),
                 pl.col("testte_saglandi").mean().alias("hedef_saglama_orani"), pl.len().alias("n_havuz")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Kantil tahminle planlama: doğrulamada seçim, testte ölçüm")
    ap.add_argument("--havuzlar", nargs="*")
    ap.add_argument("--modeller", nargs="+", default=list(MODELLER))
    ap.add_argument("--senaryolar", nargs="+", default=list(SENARYOLAR))
    ap.add_argument("--n-is", type=int, default=yaml_oku("ilp")["cozucu"]["n_jobs"])
    ap.add_argument("--denetim-yok", action="store_true")
    a = ap.parse_args()
    havuzlar = [h for h in havuzlari_listele() if not a.havuzlar or h.ad in a.havuzlar]
    gorevler = [(h, m) for h in havuzlar for m in a.modeller]
    log.info("%d havuz × %d model = %d görev, %d paralel", len(havuzlar), len(a.modeller), len(gorevler), a.n_is)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        sonuc = Parallel(n_jobs=a.n_is, verbose=5)(
            delayed(gorev)(h, m, tuple(a.senaryolar), not a.denetim_yok) for h, m in gorevler)
    d = yol("outputs/kantil")
    d.mkdir(parents=True, exist_ok=True)
    izgara = pl.DataFrame([r for s, _ in sonuc for r in s], infer_schema_length=None)
    izgara.write_csv(d / "izgara.csv")
    pl.DataFrame([r for _, k in sonuc for r in k], infer_schema_length=None).write_csv(d / "kalibrasyon.csv")
    secim = secim_tablosu(izgara)
    secim.write_csv(d / "secim.csv")
    ozet_tablosu(secim).write_csv(d / "tablo_ozet.csv")
    log.info("Bitti: %.1f dk. outputs/kantil/{izgara,secim,tablo_ozet,kalibrasyon}.csv", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
