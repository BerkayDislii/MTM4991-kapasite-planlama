"""Kayan ufuk — tam karşılaştırma: ayarlar doğrulama döneminde seçilir, test döneminde bir kez ölçülür.

Neden? Pilot (src/kayan_deney.py) en iyi ayarı test verisine bakarak gösteriyordu; bu iyimser bir tablodur.
Burada her strateji ailesi için hizmet hedefi τ (izin verilen en büyük eksik kapasiteli saat oranı) verilir:
  1. Seçim (doğrulama dönemi): ailenin ayar ızgarasından, doğrulamada eksik saat oranı ≤ τ olan en ucuz ayar.
     Hiçbiri τ'yu sağlamıyorsa eksik saati en düşük (eşitlikte en ucuz) ayar seçilir.
  2. Ölçüm (test dönemi): seçilen ayar test döneminde (gerçek veri ve sentetik senaryolar) bir kez ölçülür;
     τ'nun testte de sağlanıp sağlanmadığı raporlanır.

Aileler ve ızgaraları (α: güvenlik payı, R: yeniden planlama aralığı, saat):
  gunluk_ilp   raporun günlük ILP'si                 α ∈ ALFA            (R = 24, gün başına hizalı)
  gunluk_r2u   günlük ILP + uyumlu tampon (rapor)     α ∈ ALFA
  kayan_ilp    kayan ufuklu ILP                       α ∈ ALFA, R ∈ R_KAYAN
  kayan_r2u    kayan ufuklu ILP + uyumlu tampon       α ∈ ALFA, R ∈ R_KAYAN
  r0           tepkisel ILP (ufuk 1, girdi D_{t−1})   α ∈ ALFA            (tahmin kullanmaz)
Planlama modelleri: SARIMA ve iki naif taban (LSTM saatlik kökenler için yeniden eğitim gerektirdiğinden yok).

Isınma: seçim koşusu doğrulama döneminden ISINMA_GUN gün önce soğuk başlar; test koşusu raporla aynı biçimde
doğrulama döneminin başından soğuk başlar. Senaryolar yalnız testi bozar, bu yüzden seçim senaryodan bağımsızdır.

Çıktılar (outputs/kayan/): izgara.csv (her ayarın seçim ve test ölçütleri), secim.csv (τ başına seçilen ayarlar
ve test sonuçları), tablo_ozet.csv (grup ortalamaları)
Çalıştırma: .venv/bin/python -m src.kayan_karsilastirma [--havuzlar ...] [--n-is 14]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config

from src.baselines.tepkisel import esik_param_oku, esik_simule, tepkisel_ilp
from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.forecast.saatlik import UFUK, gun_basi_farki, kokenli_tahmin
from src.forecast.veri import bes_dk_seri
from src.ilp.kayan import kayan_plan
from src.ilp.model import Plan, param_oku, tipleri_oku
from src.sentetik import VERI_SETLERI, baslangic_sec, bozulmus_seri

log = logging.getLogger(__name__)
ALFA = (0.0, 0.05, 0.10, 0.20)
R_KAYAN = (1, 3, 6, 12)
HEDEFLER = (0.05, 0.10, 0.15, 0.20)
MODELLER = ("sarima", "son_deger", "mevsimsel_naif")
SENARYOLAR = ("kontrol", "basamak", "darbe", "dusus")
ISINMA_GUN = {"bitbrains": 2, "alibaba": 1}
AILE_SIRASI = ("gunluk_ilp", "gunluk_r2u", "kayan_ilp", "kayan_r2u", "r0")


# ---------------------------------------------------------------- yardımcılar

def senaryo_serisi(h: Havuz, senaryo: str) -> tuple[np.ndarray, int | None]:
    """Senaryonun saatlik tepe serisi ve başlangıç saati (raporun sentetik deneyiyle aynı tohum ve sıra)."""
    cfg = yaml_oku("synthetic")
    adlar = list(cfg["senaryolar"])
    sc = cfg["senaryolar"][senaryo]
    rng = np.random.default_rng([cfg["tohum"], adlar.index(senaryo), VERI_SETLERI.index(h.veri_seti)])
    bas = baslangic_sec(sc, h.sinir["test"], rng)
    if sc["tur"] == "yok":
        return saatlik_seri(h), None
    k = yaml_oku("preprocess")["grid"]["steps_per_slot"]
    return bozulmus_seri(bes_dk_seri(h), sc, h.sinir["test"], bas, k)[1], bas


def olcut(plan: Plan, D: np.ndarray, maske: np.ndarray, tipler, p) -> dict[str, float]:
    alt = Plan(plan.x[:, maske], plan.y[:, maske], plan.z[:, maske], amac=float("nan"))
    o = plan_olcutleri(alt, D[maske, 0], D[maske, 1], tipler, p)
    return {"maliyet_gunluk": o["maliyet"] / (o["n_dilim"] / 24), "eksik_saat_orani": o["eksik_saat_orani"],
            "olay_sure_max": o["olay_sure_max"], "n_acma": o["n_acma"]}


class Koşu:
    """Bir pencere ([bas, son), ölçüm maskesi) üzerinde tüm ailelerin ayarlarını çalıştırır."""

    def __init__(self, h: Havuz, y: np.ndarray, tahmin: dict | None, bas: int, son: int, olcum_bas: int):
        cfg = yaml_oku("ilp")
        self.cozucu, self.zs = cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"]
        self.tipler = tipleri_oku()
        self.h, self.y, self.tahmin, self.bas, self.son = h, y, tahmin, bas, son
        self.D, self.D_once = y[bas:son], y[bas - 1]
        self.maske = np.arange(bas, son) >= olcum_bas

    def kahin(self) -> dict:
        y_genis = np.vstack([self.y, np.repeat(self.y[-1:], UFUK, axis=0)])
        p = param_oku(self.tipler)
        plan = kayan_plan({t: y_genis[t: t + UFUK] for t in range(self.bas, self.son)}, self.bas, self.son, 24,
                          self.tipler, p, self.cozucu, self.zs).plan
        return olcut(plan, self.D, self.maske, self.tipler, p)

    def r0(self, a: float) -> dict:
        p = param_oku(self.tipler, alpha=a)
        plan, _ = tepkisel_ilp(self.D, self.D_once, self.tipler, p, self.cozucu, self.zs)
        return olcut(plan, self.D, self.maske, self.tipler, p)

    def plan_ailesi(self, a: float, R: int) -> dict[str, dict]:
        """α ve R için tamponsuz ve tamponlu sonuç (R = 24 → günlük aileler)."""
        p = param_oku(self.tipler, alpha=a)
        e = esik_param_oku(self.tipler, uyumlu=True, alpha=a)
        assert R == 24 or R in R_KAYAN
        if R == 24:
            assert self.bas % 24 == 0, "günlük ILP gün başına hizalı başlamalı"
        plan = kayan_plan(self.tahmin, self.bas, self.son, R, self.tipler, p, self.cozucu, self.zs).plan
        tamponlu = esik_simule(self.D, self.D_once, plan, self.tipler, p, e)
        ad = ("gunluk" if R == 24 else "kayan")
        return {f"{ad}_ilp": olcut(plan, self.D, self.maske, self.tipler, p),
                f"{ad}_r2u": olcut(tamponlu, self.D, self.maske, self.tipler, p)}

    def izgara(self, r0_dahil: bool) -> list[dict]:
        satir = []
        for a in ALFA:
            for R in (24,) + R_KAYAN:
                for aile, o in self.plan_ailesi(a, R).items():
                    satir.append({"aile": aile, "alfa": a, "R": R} | o)
            if r0_dahil:
                satir.append({"aile": "r0", "alfa": a, "R": 1} | self.r0(a))
        return satir


# ---------------------------------------------------------------- görev: bir havuz × model

def gorev(h: Havuz, model: str, senaryolar: tuple[str, ...], r0_dahil: bool, denetim: bool) -> list[dict]:
    t0 = time.time()
    v0, v1, t1 = h.sinir["val"][0], h.sinir["val"][1], h.sinir["test"][1]
    sec_bas = v0 - 24 * ISINMA_GUN[h.veri_seti]
    satirlar = []
    for sc in senaryolar:
        y, sc_bas = senaryo_serisi(h, sc)
        bas_k = sec_bas if sc == "kontrol" else v0
        kokenler = list(range(bas_k, t1))
        T = kokenli_tahmin(h, model, y, kokenler)
        if denetim and sc == "kontrol":
            gb = [o for o in kokenler if o >= v0]
            fark = gun_basi_farki(h, model, gb, T[len(kokenler) - len(gb):])
            assert fark < 1e-6, f"{h.ad}/{model}: gün başı tahminleri raporla aynı değil ({fark:.2e})"
        tahmin = dict(zip(kokenler, T))
        donemler = [("test", Koşu(h, y, tahmin, v0, t1, h.sinir["test"][0]))]
        if sc == "kontrol":
            donemler.insert(0, ("secim", Koşu(h, y, tahmin, sec_bas, v1, v0)))
        for donem, k in donemler:
            ortak = {"veri_seti": h.veri_seti, "havuz": h.ad, "model": model, "senaryo": sc,
                     "senaryo_bas": sc_bas, "donem": donem}
            kahin = k.kahin()
            for r in k.izgara(r0_dahil):
                r["regret_yuzde"] = 100 * (r["maliyet_gunluk"] - kahin["maliyet_gunluk"]) / kahin["maliyet_gunluk"]
                satirlar.append(ortak | r)
            satirlar.append(ortak | {"aile": "kahin", "alfa": 0.05, "R": 24} | kahin | {"regret_yuzde": 0.0})
    log.info("%s / %s: %.0f s", h.ad, model, time.time() - t0)
    return satirlar


# ---------------------------------------------------------------- seçim ve özet

def secim_tablosu(izgara: pl.DataFrame) -> pl.DataFrame:
    """Her (havuz, model, aile, τ) için doğrulamada seçilen ayar ve onun tüm test senaryolarındaki sonucu."""
    sec = izgara.filter(pl.col("donem") == "secim")
    test = izgara.filter(pl.col("donem") == "test")
    anahtar = ["havuz", "model", "aile"]
    satirlar = []
    for (havuz, model, aile), g in sec.filter(pl.col("aile") != "kahin").group_by(anahtar, maintain_order=True):
        for tau in HEDEFLER:
            uygun = g.filter(pl.col("eksik_saat_orani") <= tau + 1e-9)
            secilen = (uygun.sort("maliyet_gunluk") if uygun.height else
                       g.sort(["eksik_saat_orani", "maliyet_gunluk"])).row(0, named=True)
            t = test.filter((pl.col("havuz") == havuz) & (pl.col("model") == model) & (pl.col("aile") == aile)
                            & (pl.col("alfa") == secilen["alfa"]) & (pl.col("R") == secilen["R"]))
            for r in t.iter_rows(named=True):
                satirlar.append({"veri_seti": r["veri_seti"], "havuz": havuz, "model": model, "aile": aile,
                                 "hedef": tau, "secimde_saglandi": bool(uygun.height), "alfa": secilen["alfa"],
                                 "R": secilen["R"], "secim_maliyet": secilen["maliyet_gunluk"],
                                 "secim_eksik": secilen["eksik_saat_orani"], "senaryo": r["senaryo"],
                                 "test_maliyet": r["maliyet_gunluk"], "test_eksik": r["eksik_saat_orani"],
                                 "test_regret": r["regret_yuzde"], "test_olay_max": r["olay_sure_max"],
                                 "testte_saglandi": r["eksik_saat_orani"] <= tau + 1e-9})
    return pl.DataFrame(satirlar)


def grup_adi(havuz: str) -> str:
    if havuz.startswith("bitbrains_fastStorage_s"):
        return "bitbrains_fastStorage_alt"
    if havuz.startswith("alibaba_"):
        return "alibaba_" + havuz.split("_")[-1]
    return havuz


def ozet_tablosu(secim: pl.DataFrame) -> pl.DataFrame:
    return (secim.with_columns(pl.col("havuz").map_elements(grup_adi, return_dtype=pl.String).alias("grup"))
            .group_by(["grup", "senaryo", "model", "aile", "hedef"], maintain_order=True)
            .agg(pl.col("test_maliyet").mean().alias("maliyet_gunluk"),
                 (100 * pl.col("test_eksik")).mean().alias("eksik_saat_yuzde"),
                 pl.col("test_regret").mean().alias("regret_yuzde"),
                 pl.col("test_olay_max").mean().alias("olay_sure_max"),
                 pl.col("testte_saglandi").mean().alias("hedef_saglama_orani"),
                 pl.len().alias("n_havuz")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Kayan ufuk: doğrulamada seçim, testte ölçüm")
    ap.add_argument("--havuzlar", nargs="*", help="yalnız bu havuzlar (varsayılan: tümü)")
    ap.add_argument("--modeller", nargs="+", default=list(MODELLER))
    ap.add_argument("--senaryolar", nargs="+", default=list(SENARYOLAR))
    ap.add_argument("--n-is", type=int, default=yaml_oku("ilp")["cozucu"]["n_jobs"])
    ap.add_argument("--denetim-yok", action="store_true", help="gün başı tahmin denetimini atla")
    a = ap.parse_args()
    havuzlar = [h for h in havuzlari_listele() if not a.havuzlar or h.ad in a.havuzlar]
    gorevler = [(h, m, i == 0) for h in havuzlar for i, m in enumerate(a.modeller)]
    log.info("%d havuz × %d model = %d görev, %d paralel", len(havuzlar), len(a.modeller), len(gorevler), a.n_is)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        sonuc = Parallel(n_jobs=a.n_is, verbose=5)(
            delayed(gorev)(h, m, tuple(a.senaryolar), r0, not a.denetim_yok) for h, m, r0 in gorevler)
    izgara = pl.DataFrame([r for s in sonuc for r in s], infer_schema_length=None)
    d = yol("outputs/kayan")
    d.mkdir(parents=True, exist_ok=True)
    izgara.write_csv(d / "izgara.csv")
    secim = secim_tablosu(izgara)
    secim.write_csv(d / "secim.csv")
    ozet_tablosu(secim).write_csv(d / "tablo_ozet.csv")
    log.info("Bitti: %.1f dk. outputs/kayan/{izgara,secim,tablo_ozet}.csv", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
