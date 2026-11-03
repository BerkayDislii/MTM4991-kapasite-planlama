"""P1 SARIMA (MIMARI.md Bölüm 5.2; kullanıcı kararı 2026-10-29, M4).

Seri başına (havuz × kaynak) seçim, yalnızca eğitim + doğrulama ile:
  1. d: eğitim döneminde ADF + KPSS. d = 0 ancak ADF birim kökü reddeder VE KPSS durağanlığı reddetmezse;
     aksi hâlde fark alınıp tekrar test edilir (en fazla max_d).
  2. D: eğitim döneminde STL mevsimsellik gücü F_s >= eşik ise 1, değilse 0.
  3. İki aile: mevsimsel olmayan (p,d,q)(0,0,0)_24 ve mevsimsel (p,d,q)(P,D,Q)_24, (P,Q) != (0,0).
     Her aile içinde (aynı d, D) AIC ile kazanan seçilir (eğitime uydurulmuş).
  4. Aile kazananları arasında nihai seçim: doğrulama dönemi gün-ileri MAE'si (parametreler eğitimden sabit).
Son model: seçilen derece eğitim+doğrulamaya yeniden uydurulur; test boyunca parametreler sabit, durum
refit=False ile yeni gözlemlerle güncellenir (results.apply, Bölüm 5.1).

Gün-ileri tahmin: filtre tüm seriye uygulanır, köken o'dan dinamik tahmin yalnız y[:o] bilgisini kullanır.

Çalıştırma: .venv/bin/python -m src.forecast.sarima
"""
import itertools
import logging
import time
import warnings
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import adfuller, kpss

from src.common.config import yaml_oku
from src.forecast.ortak import (BOLUMLER, Havuz, bolum_olcutleri, havuzlari_listele, modeller_dizini, saatlik_seri,
                                tahminleri_kaydet)

log = logging.getLogger(__name__)

KAYNAKLAR: tuple[str, ...] = ("cpu", "mem")


# ---------------------------------------------------------------- durağanlık ve mevsimsel fark

def duraganlik_d(x: np.ndarray, a: float, max_d: int) -> tuple[int, list[dict[str, Any]]]:
    """ADF + KPSS ile fark derecesi d; her denenen derece için test kaydı döner."""
    kayitlar = []
    for d in range(max_d + 1):
        seri = np.diff(x, n=d) if d else x
        adf_p = float(adfuller(seri, autolag="AIC", regression="c", result_object=False)[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_p = float(kpss(seri, regression="c", nlags="auto", result_object=False)[1])
        duragan = adf_p < a and kpss_p >= a
        kayitlar.append({"d_test": d, "adf_p": round(adf_p, 4), "kpss_p": round(kpss_p, 4), "duragan": duragan})
        if duragan:
            return d, kayitlar
    return max_d, kayitlar


def mevsimsellik_gucu(x: np.ndarray, m: int) -> float:
    """STL mevsimsellik gücü F_s = max(0, 1 - Var(R) / Var(S + R)) (Wang, Smith ve Hyndman, 2006)."""
    s = STL(x, period=m, robust=True).fit()
    return float(max(0.0, 1.0 - np.var(s.resid) / np.var(s.seasonal + s.resid)))


# ---------------------------------------------------------------- uydurma ve gün-ileri tahmin

def uydur(y: np.ndarray, order: tuple[int, int, int], sorder: tuple[int, int, int, int],
          maxiter: int) -> tuple[SARIMAXResults | None, bool]:
    """SARIMAX uydurur; d + D = 0 iken sabit terim eklenir. Dönüş: (sonuç | None, yakınsadı mı)."""
    trend = "c" if order[1] + sorder[1] == 0 else "n"
    with warnings.catch_warnings(record=True) as uyari:
        warnings.simplefilter("always")
        try:
            res = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend).fit(disp=False, maxiter=maxiter)
        except (np.linalg.LinAlgError, ValueError):
            return None, False
    yakinsadi = not any(issubclass(w.category, ConvergenceWarning) for w in uyari)
    if not np.isfinite(res.aic):
        return None, False
    return res, yakinsadi


def gun_ileri(res: SARIMAXResults, y_tum: np.ndarray, gunler: list[int], ufuk: int = 24) -> np.ndarray:
    """Sabit parametrelerle gün kökenlerinde gün-ileri tahmin: (n_gun, ufuk).

    apply (refit=False) tüm seriye filtre uygular; köken o'dan dinamik tahmin yalnız y[:o]'yu kullanır.
    """
    son = 24 * gunler[-1] + ufuk
    r = res.apply(y_tum[:son])
    return np.stack([r.get_prediction(start=24 * g, end=24 * g + ufuk - 1, dynamic=True).predicted_mean
                     for g in gunler])


def adaylar(d: int, D: int, s: dict[str, Any], m: int) -> dict[str, list[tuple]]:
    """Aile adı -> [(order, seasonal_order)] aday listesi."""
    ns = [((p, d, q), (0, 0, 0, 0)) for p, q in itertools.product(s["p"], s["q"])]
    ms = [((p, d, q), (P, D, Q, m)) for p, q, P, Q in itertools.product(s["p"], s["q"], s["P"], s["Q"])
          if (P, Q) != (0, 0)]
    return {"mevsimsel_olmayan": ns, "mevsimsel": ms}


def seri_calistir(h: Havuz, j: int) -> dict[str, Any]:
    """Bir havuzun tek kaynağı için seçim + son model; tüm bölüm tahminlerini ve seçim kayıtlarını döner."""
    cfg = yaml_oku("forecast")
    s, m = cfg["sarima"], cfg["season_hours"]
    y_ham = saatlik_seri(h)[:, j]
    t_son, v_son = h.sinir["train"][1], h.sinir["val"][1]
    # Sayısal kararlılık için eğitim ortalamasıyla ölçekleme (sabit çarpan: AIC sıralamasını değiştirmez)
    olcek = float(y_ham[:t_son].mean())
    y = y_ham / olcek
    gunler = h.degerlendirme_gunleri
    t0 = time.time()

    d, d_kayit = duraganlik_d(y[:t_son], s["significance"], s["max_d"])
    fs = mevsimsellik_gucu(y[:t_son], m)
    D = int(fs >= s["seasonal_strength_D1"])

    aday_kayit, kazananlar = [], {}
    for aile, liste in adaylar(d, D, s, m).items():
        en_iyi = None
        for order, sorder in liste:
            res, yak = uydur(y[:t_son], order, sorder, s["maxiter"])
            aday_kayit.append({"havuz": h.ad, "kaynak": KAYNAKLAR[j], "aile": aile, "order": str(order),
                               "seasonal_order": str(sorder), "aic": None if res is None else round(res.aic, 3),
                               "yakinsadi": yak})
            # Yakınsamayan uydurmalar AIC karşılaştırmasına alınmaz
            if res is not None and yak and (en_iyi is None or res.aic < en_iyi[0].aic):
                en_iyi = (res, order, sorder)
        if en_iyi is not None:
            res, order, sorder = en_iyi
            val_t = gun_ileri(res, y, gunler["val"]) * olcek
            val_g = np.stack([y_ham[24 * g: 24 * g + 24] for g in gunler["val"]])
            kazananlar[aile] = {"res": res, "order": order, "sorder": sorder, "aic": res.aic,
                                "val_mae": float(np.mean(np.abs(val_t - val_g)))}
    assert kazananlar, f"{h.ad}/{KAYNAKLAR[j]}: hiçbir SARIMA adayı yakınsamadı"
    secilen_aile = min(kazananlar, key=lambda a: kazananlar[a]["val_mae"])
    k = kazananlar[secilen_aile]

    # Eğitim ve doğrulama tahminleri: eğitime uydurulmuş seçilen model (doğrulama örneklem dışı)
    tahmin = {b: gun_ileri(k["res"], y, gunler[b]) * olcek for b in ("train", "val")}
    # Son model: eğitim + doğrulama ile yeniden uydurma, test boyunca parametreler sabit
    son_res, son_yak = uydur(y[:v_son], k["order"], k["sorder"], s["maxiter"])
    if son_res is None:
        raise RuntimeError(f"{h.ad}/{KAYNAKLAR[j]}: son model eğitim+doğrulamada uydurulamadı")
    tahmin["test"] = gun_ileri(son_res, y, gunler["test"]) * olcek

    secim = {"veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "kaynak": KAYNAKLAR[j], "d": d, "D": D,
             "stl_mevsimsellik_gucu": round(fs, 4),
             "duraganlik_testleri": "; ".join(f"d={x['d_test']}: ADF p={x['adf_p']}, KPSS p={x['kpss_p']}"
                                              for x in d_kayit),
             "secilen_aile": secilen_aile, "order": str(k["order"]), "seasonal_order": str(k["sorder"]),
             "son_model_yakinsadi": son_yak, "sure_s": round(time.time() - t0, 1)}
    for aile in ("mevsimsel_olmayan", "mevsimsel"):
        w = kazananlar.get(aile)
        secim[f"{aile}_aday"] = None if w is None else f"{w['order']}x{w['sorder']}"
        secim[f"{aile}_aic"] = None if w is None else round(w["aic"], 3)
        secim[f"{aile}_val_mae"] = None if w is None else round(w["val_mae"], 4)
    return {"havuz": h.ad, "j": j, "tahmin": tahmin, "secim": secim, "adaylar": aday_kayit}


def main() -> None:
    """Tüm havuz × kaynak serilerinde SARIMA'yı paralel çalıştırır; ölçüt, seçim ve aday tablolarını yazar."""
    cfg = yaml_oku("forecast")["sarima"]
    havuzlar = havuzlari_listele()
    gorevler = [(h, j) for h in havuzlar for j in range(len(KAYNAKLAR))]
    t0 = time.time()
    # Her süreç tek iş parçacığı kullanır (BLAS aşırı abonelik olmasın)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        sonuclar = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(delayed(seri_calistir)(h, j) for h, j in gorevler)
    log.info("SARIMA: %d seri %.1f dk", len(gorevler), (time.time() - t0) / 60)

    harita = {(r["havuz"], r["j"]): r for r in sonuclar}
    satirlar = []
    for h in havuzlar:
        y = saatlik_seri(h)
        # Kaynak tahminlerini (n_gun, 24, 2) olarak birleştir
        tahmin = {b: np.stack([harita[(h.ad, j)]["tahmin"][b] for j in range(2)], axis=-1) for b in BOLUMLER}
        tahminleri_kaydet(h, "sarima", tahmin)
        satirlar += bolum_olcutleri(h, y, tahmin, "sarima")
    md = modeller_dizini()
    pl.DataFrame(satirlar, infer_schema_length=None).write_csv(md / "metrikler_sarima.csv")
    pl.DataFrame([r["secim"] for r in sonuclar], infer_schema_length=None).write_csv(md / "sarima_secim.csv")
    pl.DataFrame([a for r in sonuclar for a in r["adaylar"]], infer_schema_length=None).write_csv(md / "sarima_adaylar.csv")
    log.info("SARIMA ölçüt / seçim / aday tabloları yazıldı")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
