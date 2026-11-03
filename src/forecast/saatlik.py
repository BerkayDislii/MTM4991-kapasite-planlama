"""Saatlik kökenli 24 saat ileri tahminler (kayan ufuklu ILP için).

Raporun gün-ileri tahminleriyle aynı modeller ve aynı protokol (MIMARI.md Bölüm 5); tek fark, kökenin
yalnız gün başı (o = 24·g) değil her saat başı olmasıdır. Köken o'daki tahmin yalnız y[:o]'yu kullanır.

  - SARIMA (P1): M4'te seçilen derece (outputs/models/sarima_secim.csv). Doğrulama dönemi kökenlerinde eğitime,
    test dönemi kökenlerinde eğitim + doğrulamaya uydurulmuş parametreler sabittir; durum yeni gözlemlerle
    güncellenir (results.apply, refit yok) — src/forecast/sarima.py ile birebir aynı kural.
  - Son değer (P0)     : ŷ[o+h] = y[o−1]
  - Mevsimsel naif (P0): ŷ[o+h] = y[o+h−24]

Denetim: gün başı kökenlerindeki (o = 24·g) tahminler raporun forecast_<g>.parquet dosyalarıyla aynı olmalıdır
(gun_basi_farki). Sentetik senaryolarda parametreler bozulmamış seriden gelir, durum bozulmuş seride ilerler
(src/sentetik.py ile aynı).

Çıktı: outputs/kayan/tahmin/<senaryo>/<veri_seti>/<havuz>/<model>.parquet — sütunlar koken, h, cpu_hat, mem_hat
"""
from __future__ import annotations

import ast

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol
from src.forecast.ortak import Havuz, saatlik_seri

UFUK = 24
MODELLER = ("sarima", "son_deger", "mevsimsel_naif")


def sarima_secimi(h: Havuz) -> dict[int, dict]:
    """Kaynak dizini (0 = cpu, 1 = mem) → M4 seçim satırı."""
    df = pl.read_csv(yol(yaml_oku("forecast")["outputs"]["models_dir"]) / "sarima_secim.csv")
    out = {}
    for j, kaynak in enumerate(("cpu", "mem")):
        r = df.filter((pl.col("havuz") == h.ad) & (pl.col("kaynak") == kaynak))
        assert r.height == 1, f"{h.ad}/{kaynak}: sarima_secim.csv satırı yok"
        out[j] = r.row(0, named=True)
    return out


def sarima_kokenler(h: Havuz, y_seri: np.ndarray, kokenler: list[int]) -> np.ndarray:
    """SARIMA ile köken başına 24 saat ileri tahmin: (n_koken, 24, 2).

    y_seri: durumun ilerletileceği (gerçek ya da bozulmuş) saatlik seri; parametreler her zaman bozulmamış
    gerçek seriden uydurulur. Kökenin bulunduğu bölüm uydurma verisini belirler (val → eğitim, test → eğitim+val).
    """
    from src.forecast.sarima import uydur
    maxiter = yaml_oku("forecast")["sarima"]["maxiter"]
    y0 = saatlik_seri(h)
    t_son, v_son = h.sinir["train"][1], h.sinir["val"][1]
    secim = sarima_secimi(h)
    out = np.zeros((len(kokenler), UFUK, 2))
    for j in (0, 1):
        olcek = float(y0[:t_son, j].mean())
        order, sorder = ast.literal_eval(secim[j]["order"]), ast.literal_eval(secim[j]["seasonal_order"])
        # Köken doğrulama dönemi bitmeden önceyse (ısınma dahil) eğitime, sonra eğitim+doğrulamaya uydurulmuş model
        donem = ["test" if o >= v_son else "val" for o in kokenler]
        uyum = {}
        for ad, son in (("val", t_son), ("test", v_son)):
            if ad in donem:
                res, _ = uydur(y0[:son, j] / olcek, order, sorder, maxiter)
                assert res is not None, f"{h.ad}/{j}: SARIMA uydurulamadı"
                uyum[ad] = res.apply(y_seri[:, j] / olcek)        # durum tüm seri boyunca, parametreler sabit
        for i, o in enumerate(kokenler):
            r = uyum[donem[i]]
            out[i, :, j] = r.get_prediction(start=o, end=o + UFUK - 1, dynamic=True).predicted_mean * olcek
    return out


def naif_kokenler(model: str, y_seri: np.ndarray, kokenler: list[int]) -> np.ndarray:
    if model == "son_deger":
        return np.stack([np.repeat(y_seri[o - 1][None, :], UFUK, axis=0) for o in kokenler])
    assert model == "mevsimsel_naif"
    return np.stack([y_seri[o - 24: o - 24 + UFUK] for o in kokenler])


def kokenli_tahmin(h: Havuz, model: str, y_seri: np.ndarray, kokenler: list[int]) -> np.ndarray:
    """Model adına göre (n_koken, 24, 2) tahmin."""
    if model == "sarima":
        return sarima_kokenler(h, y_seri, kokenler)
    return naif_kokenler(model, y_seri, kokenler)


def gun_basi_farki(h: Havuz, model: str, kokenler: list[int], tahmin: np.ndarray) -> float:
    """Gün başı kökenlerinde raporun forecast_<g>.parquet tahminlerinden en büyük göreli fark."""
    d = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) / h.veri_seti / h.ad / model
    en_buyuk = 0.0
    for i, o in enumerate(kokenler):
        if o % 24:
            continue
        rapor = pl.read_parquet(d / f"forecast_{o // 24}.parquet").sort("slot").select("cpu_hat", "mem_hat").to_numpy()
        en_buyuk = max(en_buyuk, float(np.max(np.abs(rapor - tahmin[i]) / np.maximum(np.abs(rapor), 1e-9))))
    return en_buyuk


def tahmin_yaz(yol_, kokenler: list[int], tahmin: np.ndarray) -> None:
    yol_.parent.mkdir(parents=True, exist_ok=True)
    n = len(kokenler)
    pl.DataFrame({"koken": np.repeat(kokenler, UFUK).astype(np.int64), "h": np.tile(np.arange(UFUK), n),
                  "cpu_hat": tahmin[:, :, 0].ravel(), "mem_hat": tahmin[:, :, 1].ravel()}).write_parquet(yol_)


def tahmin_oku(yol_) -> dict[int, np.ndarray]:
    df = pl.read_parquet(yol_).sort("koken", "h")
    a = df.select("cpu_hat", "mem_hat").to_numpy().reshape(-1, UFUK, 2)
    return dict(zip(df["koken"].unique(maintain_order=True).to_list(), a))
