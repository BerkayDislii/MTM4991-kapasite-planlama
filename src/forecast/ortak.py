"""Tahmin modelleri için ortak yardımcılar (MIMARI.md Bölüm 5): havuz listesi, bölmeler, gün-ileri kökenler,
ölçütler ve tahmin dosyası sözleşmesi.

Gün-ileri walk-forward: gün d'nin kökeni o = 24·d; o anında yalnızca y[:o] bilinir, ŷ[o : o+24] üretilir.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol

BOLUMLER: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Havuz:
    """Bir tahmin görevinin tanımı: havuz, veri seti ve dilim sınırları (saat cinsinden, [bas, son))."""
    ad: str
    veri_seti: str                 # bitbrains | alibaba
    sinir: dict[str, tuple[int, int]]
    lookback: int                  # LSTM bakış penceresi (saat)
    uye_anahtari: str              # pool_members.json anahtarı
    r: float | None = None         # Alibaba M_ref oranı (GB/çekirdek)

    @property
    def degerlendirme_gunleri(self) -> dict[str, list[int]]:
        """Bölüm başına kökeni değerlendirilen günler; eğitimde bakış penceresi dolmadan gün alınmaz."""
        ilk = max(1, math.ceil(self.lookback / 24))
        gunler = {b: list(range(s // 24, e // 24)) for b, (s, e) in self.sinir.items()}
        gunler["train"] = [g for g in gunler["train"] if g >= ilk]
        return gunler


def havuzlari_listele() -> list[Havuz]:
    """forecast.yaml + preprocess.yaml'dan tüm havuz görevlerini üretir (Bitbrains 7, Alibaba 8 × 3 r)."""
    fcfg, pcfg = yaml_oku("forecast"), yaml_oku("preprocess")
    gorevler: list[Havuz] = []
    for vs in ("bitbrains", "alibaba"):
        b = pcfg[vs]["split_days"]
        t, v, s = 24 * b["train"], 24 * (b["train"] + b["val"]), 24 * (b["train"] + b["val"] + b["test"])
        sinir = {"train": (0, t), "val": (t, v), "test": (v, s)}
        lb = fcfg["datasets"][vs]["lookback_hours"]
        if vs == "bitbrains":
            gorevler += [Havuz(ad, vs, sinir, lb, ad) for ad in fcfg["datasets"][vs]["pools"]]
        else:
            n_havuz = sum(1 for k in _uyeler() if k.startswith("alibaba_p"))
            gorevler += [Havuz(f"alibaba_p{i}_r{r}", vs, sinir, lb, f"alibaba_p{i}", float(r))
                         for i in range(n_havuz) for r in pcfg["alibaba"]["r_grid"]]
    return gorevler


def _uyeler() -> dict[str, Any]:
    """pool_members.json içeriği."""
    return json.loads(yol("data/processed/pool_members.json").read_text(encoding="utf-8"))


def saatlik_seri(h: Havuz) -> np.ndarray:
    """pool_<ad>.parquet'ten (n_dilim, 2) dizisi: sütunlar cpu_peak, mem_peak. Bölme dışı fazla dilimler atılır."""
    df = pl.read_parquet(yol(f"data/processed/pool_{h.ad}.parquet")).sort("slot")
    n = h.sinir["test"][1]
    assert df.height >= n, f"{h.ad}: {df.height} dilim < {n}"
    return df.select("cpu_peak", "mem_peak").to_numpy()[:n].astype(np.float64)


def gun_ileri_hedef(y: np.ndarray, gunler: list[int], ufuk: int = 24) -> np.ndarray:
    """Gün listesindeki gerçek değerler: (n_gun, ufuk, 2)."""
    return np.stack([y[24 * g: 24 * g + ufuk] for g in gunler])


# ---------------------------------------------------------------- ölçütler (Bölüm 5.3)

def mase_olcegi(y: np.ndarray, egitim_son: int, m: int = 24) -> np.ndarray:
    """MASE paydası: eğitim dönemi örneklem-içi mevsimsel naif MAE'si, kaynak başına (2,)."""
    e = y[:egitim_son]
    return np.mean(np.abs(e[m:] - e[:-m]), axis=0)


def olcutler(gercek: np.ndarray, tahmin: np.ndarray, olcek: np.ndarray) -> list[dict[str, float]]:
    """Kaynak başına MAE, RMSE, MAPE (%), MASE, eksik tahmin oranı (ŷ < y) ve ortalama eksik miktar.

    gercek/tahmin: (..., 2) -> son eksen kaynak (cpu, mem). MAPE için y > 0 olmalı (havuz tepeleri pozitif).
    """
    g, t = gercek.reshape(-1, 2), tahmin.reshape(-1, 2)
    assert np.isfinite(t).all(), "Tahminde NaN/inf var"
    assert (g > 0).all(), "MAPE için gerçek değerler pozitif olmalı"
    hata = t - g
    satirlar = []
    for j, k in enumerate(("cpu", "mem")):
        e = hata[:, j]
        eksik = e < 0
        satirlar.append({
            "kaynak": k, "n": int(e.size),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mape": float(100 * np.mean(np.abs(e) / g[:, j])),
            "mase": float(np.mean(np.abs(e)) / olcek[j]),
            "eksik_tahmin_orani": float(eksik.mean()),
            # Eksik kalan dilimlerde ortalama açık (yetersiz kapasite büyüklüğü için bilgi)
            "ortalama_eksik": float(-e[eksik].mean()) if eksik.any() else 0.0,
        })
    return satirlar


def bolum_olcutleri(h: Havuz, y: np.ndarray, tahminler: dict[str, np.ndarray], model: str,
                    tohum: int | None = None) -> list[dict[str, Any]]:
    """Bölüm -> (n_gun, 24, 2) tahmin sözlüğünden ölçüt satırları üretir."""
    olcek = mase_olcegi(y, h.sinir["train"][1])
    satirlar = []
    for b, t in tahminler.items():
        gercek = gun_ileri_hedef(y, h.degerlendirme_gunleri[b])
        for s in olcutler(gercek, t, olcek):
            satirlar.append({"veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "model": model,
                             "tohum": tohum, "bolum": b, "n_gun": t.shape[0], **s})
    return satirlar


# ---------------------------------------------------------------- çıktı sözleşmesi (Bölüm 5.4)

def tahmin_yaz(h: Havuz, model: str, gunler: list[int], tahmin: np.ndarray, kok: Path | None = None) -> None:
    """<kok>/<dataset>/<pool>/<model>/forecast_<gün>.parquet (slot, cpu_hat, mem_hat).

    kok: VARSAYILAN outputs/forecasts; sentetik senaryolar kendi kök dizinini verir.
    """
    kok = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) if kok is None else kok
    d: Path = kok / h.veri_seti / h.ad / model
    d.mkdir(parents=True, exist_ok=True)
    for g, t in zip(gunler, tahmin):
        pl.DataFrame({"slot": np.arange(24 * g, 24 * g + t.shape[0], dtype=np.int64),
                      "cpu_hat": t[:, 0], "mem_hat": t[:, 1]}).write_parquet(d / f"forecast_{g}.parquet")


def tahminleri_kaydet(h: Havuz, model: str, tahminler: dict[str, np.ndarray]) -> None:
    """write_forecast_splits bölümlerindeki gün-ileri tahminleri sözleşme dosyalarına yazar."""
    for b in yaml_oku("forecast")["outputs"]["write_forecast_splits"]:
        tahmin_yaz(h, model, h.degerlendirme_gunleri[b], tahminler[b])


def modeller_dizini() -> Path:
    """outputs/models dizini (yoksa oluşturur)."""
    d = yol(yaml_oku("forecast")["outputs"]["models_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d
