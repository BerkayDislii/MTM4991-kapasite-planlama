"""Bitbrains havuzlarını oluşturur: 5 dk havuz toplamı -> saatlik TEPE (MIMARI.md Bölüm 4.4–4.6).

Girdi : data/interim/bitbrains_<veri_seti>.parquet (src.ingest çıktısı)
Çıktı : data/processed/pool_<ad>.parquet  (slot, cpu_peak, mem_peak, cpu_mean, mem_mean, n_active)
        data/processed/pool_members.json  (havuz üyelikleri)
        outputs/eda/quality_report.json   ("havuzlar" bölümü eklenir)

Çalıştırma: .venv/bin/python -m src.preprocess
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol
from src.ingest import KALITE_RAPORU, VeriDogrulamaHatasi

log = logging.getLogger(__name__)


def havuz_bes_dk_toplam(df: pl.DataFrame, uyeler: list[int], c0: int, c1: int) -> pl.DataFrame:
    """Üye VM'lerin 5 dk talebini her hücrede toplar (D5_t). Uzun boşluk NaN'ları 0 sayılır (skipna).

    Izgara [c0, c1] aralığının tamamını kapsar; hiç aktif VM olmayan hücrede talep 0'dır.
    n_active: o hücrede yaşam süresi içinde olan VM sayısı.
    """
    toplam = (df.filter(pl.col("vm_id").is_in(uyeler))
                .group_by("cell")
                .agg(pl.col("cpu_cores").sum(), pl.col("mem_gb").sum(), pl.len().alias("n_active")))
    izgara = pl.DataFrame({"cell": np.arange(c0, c1 + 1, dtype=np.int64)})
    return (izgara.join(toplam, on="cell", how="left")
                  .with_columns(pl.col("cpu_cores", "mem_gb").fill_null(0.0),
                                pl.col("n_active").fill_null(0).cast(pl.Int64))
                  .sort("cell"))


def saatlik_tepe(d5: pl.DataFrame, c0: int, adim_dilim: int) -> tuple[pl.DataFrame, int]:
    """5 dk havuz toplamından saatlik tepe ve ortalamayı hesaplar: D_h = max(D5_t : t ∈ h).

    ÖNCE toplam, SONRA tepe (Bölüm 4.4). Eksik (12 adımdan kısa) son dilim atılır.
    Dönüş: (saatlik tablo, atılan eksik dilim sayısı).
    """
    saat = (d5.with_columns(((pl.col("cell") - c0) // adim_dilim).alias("slot"))
              .group_by("slot")
              .agg(pl.col("cpu_cores").max().alias("cpu_peak"),
                   pl.col("mem_gb").max().alias("mem_peak"),
                   pl.col("cpu_cores").mean().alias("cpu_mean"),
                   pl.col("mem_gb").mean().alias("mem_mean"),
                   # Saat içindeki en yüksek eş zamanlı aktif VM sayısı
                   pl.col("n_active").max().alias("n_active"),
                   pl.len().alias("_adim"))
              .sort("slot"))
    eksik = int((saat["_adim"] < adim_dilim).sum())
    saat = saat.filter(pl.col("_adim") == adim_dilim).drop("_adim")
    # Eksik dilim yalnızca sonda olabilir, ızgara kesintisiz olmalı
    assert saat["slot"].to_list() == list(range(saat.height)), "Saatlik dilimler kesintisiz değil"
    return saat, eksik


def havuz_uyeleri(vm_idler: dict[str, list[int]], pcfg: dict[str, Any]) -> dict[str, tuple[str, list[int]]]:
    """pools.yaml tanımından havuz adı -> (veri seti, üye listesi) sözlüğünü üretir (tohumlu, ayrık)."""
    havuzlar: dict[str, tuple[str, list[int]]] = {}
    for ds in pcfg["natural"]:
        havuzlar[f"bitbrains_{ds}"] = (ds, sorted(vm_idler[ds]))
    alt = pcfg["subpools"]
    rng = np.random.default_rng(alt["seed"])
    karisik = rng.permutation(np.array(sorted(vm_idler[alt["source"]])))
    for i, parca in enumerate(np.array_split(karisik, alt["n_pools"])):
        havuzlar[f"bitbrains_{alt['source']}_s{i}"] = (alt["source"], sorted(int(v) for v in parca))
    return havuzlar


def main() -> None:
    """Tüm Bitbrains havuzlarını üretir ve kalite raporuna havuz bilgilerini ekler."""
    cfg, pcfg = yaml_oku("preprocess"), yaml_oku("pools")["bitbrains"]
    adim_s, adim_dilim = cfg["grid"]["step_seconds"], cfg["grid"]["steps_per_slot"]
    bolme = cfg["bitbrains"]["split_days"]
    gereken_dilim = 24 * sum(bolme.values())

    rapor = json.loads(KALITE_RAPORU.read_text(encoding="utf-8"))
    asan = [k for k, v in rapor["bitbrains"].items() if v["esik_asildi"]]
    if asan:
        raise VeriDogrulamaHatasi(f"DUR: {asan} için uzun boşluk oranı eşiği aşılmış; önce kullanıcıya danışın")

    veri = {ds: pl.read_parquet(yol(f"data/interim/bitbrains_{ds}.parquet")) for ds in cfg["bitbrains"]["datasets"]}
    havuzlar = havuz_uyeleri({ds: d["vm_id"].unique().to_list() for ds, d in veri.items()}, pcfg)

    # Ayrıklık ve kapsama denetimi: alt havuzlar ayrık ve kaynağın tamamını kapsar
    kaynak = pcfg["subpools"]["source"]
    altlar = [set(u) for ad, (ds, u) in havuzlar.items() if ad.startswith(f"bitbrains_{kaynak}_s")]
    assert sum(len(s) for s in altlar) == len(set().union(*altlar)) == veri[kaynak]["vm_id"].n_unique()

    cikti_dir = yol("data/processed")
    cikti_dir.mkdir(parents=True, exist_ok=True)
    rapor["havuzlar"] = {}
    for ad, (ds, uyeler) in havuzlar.items():
        df = veri[ds]
        # Dilim kökeni veri seti düzeyinde ortaktır (alt havuzlar doğal havuzla aynı dilimlere hizalı)
        c0, c1 = int(df["cell"].min()), int(df["cell"].max())
        d5 = havuz_bes_dk_toplam(df, uyeler, c0, c1)
        saat, eksik = saatlik_tepe(d5, c0, adim_dilim)
        if saat.height < gereken_dilim:
            raise VeriDogrulamaHatasi(f"{ad}: {saat.height} dilim < bölme için gereken {gereken_dilim}")
        # Tepe tanımı denetimi: tepe >= ortalama ve tepe = toplamın tepesi
        assert (saat["cpu_peak"] >= saat["cpu_mean"] - 1e-9).all() and (saat["mem_peak"] >= saat["mem_mean"] - 1e-9).all()
        saat.write_parquet(cikti_dir / f"pool_{ad}.parquet")
        rapor["havuzlar"][ad] = {
            "veri_seti": ds, "vm_sayisi": len(uyeler), "dilim_sayisi": saat.height,
            "atilan_eksik_son_dilim": eksik,
            "dilim0_baslangic_utc": datetime.fromtimestamp(c0 * adim_s, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "bolme_dilimleri": {"egitim": [0, 24 * bolme["train"] - 1],
                                "dogrulama": [24 * bolme["train"], 24 * (bolme["train"] + bolme["val"]) - 1],
                                "test": [24 * (bolme["train"] + bolme["val"]), gereken_dilim - 1]},
            "fazla_dilim_bolme_disi": saat.height - gereken_dilim,
            "cpu_peak_ozet": {"min": saat["cpu_peak"].min(), "ortalama": saat["cpu_peak"].mean(), "max": saat["cpu_peak"].max()},
            "mem_peak_ozet": {"min": saat["mem_peak"].min(), "ortalama": saat["mem_peak"].mean(), "max": saat["mem_peak"].max()},
            "tepe_ortalama_orani_medyan": {"cpu": float((saat["cpu_peak"] / saat["cpu_mean"]).median()),
                                           "mem": float((saat["mem_peak"] / saat["mem_mean"]).median())},
        }
        log.info("%s: %d VM, %d dilim yazıldı", ad, len(uyeler), saat.height)

    # Diğer veri setlerinin (ör. alibaba_*) üyelikleri korunur, yalnız bitbrains_* yeniden yazılır
    uye_dosya = cikti_dir / "pool_members.json"
    uyeler_json = json.loads(uye_dosya.read_text(encoding="utf-8")) if uye_dosya.exists() else {}
    uyeler_json = {k: v for k, v in uyeler_json.items() if not k.startswith("bitbrains_")}
    uyeler_json.update({ad: {"veri_seti": ds, "uyeler": u} for ad, (ds, u) in havuzlar.items()})
    uye_dosya.write_text(json.dumps(uyeler_json), encoding="utf-8")
    KALITE_RAPORU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
