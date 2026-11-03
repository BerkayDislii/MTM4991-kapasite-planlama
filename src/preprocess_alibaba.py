"""Alibaba havuzlarını oluşturur: 5 dk havuz toplamı -> saatlik TEPE, M_ref ızgarası için (MIMARI.md Bölüm 4.2, 4.4–4.6).

Girdi : data/interim/alibaba.parquet (src.ingest_alibaba çıktısı; vm_id = makine no, mem_gb_r1)
Çıktı : data/processed/pool_alibaba_p<i>_r<r>.parquet  (slot, cpu_peak, mem_peak, cpu_mean, mem_mean, n_active)
        data/processed/pool_members.json                ("alibaba_p<i>" havuzları eklenir)
        outputs/eda/quality_report.json                 ("havuzlar_alibaba" ve "mref_alibaba" bölümleri)
        outputs/eda/tablo_mref_alibaba.csv              (havuz × r bazında M_ref istatistikleri, eğitim dönemi)
        outputs/eda/tablo_mref_alibaba_ozet.csv         (havuzlar üzerinden ortalama ± std)

Analiz penceresi (kullanıcı kararı 2026-10-29): start_day'den itibaren; dilim 0 = bu günün ilk saati.
Bellek doğrusal ölçeklendiği için havuz toplamı bir kez r = 1 ile hesaplanır, sonra bellek sütunları r ile çarpılır:
max(r·x) = r·max(x) ve ort(r·x) = r·ort(x), r > 0.

Çalıştırma: .venv/bin/python -m src.preprocess_alibaba
"""
import json
import logging
from typing import Any

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol
from src.ingest import KALITE_RAPORU, VeriDogrulamaHatasi
from src.ingest_alibaba import INTERIM
from src.preprocess import havuz_bes_dk_toplam, saatlik_tepe

log = logging.getLogger(__name__)

KAYNAKLAR = ("cpu", "mem")


def alibaba_havuz_uyeleri(makineler: list[int], seed: int, boyut: int) -> dict[str, list[int]]:
    """Sabit tohumla rastgele ayrık makine havuzları (Bölüm 4.5); boyuta tamamlanamayan artan makineler dışarıda kalır."""
    karisik = np.random.default_rng(seed).permutation(np.array(sorted(makineler)))
    n_havuz = len(karisik) // boyut
    return {f"alibaba_p{i}": sorted(int(m) for m in karisik[i * boyut:(i + 1) * boyut]) for i in range(n_havuz)}


def r_ile_olcekle(saat_r1: pl.DataFrame, r: float) -> pl.DataFrame:
    """r = 1 GB/çekirdek ile hesaplanmış saatlik tabloda bellek sütunlarını r ile çarpar (M_ref = cpu_num × r)."""
    return saat_r1.with_columns(pl.col("mem_peak", "mem_mean") * r)


def mref_istatistikleri(ad: str, saat: pl.DataFrame, r: float, egitim: tuple[int, int],
                        aileler: list[float]) -> dict[str, Any]:
    """Eğitim döneminde bir havuz × r için talep büyüklüğü, tepe/ortalama ve GB/çekirdek oranı özetleri."""
    p = saat.filter(pl.col("slot").is_between(egitim[0], egitim[1]))
    oran = (p["mem_peak"] / p["cpu_peak"]).to_numpy()
    satir: dict[str, Any] = {"havuz": ad, "r": r}
    for k in KAYNAKLAR:
        satir[f"{k}_peak_ortalama"] = round(float(p[f"{k}_peak"].mean()), 3)
        satir[f"{k}_peak_max"] = round(float(p[f"{k}_peak"].max()), 3)
        # Σ tepe / Σ ortalama: ortalamaya göre planlansaydı eksik kalacak kapasite oranı (Bitbrains EDA ile aynı)
        satir[f"{k}_toplam_tepe_toplam_ortalama"] = round(float(p[f"{k}_peak"].sum() / p[f"{k}_mean"].sum()), 4)
    satir.update({"gb_cekirdek_medyan": round(float(np.median(oran)), 3),
                  "gb_cekirdek_p5": round(float(np.percentile(oran, 5)), 3),
                  "gb_cekirdek_p95": round(float(np.percentile(oran, 95)), 3)})
    for f in aileler:
        # Talep oranı aile oranından büyükse o ailede bellek kısıtı, küçükse CPU kısıtı bağlayıcı
        satir[f"bellek_baglayici_orani_aile{f:g}"] = round(float((oran > f).mean()), 4)
    return satir


def main() -> None:
    """Alibaba havuzlarını 3 r değeri için üretir, M_ref tablolarını ve kalite raporu bölümlerini yazar."""
    cfg, pcfg = yaml_oku("preprocess"), yaml_oku("pools")["alibaba"]
    ali, aileler = cfg["alibaba"], yaml_oku("eda")["reference_gb_per_core"]
    adim_s, adim_dilim = cfg["grid"]["step_seconds"], cfg["grid"]["steps_per_slot"]
    bolme = ali["split_days"]
    gereken_dilim = 24 * sum(bolme.values())
    egitim = (0, 24 * bolme["train"] - 1)

    rapor = json.loads(KALITE_RAPORU.read_text(encoding="utf-8"))
    if rapor["alibaba"]["esik_asildi"]:
        raise VeriDogrulamaHatasi("DUR: Alibaba analiz penceresinde uzun boşluk oranı eşiği aşılmış; önce kullanıcıya danışın")

    df = pl.read_parquet(INTERIM)
    c0 = ali["start_day"] * 86400 // adim_s
    c1 = int(df["cell"].max())
    # Analiz penceresi: start_day öncesi hücreler atılır (Bölüm 4.6, kullanıcı kararı 2026-10-29)
    df = df.filter(pl.col("cell") >= c0).rename({"mem_gb_r1": "mem_gb"})
    havuzlar = alibaba_havuz_uyeleri(df["vm_id"].unique().to_list(), pcfg["seed"], pcfg["pool_size"])
    tum = [m for u in havuzlar.values() for m in u]
    assert len(tum) == len(set(tum)), "Alibaba havuzları ayrık değil"

    cikti_dir = yol("data/processed")
    cikti_dir.mkdir(parents=True, exist_ok=True)
    rapor["havuzlar_alibaba"], satirlar = {}, []
    for ad, uyeler in havuzlar.items():
        saat_r1, eksik = saatlik_tepe(havuz_bes_dk_toplam(df, uyeler, c0, c1), c0, adim_dilim)
        if saat_r1.height < gereken_dilim:
            raise VeriDogrulamaHatasi(f"{ad}: {saat_r1.height} dilim < bölme için gereken {gereken_dilim}")
        assert (saat_r1["cpu_peak"] >= saat_r1["cpu_mean"] - 1e-9).all() and (saat_r1["mem_peak"] >= saat_r1["mem_mean"] - 1e-9).all()
        for r in ali["r_grid"]:
            saat = r_ile_olcekle(saat_r1, r)
            saat.write_parquet(cikti_dir / f"pool_{ad}_r{r}.parquet")
            satirlar.append(mref_istatistikleri(ad, saat, r, egitim, aileler))
        rapor["havuzlar_alibaba"][ad] = {
            "makine_sayisi": len(uyeler), "dilim_sayisi": saat_r1.height, "atilan_eksik_son_dilim": eksik,
            "dilim0_hucre": int(c0), "dilim0_gun": ali["start_day"],
            "bolme_dilimleri": {"egitim": list(egitim),
                                "dogrulama": [24 * bolme["train"], 24 * (bolme["train"] + bolme["val"]) - 1],
                                "test": [24 * (bolme["train"] + bolme["val"]), gereken_dilim - 1]},
            "fazla_dilim_bolme_disi": saat_r1.height - gereken_dilim,
            "n_active_min_max": [int(saat_r1["n_active"].min()), int(saat_r1["n_active"].max())],
            "cpu_peak_ozet": {"min": saat_r1["cpu_peak"].min(), "ortalama": saat_r1["cpu_peak"].mean(), "max": saat_r1["cpu_peak"].max()},
            "mem_peak_r1_ozet": {"min": saat_r1["mem_peak"].min(), "ortalama": saat_r1["mem_peak"].mean(), "max": saat_r1["mem_peak"].max()},
            "dosyalar": [f"data/processed/pool_{ad}_r{r}.parquet" for r in ali["r_grid"]],
        }
        log.info("%s: %d makine, %d dilim, r=%s yazıldı", ad, len(uyeler), saat_r1.height, ali["r_grid"])

    tablo = pl.DataFrame(satirlar)
    tablo.write_csv(yol("outputs/eda/tablo_mref_alibaba.csv"))
    sayisal = [c for c in tablo.columns if c not in ("havuz", "r")]
    ozet = (tablo.group_by("r").agg([pl.col(c).mean().round(4).alias(f"{c}_ort") for c in sayisal]
                                    + [pl.col(c).std().round(4).alias(f"{c}_std") for c in sayisal]).sort("r"))
    ozet.write_csv(yol("outputs/eda/tablo_mref_alibaba_ozet.csv"))

    uye_dosya = cikti_dir / "pool_members.json"
    uyeler_json = json.loads(uye_dosya.read_text(encoding="utf-8")) if uye_dosya.exists() else {}
    uyeler_json = {k: v for k, v in uyeler_json.items() if not k.startswith("alibaba_")}
    uyeler_json.update({ad: {"veri_seti": "alibaba", "uyeler": u} for ad, u in havuzlar.items()})
    uye_dosya.write_text(json.dumps(uyeler_json), encoding="utf-8")

    rapor["mref_alibaba"] = {
        "tanim": "mem_gb = mem_util/100 x M_ref, M_ref = cpu_num x r (Bolum 4.2); sonuclar M_ref varsayimina bagli",
        "cpu_num": rapor["alibaba"]["dogrulama"]["machine_meta"]["cpu_num_degerleri"],
        "r_grid": ali["r_grid"], "r_merkez": ali["r_center"],
        "makine_basina_M_ref_gb": {str(r): [int(k) * r for k in rapor["alibaba"]["dogrulama"]["machine_meta"]["cpu_num_degerleri"]]
                                    for r in ali["r_grid"]},
        "donem": "egitim (dilim 0-%d)" % egitim[1],
        "havuz_sayisi": len(havuzlar), "havuz_disi_makine": df["vm_id"].n_unique() - len(tum),
        "ozet_havuzlar_ortalama": {str(row["r"]): {k: row[f"{k}_ort"] for k in sayisal} for row in ozet.iter_rows(named=True)},
    }
    KALITE_RAPORU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Kalite raporu ve M_ref tabloları yazıldı")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
