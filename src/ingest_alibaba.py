"""Alibaba cluster-trace-v2018 ham verisini okur, birim dönüştürür ve makine bazında 5 dk ızgarasına hizalar.

Adımlar (MIMARI.md Bölüm 3, 4.1–4.3):
1. machine_usage.csv (~9 GB) akış (streaming) ile okunur; yalnızca gerekli 4 sütun Parquet önbelleğine yazılır.
2. Doğrulamalar: sütun sayısı, örnekleme aralığı, cpu_num homojenliği, machine_meta'nın zamanla değişimi.
3. [0,100] dışı yüzdeler NaN; 5 dk hücre ortalaması; kısa boşluk enterpolasyonu; yaşam süresi (Bitbrains ile aynı kod).

4. DUR denetimi (uzun boşluk oranı) start_day'den başlayan analiz penceresinde yapılır (Bölüm 4.6).

Bellek r'den bağımsız saklanır: mem_gb_r1 = mem_util/100 × cpu_num (r = 1 GB/çekirdek iken GB).
Herhangi bir r için mem_gb = r × mem_gb_r1 (M_ref = cpu_num × r, Bölüm 4.2); çarpım src.preprocess_alibaba'da yapılır.

Çıktılar:
- data/interim/alibaba_machine_usage_raw.parquet  (machine_id, time_stamp, cpu_util_percent, mem_util_percent)
- data/interim/alibaba.parquet                    (vm_id = makine no, cell, cpu_cores, mem_gb_r1, filled)
- outputs/eda/quality_report.json                 ("alibaba" bölümü eklenir/güncellenir)

Çalıştırma: .venv/bin/python -m src.ingest_alibaba
"""
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol
from src.ingest import KALITE_RAPORU, VeriDogrulamaHatasi, vm_izgaraya_hizala, zaman_birimi_tespit

log = logging.getLogger(__name__)

HAM_PARQUET: Path = yol("data/interim/alibaba_machine_usage_raw.parquet")
INTERIM: Path = yol("data/interim/alibaba.parquet")
GEREKLI = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"]


def makine_no(sutun: str = "machine_id") -> pl.Expr:
    """'m_123' biçimindeki kimliği tamsayıya çevirir; biçim dışı kimlikte hata verir (strict cast)."""
    return pl.col(sutun).str.strip_prefix("m_").cast(pl.Int32, strict=True)


def basliksiz_csv_oku(dosya: Path, sutunlar: list[str]) -> pl.DataFrame:
    """Küçük başlıksız CSV'yi okur; sütun sayısının şemayla aynı olduğunu doğrular."""
    df = pl.read_csv(dosya, has_header=False, infer_schema_length=10000)
    if df.width != len(sutunlar):
        raise VeriDogrulamaHatasi(f"{dosya}: {df.width} sütun, şemada {len(sutunlar)} bekleniyordu")
    df.columns = sutunlar
    return df


def ham_parquet_uret(csv: Path, sutunlar: list[str]) -> None:
    """machine_usage CSV'sini akış ile okuyup gerekli 4 sütunu Parquet'e yazar (tek seferlik önbellek)."""
    ornek = pl.read_csv(csv, has_header=False, n_rows=10000)
    if ornek.width != len(sutunlar):
        raise VeriDogrulamaHatasi(f"{csv}: {ornek.width} sütun, şemada {len(sutunlar)} bekleniyordu")
    sema = {ad: (pl.String if ad == "machine_id" else pl.Float64) for ad in sutunlar}
    sema["time_stamp"] = pl.Int64
    lf = pl.scan_csv(csv, has_header=False, new_columns=sutunlar, schema_overrides=sema)
    HAM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    (lf.select(makine_no(), pl.col("time_stamp"), pl.col("cpu_util_percent"), pl.col("mem_util_percent"))
       .sink_parquet(HAM_PARQUET))
    log.info("Ham Parquet yazıldı: %s", HAM_PARQUET)


def ornekleme_dogrula(lf: pl.LazyFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Örnekleme aralığını (ardışık time_stamp farkı) tohumlu makine örneğinde ölçer; ızgaradan kaba ise DUR.

    Zaman birimi: toplam süre / 86400 beklenen gün sayısına yakın olmalı (saniye), değilse DUR.
    """
    ali, adim = cfg["alibaba"], cfg["grid"]["step_seconds"]
    idler = np.sort(lf.select(pl.col("machine_id").unique()).collect()["machine_id"].to_numpy())
    rng = np.random.default_rng(ali["sampling_check_seed"])
    secilen = rng.choice(idler, min(ali["sampling_check_machines"], len(idler)), replace=False)
    fark = (lf.filter(pl.col("machine_id").is_in(secilen.tolist()))
              .select("machine_id", "time_stamp").collect()
              .sort("machine_id", "time_stamp")
              .select(pl.col("time_stamp").diff().over("machine_id").alias("d"))
              .drop_nulls()["d"].to_numpy())
    medyan = float(np.median(fark[fark > 0]))
    if medyan > adim:
        raise VeriDogrulamaHatasi(f"Örnekleme aralığı medyanı {medyan} s > ızgara adımı {adim} s")
    t = lf.select(pl.col("time_stamp").min().alias("t0"), pl.col("time_stamp").max().alias("t1")).collect().row(0)
    gun = (t[1] - t[0]) / 86400
    if abs(gun - ali["expected_days"]) > 1:
        raise VeriDogrulamaHatasi(f"time_stamp aralığı {gun:.2f} gün; saniye birimiyle {ali['expected_days']} gün bekleniyordu")
    return {"ornek_makine": len(secilen), "ardisik_fark_medyani_s": medyan,
            "ardisik_fark_p99_s": float(np.percentile(fark, 99)), "sifir_fark_sayisi": int((fark == 0).sum()),
            "zaman_birimi": "saniye", "time_stamp_araligi": list(t), "sure_gun": gun}


def meta_dogrula(meta: pl.DataFrame, usage_idler: pl.Series) -> tuple[pl.DataFrame, dict[str, Any]]:
    """machine_meta: cpu_num makine içinde zamanla sabit mi, makineler arası homojen mi; tüm makineler meta'da mı.

    Dönüş: (machine_id -> cpu_num tablosu, rapor sözlüğü). cpu_num zamanla değişirse DUR.
    """
    m = meta.with_columns(makine_no())
    grup = m.group_by("machine_id").agg(pl.col("cpu_num").n_unique().alias("n_cpu"),
                                        pl.col("mem_size").n_unique().alias("n_mem"),
                                        pl.col("status").n_unique().alias("n_status"))
    degisen = int((grup["n_cpu"] > 1).sum())
    if degisen:
        raise VeriDogrulamaHatasi(f"machine_meta: {degisen} makinede cpu_num zamanla değişiyor (Bölüm 3)")
    eksik = int((~usage_idler.is_in(m["machine_id"].unique().implode())).sum())
    if eksik:
        raise VeriDogrulamaHatasi(f"machine_usage'daki {eksik} makine machine_meta'da yok (cpu_num bilinmiyor)")
    cpu = m.group_by("machine_id").agg(pl.col("cpu_num").first())
    return cpu, {
        "satir": meta.height, "makine": grup.height,
        "cpu_num_degerleri": {str(k): v for k, v in cpu["cpu_num"].value_counts().sort("cpu_num").iter_rows()},
        "cpu_num_homojen": cpu["cpu_num"].n_unique() == 1,
        "cpu_num_zamanla_degisen_makine": degisen,
        "mem_size_degerleri": sorted(m["mem_size"].unique().to_list()),
        "mem_size_zamanla_degisen_makine": int((grup["n_mem"] > 1).sum()),
        "status_degerleri": {k: v for k, v in m["status"].value_counts().sort("status").iter_rows()},
        "status_zamanla_degisen_makine": int((grup["n_status"] > 1).sum()),
        "usage_makinesi_meta_disi": eksik,
    }


def container_meta_ozet(df: pl.DataFrame) -> dict[str, Any]:
    """container_meta (opsiyonel, Bölüm 10 israf grafiği için) için yalnızca şema ve kapsam özeti."""
    return {"satir": df.height, "konteyner": df["container_id"].n_unique(), "makine": df["machine_id"].n_unique(),
            "uygulama": df["app_du"].n_unique(),
            "status_degerleri": {k: v for k, v in df["status"].value_counts().sort("status").iter_rows()}}


def bes_dk_ortalama(lf: pl.LazyFrame, cpu_num: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Ham 10 sn ölçümleri birim dönüştürüp makine × 5 dk hücresine ortalar (polars akış motoru).

    [0,100] dışı yüzde -> null (Bölüm 4.2); hücre ortalaması yalnız geçerli ölçümlerden alınır.
    Hücre başına bir satır: tüm ölçümleri geçersiz hücre de (null değerle) korunur ki "gözlemli" sayılsın.
    cpu_cores = cpu_util/100 × cpu_num,  mem_gb_r1 = mem_util/100 × cpu_num (M_ref'in r = 1 hâli).
    """
    alt, ust = cfg["alibaba"]["util_valid_range"]
    adim = cfg["grid"]["step_seconds"]

    def gecerli(s: str) -> pl.Expr:
        return pl.when(pl.col(s).is_between(alt, ust)).then(pl.col(s)).otherwise(None)

    sayac = lf.select(
        pl.len().alias("ham_satir"),
        pl.col("cpu_util_percent").is_null().sum().alias("cpu_bos"),
        pl.col("mem_util_percent").is_null().sum().alias("mem_bos"),
        (~pl.col("cpu_util_percent").is_between(alt, ust)).sum().alias("cpu_aralik_disi"),
        (~pl.col("mem_util_percent").is_between(alt, ust)).sum().alias("mem_aralik_disi"),
    ).collect(engine="streaming").row(0, named=True)
    # Aynı (makine, zaman) çiftinin tekrar sayısı
    sayac["yinelenen_makine_zaman"] = int(
        lf.group_by("machine_id").agg(pl.len() - pl.col("time_stamp").n_unique())
          .collect(engine="streaming").to_series(1).sum())

    hucre = (lf.with_columns(gecerli("cpu_util_percent"), gecerli("mem_util_percent"),
                             (pl.col("time_stamp") // adim).alias("cell"))
               .group_by("machine_id", "cell")
               .agg(pl.col("cpu_util_percent").mean(), pl.col("mem_util_percent").mean())
               .join(cpu_num.lazy(), on="machine_id", how="left")
               .select("machine_id", "cell",
                       (pl.col("cpu_util_percent") / 100 * pl.col("cpu_num")).alias("cpu_cores"),
                       (pl.col("mem_util_percent") / 100 * pl.col("cpu_num")).alias("mem_gb_r1"))
               .collect(engine="streaming")
               .sort("machine_id", "cell"))
    return hucre, {k: int(v) for k, v in sayac.items()}


def makineleri_hizala(hucre: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Her makinenin hücre ortalamalarına Bitbrains ile aynı yaşam süresi + kısa boşluk kuralını uygular.

    Hücre başına tek satır olduğundan vm_izgaraya_hizala'nın hücre ortalaması bu seriyi değiştirmez.
    """
    adim, max_bosluk = cfg["grid"]["step_seconds"], cfg["gaps"]["max_gap_steps"]
    sayac = dict(yasam_hucre=0, gozlemsiz_hucre=0, gecersiz_hucre=0, doldurulan_hucre=0, uzun_bosluk_hucre=0,
                 yasam_disi_hucre=0, gecerli_olcumsuz_makine=0)
    parcalar: list[pl.DataFrame] = []
    for (mid,), d in hucre.partition_by("machine_id", as_dict=True, maintain_order=True).items():
        cpu = d["cpu_cores"].fill_null(np.nan).to_numpy()
        mem = d["mem_gb_r1"].fill_null(np.nan).to_numpy()
        g = vm_izgaraya_hizala(d["cell"].to_numpy() * adim, cpu, mem, adim, max_bosluk)
        if g is None:
            sayac["gecerli_olcumsuz_makine"] += 1
            sayac["yasam_disi_hucre"] += d.height
            continue
        n = len(g["cell"])
        sayac["yasam_disi_hucre"] += int((~g["inside_rows"]).sum())
        sayac["yasam_hucre"] += n
        sayac["gozlemsiz_hucre"] += int((~g["observed"]).sum())
        sayac["gecersiz_hucre"] += int((g["observed"] & (np.isnan(g["cpu_cores"]) | g["cpu_cores_filled"])).sum())
        sayac["doldurulan_hucre"] += int(g["cpu_cores_filled"].sum())
        sayac["uzun_bosluk_hucre"] += int((np.isnan(g["cpu_cores"]) & ~g["cpu_cores_filled"]).sum())
        parcalar.append(pl.DataFrame({
            "vm_id": np.full(n, mid, dtype=np.int32), "cell": g["cell"],
            "cpu_cores": g["cpu_cores"], "mem_gb_r1": g["mem_gb"],
            "filled": g["cpu_cores_filled"] | g["mem_gb_filled"],
        }, schema_overrides={"cpu_cores": pl.Float64, "mem_gb_r1": pl.Float64}).with_columns(
            pl.col("cpu_cores", "mem_gb_r1").fill_nan(None)))
    return pl.concat(parcalar), sayac


def main() -> None:
    """Alibaba makine verisini işler, interim Parquet ve kalite raporunu yazar; eşik aşılırsa DUR."""
    cfg = yaml_oku("preprocess")
    ali = cfg["alibaba"]
    ham_dir = yol(ali["raw_dir"])
    usage_csv = ham_dir / ali["machine_usage"]["file"]
    if not HAM_PARQUET.exists():
        ham_parquet_uret(usage_csv, ali["machine_usage"]["columns"])
    lf = pl.scan_parquet(HAM_PARQUET)

    meta = basliksiz_csv_oku(ham_dir / ali["machine_meta"]["file"], ali["machine_meta"]["columns"])
    usage_idler = lf.select(pl.col("machine_id").unique()).collect()["machine_id"]
    cpu_num, meta_rapor = meta_dogrula(meta, usage_idler)
    cont = basliksiz_csv_oku(ham_dir / ali["container_meta"]["file"], ali["container_meta"]["columns"])
    ornek = ornekleme_dogrula(lf, cfg)
    log.info("Örnekleme medyanı %.0f s, cpu_num değerleri %s", ornek["ardisik_fark_medyani_s"], meta_rapor["cpu_num_degerleri"])

    hucre, ham_sayac = bes_dk_ortalama(lf, cpu_num, cfg)
    log.info("5 dk hücre tablosu: %d satır", hucre.height)
    interim, sayac = makineleri_hizala(hucre, cfg)
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    interim.write_parquet(INTERIM)
    log.info("Interim yazıldı: %s (%d satır)", INTERIM, interim.height)

    # DUR denetimi analiz penceresinde yapılır (kullanıcı kararı 2026-10-29: 0–1. günler atılır, Bölüm 4.6);
    # 8 günün tamamındaki oran bilgi amaçlı raporlanır.
    oran_tum = sayac["uzun_bosluk_hucre"] / sayac["yasam_hucre"]
    c_bas = ali["start_day"] * 86400 // cfg["grid"]["step_seconds"]
    pencere = interim.filter(pl.col("cell") >= c_bas)
    uzun_pencere = int((pencere["cpu_cores"].is_null() & ~pencere["filled"]).sum())
    oran = uzun_pencere / pencere.height
    esik = cfg["gaps"]["max_long_gap_ratio"]
    rapor = json.loads(KALITE_RAPORU.read_text(encoding="utf-8")) if KALITE_RAPORU.exists() else {}
    rapor["alibaba"] = {
        "dosyalar": {k: str((ham_dir / ali[k]["file"]).relative_to(yol("."))) for k in ("machine_usage", "machine_meta", "container_meta")},
        "dogrulama": {
            "sutun_sayisi": "tamam",
            "ham_parquet_satir_sayisi": ham_sayac["ham_satir"],
            "ornekleme": ornek,
            "machine_meta": meta_rapor,
        },
        "container_meta": container_meta_ozet(cont),
        "makine_sayisi": int(usage_idler.len()),
        "ham_satir": ham_sayac["ham_satir"],
        "cpu_util_bos_satir": ham_sayac["cpu_bos"],
        "mem_util_bos_satir": ham_sayac["mem_bos"],
        "cpu_util_aralik_disi_satir": ham_sayac["cpu_aralik_disi"],
        "mem_util_aralik_disi_satir": ham_sayac["mem_aralik_disi"],
        "yinelenen_makine_zaman_satir": ham_sayac["yinelenen_makine_zaman"],
        "yasam_suresi_tanimi": "ilk/son gecerli cpu_util olcumunun 5 dk hucresi",
        "gecerli_olcumu_olmayan_makine": sayac["gecerli_olcumsuz_makine"],
        "yasam_disi_atilan_hucre": sayac["yasam_disi_hucre"],
        "yasam_suresi_hucre": sayac["yasam_hucre"],
        "gozlemsiz_hucre": sayac["gozlemsiz_hucre"],
        "yalniz_gecersiz_olcumlu_hucre": sayac["gecersiz_hucre"],
        "enterpole_hucre": sayac["doldurulan_hucre"],
        "uzun_bosluk_nan_hucre": sayac["uzun_bosluk_hucre"],
        "uzun_bosluk_nan_orani_8_gun": oran_tum,
        "analiz_penceresi": {"start_day": ali["start_day"], "ilk_hucre": int(c_bas),
                             "yasam_suresi_hucre": pencere.height, "uzun_bosluk_nan_hucre": uzun_pencere,
                             "makine": pencere["vm_id"].n_unique()},
        "uzun_bosluk_nan_orani": oran,
        "esik": esik,
        "esik_asildi": bool(oran > esik),
        "bellek_birimi": "mem_gb_r1 = mem_util/100 x cpu_num (r=1); mem_gb = r x mem_gb_r1 (M_ref varsayimina bagli)",
    }
    KALITE_RAPORU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Kalite raporu güncellendi: %s", KALITE_RAPORU)
    if oran > esik:
        raise VeriDogrulamaHatasi(f"DUR: Alibaba uzun boşluk NaN oranı (analiz penceresi) {oran:.4%} > eşik {esik:.0%} (Bölüm 4.3)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
