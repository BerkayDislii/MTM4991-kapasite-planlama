"""Bitbrains ham verisini okur, birim dönüştürür ve VM bazında 5 dk ızgarasına hizalar.

Çıktılar:
- data/interim/bitbrains_<veri_seti>.parquet  (vm_id, cell, cpu_cores, mem_gb, filled)
- outputs/eda/quality_report.json             (veri kalitesi ve doğrulama sonuçları)

Çalıştırma: .venv/bin/python -m src.ingest
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol

log = logging.getLogger(__name__)

# Bölüm 3'te doğrulanması istenen sütunlar (strip sonrası)
SUTUN_TS = "Timestamp [ms]"
SUTUN_CORES = "CPU cores"
SUTUN_CAP = "CPU capacity provisioned [MHZ]"
SUTUN_USAGE = "CPU usage [MHZ]"
SUTUN_MEM = "Memory usage [KB]"
GEREKLI_SUTUNLAR: list[str] = [SUTUN_TS, SUTUN_CORES, SUTUN_CAP, SUTUN_USAGE, SUTUN_MEM]

KALITE_RAPORU: Path = yol("outputs/eda/quality_report.json")


class VeriDogrulamaHatasi(RuntimeError):
    """MIMARI.md'deki bir DUR koşulu tetiklendiğinde fırlatılır."""


def csv_oku(dosya: Path, ayrac: str) -> pl.DataFrame:
    """Bitbrains CSV'sini okur, sütun adlarını ve değerleri strip eder, gerekli sütunları doğrular."""
    df = pl.read_csv(dosya, separator=ayrac, infer_schema_length=0)
    df.columns = [c.strip() for c in df.columns]
    eksik = [c for c in GEREKLI_SUTUNLAR if c not in df.columns]
    if eksik:
        raise VeriDogrulamaHatasi(f"{dosya}: eksik sütun(lar) {eksik}; mevcut: {df.columns}")
    return df.select(GEREKLI_SUTUNLAR).with_columns(pl.all().str.strip_chars().cast(pl.Float64))


def zaman_birimi_tespit(medyan_fark: float, adim_s: int, tolerans: float) -> int:
    """Ardışık zaman farkı medyanından birimi bulur; saniyeye bölen katsayıyı döndürür.

    ≈ adim_s ise saniye (1), ≈ adim_s*1000 ise milisaniye (1000). Aksi hâlde DUR.
    """
    for katsayi in (1, 1000):
        if abs(medyan_fark - adim_s * katsayi) <= tolerans * adim_s * katsayi:
            return katsayi
    raise VeriDogrulamaHatasi(
        f"Zaman birimi belirlenemedi: ardışık fark medyanı {medyan_fark}, beklenen ≈{adim_s} veya ≈{adim_s * 1000}")


def birim_donustur(df: pl.DataFrame, sifir_kapasite: str = "nan") -> pl.DataFrame:
    """CPU'yu çekirdek eşdeğerine, belleği GiB'e çevirir (Bölüm 4.2).

    cpu_cores = CPU_usage_MHz × CPU_cores / CPU_capacity_MHz,  mem_gb = Memory_KB / 1024².
    Çekirdek veya kapasite <= 0 ya da negatif ölçüm olan satırlar geçersizdir: CPU ve bellek null.
    Kullanım > kapasite durumunda kırpma yapılmaz.
    """
    if sifir_kapasite != "nan":
        raise ValueError(f"Desteklenmeyen zero_capacity_policy: {sifir_kapasite}")
    gecerli = ((pl.col(SUTUN_CORES) > 0) & (pl.col(SUTUN_CAP) > 0)
               & (pl.col(SUTUN_USAGE) >= 0) & (pl.col(SUTUN_MEM) >= 0))
    return df.select(
        pl.col(SUTUN_TS).alias("ts"),
        pl.when(gecerli).then(pl.col(SUTUN_USAGE) * pl.col(SUTUN_CORES) / pl.col(SUTUN_CAP))
          .otherwise(None).alias("cpu_cores"),
        pl.when(gecerli).then(pl.col(SUTUN_MEM) / 1024**2).otherwise(None).alias("mem_gb"),
    )


def kisa_bosluklari_doldur(x: np.ndarray, max_bosluk: int) -> tuple[np.ndarray, np.ndarray]:
    """İki yanı dolu ve uzunluğu <= max_bosluk olan NaN koşularını doğrusal enterpole eder.

    Uzun koşular ve seri uçlarındaki koşular NaN kalır (kısmen bile doldurulmaz).
    Dönüş: (doldurulmuş seri, doldurulan hücre maskesi).
    """
    nan = np.isnan(x)
    doldur = np.zeros(len(x), dtype=bool)
    if not nan.any() or nan.all():
        return x.copy(), doldur
    # NaN koşularının [bas, son) sınırları
    d = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
    bas, son = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    sec = (bas > 0) & (son < len(x)) & (son - bas <= max_bosluk)
    # Seçili koşuları fark dizisi + kümülatif toplam ile işaretle (döngüsüz)
    isaret = np.zeros(len(x) + 1, dtype=np.int32)
    np.add.at(isaret, bas[sec], 1)
    np.add.at(isaret, son[sec], -1)
    doldur = np.cumsum(isaret[:-1]) > 0
    y = x.copy()
    idx = np.arange(len(x))
    y[doldur] = np.interp(idx[doldur], idx[~nan], x[~nan])
    return y, doldur


def vm_izgaraya_hizala(ts_s: np.ndarray, cpu: np.ndarray, mem: np.ndarray, adim_s: int,
                       max_bosluk: int) -> dict[str, np.ndarray] | None:
    """Tek VM'in ölçümlerini 5 dk ızgarasına hizalar (floor + hücre ortalaması) ve kısa boşlukları doldurur.

    Yaşam süresi (kullanıcı kararı 2026-10-29): ilk ve son GEÇERLİ ölçümün (CPU null değil) hücreleri arası.
    Bu aralığın dışındaki satırlar atılır (VM yok sayılır, havuz toplamında talep 0).
    Hiç geçerli ölçümü olmayan VM için None döner.
    """
    hucre_tum = np.floor_divide(ts_s, adim_s).astype(np.int64)
    gecerli_satir = ~np.isnan(cpu)
    if not gecerli_satir.any():
        return None
    c0, c1 = hucre_tum[gecerli_satir].min(), hucre_tum[gecerli_satir].max()
    icerde = (hucre_tum >= c0) & (hucre_tum <= c1)
    hucre, cpu, mem = hucre_tum[icerde], cpu[icerde], mem[icerde]
    n = int(c1 - c0 + 1)
    konum = hucre - c0
    sonuc: dict[str, np.ndarray] = {"cell": np.arange(c0, c0 + n, dtype=np.int64), "inside_rows": icerde}
    gozlem = np.bincount(konum, minlength=n)  # hücredeki ham ölçüm sayısı
    sonuc["observed"] = gozlem > 0
    for ad, deger in (("cpu_cores", cpu), ("mem_gb", mem)):
        gecerli = ~np.isnan(deger)
        toplam = np.bincount(konum[gecerli], weights=deger[gecerli], minlength=n)
        sayi = np.bincount(konum[gecerli], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            ort = np.where(sayi > 0, toplam / np.maximum(sayi, 1), np.nan)
        sonuc[ad], sonuc[f"{ad}_filled"] = kisa_bosluklari_doldur(ort, max_bosluk)
    return sonuc


def utc(t: float) -> str:
    """Unix saniyesini okunur UTC metnine çevirir."""
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def veri_seti_isle(ad: str, klasor: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Bir Bitbrains veri setini (klasördeki tüm VM CSV'leri) işler, interim Parquet yazar, kalite özetini döndürür."""
    bb, adim = cfg["bitbrains"], cfg["grid"]["step_seconds"]
    dosyalar = sorted(klasor.glob("*.csv"), key=lambda p: int(p.stem))
    if not dosyalar:
        raise VeriDogrulamaHatasi(f"{klasor} içinde CSV yok")
    log.info("%s: %d dosya okunuyor (%s)", ad, len(dosyalar), klasor)

    ham = {int(f.stem): csv_oku(f, bb["separator"]) for f in dosyalar}

    # Zaman birimi: sütun adına güvenmeden, tüm dosyalardaki pozitif ardışık farkların medyanından
    farklar = np.concatenate([np.diff(np.sort(d[SUTUN_TS].to_numpy())) for d in ham.values()])
    medyan_fark = float(np.median(farklar[farklar > 0]))
    katsayi = zaman_birimi_tespit(medyan_fark, adim, cfg["timestamp_unit"]["rel_tolerance"])
    # Dosya bazında da hiçbir dosyanın diğer birimde görünmediğini doğrula
    dosya_medyan = np.array([np.median(np.diff(np.sort(d[SUTUN_TS].to_numpy()))) if d.height > 1 else adim * katsayi
                             for d in ham.values()])
    if katsayi == 1 and (dosya_medyan > adim * 100).any():
        raise VeriDogrulamaHatasi(f"{ad}: bazı dosyalar milisaniye görünüyor: {np.flatnonzero(dosya_medyan > adim * 100)}")

    sayac = dict(satir=0, tam_kopya=0, sifir_kapasite=0, kullanim_kapasite_ustu=0, yasam_hucre=0,
                 gozlemsiz_hucre=0, gecersiz_hucre=0, doldurulan_hucre=0, uzun_bosluk_hucre=0,
                 yasam_disi_satir=0, yasam_disi_kullanim_pozitif=0, gecerli_olcumsuz_vm=0)
    parcalar: list[pl.DataFrame] = []
    t_min, t_max = np.inf, -np.inf
    for vm_id, d in ham.items():
        sayac["satir"] += d.height
        sayac["tam_kopya"] += d.height - d.unique().height
        sayac["sifir_kapasite"] += int(((d[SUTUN_CORES] <= 0) | (d[SUTUN_CAP] <= 0)).sum())
        sayac["kullanim_kapasite_ustu"] += int((d[SUTUN_USAGE] > d[SUTUN_CAP]).sum())

        b = birim_donustur(d, bb["zero_capacity_policy"])
        ts_s = b["ts"].to_numpy() / katsayi
        t_min, t_max = min(t_min, ts_s.min()), max(t_max, ts_s.max())
        g = vm_izgaraya_hizala(ts_s, b["cpu_cores"].to_numpy().astype(float),
                               b["mem_gb"].to_numpy().astype(float), adim, cfg["gaps"]["max_gap_steps"])
        if g is None:
            sayac["gecerli_olcumsuz_vm"] += 1
            sayac["yasam_disi_satir"] += d.height
            continue
        # Yaşam süresi dışında kalan (uçlardaki geçersiz) satırlar ve bunlarda sıfırdan büyük kullanım
        disari = d.filter(pl.Series(~g["inside_rows"]))
        sayac["yasam_disi_satir"] += disari.height
        sayac["yasam_disi_kullanim_pozitif"] += int(((disari[SUTUN_USAGE] > 0) | (disari[SUTUN_MEM] > 0)).sum())
        n = len(g["cell"])
        ham_nan = np.isnan(g["cpu_cores"]) & ~g["cpu_cores_filled"]
        sayac["yasam_hucre"] += n
        sayac["gozlemsiz_hucre"] += int((~g["observed"]).sum())
        sayac["gecersiz_hucre"] += int((g["observed"] & (np.isnan(g["cpu_cores"]) | g["cpu_cores_filled"])).sum())
        sayac["doldurulan_hucre"] += int(g["cpu_cores_filled"].sum())
        sayac["uzun_bosluk_hucre"] += int(ham_nan.sum())
        parcalar.append(pl.DataFrame({
            "vm_id": np.full(n, vm_id, dtype=np.int32), "cell": g["cell"],
            "cpu_cores": g["cpu_cores"], "mem_gb": g["mem_gb"],
            "filled": g["cpu_cores_filled"] | g["mem_gb_filled"],
        }, schema_overrides={"cpu_cores": pl.Float64, "mem_gb": pl.Float64}).with_columns(
            pl.col("cpu_cores", "mem_gb").fill_nan(None)))

    cikti = yol(f"data/interim/bitbrains_{ad}.parquet")
    cikti.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(parcalar).write_parquet(cikti)
    log.info("%s: %s yazıldı", ad, cikti)

    oran = sayac["uzun_bosluk_hucre"] / sayac["yasam_hucre"]
    esik = cfg["gaps"]["max_long_gap_ratio"]
    return {
        "klasor": str(klasor.relative_to(yol("."))),
        "vm_sayisi": len(ham),
        "dogrulama": {
            "gerekli_sutunlar": "tamam",
            "zaman_ardisik_fark_medyani": medyan_fark,
            "zaman_birimi": "saniye" if katsayi == 1 else "milisaniye",
            "dosya_medyan_adim_300_olmayan": int((dosya_medyan != adim * katsayi).sum()),
        },
        "zaman_araligi_utc": [utc(t_min), utc(t_max)],
        "ham_satir": sayac["satir"],
        "tam_kopya_satir": sayac["tam_kopya"],
        "sifir_kapasite_satir": sayac["sifir_kapasite"],
        "sifir_kapasite_satir_orani": sayac["sifir_kapasite"] / sayac["satir"],
        "kullanim_kapasite_ustu_satir": sayac["kullanim_kapasite_ustu"],
        "kullanim_kapasite_ustu_orani": sayac["kullanim_kapasite_ustu"] / sayac["satir"],
        "yasam_suresi_tanimi": "ilk/son gecerli olcum (cores>0, kapasite>0)",
        "gecerli_olcumu_olmayan_vm": sayac["gecerli_olcumsuz_vm"],
        "yasam_disi_atilan_satir": sayac["yasam_disi_satir"],
        "yasam_disi_kullanim_veya_bellek_pozitif_satir": sayac["yasam_disi_kullanim_pozitif"],
        "yasam_suresi_hucre": sayac["yasam_hucre"],
        "gozlemsiz_hucre": sayac["gozlemsiz_hucre"],
        "yalniz_gecersiz_olcumlu_hucre": sayac["gecersiz_hucre"],
        "enterpole_hucre": sayac["doldurulan_hucre"],
        "uzun_bosluk_nan_hucre": sayac["uzun_bosluk_hucre"],
        "uzun_bosluk_nan_orani": oran,
        "esik": esik,
        "esik_asildi": bool(oran > esik),
    }


def main() -> None:
    """Tüm Bitbrains veri setlerini işler; uzun boşluk oranı eşiği aşılırsa DUR."""
    cfg = yaml_oku("preprocess")
    bb = cfg["bitbrains"]
    # Diğer veri setlerinin (ör. "alibaba") bölümleri korunur, yalnız "bitbrains" yeniden yazılır
    rapor: dict[str, Any] = json.loads(KALITE_RAPORU.read_text(encoding="utf-8")) if KALITE_RAPORU.exists() else {}
    rapor["bitbrains"] = {}
    for ad, alt in bb["datasets"].items():
        rapor["bitbrains"][ad] = veri_seti_isle(ad, yol(bb["raw_dir"]) / alt, cfg)
    KALITE_RAPORU.parent.mkdir(parents=True, exist_ok=True)
    KALITE_RAPORU.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Kalite raporu yazıldı: %s", KALITE_RAPORU)
    asan = {k: v["uzun_bosluk_nan_orani"] for k, v in rapor["bitbrains"].items() if v["esik_asildi"]}
    if asan:
        raise VeriDogrulamaHatasi(f"DUR: uzun boşluk NaN oranı eşiği aştı: {asan} (Bölüm 4.3)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
