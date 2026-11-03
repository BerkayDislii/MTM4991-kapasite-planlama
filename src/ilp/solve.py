"""M5: tahmine dayalı çok dönemli ILP'nin kayan pencereyle çözümü ve gerçek talep üzerinde değerlendirmesi.

Her havuz × model için: ısınma (doğrulama) günlerinden başlayarak her gün (1)–(6) çözülür, sonraki günün x0 ve
devralınan başlatmaları bu planın son durumundan gelir (Bölüm 6.3). ILP yalnızca forecast_<gün>.parquet dosyasını
görür (Bölüm 5.4); ölçütler test günlerinde gerçek D ile hesaplanır (Bölüm 7). Kahin ILP girdisi gerçek D'dir.

Çıktılar (outputs/ilp/):
  plans/<dataset>/<pool>/<model>/plan_<gün>.parquet   saatlik plan, kapasite, tahmin ve gerçek talep
  gunluk.csv          havuz × model × gün ölçütleri (+ çözüm süresi, LP gevşetmesi / yuvarlama karşılaştırması)
  havuz_ozet.csv      havuz × model, test günleri toplamı (+ kahine göre regret)
  tablo_ozet.csv      grup × model, havuzlar üzerinden ortalama ve std (LSTM'de havuz × tohum)
  tablo_ozet_metin.csv  "ort ± std" biçimli

Çalıştırma: .venv/bin/python -m src.ilp.solve
"""
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.forecast.ozet import grup_esleme
from src.ilp.model import IlpParam, Plan, SunucuTipleri, coz, dogrula, lp_yuvarla, param_oku, tipleri_oku

log = logging.getLogger(__name__)
KAHIN = "kahin"


def tahmin_oku(h: Havuz, model: str, gun: int, H: int, dizin: Path | None = None) -> np.ndarray:
    """Bölüm 5.4 sözleşmesi: (H, 2) dizisi [cpu_hat, mem_hat]; dilimler assert ile denetlenir.

    dizin: tahmin kök dizini (VARSAYILAN forecast.yaml forecasts_dir; sentetik senaryolar kendi dizinini verir).
    """
    kok = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) if dizin is None else dizin
    yol_ = kok / h.veri_seti / h.ad / model / f"forecast_{gun}.parquet"
    df = pl.read_parquet(yol_).sort("slot")
    assert df["slot"].to_list() == list(range(24 * gun, 24 * gun + H)), f"{yol_}: dilimler beklenen gün değil"
    a = df.select("cpu_hat", "mem_hat").to_numpy().astype(np.float64)
    assert np.isfinite(a).all(), f"{yol_}: NaN/inf"
    return a


def plan_yaz(d: Path, gun: int, plan: Plan, tip_ad: tuple[str, ...], kap: np.ndarray, Dh: np.ndarray,
             D: np.ndarray) -> None:
    """Saatlik planı parquet olarak yazar (geniş biçim: tip başına x, y, z)."""
    d.mkdir(parents=True, exist_ok=True)
    sut = {"slot": np.arange(24 * gun, 24 * gun + plan.x.shape[1], dtype=np.int64)}
    for k, a in enumerate(tip_ad):
        for v in ("x", "y", "z"):
            sut[f"{v}_{a}"] = getattr(plan, v)[k].astype(np.int64)
    sut |= {"kap_cpu": kap[0], "kap_mem": kap[1], "cpu_hat": Dh[:, 0], "mem_hat": Dh[:, 1],
            "cpu_peak": D[:, 0], "mem_peak": D[:, 1]}
    pl.DataFrame(sut).write_parquet(d / f"plan_{gun}.parquet")


def kayan_pencere(h: Havuz, model: str, y: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
                  tahmin_dizin: Path | None = None) -> Iterator[tuple[str, int, np.ndarray, np.ndarray, Plan,
                                                                       np.ndarray, np.ndarray | None, float]]:
    """Kayan pencere çekirdeği (Bölüm 6.3): ısınma + değerlendirme günlerini sırayla çözer ve doğrular.

    y: (n_dilim, 2) gerçek talep (sentetik senaryoda bozulmuş seri). Her gün için
    (bölüm, gün, D̂, D, plan, x0, y_onceki, çözüm süresi) üretir; x0/y_onceki o günün başlangıç durumudur.
    """
    cfg = yaml_oku("ilp")
    H = cfg["model"]["horizon_h"]
    gunler = h.degerlendirme_gunleri
    x0, y_onceki = np.zeros(tipler.K), None
    for b in (cfg["kayan_pencere"]["isinma_bolumu"], cfg["kayan_pencere"]["degerlendirme_bolumu"]):
        for g in gunler[b]:
            D = y[24 * g: 24 * g + H]
            Dh = D.copy() if model == KAHIN else tahmin_oku(h, model, g, H, tahmin_dizin)
            t0 = time.perf_counter()
            plan = coz(Dh[:, 0], Dh[:, 1], tipler, p, x0, y_onceki, cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"])
            sure = time.perf_counter() - t0
            # Bölüm 6.5: bağımsız doğrulayıcı (tahmin edilen talebe göre (2)–(6))
            ihlal = dogrula(plan, Dh[:, 0], Dh[:, 1], tipler, p, x0, y_onceki)
            if ihlal:
                raise RuntimeError(f"{h.ad}/{model}/gün {g}: doğrulayıcı ihlali: {ihlal[:5]}")
            yield b, g, Dh, D, plan, x0, y_onceki, sure
            # Sonraki günün başlangıç durumu: bu planın son dilimi ve başlatmaları (Bölüm 6.3)
            x0, y_onceki = plan.x[:, -1].copy(), plan.y.copy()


def planlari_birlestir(planlar: list[Plan]) -> Plan:
    """Günlük planları zaman ekseninde birleştirir."""
    return Plan(*(np.concatenate([getattr(q, v) for q in planlar], axis=1) for v in ("x", "y", "z")),
                amac=float(sum(q.amac for q in planlar)))


def kayan_pencere_plani(h: Havuz, model: str, y: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
                        tahmin_dizin: Path | None = None) -> Plan:
    """Isınma + değerlendirme günlerinin birleşik planı (M7 ızgarası / sentetik: diske yazmadan)."""
    return planlari_birlestir([q[4] for q in kayan_pencere(h, model, y, tipler, p, tahmin_dizin)])


def havuz_model_coz(h: Havuz, model: str) -> tuple[list[dict], dict]:
    """Bir havuz × model için kayan pencere; (günlük satırlar, test toplamı satırı) döndürür."""
    cfg = yaml_oku("ilp")
    tipler = tipleri_oku()
    p = param_oku(tipler)
    tol = cfg["degerlendirme"]["tolerans"]
    y = saatlik_seri(h)
    deg = cfg["kayan_pencere"]["degerlendirme_bolumu"]
    plan_dizin = yol(cfg["outputs"]["dizin"]) / "plans" / h.veri_seti / h.ad / model
    gunluk, test_planlar, test_D = [], [], []
    for b, g, Dh, D, plan, x0, y_onceki, sure in kayan_pencere(h, model, y, tipler, p):
        # Bölüm 11(g): aynı durumdan LP gevşetmesi + yukarı yuvarlama
        lp, yuv, uygun = lp_yuvarla(Dh[:, 0], Dh[:, 1], tipler, p, x0, y_onceki)
        kap = np.stack([(p.rho_cpu * tipler.C[:, None] * plan.x).sum(0), (p.rho_mem * tipler.M[:, None] * plan.x).sum(0)])
        plan_yaz(plan_dizin, g, plan, tipler.ad, kap, Dh, D)
        gunluk.append({"veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "model": model, "bolum": b, "gun": g,
                       "amac": plan.amac, "cozum_s": sure, "lp_amac": lp.amac, "yuvarla_amac": yuv.amac,
                       "yuvarla_uygun": uygun, **plan_olcutleri(plan, D[:, 0], D[:, 1], tipler, p, tol)})
        if b == deg:
            test_planlar.append(plan)
            test_D.append(D)
    birlesik = planlari_birlestir(test_planlar)
    Dt = np.concatenate(test_D)
    toplam = {"veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "model": model, "n_gun": len(test_planlar),
              **plan_olcutleri(birlesik, Dt[:, 0], Dt[:, 1], tipler, p, tol)}
    toplam["maliyet_gunluk"] = toplam["maliyet"] / len(test_planlar)
    return gunluk, toplam


def havuz_coz(h: Havuz) -> tuple[list[dict], list[dict]]:
    """Bir havuzun tüm modelleri (+ kahin)."""
    cfg = yaml_oku("ilp")["degerlendirme"]
    modeller = list(cfg["modeller"]) + ([KAHIN] if cfg["kahin"] else [])
    gunluk, toplam = [], []
    for m in modeller:
        g, t = havuz_model_coz(h, m)
        gunluk += g
        toplam.append(t)
    log.info("%s: %d model çözüldü", h.ad, len(modeller))
    return gunluk, toplam


OZET_OLCUTLER: list[str] = ["maliyet", "maliyet_gunluk", "regret", "regret_yuzde", "n_acma", "ort_sunucu",
                            "bos_orani_cpu", "bos_orani_mem", "eksik_saat", "eksik_saat_orani",
                            "eksik_saat_cpu", "eksik_saat_mem", "karsilanmayan_cpu", "karsilanmayan_mem",
                            "karsilanmayan_cpu_yuzde", "karsilanmayan_mem_yuzde"]
MODEL_SIRASI: list[str] = ["son_deger", "mevsimsel_naif", "sarima", "lstm", KAHIN]


def ozetle(havuz: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Grup × model ortalama/std tablosu ve metin biçimi (LSTM tohumları tek model altında)."""
    df = havuz.join(grup_esleme(), on="havuz", how="left").with_columns(
        pl.when(pl.col("veri_seti") == "alibaba").then(pl.format("alibaba_r{}", pl.col("r").cast(pl.Int64)))
          .otherwise(pl.col("grup")).alias("grup"),
        pl.col("model").str.replace(r"^lstm_s\d+$", "lstm").alias("model"))
    assert df["grup"].null_count() == 0, "Gruplanmamış havuz var"
    ozet = (df.group_by("grup", "model")
              .agg([pl.len().alias("n_satir"), pl.col("havuz").n_unique().alias("n_havuz"), pl.col("n_gun").first()]
                   + [pl.col(c).mean().alias(f"{c}_ort") for c in OZET_OLCUTLER]
                   + [pl.col(c).std().alias(f"{c}_std") for c in OZET_OLCUTLER])
              .with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(MODEL_SIRASI)}).alias("_m"))
              .sort("grup", "_m").drop("_m"))

    def bicim(c: str, basamak: int) -> pl.Expr:
        return (pl.when(pl.col(f"{c}_std").is_null() | (pl.col("n_satir") == 1))
                  .then(pl.col(f"{c}_ort").round(basamak).cast(pl.String))
                  .otherwise(pl.format("{} ± {}", pl.col(f"{c}_ort").round(basamak), pl.col(f"{c}_std").round(basamak)))
                  .alias(c))
    metin = ozet.select("grup", "model", "n_havuz", "n_satir", "n_gun",
                        bicim("maliyet", 1), bicim("maliyet_gunluk", 1), bicim("regret_yuzde", 2),
                        bicim("bos_orani_cpu", 3), bicim("bos_orani_mem", 3), bicim("eksik_saat_orani", 3),
                        bicim("karsilanmayan_cpu", 1), bicim("karsilanmayan_mem", 1),
                        bicim("karsilanmayan_cpu_yuzde", 3), bicim("karsilanmayan_mem_yuzde", 3), bicim("n_acma", 1))
    return ozet, metin


def main() -> None:
    """Tüm havuzlar için ILP koşusu ve tablolar."""
    cfg = yaml_oku("ilp")
    cikti = yol(cfg["outputs"]["dizin"])
    cikti.mkdir(parents=True, exist_ok=True)
    sonuc = Parallel(n_jobs=cfg["cozucu"]["n_jobs"])(delayed(havuz_coz)(h) for h in havuzlari_listele())
    gunluk = pl.DataFrame([s for g, _ in sonuc for s in g], infer_schema_length=None)
    havuz = pl.DataFrame([s for _, t in sonuc for s in t], infer_schema_length=None)
    # Regret (Bölüm 7, ölçüt 5): aynı havuzun kahin maliyetine göre
    kahin = havuz.filter(pl.col("model") == KAHIN).select("havuz", pl.col("maliyet").alias("_kahin"))
    havuz = havuz.join(kahin, on="havuz").with_columns(
        (pl.col("maliyet") - pl.col("_kahin")).alias("regret"),
        (100 * (pl.col("maliyet") - pl.col("_kahin")) / pl.col("_kahin")).alias("regret_yuzde")).drop("_kahin")
    gunluk.write_csv(cikti / "gunluk.csv")
    havuz.write_csv(cikti / "havuz_ozet.csv")
    ozet, metin = ozetle(havuz)
    ozet.write_csv(cikti / "tablo_ozet.csv")
    metin.write_csv(cikti / "tablo_ozet_metin.csv")
    # Bölüm 11(g) özeti: ILP ile LP + yukarı yuvarlama arasındaki fark
    g = gunluk.with_columns((100 * (pl.col("yuvarla_amac") - pl.col("amac")) / pl.col("amac")).alias("yuvarla_fark_yuzde"),
                            (100 * (pl.col("amac") - pl.col("lp_amac")) / pl.col("amac")).alias("lp_bosluk_yuzde"))
    (g.group_by("veri_seti").agg(pl.len().alias("n_cozum"), pl.col("yuvarla_uygun").mean().alias("yuvarla_uygun_orani"),
                                 pl.col("yuvarla_fark_yuzde").mean().alias("yuvarla_fark_yuzde_ort"),
                                 pl.col("yuvarla_fark_yuzde").max().alias("yuvarla_fark_yuzde_max"),
                                 pl.col("lp_bosluk_yuzde").mean().alias("lp_bosluk_yuzde_ort"),
                                 pl.col("cozum_s").mean().alias("cozum_s_ort"), pl.col("cozum_s").max().alias("cozum_s_max"))
      .sort("veri_seti").write_csv(cikti / "tablo_lp_yuvarlama.csv"))
    log.info("M5 tamam: %d günlük çözüm, %d havuz × model satırı", gunluk.height, havuz.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
