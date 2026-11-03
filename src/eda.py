"""Bitbrains keşifsel veri analizi (MIMARI.md Bölüm 10). Şekil (PNG) ve tablolar (CSV) outputs/eda/ altına yazılır.

Mevsimsellik / durağanlık analizleri yalnızca eğitim döneminde, mevsimsel naif taban eğitim ve doğrulama
dönemlerinde hesaplanır; test dönemi kullanılmaz. Şekillerin içine başlık yazılmaz (raporda altyazı).

Çalıştırma: .venv/bin/python -m src.eda
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.signal import periodogram  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.tools.sm_exceptions import InterpolationWarning  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import adfuller, kpss  # noqa: E402

from src.common.config import yaml_oku, yol  # noqa: E402

log = logging.getLogger(__name__)

CIKTI: Path = yol("outputs/eda")
KAYNAKLAR: dict[str, str] = {"cpu": "CPU (çekirdek)", "mem": "Bellek (GiB)"}


# ---------------------------------------------------------------- yardımcılar

def sekil_ayarla(fcfg: dict[str, Any]) -> None:
    """Serif yazı tipi ve rapor boyutları için matplotlib ayarları."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [fcfg["font"], "DejaVu Serif"],
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": fcfg["dpi"],
        "savefig.bbox": "tight", "axes.unicode_minus": False, "figure.constrained_layout.use": True,
    })


def kaydet(fig: plt.Figure, ad: str) -> None:
    """Şekli outputs/eda/<ad>.png olarak kaydeder ve kapatır."""
    fig.savefig(CIKTI / f"{ad}.png")
    plt.close(fig)
    log.info("Şekil: %s.png", ad)


def bolme_sinirlari(pcfg: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Gün bölmesinden [baş, son) dilim aralıklarını üretir."""
    e, d, t = (24 * pcfg[k] for k in ("train", "val", "test"))
    return {"egitim": (0, e), "dogrulama": (e, e + d), "test": (e + d, e + d + t)}


def havuz_oku(ad: str) -> pl.DataFrame:
    """data/processed/pool_<ad>.parquet dosyasını okur."""
    return pl.read_parquet(yol(f"data/processed/pool_{ad}.parquet"))


# ---------------------------------------------------------------- 1) kapsama ve yaşam süresi

def kapsama_analizi(fcfg: dict[str, Any], adim_dilim: int) -> pl.DataFrame:
    """VM yaşam süresi dağılımı ve kapsama tablosu; yaşam süresi histogramı."""
    rapor = json.loads((CIKTI / "quality_report.json").read_text(encoding="utf-8"))["bitbrains"]
    satirlar, fig = [], plt.figure(figsize=(fcfg["width_in"], fcfg["height_in"]))
    ax = fig.gca()
    for ds, q in rapor.items():
        df = pl.read_parquet(yol(f"data/interim/bitbrains_{ds}.parquet"))
        toplam_hucre = df["cell"].max() - df["cell"].min() + 1
        yasam = df.group_by("vm_id").agg(pl.len().alias("n"))["n"].to_numpy() / (adim_dilim * 24)
        ax.hist(yasam, bins=np.arange(0, 31, 1), alpha=0.6, label=ds)
        satirlar.append({
            "veri_seti": ds, "vm_sayisi": q["vm_sayisi"], "donem_gun": round(toplam_hucre / (adim_dilim * 24), 2),
            "yasam_gun_min": round(float(yasam.min()), 2), "yasam_gun_medyan": round(float(np.median(yasam)), 2),
            "yasam_gun_max": round(float(yasam.max()), 2),
            "tum_donemi_kapsayan_vm_orani": round(float((yasam >= 0.99 * toplam_hucre / (adim_dilim * 24)).mean()), 4),
            "enterpole_hucre_orani": round(q["enterpole_hucre"] / q["yasam_suresi_hucre"], 5),
            "uzun_bosluk_nan_orani": round(q["uzun_bosluk_nan_orani"], 5),
            "yasam_disi_atilan_satir": q["yasam_disi_atilan_satir"],
        })
    ax.set_xlabel("VM yaşam süresi (gün)")
    ax.set_ylabel("VM sayısı")
    ax.legend()
    kaydet(fig, "vm_yasam_suresi")
    return pl.DataFrame(satirlar)


# ---------------------------------------------------------------- 2) zaman grafikleri ve tepe/ortalama

def zaman_grafigi(ad: str, df: pl.DataFrame, sinir: dict[str, tuple[int, int]], fcfg: dict[str, Any]) -> None:
    """Saatlik tepe ve ortalama talep (CPU ve bellek), dönem sınırları işaretli."""
    fig, eksen = plt.subplots(2, 1, figsize=(fcfg["width_in"], fcfg["height_in"] * 1.5), sharex=True)
    gun = df["slot"].to_numpy() / 24
    for ax, (k, etiket) in zip(eksen, KAYNAKLAR.items()):
        ax.plot(gun, df[f"{k}_peak"], lw=0.7, label="Saatlik tepe")
        ax.plot(gun, df[f"{k}_mean"], lw=0.7, label="Saatlik ortalama")
        for bas, _ in list(sinir.values())[1:]:
            ax.axvline(bas / 24, color="k", ls="--", lw=0.8)
        ax.set_ylabel(etiket)
    ymax = eksen[0].get_ylim()[1]
    for d_ad, (bas, son) in zip(["Eğitim", "Doğrulama", "Test"], sinir.values()):
        eksen[0].text((bas + son) / 48, ymax * 0.97, d_ad, ha="center", va="top", fontsize=8)
    eksen[0].legend(loc="lower left", fontsize=8)
    eksen[1].set_xlabel("Gün (göreli)")
    kaydet(fig, f"talep_zaman_{ad}")


def tepe_ortalama_tablosu(havuzlar: dict[str, pl.DataFrame], sinir: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """Saatlik tepe/ortalama oranı (dönem bazında) ve tepe planlamasının ortalama planlamaya göre ek kapasitesi."""
    satirlar = []
    for ad, df in havuzlar.items():
        for donem, (bas, son) in sinir.items():
            p = df.filter(pl.col("slot").is_between(bas, son - 1))
            for k in KAYNAKLAR:
                oran = (p[f"{k}_peak"] / p[f"{k}_mean"]).to_numpy()
                satirlar.append({
                    "havuz": ad, "donem": donem, "kaynak": k,
                    "oran_medyan": round(float(np.median(oran)), 4), "oran_p95": round(float(np.percentile(oran, 95)), 4),
                    "oran_max": round(float(oran.max()), 4),
                    # Σ tepe / Σ ortalama: tepe yerine ortalamaya göre planlansaydı eksik kalacak kapasite oranı
                    "toplam_tepe_toplam_ortalama": round(float(p[f"{k}_peak"].sum() / p[f"{k}_mean"].sum()), 4),
                })
    return pl.DataFrame(satirlar)


# ---------------------------------------------------------------- 3) mevsimsellik: ACF/PACF, periyodogram, STL

def acf_pacf_grafigi(ad: str, egitim: pl.DataFrame, cfg: dict[str, Any]) -> None:
    """Eğitim dönemi saatlik tepe serileri için ACF ve PACF (CPU ve bellek)."""
    fcfg = cfg["figure"]
    fig, eksen = plt.subplots(2, 2, figsize=(fcfg["width_in"], fcfg["height_in"] * 1.4))
    for i, (k, etiket) in enumerate(KAYNAKLAR.items()):
        x = egitim[f"{k}_peak"].to_numpy()
        plot_acf(x, lags=cfg["acf_lags"], ax=eksen[i, 0], title="", markersize=2)
        plot_pacf(x, lags=cfg["acf_lags"], ax=eksen[i, 1], title="", method="ywm", markersize=2)
        eksen[i, 0].set_ylabel(f"{etiket.split(' ')[0]}\nACF")
        eksen[i, 1].set_ylabel("PACF")
        for ax in eksen[i]:
            for g in range(cfg["season_hours"], cfg["acf_lags"] + 1, cfg["season_hours"]):
                ax.axvline(g, color="gray", ls=":", lw=0.8)
    for ax in eksen[1]:
        ax.set_xlabel("Gecikme (saat)")
    kaydet(fig, f"acf_pacf_{ad}")


def periyodogram_analizi(ad: str, egitim: pl.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Ortalaması çıkarılmış eğitim serisinin periyodogramı; en güçlü K periyot tabloya."""
    fcfg = cfg["figure"]
    fig, ax = plt.subplots(figsize=(fcfg["width_in"], fcfg["height_in"]))
    satirlar = []
    for k, etiket in KAYNAKLAR.items():
        x = egitim[f"{k}_peak"].to_numpy()
        f, g = periodogram(x - x.mean(), detrend="linear")
        f, g = f[1:], g[1:]                       # sıfır frekansı at
        ax.semilogy(1 / f, g, lw=0.8, label=etiket.split(" ")[0])
        for sira, i in enumerate(np.argsort(g)[::-1][: cfg["periodogram_top_k"]], start=1):
            satirlar.append({"havuz": ad, "kaynak": k, "sira": sira, "periyot_saat": round(float(1 / f[i]), 2),
                             "guc_payi": round(float(g[i] / g.sum()), 4)})
    ax.set_xscale("log")
    ax.axvline(cfg["season_hours"], color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Periyot (saat, log ölçek)")
    ax.set_ylabel("Spektral yoğunluk")
    ax.legend()
    kaydet(fig, f"periyodogram_{ad}")
    return satirlar


def stl_analizi(ad: str, egitim: pl.DataFrame, cfg: dict[str, Any], sekil: bool = True) -> list[dict[str, Any]]:
    """STL ayrışımı (periyot 24); mevsimsellik ve trend gücü (Wang vd. F ölçütleri) tabloya, istenirse şekil."""
    fcfg, satirlar = cfg["figure"], []
    for k, etiket in KAYNAKLAR.items():
        x = egitim[f"{k}_peak"].to_numpy()
        s = STL(x, period=cfg["stl"]["period"], robust=cfg["stl"]["robust"]).fit()
        if sekil:
            stl_sekli(ad, k, etiket, x, s, fcfg)
        # F_s = max(0, 1 - Var(R)/Var(S+R)),  F_t = max(0, 1 - Var(R)/Var(T+R))
        satirlar.append({
            "havuz": ad, "kaynak": k,
            "mevsimsellik_gucu": round(max(0.0, 1 - np.var(s.resid) / np.var(s.seasonal + s.resid)), 4),
            "trend_gucu": round(max(0.0, 1 - np.var(s.resid) / np.var(s.trend + s.resid)), 4),
            "mevsimsel_genlik": round(float(s.seasonal.max() - s.seasonal.min()), 3),
            "seri_ortalama": round(float(x.mean()), 3),
        })
    return satirlar


def stl_sekli(ad: str, k: str, etiket: str, x: np.ndarray, s: Any, fcfg: dict[str, Any]) -> None:
    """STL bileşenlerinin (gözlem, trend, mevsimsel, kalıntı) dört panelli şekli."""
    fig, eksen = plt.subplots(4, 1, figsize=(fcfg["width_in"], fcfg["height_in"] * 1.6), sharex=True)
    gun = np.arange(len(x)) / 24
    for ax, seri, et in zip(eksen, [x, s.trend, s.seasonal, s.resid], ["Gözlem", "Trend", "Mevsimsel", "Kalıntı"]):
        ax.plot(gun, seri, lw=0.7)
        ax.set_ylabel(et)
    eksen[-1].set_xlabel(f"Gün (eğitim dönemi) — {etiket}")
    kaydet(fig, f"stl_{ad}_{k}")


# ---------------------------------------------------------------- 4) durağanlık

def duraganlik_testleri(ad: str, egitim: pl.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """ADF (H0: birim kök) ve KPSS (H0: durağan) — düzey, 1. fark ve 24 saatlik mevsimsel fark."""
    a, m = cfg["significance"], cfg["season_hours"]
    satirlar = []
    for k in KAYNAKLAR:
        x = egitim[f"{k}_peak"].to_numpy()
        for donusum, seri in [("duzey", x), ("fark1", np.diff(x)), (f"mevsimsel_fark{m}", x[m:] - x[:-m])]:
            adf = adfuller(seri, autolag="AIC", regression="c", result_object=False)
            with warnings.catch_warnings(record=True) as uyari:
                warnings.simplefilter("always", InterpolationWarning)
                kp = kpss(seri, regression="c", nlags="auto")
            # KPSS p değeri tablo sınırında kırpılmışsa işaretle
            sinirda = any(issubclass(w.category, InterpolationWarning) for w in uyari)
            adf_d, kpss_d = bool(adf[1] < a), bool(kp[1] >= a)
            satirlar.append({
                "havuz": ad, "kaynak": k, "donusum": donusum,
                "adf_istatistik": round(adf[0], 4), "adf_p": round(adf[1], 4), "adf_gecikme": adf[2],
                "kpss_istatistik": round(kp[0], 4), "kpss_p": round(kp[1], 4), "kpss_p_tablo_sinirinda": sinirda,
                "adf_duragan": adf_d, "kpss_duragan": kpss_d,
                "sonuc": "duragan" if adf_d and kpss_d else ("duragan_degil" if not adf_d and not kpss_d else "celisik"),
            })
    return satirlar


# ---------------------------------------------------------------- 5) mevsimsel naif taban

def mevsimsel_naif(havuzlar: dict[str, pl.DataFrame], sinir: dict[str, tuple[int, int]], m: int) -> pl.DataFrame:
    """Gün-ileri mevsimsel naif (ŷ_t = y_{t-24}) hataları; eğitim ve doğrulama dönemi (test M4'te)."""
    satirlar = []
    for ad, df in havuzlar.items():
        for donem in ("egitim", "dogrulama"):
            bas, son = sinir[donem]
            for k in KAYNAKLAR:
                y = df[f"{k}_peak"].to_numpy()
                t = np.arange(max(bas, m), son)
                gercek, tahmin = y[t], y[t - m]
                hata = tahmin - gercek
                satirlar.append({
                    "havuz": ad, "donem": donem, "kaynak": k, "n": len(t),
                    "mae": round(float(np.abs(hata).mean()), 4), "rmse": round(float(np.sqrt((hata ** 2).mean())), 4),
                    "goreli_mae": round(float(np.abs(hata).mean() / gercek.mean()), 4),
                    "eksik_tahmin_orani": round(float((tahmin < gercek).mean()), 4),
                    "ortalama_eksik_tahmin": round(float(np.clip(gercek - tahmin, 0, None).mean()), 4),
                })
    return pl.DataFrame(satirlar)


# ---------------------------------------------------------------- 6) kaynak oranı

def kaynak_orani(havuzlar: dict[str, pl.DataFrame], sinir: dict[str, tuple[int, int]],
                 cfg: dict[str, Any]) -> pl.DataFrame:
    """Talebin GB/çekirdek oranı (eğitim) ve referans aile oranlarıyla kıyas; histogram şekli."""
    bas, son = sinir["egitim"]
    fcfg, ref = cfg["figure"], cfg["reference_gb_per_core"]
    fig, ax = plt.subplots(figsize=(fcfg["width_in"], fcfg["height_in"]))
    satirlar = []
    for ad, df in havuzlar.items():
        p = df.filter(pl.col("slot").is_between(bas, son - 1))
        oran = (p["mem_peak"] / p["cpu_peak"]).to_numpy()
        if ad in cfg["figure_pools"]:
            ax.hist(oran, bins=50, alpha=0.6, label=ad.replace("bitbrains_", ""))
        satir = {"havuz": ad, "oran_medyan": round(float(np.median(oran)), 3),
                 "oran_p5": round(float(np.percentile(oran, 5)), 3), "oran_p95": round(float(np.percentile(oran, 95)), 3)}
        for r in ref:
            # Talep oranı r'den büyükse bellek kısıtı, küçükse CPU kısıtı bağlayıcı olur
            satir[f"bellek_baglayici_orani_r{r}"] = round(float((oran > r).mean()), 4)
        satirlar.append(satir)
    for r in ref:
        ax.axvline(r, color="k", ls="--", lw=0.8)
        ax.text(r, ax.get_ylim()[1] * 0.95, f" {r} GB/çekirdek", fontsize=7, va="top")
    ax.set_xlabel("Talep oranı: bellek tepesi / CPU tepesi (GiB / çekirdek)")
    ax.set_ylabel("Saat sayısı (eğitim)")
    ax.legend()
    kaydet(fig, "kaynak_orani")
    return pl.DataFrame(satirlar)


# ---------------------------------------------------------------- 7) VM ve toplam günlük otokorelasyon

def vm_otokorelasyon(c_egitim_sonu: dict[str, int], cfg: dict[str, Any], adim_dilim: int) -> pl.DataFrame:
    """5 dk serilerinde 1 günlük gecikmeli otokorelasyon: VM bazında dağılım vs havuz toplamı (eğitim)."""
    g, fcfg = cfg["vm_acf"]["lag_steps"], cfg["figure"]
    min_hucre = cfg["vm_acf"]["min_days"] * 24 * adim_dilim
    fig, eksen = plt.subplots(1, 2, figsize=(fcfg["width_in"], fcfg["height_in"]), sharey=True)
    satirlar = []
    for ax, k in zip(eksen, ["cpu_cores", "mem_gb"]):
        for ds, son in c_egitim_sonu.items():
            df = (pl.read_parquet(yol(f"data/interim/bitbrains_{ds}.parquet"), columns=["vm_id", "cell", k])
                    .filter(pl.col("cell") < son).sort("vm_id", "cell"))
            # VM ızgaraları yaşam süresi içinde kesintisiz olduğundan shift(g) tam 1 gün öncesidir
            vm = (df.with_columns(pl.col(k).shift(g).over("vm_id").alias("gec"))
                    .group_by("vm_id").agg(pl.len().alias("n"), pl.corr(k, "gec").alias("r"))
                    .filter(pl.col("n") >= min_hucre))
            r = vm["r"].fill_nan(None).drop_nulls().to_numpy()
            toplam = df.group_by("cell").agg(pl.col(k).sum()).sort("cell")
            izgara = pl.DataFrame({"cell": np.arange(toplam["cell"].min(), son, dtype=np.int64)})
            x = izgara.join(toplam, on="cell", how="left")[k].fill_null(0.0).to_numpy()
            r_toplam = float(np.corrcoef(x[g:], x[:-g])[0, 1])
            ax.hist(r, bins=40, alpha=0.5, label=ds)
            ax.axvline(r_toplam, ls="--", lw=1, color=ax.patches[-1].get_facecolor()[:3])
            satirlar.append({"veri_seti": ds, "kaynak": k, "vm_sayisi": len(r),
                             "sabit_seri_vm": int(vm.height - len(r)),
                             "vm_r_medyan": round(float(np.median(r)), 4), "vm_r_p25": round(float(np.percentile(r, 25)), 4),
                             "vm_r_p75": round(float(np.percentile(r, 75)), 4),
                             "vm_r_gt_0_5_orani": round(float((r > 0.5).mean()), 4), "toplam_r": round(r_toplam, 4)})
        ax.set_xlabel(f"1 günlük gecikmeli otokorelasyon ({'CPU' if k == 'cpu_cores' else 'Bellek'})")
    eksen[0].set_ylabel("VM sayısı")
    eksen[0].legend(fontsize=8)
    kaydet(fig, "vm_otokorelasyon")
    return pl.DataFrame(satirlar)


# ---------------------------------------------------------------- ana akış

def main() -> None:
    """Tüm EDA şekil ve tablolarını üretir."""
    cfg, pcfg = yaml_oku("eda"), yaml_oku("preprocess")
    sekil_ayarla(cfg["figure"])
    CIKTI.mkdir(parents=True, exist_ok=True)
    adim_dilim = pcfg["grid"]["steps_per_slot"]
    sinir = bolme_sinirlari(pcfg["bitbrains"]["split_days"])
    uyeler = json.loads(yol("data/processed/pool_members.json").read_text(encoding="utf-8"))
    havuzlar = {ad: havuz_oku(ad) for ad in uyeler}
    egitim = {ad: df.filter(pl.col("slot") < sinir["egitim"][1]) for ad, df in havuzlar.items()}

    def yaz(tablo: pl.DataFrame, ad: str) -> None:
        tablo.write_csv(CIKTI / f"{ad}.csv")
        log.info("Tablo: %s.csv", ad)

    yaz(kapsama_analizi(cfg["figure"], adim_dilim), "tablo_kapsama")
    for ad in cfg["figure_pools"]:
        zaman_grafigi(ad, havuzlar[ad], sinir, cfg["figure"])
        acf_pacf_grafigi(ad, egitim[ad], cfg)
    yaz(tepe_ortalama_tablosu(havuzlar, sinir), "tablo_tepe_ortalama")
    yaz(pl.DataFrame([r for ad in cfg["figure_pools"] for r in periyodogram_analizi(ad, egitim[ad], cfg)]),
        "tablo_periyodogram")
    stl_satir = [r for ad in havuzlar for r in stl_analizi(ad, egitim[ad], cfg, sekil=ad in cfg["figure_pools"])]
    yaz(pl.DataFrame(stl_satir), "tablo_stl_guc")
    yaz(pl.DataFrame([r for ad in havuzlar for r in duraganlik_testleri(ad, egitim[ad], cfg)]), "tablo_duraganlik")
    yaz(mevsimsel_naif(havuzlar, sinir, cfg["season_hours"]), "tablo_mevsimsel_naif")
    yaz(kaynak_orani(havuzlar, sinir, cfg), "tablo_kaynak_orani")
    # VM düzeyi analiz için eğitim sonu hücresi: veri setinin ilk hücresi + eğitim dilimi × 12
    c_son = {ds: int(pl.read_parquet(yol(f"data/interim/bitbrains_{ds}.parquet"), columns=["cell"])["cell"].min())
             + sinir["egitim"][1] * adim_dilim for ds in pcfg["bitbrains"]["datasets"]}
    yaz(vm_otokorelasyon(c_son, cfg, adim_dilim), "tablo_vm_otokorelasyon")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
