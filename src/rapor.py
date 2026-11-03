"""M8: rapor şekil ve tabloları (MIMARI.md Bölüm 9 ve 13; şablon: Tasarim_Uygulamasi_Rapor_Sablonu.docx).

Yalnız önceki taşların ürettiği çıktıları okur (yeni hesap yapmaz). Numaralar configs/rapor.yaml katalog sırasından
bölüm başına verilir. Şekillerde başlık yoktur (altyazı rapora girer); serif yazı tipi, Türkçe ondalık virgül.
Her şeklin çizildiği veri ayrıca CSV olarak yazılır (tablo görünümü).

Çıktılar (outputs/rapor/):
  sekil_<b>_<n>_<id>.png (+ _veri.csv)   tablo_<b>_<n>_<id>.csv   katalog.csv   katalog.md (önizleme)

Çalıştırma: .venv/bin/python -m src.rapor
"""
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.common.config import yaml_oku, yol  # noqa: E402
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri  # noqa: E402

log = logging.getLogger(__name__)
CFG: dict[str, Any] = yaml_oku("rapor")
AD: dict[str, dict[str, str]] = CFG["ad"]


# ---------------------------------------------------------------- biçim yardımcıları

def sayi(x: float | None, basamak: int) -> str:
    """Türkçe ondalık virgüllü sayı; None/NaN → '—'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{basamak}f}".replace(".", CFG["ondalik"])


def ort_std(df: pl.DataFrame, c: str, basamak: int, carpan: float = 1.0) -> list[str]:
    """<c>_ort ve <c>_std sütunlarından 'ort ± std' (tek satırlı grupta yalnız ort)."""
    out = []
    for r in df.iter_rows(named=True):
        o, s = r[f"{c}_ort"], r.get(f"{c}_std")
        tek = r.get("n_satir", 2) == 1 or s is None
        out.append(sayi(o * carpan, basamak) if tek else f"{sayi(o * carpan, basamak)} ± {sayi(s * carpan, basamak)}")
    return out


def adlandir(df: pl.DataFrame) -> pl.DataFrame:
    """grup / model / strateji / kaynak / senaryo / veri_seti sütunlarını görünen adlara çevirir."""
    eslem = {"grup": "grup", "model": "model", "strateji": "strateji", "kaynak": "kaynak", "senaryo": "senaryo",
             "veri_seti": "veri_seti"}
    return df.with_columns([pl.col(c).replace_strict(AD[a], default=pl.col(c)) for c, a in eslem.items()
                            if c in df.columns])


def havuz_bul(ad: str) -> Havuz:
    """Havuz adından Havuz tanımı."""
    return next(h for h in havuzlari_listele() if h.ad == ad)


def stil(anahtar: str, tur: str) -> dict[str, Any]:
    """Strateji/model/tip için sabit renk ve çizgi stili (kimlik her şekilde aynı)."""
    i = CFG["renk"][tur][anahtar]
    ls = CFG["cizgi"][i]
    return {"color": CFG["renk"]["kategorik"][i], "linestyle": tuple(ls) if isinstance(ls, list) else ls}


def strateji_stili(ad: str) -> tuple[dict[str, Any], str]:
    """'r2_hibrit_uyumlu_sarima' gibi sim adından (stil, görünen ad)."""
    for s in sorted(CFG["renk"]["strateji"], key=len, reverse=True):
        if ad == s or ad.startswith(s + "_"):
            model = ad[len(s) + 1:]
            etiket = AD["strateji"][s] + (f", {AD['model'][model.split('_s')[0] if model.startswith('lstm') else model]}"
                                          if model else "")
            return stil(s, "strateji"), etiket
    raise KeyError(ad)


def sekil_hazirla() -> dict[str, Any]:
    """EDA ile aynı şekil ayarları (configs/eda.yaml figure)."""
    f = yaml_oku("eda")["figure"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [f["font"], "DejaVu Serif"], "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "savefig.dpi": f["dpi"],
        "savefig.bbox": "tight", "axes.unicode_minus": False, "figure.constrained_layout.use": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "legend.frameon": False})
    return f


def virgul(ax: plt.Axes, eksen: str = "both") -> None:
    """Eksen işaretlerinde ondalık virgül."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", CFG["ondalik"]))
    if eksen in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if eksen in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def sim_oku(kok: Path, h: Havuz, ad: str) -> pl.DataFrame:
    """Saatlik strateji simülasyonu (M6/M7b sim dosyası)."""
    return pl.read_parquet(kok / h.veri_seti / h.ad / f"{ad}.parquet").sort("slot")


def tahmin_oku(h: Havuz, model: str, gunler: list[int]) -> np.ndarray:
    """M4 tahmin dosyalarından (n_dilim, 2)."""
    d = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) / h.veri_seti / h.ad / model
    return np.concatenate([pl.read_parquet(d / f"forecast_{g}.parquet").sort("slot")
                           .select("cpu_hat", "mem_hat").to_numpy() for g in gunler])


# ---------------------------------------------------------------- tablolar

def t_sunucu_tipleri() -> pl.DataFrame:
    st = yaml_oku("server_types")
    return pl.DataFrame([{"Tip": t["ad"], "Aile": {"Compute optimized": "Hesaplama optimize",
                                                   "General purpose": "Genel amaçlı",
                                                   "Memory optimized": "Bellek optimize"}[t["aile"]],
                          "vCPU (çekirdek)": str(t["vcpu"]), "Bellek (GiB)": sayi(t["bellek_gib"], 0),
                          "GB/vCPU": sayi(t["gb_per_vcpu"], 0), "Saatlik maliyet (USD)": sayi(t["saatlik_usd"], 5)}
                         for t in st["tipler"]])


def t_tahmin_olcutleri() -> pl.DataFrame:
    d = (pl.read_csv(yol("outputs/models/tablo_ozet.csv")).filter(pl.col("bolum") == "test")
           .with_columns(pl.col("grup").replace_strict({g: i for i, g in enumerate(AD["grup"])}).alias("_g"),
                         pl.col("model").replace_strict({m: i for i, m in enumerate(CFG["renk"]["model"])}).alias("_m"),
                         pl.col("kaynak").replace_strict({"cpu": 0, "mem": 1}).alias("_k"))
           .sort("_g", "_k", "_m"))
    return adlandir(d).select(pl.col("grup").alias("Grup"), pl.col("kaynak").alias("Kaynak"),
                              pl.col("model").alias("Model"), pl.Series("MAE", ort_std(d, "mae", 1)),
                              pl.Series("RMSE", ort_std(d, "rmse", 1)), pl.Series("MAPE (%)", ort_std(d, "mape", 1)),
                              pl.Series("MASE", ort_std(d, "mase", 2)),
                              pl.Series("Eksik tahmin (%)", ort_std(d, "eksik_tahmin_orani", 1, 100)))


def t_sarima_secim() -> pl.DataFrame:
    d = pl.read_csv(yol("outputs/models/sarima_secim.csv")).filter(pl.col("veri_seti") == "bitbrains")
    return adlandir(d).select(
        pl.col("havuz").str.replace("bitbrains_", "").alias("Havuz"), pl.col("kaynak").alias("Kaynak"),
        pl.col("d").cast(pl.String), pl.col("D").cast(pl.String),
        pl.col("stl_mevsimsellik_gucu").map_elements(lambda v: sayi(v, 3), return_dtype=pl.String).alias("STL F_s"),
        pl.col("secilen_aile").replace({"mevsimsel": "Mevsimsel", "mevsimsel_olmayan": "Mevsimsel olmayan"})
          .alias("Seçilen aile"),
        pl.format("SARIMA{}×{}", pl.col("order"), pl.col("seasonal_order")).alias("Model"))


def karar_tablosu(d: pl.DataFrame, regret: str = "regret_yuzde") -> pl.DataFrame:
    """Strateji/ILP sonuç tablosu ortak sütunları (tablo_ozet biçimli girdi)."""
    return adlandir(d).select(
        pl.col("grup").alias("Grup"), pl.col("strateji").alias("Strateji"), pl.col("model").alias("Tahmin modeli"),
        pl.Series("Günlük maliyet (USD)", ort_std(d, "maliyet_gunluk", 0)),
        pl.Series("Regret (%)", ort_std(d, regret, 1)),
        pl.Series("Boşta işlemci", ort_std(d, "bos_orani_cpu", 2)),
        pl.Series("Boşta bellek", ort_std(d, "bos_orani_mem", 2)),
        pl.Series("Eksik saat (%)", ort_std(d, "eksik_saat_orani", 1, 100)),
        pl.Series("Karşılanmayan işlemci (%)", ort_std(d, "karsilanmayan_cpu_yuzde", 2)),
        pl.Series("Karşılanmayan bellek (%)", ort_std(d, "karsilanmayan_mem_yuzde", 2)),
        pl.Series("En uzun olay (sa)", ort_std(d, "olay_sure_max", 1)),
        pl.Series("Açılış", ort_std(d, "n_acma", 0)))


def _m6() -> pl.DataFrame:
    return pl.read_csv(yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "tablo_ozet.csv")


def _sirala(d: pl.DataFrame) -> pl.DataFrame:
    so = {"kahin": 0, "ilp": 1, "r0_tepkisel_ilp": 2, "r1_esik": 3, "r2_hibrit": 4, "r2_hibrit_uyumlu": 5}
    mo = {"son_deger": 0, "mevsimsel_naif": 1, "sarima": 2, "lstm": 3, "-": 4, "kahin": 4}
    return (d.with_columns(pl.col("grup").replace_strict({g: i for i, g in enumerate(AD["grup"])}).alias("_g"),
                           pl.col("strateji").replace_strict(so).alias("_s"), pl.col("model").replace_strict(mo).alias("_m"))
             .sort("_g", "_s", "_m").drop("_g", "_s", "_m"))


def t_ilp_sonuclari() -> pl.DataFrame:
    return karar_tablosu(_sirala(_m6().filter(pl.col("strateji").is_in(["ilp", "kahin"]))))


def t_lp_yuvarlama() -> pl.DataFrame:
    d = pl.read_csv(yol(yaml_oku("ilp")["outputs"]["dizin"]) / "tablo_lp_yuvarlama.csv")
    return adlandir(d).select(
        pl.col("veri_seti").alias("Veri seti"), pl.col("n_cozum").cast(pl.String).alias("Çözüm sayısı"),
        *[pl.col(c).map_elements(lambda v, b=b, k=k: sayi(v * k, b), return_dtype=pl.String).alias(a)
          for c, a, b, k in [("yuvarla_uygun_orani", "Yuvarlanmış plan (5)'e uygun (%)", 1, 100),
                             ("yuvarla_fark_yuzde_ort", "Yuvarlama maliyet farkı, ort. (%)", 3, 1),
                             ("yuvarla_fark_yuzde_max", "Yuvarlama maliyet farkı, en büyük (%)", 2, 1),
                             ("lp_bosluk_yuzde_ort", "LP alt sınır boşluğu, ort. (%)", 3, 1),
                             ("cozum_s_ort", "ILP çözüm süresi, ort. (s)", 3, 1),
                             ("cozum_s_max", "ILP çözüm süresi, en büyük (s)", 2, 1)]])


def _izgara() -> pl.DataFrame:
    return pl.read_csv(yol(yaml_oku("ilp")["izgara"]["dizin"]) / "tablo_ozet.csv")


def t_alfa_duyarlilik() -> pl.DataFrame:
    d = _sirala(_izgara().filter((pl.col("strateji") == "ilp") & (pl.col("model") == "sarima")
                                 & (pl.col("w") == 0.25) & (pl.col("L") == 2))).sort("alpha", maintain_order=True)
    d = d.sort(pl.col("grup").replace_strict({g: i for i, g in enumerate(AD["grup"])}), "alpha")
    return adlandir(d).select(
        pl.col("grup").alias("Grup"), pl.col("alpha").map_elements(lambda v: sayi(v, 2), return_dtype=pl.String).alias("α"),
        pl.Series("Günlük maliyet (USD)", ort_std(d, "maliyet_gunluk", 0)),
        pl.Series("Regret, aynı α kahinine göre (%)", ort_std(d, "regret_yuzde", 1)),
        pl.Series("Regret, α = 0 kahinine göre (%)", ort_std(d, "regret_a0_yuzde", 1)),
        pl.Series("Eksik saat (%)", ort_std(d, "eksik_saat_orani", 1, 100)),
        pl.Series("Karşılanmayan işlemci (%)", ort_std(d, "karsilanmayan_cpu_yuzde", 2)),
        pl.Series("Karşılanmayan bellek (%)", ort_std(d, "karsilanmayan_mem_yuzde", 2)),
        pl.Series("En uzun olay (sa)", ort_std(d, "olay_sure_max", 1)))


def t_w_L_etkisi() -> pl.DataFrame:
    d = _izgara().filter((pl.col("strateji") == "ilp") & (pl.col("model") == "sarima") & (pl.col("alpha") == 0.05))
    satir = []
    for g in AD["grup"]:
        for w in sorted(d["w"].unique()):
            s = {"Grup": AD["grup"][g], "w": sayi(w, 2)}
            for L in sorted(d["L"].unique()):
                r = d.filter((pl.col("grup") == g) & (pl.col("w") == w) & (pl.col("L") == L))
                s[f"Açılış (L = {L})"] = ort_std(r, "n_acma", 0)[0]
            r2 = d.filter((pl.col("grup") == g) & (pl.col("w") == w) & (pl.col("L") == 2))
            s["Günlük maliyet, L = 2 (USD)"] = ort_std(r2, "maliyet_gunluk", 0)[0]
            s["Eksik saat, L = 2 (%)"] = ort_std(r2, "eksik_saat_orani", 1, 100)[0]
            satir.append(s)
    return pl.DataFrame(satir)


def _karsilastirma(veri_seti: str) -> pl.DataFrame:
    d = _m6().filter(pl.col("grup").str.starts_with(veri_seti)
                     & ((pl.col("strateji").is_in(["kahin", "ilp", "r0_tepkisel_ilp", "r1_esik"]))
                        | (pl.col("strateji").str.starts_with("r2_") & (pl.col("model") == "sarima"))))
    return karar_tablosu(_sirala(d))


def t_karsilastirma_bitbrains() -> pl.DataFrame:
    return _karsilastirma("bitbrains")


def t_karsilastirma_alibaba() -> pl.DataFrame:
    return _karsilastirma("alibaba")


SENTETIK_STRATEJI: list[tuple[str, str]] = [("ilp", "sarima"), ("r0_tepkisel_ilp", "-"), ("r1_esik", "-"),
                                            ("r2_hibrit", "sarima"), ("r2_hibrit_uyumlu", "sarima")]


def t_sentetik_ozet() -> pl.DataFrame:
    s = pl.read_csv(yol(yaml_oku("synthetic")["dizin"]) / "havuz_ozet.csv")
    satir = []
    for vs in AD["veri_seti"]:
        for sc in AD["senaryo"]:
            r = {"Veri seti": AD["veri_seti"][vs], "Senaryo": AD["senaryo"][sc]}
            for st, m in SENTETIK_STRATEJI:
                f = s.filter((pl.col("veri_seti") == vs) & (pl.col("senaryo") == sc) & (pl.col("strateji") == st)
                             & (pl.col("model") == m))
                assert f.height > 0, (vs, sc, st, m)
                r[f"{AD['strateji'][st]}: regret (%)"] = sayi(f["regret_yuzde"].mean(), 1)
                r[f"{AD['strateji'][st]}: eksik saat (%)"] = sayi(100 * f["eksik_saat_orani"].mean(), 1)
            satir.append(r)
    return pl.DataFrame(satir)


def t_sentetik_tahmin() -> pl.DataFrame:
    m = (pl.read_csv(yol(yaml_oku("synthetic")["dizin"]) / "tahmin_olcutleri.csv")
           .with_columns(pl.col("model").str.replace(r"^lstm_s\d+$", "lstm")))
    satir = []
    for vs in AD["veri_seti"]:
        for sc in AD["senaryo"]:
            r = {"Veri seti": AD["veri_seti"][vs], "Senaryo": AD["senaryo"][sc]}
            for k in ("cpu", "mem"):
                for mod in CFG["renk"]["model"]:
                    f = m.filter((pl.col("veri_seti") == vs) & (pl.col("senaryo") == sc) & (pl.col("kaynak") == k)
                                 & (pl.col("model") == mod))
                    r[f"{AD['kaynak'][k]}: {AD['model'][mod]}"] = sayi(f["mase"].mean(), 2)
            satir.append(r)
    return pl.DataFrame(satir)


# ---------------------------------------------------------------- şekiller

def s_tahmin(havuz: str, bolumler: tuple[str, ...]) -> tuple[plt.Figure, pl.DataFrame]:
    """Gerçek ve tahmin edilen saatlik tepe (işlemci ve bellek panelleri)."""
    f = sekil_hazirla()
    h = havuz_bul(havuz)
    gunler = [g for b in bolumler for g in h.degerlendirme_gunleri[b]]
    y = saatlik_seri(h)[24 * gunler[0]: 24 * (gunler[-1] + 1)]
    t = (np.arange(len(y)) + 24 * gunler[0] - h.sinir["test"][0]) / 24
    veri = {"gun_test_basindan": t, "cpu_gercek": y[:, 0], "mem_gercek": y[:, 1]}
    fig, eks = plt.subplots(2, 1, figsize=(f["width_in"], f["height_in"] * 1.45), sharex=True)
    for j, (ax, k) in enumerate(zip(eks, ("cpu", "mem"))):
        ax.plot(t, y[:, j], color=CFG["renk"]["murekkep"], lw=1.4, label="Gerçek", drawstyle="steps-post")
        for m in CFG["sekil"]["tahmin_modeller"]:
            mk = m.split("_s")[0] if m.startswith("lstm") else m
            yh = tahmin_oku(h, m, gunler)
            veri[f"{k}_{m}"] = yh[:, j]
            ax.plot(t, yh[:, j], label=AD["model"][mk] + (" (tohum 0)" if m.startswith("lstm") else ""),
                    drawstyle="steps-post", **stil(mk, "model"))
        if t[0] < 0:   # bölüm sınırı; etiketler veri koordinatında, çizginin iki yanında
            ax.axvline(0, color=CFG["renk"]["murekkep"], lw=0.6, alpha=0.6)
            ax.text(-0.03, 0.97, "doğrulama", transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8)
            ax.text(0.03, 0.97, "test", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=8)
        ax.set_ylabel("Tepe işlemci (çekirdek)" if k == "cpu" else "Tepe bellek (GB)")
        virgul(ax)
    eks[0].legend(ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
    eks[1].set_xlabel("Test başlangıcından itibaren gün")
    return fig, pl.DataFrame(veri)


def f_tahmin_bitbrains() -> tuple[plt.Figure, pl.DataFrame]:
    return s_tahmin(CFG["sekil"]["tahmin_havuz_bitbrains"], ("test",))


def f_tahmin_alibaba() -> tuple[plt.Figure, pl.DataFrame]:
    return s_tahmin(CFG["sekil"]["tahmin_havuz_alibaba"], ("val", "test"))


def f_ilp_plan() -> tuple[plt.Figure, pl.DataFrame]:
    """Üç panel: tip başına sunucu sayısı (yığılmış), işlemci ve bellek kapasitesi ↔ gerçek talep ↔ (1+α)·tahmin."""
    f = sekil_hazirla()
    h = havuz_bul(CFG["sekil"]["plan_havuz"])
    m = CFG["sekil"]["plan_model"]
    alpha = yaml_oku("ilp")["model"]["alpha"]
    d = sim_oku(yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "sim", h, f"ilp_{m}").filter(pl.col("bolum") == "test")
    gunler = h.degerlendirme_gunleri["test"]
    yh = tahmin_oku(h, m, gunler)
    t = (d["slot"].to_numpy() - h.sinir["test"][0]) / 24
    tipler = [c[2:] for c in d.columns if c.startswith("x_")]
    fig, eks = plt.subplots(3, 1, figsize=(f["width_in"], f["height_in"] * 1.9), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    alt = np.zeros(len(t))
    for tip in tipler:
        x = d[f"x_{tip}"].to_numpy().astype(float)
        if x.sum() == 0:
            continue
        renk = stil(tip, "tip")["color"]
        eks[0].fill_between(t, alt, alt + x, step="post", color=renk, alpha=0.85, lw=0, label=tip)
        alt = alt + x
    eks[0].set_ylabel("Sunucu sayısı")
    eks[0].legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
    veri = {"gun_test_basindan": t, **{f"x_{tip}": d[f"x_{tip}"].to_numpy() for tip in tipler}}
    for j, (ax, k) in enumerate(zip(eks[1:], ("cpu", "mem"))):
        kap, D = d[f"kap_{k}"].to_numpy(), d[f"{k}_peak"].to_numpy()
        ax.plot(t, kap, drawstyle="steps-post", label="ILP kapasitesi", **stil("ilp", "strateji"))
        ax.plot(t, (1 + alpha) * yh[:, j], drawstyle="steps-post", color=stil("sarima", "model")["color"], lw=0.8,
                linestyle=":", label="(1 + α) × SARIMA tahmini")
        ax.plot(t, D, drawstyle="steps-post", color=CFG["renk"]["murekkep"], lw=1.0, label="Gerçek talep")
        eksik = D > kap
        ax.fill_between(t, kap, D, where=eksik, step="post", color=CFG["renk"]["murekkep"], alpha=0.25, lw=0,
                        label="Eksik kapasite")
        ax.set_ylabel("İşlemci (çekirdek)" if k == "cpu" else "Bellek (GB)")
        virgul(ax)
        veri |= {f"kap_{k}": kap, f"{k}_gercek": D, f"{k}_tahmin_alfali": (1 + alpha) * yh[:, j]}
    eks[1].legend(ncol=4, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    eks[2].set_xlabel("Test başlangıcından itibaren gün")
    return fig, pl.DataFrame(veri)


def zaman_paneli(ax: plt.Axes, kok: Path, h: Havuz, adlar: list[str], aralik: tuple[int, int], t0: int,
                 kaynak: str = "cpu") -> dict[str, np.ndarray]:
    """Stratejilerin kapasitesi ↔ gerçek talep (tek kaynak); çizilen veriyi döndürür."""
    veri: dict[str, np.ndarray] = {}
    D = None
    for ad in adlar:
        d = sim_oku(kok, h, ad).filter((pl.col("slot") >= aralik[0]) & (pl.col("slot") < aralik[1]))
        t = (d["slot"].to_numpy() - t0) / 24 if t0 else d["slot"].to_numpy()
        st, etiket = strateji_stili(ad)
        ax.plot(t, d[f"kap_{kaynak}"].to_numpy(), drawstyle="steps-post", label=etiket, **st)
        veri[f"kap_{ad}"] = d[f"kap_{kaynak}"].to_numpy()
        D, tt = d[f"{kaynak}_peak"].to_numpy(), t
    ax.plot(tt, D, drawstyle="steps-post", color=CFG["renk"]["murekkep"], lw=1.6, label="Gerçek talep", zorder=5)
    virgul(ax)
    return {"t": tt, "gercek": D, **veri}


def f_karsilastirma_zaman() -> tuple[plt.Figure, pl.DataFrame]:
    f = sekil_hazirla()
    s = CFG["sekil"]
    h = havuz_bul(s["karsilastirma_havuz"])
    g = s["karsilastirma_gunler"]
    fig, ax = plt.subplots(figsize=(f["width_in"], f["height_in"] * 1.1))
    v = zaman_paneli(ax, yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "sim", h, s["karsilastirma_stratejiler"],
                     (24 * g[0], 24 * (g[-1] + 1)), h.sinir["test"][0])
    ax.set_xlabel("Test başlangıcından itibaren gün")
    ax.set_ylabel("İşlemci (çekirdek)")
    ax.legend(ncol=3, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    return fig, pl.DataFrame(v).rename({"t": "gun_test_basindan"})


def f_maliyet_yetersizlik() -> tuple[plt.Figure, pl.DataFrame]:
    f = sekil_hazirla()
    d = _izgara().filter((pl.col("w") == 0.25) & (pl.col("L") == 2))
    egriler = [("ilp", "sarima"), ("r0_tepkisel_ilp", "-"), ("r2_hibrit_uyumlu", "sarima"), ("r1_esik", "-")]
    gr = CFG["sekil"]["egri_gruplar"]
    fig, eks = plt.subplots(2, 2, figsize=(f["width_in"], f["height_in"] * 1.7))
    veri = []
    for i, (ax, g) in enumerate(zip(eks.flat, gr)):
        for st, m in egriler:
            r = d.filter((pl.col("grup") == g) & (pl.col("strateji") == st) & (pl.col("model") == m)).sort("alpha")
            x, y = 100 * r["eksik_saat_orani_ort"].to_numpy(), r["regret_a0_yuzde_ort"].to_numpy()
            sty = stil(st, "strateji")
            etiket = AD["strateji"][st] + (", SARIMA" if m == "sarima" else "")
            ax.plot(x, y, marker="o", ms=4, mec="white", mew=0.8, label=etiket, **sty)
            if st == "ilp" and i == 0:   # α etiketleri yalnız ilk panelde (diğerlerinde çakışıyor; sıra altyazıda)
                for xi, yi, a in zip(x, y, r["alpha"].to_list()):
                    ax.annotate(f"α={sayi(a, 2)}", (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
            veri += [{"grup": g, "strateji": st, "model": m, "alpha": a, "eksik_saat_yuzde": xi, "regret_a0_yuzde": yi}
                     for a, xi, yi in zip(r["alpha"].to_list(), x, y)]
        ax.axhline(0, color=CFG["renk"]["murekkep"], lw=0.6, alpha=0.6)
        ax.set_title(AD["grup"][g], fontsize=9)   # panel adı (şekil başlığı değil)
        virgul(ax)
    for ax in eks[1]:
        ax.set_xlabel("Eksik kapasiteli saat oranı (%)")
    for ax in eks[:, 0]:
        ax.set_ylabel("Regret, α = 0 kahinine göre (%)")
    h_, l_ = eks[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return fig, pl.DataFrame(veri)


def f_sentetik_zaman() -> tuple[plt.Figure, pl.DataFrame]:
    f = sekil_hazirla()
    s = CFG["sekil"]
    h = havuz_bul(s["sentetik_havuz"])
    bas = pl.read_csv(yol(yaml_oku("synthetic")["dizin"]) / "senaryolar.csv")
    fig, eks = plt.subplots(len(s["sentetik_senaryolar"]), 1,
                            figsize=(f["width_in"], f["height_in"] * 0.85 * len(s["sentetik_senaryolar"])))
    parcalar = []
    for ax, sc in zip(np.atleast_1d(eks), s["sentetik_senaryolar"]):
        b0 = bas.filter((pl.col("senaryo") == sc) & (pl.col("veri_seti") == h.veri_seti))["baslangic_dilim"].item()
        a, b = b0 + s["sentetik_pencere_saat"][0], b0 + s["sentetik_pencere_saat"][1]
        a, b = max(a, h.sinir["test"][0]), min(b, h.sinir["test"][1])
        v = zaman_paneli(ax, yol(yaml_oku("synthetic")["dizin"]) / "sim" / sc, h, s["sentetik_stratejiler"], (a, b), 0)
        v["t"] = v["t"] - b0
        for ln in ax.get_lines():
            ln.set_xdata(np.asarray(ln.get_xdata()) - b0)
        ax.relim()
        ax.autoscale_view()
        ax.axvline(0, color=CFG["renk"]["murekkep"], lw=0.6, alpha=0.6)
        ax.text(0.01, 0.97, AD["senaryo"][sc], transform=ax.transAxes, va="top", fontsize=8)
        ax.set_ylabel("İşlemci (çekirdek)")
        parcalar.append(pl.DataFrame(v).rename({"t": "saat_bozulma_basindan"}).with_columns(pl.lit(sc).alias("senaryo")))
    np.atleast_1d(eks)[-1].set_xlabel("Bozulma başlangıcından itibaren saat")
    np.atleast_1d(eks)[0].legend(ncol=2, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    return fig, pl.concat(parcalar, how="diagonal")


# ---------------------------------------------------------------- katalog

URETICI: dict[str, Callable[[], Any]] = {
    "sunucu_tipleri": t_sunucu_tipleri, "tahmin_olcutleri": t_tahmin_olcutleri, "sarima_secim": t_sarima_secim,
    "ilp_sonuclari": t_ilp_sonuclari, "lp_yuvarlama": t_lp_yuvarlama, "alfa_duyarlilik": t_alfa_duyarlilik,
    "w_L_etkisi": t_w_L_etkisi, "karsilastirma_bitbrains": t_karsilastirma_bitbrains,
    "karsilastirma_alibaba": t_karsilastirma_alibaba, "sentetik_ozet": t_sentetik_ozet,
    "sentetik_tahmin": t_sentetik_tahmin, "tahmin_bitbrains": f_tahmin_bitbrains, "tahmin_alibaba": f_tahmin_alibaba,
    "ilp_plan": f_ilp_plan, "karsilastirma_zaman": f_karsilastirma_zaman,
    "maliyet_yetersizlik": f_maliyet_yetersizlik, "sentetik_zaman": f_sentetik_zaman}


def md_tablo(df: pl.DataFrame) -> str:
    """Önizleme için Markdown tablo."""
    satirlar = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    satirlar += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(satirlar)


def main() -> None:
    """Katalogdaki tüm şekil ve tabloları üretir."""
    cikti = yol(CFG["dizin"])
    cikti.mkdir(parents=True, exist_ok=True)
    for eski in list(cikti.glob("sekil_*")) + list(cikti.glob("tablo_*")):
        eski.unlink()   # numaralar değişebileceği için önceki üretim temizlenir
    sayac: dict[tuple[int, str], int] = {}
    katalog, md = [], ["# Rapor şekil ve tablo kataloğu (M8)", "",
                       "Otomatik üretildi: `src/rapor.py`. Altyazılar öneridir. Sıra: `configs/rapor.yaml`.", ""]
    for o in CFG["katalog"]:
        anahtar = (o["bolum"], o["tur"])
        sayac[anahtar] = sayac.get(anahtar, 0) + 1
        no = f"{o['bolum']}.{sayac[anahtar]}"
        on = "Şekil" if o["tur"] == "sekil" else "Tablo"
        dosya = f"{o['tur']}_{no.replace('.', '_')}_{o['id']}"
        sonuc = URETICI[o["id"]]()
        if o["tur"] == "sekil":
            fig, veri = sonuc
            fig.savefig(cikti / f"{dosya}.png")
            plt.close(fig)
            veri.write_csv(cikti / f"{dosya}_veri.csv")
            dosyalar = f"{dosya}.png; {dosya}_veri.csv"
            md += [f"## {on} {no} {o['baslik']}", f"*{o['alt']}* — `{dosya}.png`", "", f"![{on} {no}]({dosya}.png)", ""]
        else:
            sonuc.write_csv(cikti / f"{dosya}.csv")
            dosyalar = f"{dosya}.csv"
            md += [f"## {on} {no} {o['baslik']}", f"*{o['alt']}* — `{dosya}.csv`", "", md_tablo(sonuc), ""]
        katalog.append({"numara": f"{on} {no}", "bolum": o["bolum"], "alt_baslik": o["alt"], "baslik_onerisi": o["baslik"],
                        "not": o.get("not", ""), "dosya": dosyalar})
        if o.get("not"):
            md.insert(len(md) - 1, f"> Not: {o['not']}")
        log.info("%s %s: %s", on, no, dosya)
    pl.DataFrame(katalog).write_csv(cikti / "katalog.csv")
    (cikti / "katalog.md").write_text("\n".join(md), encoding="utf-8")
    log.info("M8: %d öğe yazıldı", len(katalog))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
