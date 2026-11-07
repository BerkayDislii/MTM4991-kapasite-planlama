"""İstatistiksel testler: bulguların şans eseri olup olmadığı.

A) Tahmin karşılaştırması — Diebold–Mariano testi (Diebold ve Mariano, 1995; küçük örnek düzeltmesi Harvey,
   Leybourne ve Newbold, 1997). Test dönemi gün-ileri tahmin hataları üzerinden, kayıp = mutlak hata;
   d_t = |e_A,t| − |e_B,t|, uzun dönem varyansı Newey–West (Bartlett, gecikme h − 1 = 23; gün-ileri tahminin
   hataları 24 saate kadar ilişkili olabilir). H0: iki modelin beklenen kaybı eşit. Havuz × kaynak başına.
   Karşılaştırmalar: her model ↔ mevsimsel naif (taban) ve SARIMA ↔ son değer.

B) Planlama karşılaştırması — eşleştirilmiş gün blok bootstrap (Bitbrains, SARIMA, gerçek veri).
   Her ailenin her hedef τ için doğrulamada seçilmiş ayarı (outputs/{kayan,kantil,stokastik}/secim.csv) test
   döneminde yeniden çalıştırılır; saat saat maliyet ve eksik kapasite kaydedilir. 6 test günü iadeli yeniden
   örneklenir; aynı günler bütün havuzlara ve ailelere uygulanır (eşleştirilmiş; havuzlar arası ve aileler arası
   ilişki korunur). Her örneklemde aile başına ortalama regret, eksik saat oranı ve "hedef tutturma oranı"
   hesaplanır; referansa (sabit pay + tampon, raporun yöntemi) göre farkların %95 yüzdelik güven aralığı verilir.
   Aralık 0'ı içermiyorsa fark bu veriyle anlamlıdır. Not: 6 gün az olduğu için aralıklar geniştir; Alibaba'nın
   tek test günü bootstrap'a izin vermediğinden B bölümü yalnız Bitbrains'tir.

Çıktılar (outputs/istatistik/): dm_testleri.csv, dm_ozet.csv, bootstrap_gunluk.csv, bootstrap_fark.csv
Çalıştırma: .venv/bin/python -m src.istatistik [--yalniz dm|bootstrap] [--stokastik-yok]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import polars as pl
from scipy import stats

from src.baselines.tepkisel import esik_param_oku, esik_simule, tepkisel_ilp
from src.common.config import yaml_oku, yol
from src.forecast.kantil import konformal_kantil, sarima_aralik_kantil
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.forecast.saatlik import sarima_kokenler
from src.ilp.kayan import kayan_plan
from src.ilp.model import Plan, param_oku, tipleri_oku

log = logging.getLogger(__name__)
DM_MODELLER = ("son_deger", "mevsimsel_naif", "sarima", "lstm_s0", "lstm_s1", "lstm_s2")
B_ORNEK = 10_000
TOHUM = 2026
REFERANS = "sabit_alfa_r2u"


# ================================================================ A) Diebold–Mariano

def diebold_mariano(eA: np.ndarray, eB: np.ndarray, h: int = 24) -> tuple[float, float, float]:
    """İki taraflı DM testi (mutlak hata kaybı, HLN düzeltmeli). Döndürür: (istatistik, p, ortalama fark).

    Ortalama fark < 0: A'nın hatası daha küçük.
    """
    d = np.abs(eA) - np.abs(eB)
    T = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T                # Bartlett ağırlıkları
    if lrv <= 0:
        return float("nan"), float("nan"), float(d.mean())
    dm = d.mean() / np.sqrt(lrv / T)
    duz = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)               # Harvey–Leybourne–Newbold
    s = dm * duz
    return float(s), float(2 * stats.t.sf(abs(s), df=T - 1)), float(d.mean())


def test_tahminleri(h: Havuz, model: str) -> np.ndarray | None:
    d = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) / h.veri_seti / h.ad / model
    gunler = h.degerlendirme_gunleri["test"]
    if not all((d / f"forecast_{g}.parquet").exists() for g in gunler):
        return None
    return np.concatenate([pl.read_parquet(d / f"forecast_{g}.parquet").sort("slot")
                           .select("cpu_hat", "mem_hat").to_numpy() for g in gunler])


def dm_bolumu(havuzlar: list[Havuz]) -> pl.DataFrame:
    satir = []
    for h in havuzlar:
        y = saatlik_seri(h)
        gunler = h.degerlendirme_gunleri["test"]
        gercek = np.concatenate([y[24 * g: 24 * g + 24] for g in gunler])
        tah = {m: t for m in DM_MODELLER if (t := test_tahminleri(h, m)) is not None}
        ciftler = [(m, "mevsimsel_naif") for m in tah if m != "mevsimsel_naif"] + [("sarima", "son_deger")]
        for A, B in ciftler:
            if A not in tah or B not in tah:
                continue
            for j, kaynak in enumerate(("cpu", "mem")):
                s, p, fark = diebold_mariano(gercek[:, j] - tah[A][:, j], gercek[:, j] - tah[B][:, j])
                satir.append({"veri_seti": h.veri_seti, "havuz": h.ad, "kaynak": kaynak, "A": A, "B": B,
                              "dm": s, "p": p, "ort_fark_mae": fark, "n": len(gercek),
                              "sonuc": ("A daha iyi" if fark < 0 else "B daha iyi") if p < 0.05 else "fark yok"})
    return pl.DataFrame(satir)


def dm_ozeti(dm: pl.DataFrame) -> pl.DataFrame:
    """Veri seti × karşılaştırma × kaynak: kaç seride A anlamlı iyi / fark yok / B anlamlı iyi."""
    return (dm.with_columns(pl.col("A").str.replace(r"lstm_s\d", "lstm"))
            .group_by(["veri_seti", "A", "B", "kaynak"], maintain_order=True)
            .agg((pl.col("sonuc") == "A daha iyi").sum().alias("A_iyi"),
                 (pl.col("sonuc") == "fark yok").sum().alias("fark_yok"),
                 (pl.col("sonuc") == "B daha iyi").sum().alias("B_iyi"), pl.len().alias("n_seri"))
            .sort("veri_seti", "A", "B", "kaynak"))


# ================================================================ B) eşleştirilmiş gün bootstrap

def secilen_ayarlar(havuzlar: set[str], stokastik: bool) -> pl.DataFrame:
    """(havuz, aile, hedef) → ayar; Bitbrains, SARIMA, gerçek veri, doğrulamada seçim."""
    def oku(dosya: str) -> pl.DataFrame:
        return pl.read_csv(yol(f"outputs/{dosya}/secim.csv")).filter(
            (pl.col("veri_seti") == "bitbrains") & (pl.col("model") == "sarima") & (pl.col("senaryo") == "kontrol")
            & pl.col("havuz").is_in(list(havuzlar)))
    k = oku("kayan").select("havuz", "aile", "hedef", "alfa", "R")
    parcalar = [k.filter(pl.col("aile").is_in(["kayan_r2u", "r0"])).with_columns(pl.col("alfa").alias("ayar"))]
    q = oku("kantil").filter(pl.col("secim_yolu") == "dogrulama").select("havuz", "aile", "hedef", "ayar")
    parcalar.append(q.filter(pl.col("aile").is_in(["sabit_alfa_r2u", "sabit_alfa", "konformal_r2u",
                                                   "sarima_aralik_r2u"]))
                    .with_columns(pl.lit(None, pl.Float64).alias("alfa"), pl.lit(24).alias("R")))
    if stokastik:
        s = oku("stokastik").filter(pl.col("aile") == "stokastik_r2u").select("havuz", "aile", "hedef", "ayar")
        parcalar.append(s.with_columns(pl.lit(None, pl.Float64).alias("alfa"), pl.lit(24).alias("R")))
    cols = ["havuz", "aile", "hedef", "ayar", "R"]
    tip = {"havuz": pl.String, "aile": pl.String, "hedef": pl.Float64, "ayar": pl.Float64, "R": pl.Int64}
    return pl.concat([p.select([pl.col(c).cast(tip[c]) for c in cols]) for p in parcalar])


def saatlik_sonuc(plan: Plan, D: np.ndarray, tipler, p) -> tuple[np.ndarray, np.ndarray]:
    """Saat başına maliyet ve eksik kapasite göstergesi."""
    maliyet = (tipler.c[:, None] * plan.x + p.s[:, None] * plan.y + p.d[:, None] * plan.z).sum(0)
    kap = np.stack([(p.rho_cpu * tipler.C[:, None] * plan.x).sum(0), (p.rho_mem * tipler.M[:, None] * plan.x).sum(0)], 1)
    return maliyet, ((D - kap) > 1e-9).any(1)


def havuz_gunluk(h: Havuz, ayarlar: pl.DataFrame) -> list[dict]:
    """Bir havuzda seçilmiş her ayarı test koşusunda çalıştırır; test günlerine göre maliyet ve eksik saat."""
    from src.ilp.stokastik import gunluk_stokastik_plan
    from src.kantil_karsilastirma import PENCERE_GUN
    from src.stokastik_karsilastirma import PENCERE_GUN as S_PENCERE, S_SENARYO, TOHUM as S_TOHUM
    cfg = yaml_oku("ilp")
    cozucu, zs = cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"]
    tipler = tipleri_oku()
    y = saatlik_seri(h)
    v0, t0_, t1 = h.sinir["val"][0], h.sinir["test"][0], h.sinir["test"][1]
    D, D_once = y[v0:t1], y[v0 - 1]
    test = np.arange(v0, t1) >= t0_
    kokenler = list(range(v0 - 24 * PENCERE_GUN, t1))
    T, se = sarima_kokenler(h, y, kokenler, se_dondur=True)
    nokta, sed = dict(zip(kokenler, T)), dict(zip(kokenler, se))
    gun = list(range(v0, t1, 24))

    def gunluk(maliyet, eksik):
        g = (np.arange(v0, t1)[test] - t0_) // 24
        return (np.bincount(g, weights=maliyet[test]), np.bincount(g, weights=eksik[test].astype(float)))

    sonuc = []
    p_ref = param_oku(tipler)
    y_genis = np.vstack([y, np.repeat(y[-1:], 24, axis=0)])
    kahin = kayan_plan({t: y_genis[t: t + 24] for t in range(v0, t1)}, v0, t1, 24, tipler, p_ref, cozucu, zs).plan
    km, _ = gunluk(*saatlik_sonuc(kahin, D, tipler, p_ref))
    onbellek: dict[tuple, tuple] = {}
    for r in ayarlar.iter_rows(named=True):
        aile, ayar, R = r["aile"], float(r["ayar"]), int(r["R"])
        anahtar = (aile, ayar, R)
        if anahtar not in onbellek:
            alfa = ayar if aile in ("sabit_alfa", "sabit_alfa_r2u", "kayan_r2u", "r0") else 0.0
            p = param_oku(tipler, alpha=alfa)
            e = esik_param_oku(tipler, uyumlu=True, alpha=alfa)
            if aile == "r0":
                plan, _ = tepkisel_ilp(D, D_once, tipler, p, cozucu, zs)
            else:
                if aile.startswith("sabit_alfa"):
                    tah = {O: nokta[O] for O in gun}
                elif aile == "kayan_r2u":
                    tah = {t: nokta[t] for t in range(v0, t1)}
                elif aile == "konformal_r2u":
                    tah = konformal_kantil(nokta, y, gun, ayar, PENCERE_GUN)
                elif aile == "sarima_aralik_r2u":
                    tah = sarima_aralik_kantil(nokta, sed, gun, ayar)
                if aile == "stokastik_r2u":
                    plan, _ = gunluk_stokastik_plan(nokta, y, v0, t1, tipler, p, e, ayar, S_SENARYO, S_PENCERE,
                                                    S_TOHUM, cozucu)
                else:
                    plan = kayan_plan(tah, v0, t1, R, tipler, p, cozucu, zs).plan
                if aile.endswith("_r2u"):
                    plan = esik_simule(D, D_once, plan, tipler, p, e)
            onbellek[anahtar] = gunluk(*saatlik_sonuc(plan, D, tipler, p))
        m, e_ = onbellek[anahtar]
        for d in range(len(m)):
            sonuc.append({"havuz": h.ad, "aile": aile, "hedef": r["hedef"], "ayar": ayar, "R": R, "gun": d,
                          "maliyet": m[d], "kahin_maliyet": km[d], "eksik_saat": e_[d]})
    return sonuc


def bootstrap(gunluk: pl.DataFrame) -> pl.DataFrame:
    """Eşleştirilmiş gün blok bootstrap: aile × hedef için düzeyler ve referansa göre farkların %95 aralıkları."""
    aileler = gunluk["aile"].unique(maintain_order=True).to_list()
    hedefler = sorted(gunluk["hedef"].unique().to_list())
    havuzlar = gunluk["havuz"].unique(maintain_order=True).to_list()
    n_gun = int(gunluk["gun"].max()) + 1
    # (aile, hedef, havuz, gün) dizileri
    idx = {(a, t): (gunluk.filter((pl.col("aile") == a) & (pl.col("hedef") == t)).sort("havuz", "gun"))
           for a in aileler for t in hedefler}
    rng = np.random.default_rng(TOHUM)
    ornek = rng.integers(0, n_gun, size=(B_ORNEK, n_gun))
    ornek[0] = np.arange(n_gun)                                     # 0. örnek: gerçek test dönemi (nokta tahmin)

    def olcut(df: pl.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = len(havuzlar)
        m = df["maliyet"].to_numpy().reshape(H, n_gun)
        k = df["kahin_maliyet"].to_numpy().reshape(H, n_gun)
        e = df["eksik_saat"].to_numpy().reshape(H, n_gun)
        M, K, E = m[:, ornek].sum(2), k[:, ornek].sum(2), e[:, ornek].sum(2) / (24 * n_gun)  # (H, B)
        regret = 100 * (M - K) / K
        return regret.mean(0), 100 * E.mean(0), 100 * (E <= tau + 1e-9).mean(0)

    sonuc = []
    for t in hedefler:
        ref = olcut(idx[(REFERANS, t)], t)
        for a in aileler:
            df = idx[(a, t)]
            if df.height != len(havuzlar) * n_gun:
                continue
            o = olcut(df, t)
            satir = {"aile": a, "hedef": t}
            for ad, v, r in zip(("regret", "eksik_saat", "tutturma"), o, ref):
                satir |= {f"{ad}": float(v[0]), f"{ad}_alt": float(np.percentile(v, 2.5)),
                          f"{ad}_ust": float(np.percentile(v, 97.5))}
                if a != REFERANS:
                    f = v - r
                    satir |= {f"fark_{ad}": float(f[0]), f"fark_{ad}_alt": float(np.percentile(f, 2.5)),
                              f"fark_{ad}_ust": float(np.percentile(f, 97.5))}
            sonuc.append(satir)
    return pl.DataFrame(sonuc, infer_schema_length=None)


def main() -> None:
    ap = argparse.ArgumentParser(description="İstatistiksel testler (Diebold–Mariano ve gün bootstrap)")
    ap.add_argument("--yalniz", choices=["dm", "bootstrap"])
    ap.add_argument("--havuzlar", nargs="*")
    ap.add_argument("--stokastik-yok", action="store_true", help="stokastik ILP'yi bootstrap'a alma (yavaş)")
    ap.add_argument("--n-is", type=int, default=yaml_oku("ilp")["cozucu"]["n_jobs"])
    a = ap.parse_args()
    havuzlar = [h for h in havuzlari_listele() if not a.havuzlar or h.ad in a.havuzlar]
    d = yol("outputs/istatistik")
    d.mkdir(parents=True, exist_ok=True)
    if a.yalniz in (None, "dm"):
        dm = dm_bolumu(havuzlar)
        dm.write_csv(d / "dm_testleri.csv")
        dm_ozeti(dm).write_csv(d / "dm_ozet.csv")
        log.info("DM: %d test yazıldı", dm.height)
    if a.yalniz in (None, "bootstrap"):
        from joblib import Parallel, delayed, parallel_config
        bb = [h for h in havuzlar if h.veri_seti == "bitbrains"]
        ayar = secilen_ayarlar({h.ad for h in bb}, not a.stokastik_yok)
        t0 = time.time()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            parca = Parallel(n_jobs=a.n_is, verbose=5)(
                delayed(havuz_gunluk)(h, ayar.filter(pl.col("havuz") == h.ad)) for h in bb)
        gunluk = pl.DataFrame([r for p in parca for r in p])
        gunluk.write_csv(d / "bootstrap_gunluk.csv")
        bootstrap(gunluk).write_csv(d / "bootstrap_fark.csv")
        log.info("Bootstrap: %.1f dk", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
