"""M6: tepkisel ve hibrit ölçekleme stratejileri (MIMARI.md Bölüm 7, kullanıcı kararı 2026-10-31).

Stratejiler (hepsi saatlik dilimlerle, t diliminin kararı yalnızca t−1'e kadar gözlenen GERÇEK talebi görür):
  R0  Tepkisel-ILP: her dilimde ufuk = 1 ile (1)–(6) çözülür, girdi D_{t-1} (tahminin değerini yalıtır).
  R1  Eşik kuralı (HPA benzeri): u_{t-1} = max_r D^r_{t-1} / kap^r_{t-1}. u > üst eşik ise sunucu sayısı hedef
      kullanıma göre büyütülür; u < alt eşik ve soğuma dolmuşsa küçültülür (L_k'ye uyarak). Tek ölçek tipi.
  R2  Hibrit: M5'in tahmine dayalı ILP planı taban alınır; üzerine R1 ile aynı kuralla yönetilen tampon eklenir.
      Tampon 0'ın altına inmez, yani sunucu sayısı plandan hiç az olmaz. İki varyant: r2_hibrit (R1'in HPA
      eşikleri) ve r2_hibrit_uyumlu (ILP planının hedef kullanımına uyumlu eşikler, ilp.yaml tepkisel.hibrit_uyumlu).

R1 ve R2 aynı motoru (esik_simule) kullanır: R1 = sıfır taban plan + en az r1_min_sunucu tampon.
Simülasyon ısınma (doğrulama) bölümünün ilk diliminden soğuk başlangıçla (x = 0) başlar, ölçütler yalnız
değerlendirme (test) dilimlerinde hesaplanır (M5 ile aynı).

Çıktılar (outputs/reactive/):
  sim/<dataset>/<pool>/<strateji>[_<model>].parquet  saatlik sunucu sayıları, kapasite, gerçek talep (ILP/kahin dahil)
  havuz_ozet.csv        havuz × strateji × model (ILP ve kahin satırları M5 planlarından aynı ölçütlerle)
  tablo_ozet.csv        grup × strateji × model, havuzlar üzerinden ortalama ve std (LSTM'de havuz × tohum)
  tablo_ozet_metin.csv  "ort ± std" biçimli

Çalıştırma: .venv/bin/python -m src.baselines.tepkisel
"""
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.forecast.ozet import grup_esleme
from src.ilp.model import IlpParam, Plan, SunucuTipleri, coz, dogrula, param_oku, tipleri_oku

log = logging.getLogger(__name__)
KAHIN = "kahin"
YOK = "-"                       # Model kullanmayan stratejilerin model sütunu


@dataclass(frozen=True)
class EsikParam:
    """Eşik kuralı parametreleri (ilp.yaml tepkisel bölümü)."""
    tip: int                    # ölçek tipinin dizini
    ust: float
    hedef: float
    alt: float
    soguma: int


def esik_param_oku(tipler: SunucuTipleri, uyumlu: bool = False, alpha: float | None = None) -> EsikParam:
    """ilp.yaml tepkisel bölümünden EsikParam. uyumlu=True: hibrit_uyumlu eşikleri (hedef null → 1/(1+α)).

    alpha: hedefin bağlı olduğu α (VARSAYILAN model.alpha; M7 ızgarasında ızgara noktasının α'sı).
    α = 0 iken uyumlu hedef 1 = üst eşik olabilir, bu yüzden hedef ≤ üst kabul edilir.
    """
    cfg = yaml_oku("ilp")
    c = cfg["tepkisel"]
    assert c["olcek_tipi"] in tipler.ad, f"{c['olcek_tipi']} sunucu tipleri arasında yok"
    e = c["hibrit_uyumlu"] if uyumlu else c
    a = cfg["model"]["alpha"] if alpha is None else alpha
    hedef = 1 / (1 + a) if e["esik_hedef"] is None else float(e["esik_hedef"])
    assert 0 < e["esik_alt"] < hedef <= e["esik_ust"] <= 1, "Eşikler alt < hedef ≤ üst ≤ 1 olmalı"
    return EsikParam(tipler.ad.index(c["olcek_tipi"]), float(e["esik_ust"]), hedef,
                     float(e["esik_alt"]), int(e["soguma"]))


def kapasite(x: np.ndarray, tipler: SunucuTipleri, p: IlpParam) -> np.ndarray:
    """(K, T) sunucu sayısından (2, T) kapasite [cpu, mem] (α payı yok)."""
    return np.stack([(p.rho_cpu * tipler.C[:, None] * x).sum(0), (p.rho_mem * tipler.M[:, None] * x).sum(0)])


def esik_simule(D: np.ndarray, D_once: np.ndarray, baz: Plan, tipler: SunucuTipleri, p: IlpParam, e: EsikParam,
                min_tampon: int = 0) -> Plan:
    """Eşik kuralıyla yönetilen tamponu baz planın üzerine ekler (R1: baz = 0, R2: baz = ILP planı).

    D: (T, 2) gerçek talep; D_once: (2,) ilk dilimden önceki gözlem. t dilimindeki karar yalnızca D_{t-1} ve
    t−1'deki toplam kapasiteyi kullanır. Tampon büyüklüğü, t'deki baz kapasiteyle birlikte D_{t-1}'i hedef
    kullanımda karşılayacak en küçük ölçek tipi sayısıdır. Döndürülen plan: x = baz.x + tampon, y/z de toplanır
    (baz ve tampon açılışları ayrı sayılır, maliyet toplanabilir kalır).
    """
    T = len(D)
    k, L = e.tip, int(p.L[e.tip])
    birim = np.array([p.rho_cpu * tipler.C[k], p.rho_mem * tipler.M[k]])
    kap_baz = kapasite(baz.x, tipler, p)                     # (2, T)
    b, yb, zb = np.zeros(T), np.zeros(T), np.zeros(T)
    b_once, kap_once, son_degisim = 0.0, np.zeros(2), -10 ** 9
    for t in range(T):
        d_once = D_once if t == 0 else D[t - 1]
        # Bir önceki dilimin kullanımı (kapasite 0 ve talep > 0 ise sonsuz)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(kap_once > 0, d_once / kap_once, np.where(d_once > 0, np.inf, 0.0)).max()
        gerek = max(0.0, math.ceil(((d_once / e.hedef - kap_baz[:, t]) / birim).max() - 1e-9))
        # (5): son L−1 dilimde açılan tampon sunucuları kapatılamaz
        L_alt = yb[max(0, t - L + 1): t].sum()
        if u > e.ust:
            yeni = max(b_once, gerek)
        elif u < e.alt and t - son_degisim >= e.soguma:
            yeni = min(b_once, gerek)
        else:
            yeni = b_once
        yeni = max(yeni, L_alt, min_tampon)
        b[t], yb[t], zb[t] = yeni, max(yeni - b_once, 0), max(b_once - yeni, 0)
        if yeni != b_once:
            son_degisim = t
        b_once = yeni
        kap_once = kap_baz[:, t] + yeni * birim
    x, y, z = baz.x.copy(), baz.y.copy(), baz.z.copy()
    x[k] += b
    y[k] += yb
    z[k] += zb
    return Plan(x, y, z, amac=float("nan"))


def tepkisel_ilp(D: np.ndarray, D_once: np.ndarray, tipler: SunucuTipleri, p: IlpParam, cozucu: str,
                 zaman_siniri_s: float) -> tuple[Plan, np.ndarray]:
    """R0: her dilimde ufuk = 1 ILP; girdi D_{t-1}. Döndürür: (plan, ILP'nin gördüğü girdi (T, 2))."""
    T, K = len(D), tipler.K
    girdi = np.vstack([D_once[None, :], D[:-1]])
    Lmax = int(p.L.max())
    X, Y, Z = np.zeros((K, T)), np.zeros((K, T)), np.zeros((K, T))
    x0 = np.zeros(K)
    for t in range(T):
        y_onceki = Y[:, max(0, t - Lmax + 1): t] if t > 0 else None
        q = coz(girdi[t: t + 1, 0], girdi[t: t + 1, 1], tipler, p, x0, y_onceki, cozucu, zaman_siniri_s)
        X[:, t], Y[:, t], Z[:, t] = q.x[:, 0], q.y[:, 0], q.z[:, 0]
        x0 = q.x[:, 0]
    return Plan(X, Y, Z, amac=float("nan")), girdi


def ilp_plan_oku(h: Havuz, model: str, gunler: list[int], tipler: SunucuTipleri) -> Plan:
    """M5'in kaydettiği günlük planları (outputs/ilp/plans) art arda birleştirir."""
    d = yol(yaml_oku("ilp")["outputs"]["dizin"]) / "plans" / h.veri_seti / h.ad / model
    df = pl.concat([pl.read_parquet(d / f"plan_{g}.parquet") for g in gunler]).sort("slot")
    assert df["slot"].to_list() == list(range(24 * gunler[0], 24 * (gunler[-1] + 1))), f"{d}: dilimler eksik"
    dizi = {v: np.stack([df[f"{v}_{a}"].to_numpy() for a in tipler.ad]).astype(float) for v in ("x", "y", "z")}
    return Plan(dizi["x"], dizi["y"], dizi["z"], amac=float("nan"))


def sim_yaz(d: Path, ad: str, plan: Plan, slot: np.ndarray, bolum: np.ndarray, D: np.ndarray,
            tipler: SunucuTipleri, p: IlpParam) -> None:
    """Saatlik simülasyonu parquet olarak yazar."""
    d.mkdir(parents=True, exist_ok=True)
    kap = kapasite(plan.x, tipler, p)
    sut = {"slot": slot.astype(np.int64), "bolum": bolum}
    for k, a in enumerate(tipler.ad):
        for v in ("x", "y", "z"):
            sut[f"{v}_{a}"] = getattr(plan, v)[k].astype(np.int64)
    sut |= {"kap_cpu": kap[0], "kap_mem": kap[1], "cpu_peak": D[:, 0], "mem_peak": D[:, 1]}
    pl.DataFrame(sut).write_parquet(d / f"{ad}.parquet")


def simulasyon_gunleri(h: Havuz) -> list[int]:
    """Isınma + değerlendirme günleri (simülasyon aralığı bölmelerle birebir örtüşmeli)."""
    kp = yaml_oku("ilp")["kayan_pencere"]
    isinma, deg = kp["isinma_bolumu"], kp["degerlendirme_bolumu"]
    gunler = h.degerlendirme_gunleri[isinma] + h.degerlendirme_gunleri[deg]
    assert 24 * gunler[0] == h.sinir[isinma][0] and 24 * (gunler[-1] + 1) == h.sinir[deg][1], \
        f"{h.ad}: simülasyon aralığı bölmelerle uyuşmuyor"
    return gunler


def havuz_deneyi(h: Havuz, y: np.ndarray, tipler: SunucuTipleri, p: IlpParam, e: EsikParam, e_uyumlu: EsikParam,
                 ilp_planlari: dict[str, Plan], sim_dizin: Path | None = None,
                 ek: dict | None = None) -> list[dict]:
    """Çekirdek: R0, R1, R2 (iki varyant, her model), ILP ve kahin satırları (test dilimleri).

    y: (n_dilim, 2) gerçek talep; ilp_planlari: model → ısınma+değerlendirme birleşik ILP planı (kahin dahil).
    sim_dizin verilirse tepkisel simülasyonlar yazılır. ek: her satıra eklenecek sütunlar (ör. ızgara noktası).
    M6 ana koşusu, M7 ızgarası ve sentetik senaryolar bu fonksiyonu kullanır.
    """
    cfg = yaml_oku("ilp")
    tc = cfg["tepkisel"]
    tol = cfg["degerlendirme"]["tolerans"]
    deg = cfg["kayan_pencere"]["degerlendirme_bolumu"]
    gunler = simulasyon_gunleri(h)
    bas, son = 24 * gunler[0], 24 * (gunler[-1] + 1)
    D, D_once = y[bas:son], y[bas - 1]
    slot = np.arange(bas, son)
    test = slot >= h.sinir[deg][0]
    bolum = np.where(test, deg, cfg["kayan_pencere"]["isinma_bolumu"])
    sifir = Plan(*(np.zeros((tipler.K, len(D))) for _ in range(3)), amac=0.0)

    planlar: list[tuple[str, str, Plan]] = []
    r0, girdi = tepkisel_ilp(D, D_once, tipler, p, cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"])
    # Bölüm 6.5: R0'ın her dilimi ILP'dir → gördüğü girdiye göre (2)–(6) bağımsız denetlenir
    assert not dogrula(r0, girdi[:, 0], girdi[:, 1], tipler, p), f"{h.ad}: R0 doğrulayıcı ihlali"
    planlar.append(("r0_tepkisel_ilp", YOK, r0))
    planlar.append(("r1_esik", YOK, esik_simule(D, D_once, sifir, tipler, p, e, int(tc["r1_min_sunucu"]))))
    for m in tc["hibrit_modeller"]:
        baz = ilp_planlari[m]
        assert baz.x.shape[1] == len(D), f"{h.ad}/{m}: ILP planı simülasyon aralığını kapsamıyor"
        planlar.append(("r2_hibrit", m, esik_simule(D, D_once, baz, tipler, p, e)))
        planlar.append(("r2_hibrit_uyumlu", m, esik_simule(D, D_once, baz, tipler, p, e_uyumlu)))
        planlar.append(("ilp", m, baz))
    planlar.append((KAHIN, KAHIN, ilp_planlari[KAHIN]))

    satirlar = []
    sifir_D = np.zeros(len(D))
    for strateji, model, plan in planlar:
        # (4)–(6) ve (5) devamlılığı tüm simülasyon boyunca (kapasite denetimi talep 0 ile etkisiz)
        ihlal = dogrula(plan, sifir_D, sifir_D, tipler, p)
        assert not ihlal, f"{h.ad}/{strateji}/{model}: {ihlal[:3]}"
        if sim_dizin is not None:   # tüm stratejiler aynı biçimde (rapor şekilleri bu dosyaları okur)
            ad = strateji if model in (YOK, strateji) else f"{strateji}_{model}"
            sim_yaz(sim_dizin, ad, plan, slot, bolum, D, tipler, p)
        pt = Plan(plan.x[:, test], plan.y[:, test], plan.z[:, test], amac=float("nan"))
        n_gun = int(test.sum() // 24)
        o = plan_olcutleri(pt, D[test, 0], D[test, 1], tipler, p, tol)
        satirlar.append({**(ek or {}), "veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "strateji": strateji,
                         "model": model, "n_gun": n_gun, **o, "maliyet_gunluk": o["maliyet"] / n_gun})
    return satirlar


def havuz_coz(h: Havuz) -> list[dict]:
    """M6 ana koşusu: merkez parametreler, ILP planları M5 çıktısından (outputs/ilp/plans)."""
    cfg = yaml_oku("ilp")
    tipler = tipleri_oku()
    gunler = simulasyon_gunleri(h)
    planlar = {m: ilp_plan_oku(h, m, gunler, tipler) for m in [*cfg["tepkisel"]["hibrit_modeller"], KAHIN]}
    satirlar = havuz_deneyi(h, saatlik_seri(h), tipler, param_oku(tipler), esik_param_oku(tipler),
                            esik_param_oku(tipler, uyumlu=True), planlar,
                            yol(cfg["tepkisel"]["dizin"]) / "sim" / h.veri_seti / h.ad)
    log.info("%s: %d strateji satırı", h.ad, len(satirlar))
    return satirlar


def regret_ekle(df: pl.DataFrame, anahtar: list[str]) -> pl.DataFrame:
    """Regret (Bölüm 7, ölçüt 5): aynı anahtar (havuz [+ ızgara noktası/senaryo]) için kahin maliyetine göre."""
    kahin = df.filter(pl.col("strateji") == KAHIN).select(*anahtar, pl.col("maliyet").alias("_kahin"))
    return df.join(kahin, on=anahtar).with_columns(
        (pl.col("maliyet") - pl.col("_kahin")).alias("regret"),
        (100 * (pl.col("maliyet") - pl.col("_kahin")) / pl.col("_kahin")).alias("regret_yuzde")).drop("_kahin")


def grup_ekle(df: pl.DataFrame) -> pl.DataFrame:
    """Özet grubu (Bitbrains: forecast.yaml summary_groups, Alibaba: alibaba_r<r>) ve LSTM tohumlarını birleştirir."""
    out = df.join(grup_esleme(), on="havuz", how="left").with_columns(
        pl.when(pl.col("veri_seti") == "alibaba").then(pl.format("alibaba_r{}", pl.col("r").cast(pl.Int64)))
          .otherwise(pl.col("grup")).alias("grup"),
        pl.col("model").str.replace(r"^lstm_s\d+$", "lstm").alias("model"))
    assert out["grup"].null_count() == 0, "Gruplanmamış havuz var"
    return out


OZET_OLCUTLER: list[str] = ["maliyet", "maliyet_gunluk", "regret", "regret_yuzde", "n_acma", "ort_sunucu",
                            "bos_orani_cpu", "bos_orani_mem", "eksik_saat", "eksik_saat_orani",
                            "karsilanmayan_cpu", "karsilanmayan_mem", "karsilanmayan_cpu_yuzde",
                            "karsilanmayan_mem_yuzde", "n_olay", "olay_sure_ort", "olay_sure_max"]
STRATEJI_SIRASI: list[str] = ["ilp", "r2_hibrit", "r2_hibrit_uyumlu", "r0_tepkisel_ilp", "r1_esik", KAHIN]
MODEL_SIRASI: list[str] = ["son_deger", "mevsimsel_naif", "sarima", "lstm", YOK, KAHIN]


def ozetle(havuz: pl.DataFrame, ek_anahtar: list[str] | None = None,
           ek_olcut: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Grup × strateji × model [× ek anahtar] ortalama/std tablosu ve metin biçimi (LSTM tohumları tek model)."""
    ek = ek_anahtar or []
    olcutler = OZET_OLCUTLER + (ek_olcut or [])
    df = grup_ekle(havuz)
    ozet = (df.group_by(*ek, "grup", "strateji", "model")
              .agg([pl.len().alias("n_satir"), pl.col("havuz").n_unique().alias("n_havuz"), pl.col("n_gun").first()]
                   + [pl.col(c).mean().alias(f"{c}_ort") for c in olcutler]
                   + [pl.col(c).std().alias(f"{c}_std") for c in olcutler])
              .with_columns(pl.col("strateji").replace_strict({s: i for i, s in enumerate(STRATEJI_SIRASI)}).alias("_s"),
                            pl.col("model").replace_strict({m: i for i, m in enumerate(MODEL_SIRASI)}).alias("_m"))
              .sort(*ek, "grup", "_s", "_m").drop("_s", "_m"))

    def bicim(c: str, basamak: int) -> pl.Expr:
        return (pl.when(pl.col(f"{c}_std").is_null() | (pl.col("n_satir") == 1))
                  .then(pl.col(f"{c}_ort").round(basamak).cast(pl.String))
                  .otherwise(pl.format("{} ± {}", pl.col(f"{c}_ort").round(basamak), pl.col(f"{c}_std").round(basamak)))
                  .alias(c))
    metin = ozet.select(*ek, "grup", "strateji", "model", "n_havuz", "n_satir", "n_gun",
                        bicim("maliyet_gunluk", 1), bicim("regret_yuzde", 2), bicim("bos_orani_cpu", 3),
                        bicim("bos_orani_mem", 3), bicim("eksik_saat_orani", 3), bicim("karsilanmayan_cpu_yuzde", 3),
                        bicim("karsilanmayan_mem_yuzde", 3), bicim("n_olay", 1), bicim("olay_sure_ort", 2),
                        bicim("olay_sure_max", 1), bicim("n_acma", 1), *[bicim(c, 2) for c in ek_olcut or []])
    return ozet, metin


def main() -> None:
    """Tüm havuzlar için M6 koşusu ve tablolar."""
    cfg = yaml_oku("ilp")
    cikti = yol(cfg["tepkisel"]["dizin"])
    cikti.mkdir(parents=True, exist_ok=True)
    sonuc = Parallel(n_jobs=cfg["cozucu"]["n_jobs"])(delayed(havuz_coz)(h) for h in havuzlari_listele())
    havuz = pl.DataFrame([s for satirlar in sonuc for s in satirlar], infer_schema_length=None)
    # Tutarlılık: ILP/kahin satırları M5'in havuz_ozet.csv maliyetleriyle aynı olmalı (aynı planlar, aynı ölçüt)
    m5 = pl.read_csv(yol(cfg["outputs"]["dizin"]) / "havuz_ozet.csv").select("havuz", "model", pl.col("maliyet").alias("_m5"))
    kar = havuz.filter(pl.col("strateji").is_in(["ilp", KAHIN])).join(m5, on=["havuz", "model"], how="left")
    assert kar["_m5"].null_count() == 0 and np.allclose(kar["maliyet"], kar["_m5"], rtol=1e-9), \
        "ILP maliyetleri M5 havuz_ozet.csv ile uyuşmuyor"
    havuz = regret_ekle(havuz, ["havuz"])
    havuz.write_csv(cikti / "havuz_ozet.csv")
    ozet, metin = ozetle(havuz)
    ozet.write_csv(cikti / "tablo_ozet.csv")
    metin.write_csv(cikti / "tablo_ozet_metin.csv")
    log.info("M6 tamam: %d havuz × strateji satırı", havuz.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
