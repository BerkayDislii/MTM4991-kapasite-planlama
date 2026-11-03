"""Kayan ufuk deneyi (pilot): günlük ILP ile saatlik/çok saatlik yeniden planlamanın karşılaştırması.

Her (senaryo, havuz, model, R) için:
  kayan_ilp      kayan ufuklu ILP (R saatte bir yeniden çözüm; R = 24 → raporun günlük ILP'si)
  kayan_r2u      aynı plan + plana uyumlu eşikli tepkisel tampon (src/baselines/tepkisel.py, R2 uyumlu)
ve referanslar: kahin (gerçek talebi bilen günlük ILP), r0 (tepkisel ILP, ufuk 1).
Senaryolar: kontrol (gerçek veri) ve raporun sentetik çarpan senaryoları (configs/synthetic.yaml; başlangıç
saatleri raporla aynı tohumla seçilir). Ölçütler yalnız test saatlerinde, gerçek (ya da bozulmuş) talep üzerinden.

Denetimler: gün başı tahminleri raporla aynı (göreli fark < 1e-6); kontrol senaryosunda R = 24 planı raporun
kayıtlı ILP planlarıyla aynı.

Çıktılar (outputs/kayan/): tahmin/…parquet, sim/<senaryo>/<havuz>/<strateji>_<model>_R<R>.parquet, havuz_ozet.csv
Çalıştırma: .venv/bin/python -m src.kayan_deney [--havuz bitbrains_fastStorage] [--modeller sarima son_deger ...]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import polars as pl

from src.baselines.tepkisel import esik_param_oku, esik_simule, tepkisel_ilp
from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.forecast.ortak import Havuz, saatlik_seri
from src.forecast.saatlik import UFUK, gun_basi_farki, kokenli_tahmin, tahmin_oku, tahmin_yaz
from src.forecast.veri import bes_dk_seri
from src.ilp.kayan import kayan_plan
from src.ilp.model import Plan, param_oku, tipleri_oku
from src.sentetik import baslangic_sec, bozulmus_seri

log = logging.getLogger(__name__)
SENARYOLAR = ("kontrol", "basamak", "darbe", "dusus")
R_IZGARA = (1, 3, 6, 12, 24)


def havuz_bul(ad: str) -> Havuz:
    """Bitbrains havuz tanımı (src.forecast.ortak.havuzlari_listele ile aynı bölme)."""
    b = yaml_oku("preprocess")["bitbrains"]["split_days"]
    t, v, s = 24 * b["train"], 24 * (b["train"] + b["val"]), 24 * (b["train"] + b["val"] + b["test"])
    lb = yaml_oku("forecast")["datasets"]["bitbrains"]["lookback_hours"]
    return Havuz(ad, "bitbrains", {"train": (0, t), "val": (t, v), "test": (v, s)}, lb, ad)


def senaryo_serisi(h: Havuz, senaryo: str) -> tuple[np.ndarray, int | None]:
    """Senaryonun saatlik tepe serisi ve başlangıç saati (raporun sentetik deneyiyle aynı tohum ve sıra)."""
    cfg = yaml_oku("synthetic")
    adlar = list(cfg["senaryolar"])
    i, j = adlar.index(senaryo), 0                                   # j = 0: bitbrains (VERI_SETLERI sırası)
    sc = cfg["senaryolar"][senaryo]
    bas = baslangic_sec(sc, h.sinir["test"], np.random.default_rng([cfg["tohum"], i, j]))
    if sc["tur"] == "yok":
        return saatlik_seri(h), None
    k = yaml_oku("preprocess")["grid"]["steps_per_slot"]
    _, y = bozulmus_seri(bes_dk_seri(h), sc, h.sinir["test"], bas, k)
    return y, bas


def olcut_satiri(plan: Plan, D: np.ndarray, test: np.ndarray, tipler, p) -> dict[str, float]:
    alt = Plan(plan.x[:, test], plan.y[:, test], plan.z[:, test], amac=float("nan"))
    o = plan_olcutleri(alt, D[test, 0], D[test, 1], tipler, p)
    gun = o["n_dilim"] / 24
    return {"maliyet_gunluk": o["maliyet"] / gun, "eksik_saat_orani": o["eksik_saat_orani"],
            "eksik_saat": o["eksik_saat"], "olay_sure_max": o["olay_sure_max"], "n_acma": o["n_acma"],
            "karsilanmayan_cpu_yuzde": o["karsilanmayan_cpu_yuzde"],
            "karsilanmayan_mem_yuzde": o["karsilanmayan_mem_yuzde"]}


def sim_yaz(d, ad: str, plan: Plan, slot: np.ndarray, D: np.ndarray, tipler) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sut = {"slot": slot.astype(np.int64)}
    for k, a in enumerate(tipler.ad):
        for v in ("x", "y", "z"):
            sut[f"{v}_{a}"] = getattr(plan, v)[k].astype(np.int64)
    sut |= {"kap_cpu": (tipler.C[:, None] * plan.x).sum(0), "kap_mem": (tipler.M[:, None] * plan.x).sum(0),
            "cpu_peak": D[:, 0], "mem_peak": D[:, 1]}
    pl.DataFrame(sut).write_parquet(d / f"{ad}.parquet")


def havuz_senaryo(h: Havuz, senaryo: str, modeller: list[str], R_listesi: tuple[int, ...]) -> list[dict]:
    cfg = yaml_oku("ilp")
    cozucu, zs = cfg["cozucu"]["ad"], cfg["cozucu"]["zaman_siniri_s"]
    tipler = tipleri_oku()
    p = param_oku(tipler)
    e = esik_param_oku(tipler, uyumlu=True)
    kok = yol("outputs/kayan")
    y, sc_bas = senaryo_serisi(h, senaryo)
    bas, son = h.sinir["val"][0], h.sinir["test"][1]
    slot = np.arange(bas, son)
    test = slot >= h.sinir["test"][0]
    D, D_once = y[bas:son], y[bas - 1]
    sim_d = kok / "sim" / senaryo / h.ad
    ortak = {"senaryo": senaryo, "senaryo_bas": sc_bas, "havuz": h.ad}

    # Kahin: gerçek (bozulmuş) talebi bilen günlük ILP — regret referansı
    y_genis = np.vstack([y, np.repeat(y[-1:], UFUK, axis=0)])        # son kökenlerin ufku için dolgu
    kahin = kayan_plan({t: y_genis[t: t + UFUK] for t in range(bas, son)}, bas, son, 24, tipler, p, cozucu, zs).plan
    o_kahin = olcut_satiri(kahin, D, test, tipler, p)
    satirlar = [ortak | {"strateji": "kahin", "model": "-", "R": 24} | o_kahin]
    sim_yaz(sim_d, "kahin", kahin, slot, D, tipler)

    r0, _ = tepkisel_ilp(D, D_once, tipler, p, cozucu, zs)
    satirlar.append(ortak | {"strateji": "r0", "model": "-", "R": 1} | olcut_satiri(r0, D, test, tipler, p))

    for m in modeller:
        t0 = time.time()
        kokenler = list(range(bas, son))
        T = kokenli_tahmin(h, m, y, kokenler)
        tahmin_yaz(kok / "tahmin" / senaryo / h.veri_seti / h.ad / f"{m}.parquet", kokenler, T)
        if senaryo == "kontrol":
            fark = gun_basi_farki(h, m, kokenler, T)
            assert fark < 1e-6, f"{h.ad}/{m}: gün başı tahminleri raporla aynı değil (göreli fark {fark:.2e})"
        tahmin = dict(zip(kokenler, T))
        log.info("%s/%s/%s: tahminler %.1f s", senaryo, h.ad, m, time.time() - t0)
        for R in R_listesi:
            s = kayan_plan(tahmin, bas, son, R, tipler, p, cozucu, zs)
            r2 = esik_simule(D, D_once, s.plan, tipler, p, e)
            for ad, plan in (("kayan_ilp", s.plan), ("kayan_r2u", r2)):
                satirlar.append(ortak | {"strateji": ad, "model": m, "R": R, "n_cozum": s.n_cozum,
                                         "cozum_ms_ort": float(np.mean(s.sure_ms)),
                                         "cozum_ms_max": float(np.max(s.sure_ms))}
                                | olcut_satiri(plan, D, test, tipler, p))
                sim_yaz(sim_d, f"{ad}_{m}_R{R}", plan, slot, D, tipler)
    for r in satirlar:
        r["regret_yuzde"] = 100 * (r["maliyet_gunluk"] - o_kahin["maliyet_gunluk"]) / o_kahin["maliyet_gunluk"]
    return satirlar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--havuz", default="bitbrains_fastStorage")
    ap.add_argument("--modeller", nargs="+", default=["sarima", "son_deger", "mevsimsel_naif"])
    ap.add_argument("--senaryolar", nargs="+", default=list(SENARYOLAR))
    a = ap.parse_args()
    h = havuz_bul(a.havuz)
    satirlar = []
    for sc in a.senaryolar:
        satirlar += havuz_senaryo(h, sc, a.modeller, R_IZGARA)
    df = pl.DataFrame(satirlar, infer_schema_length=None)
    hedef = yol("outputs/kayan")
    hedef.mkdir(parents=True, exist_ok=True)
    df.write_csv(hedef / "havuz_ozet.csv")
    log.info("Yazıldı: %s (%d satır)", hedef / "havuz_ozet.csv", df.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
