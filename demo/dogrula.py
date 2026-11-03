"""Demonun ölçtüğünü raporun benzetimiyle karşılaştırır.

Denetlenenler (aynı saatler için):
  1. Düğüm sayıları: kümede çalışan sunucu sayıları benzetimin x değerleriyle her saat aynı mı?
  2. Eksik kapasite: kümede yer bulamayan pod olan saatler, benzetimin "talep > kapasite" saatleriyle aynı mı?
  3. Maliyet: çalışma maliyeti aynı mı? Açma maliyeti neden farklı olabilir (aşağıda)?
  4. Raporun ölçütleri: aynı pencere için maliyet, eksik saat oranı ve en uzun olay.

Açma maliyeti notu: benzetim, plan ve tamponun açılışlarını ayrı sayar (plan bir m8i'yi kapatırken tampon
bir m8i açarsa iki işlem). Gerçek kümede bu ikisi birbirini götürür ve hiçbir düğüm açılmaz; bu yüzden
demonun açma maliyeti benzetiminkine eşit ya da ondan küçüktür.

Çalıştırma: .venv/bin/python -m demo.dogrula [--cikti demo/cikti]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from demo.cekirdek import PROJE_KOKU, ayar_oku, havuz_olustur
from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.ilp.model import Plan, param_oku, tipleri_oku

SIM_ADI = {"ilp": "ilp_{model}", "r2_uyumlu": "r2_hibrit_uyumlu_{model}"}


def olcutler(x: np.ndarray, y: np.ndarray, D: np.ndarray, tipler, p) -> dict[str, float]:
    o = plan_olcutleri(Plan(x, y, np.zeros_like(x), amac=float("nan")), D[:, 0], D[:, 1], tipler, p)
    gun = o["n_dilim"] / 24
    return {"maliyet_gunluk": o["maliyet"] / gun, "maliyet_calisma_gunluk": o["maliyet_calisma"] / gun,
            "maliyet_acma_gunluk": o["maliyet_acma"] / gun, "eksik_saat": o["eksik_saat"],
            "eksik_saat_orani_%": 100 * o["eksik_saat_orani"], "olay_sure_max": o["olay_sure_max"],
            "n_acma": o["n_acma"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo sonuçlarını benzetimle karşılaştırır")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()
    ayar = ayar_oku()
    cikti = PROJE_KOKU / (a.cikti or ayar["cikti"])
    meta = json.loads((cikti / "durum.json").read_text(encoding="utf-8"))
    demo = pl.read_csv(cikti / "saat.csv")
    if demo.is_empty():
        raise SystemExit("saat.csv boş: demo henüz bir saati tamamlamadı")

    tipler, h = tipleri_oku(), havuz_olustur(meta["havuz"])
    p = param_oku(tipler)
    kisa = [t.split(".")[0] for t in tipler.ad]
    sim_dosya = (yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "sim" / h.veri_seti / h.ad
                 / f"{SIM_ADI[meta['strateji']].format(model=meta['model'])}.parquet")
    sim = pl.read_parquet(sim_dosya).filter(pl.col("slot").is_in(demo["slot"].to_list())).sort("slot")
    assert sim["slot"].to_list() == demo["slot"].to_list(), "benzetim bu saatleri kapsamıyor"
    n = demo.height

    x_demo = np.stack([demo[f"x_{k}"].to_numpy() for k in kisa]).astype(float)
    x_sim = np.stack([sim[f"x_{a}"].to_numpy() for a in tipler.ad]).astype(float)
    y_demo = np.stack([demo[f"acilan_{k}"].to_numpy() for k in kisa]).astype(float)
    y_sim = np.stack([sim[f"y_{a}"].to_numpy() for a in tipler.ad]).astype(float)
    D = sim.select("cpu_peak", "mem_peak").to_numpy()
    tol = yaml_oku("ilp")["degerlendirme"]["tolerans"]

    print(f"\nDemo: {meta['havuz']} · {meta['model']} · {meta['strateji']} · {n} saat "
          f"(slot {demo['slot'][0]}–{demo['slot'][-1]})\nBenzetim: {sim_dosya.relative_to(PROJE_KOKU)}\n")

    # 1. Düğüm sayıları
    fark = np.abs(x_demo - x_sim).sum(0) > 0
    print(f"1. Düğüm sayıları   : {n - fark.sum()}/{n} saatte birebir aynı"
          + ("" if not fark.any() else f"  ← farklı saatler: {demo['slot'].to_numpy()[fark][:10].tolist()}"))

    # 2. Eksik kapasite saatleri
    kap_sim = np.stack([(tipler.C[:, None] * x_sim).sum(0), (tipler.M[:, None] * x_sim).sum(0)], axis=1)
    eksik_sim = ((D - kap_sim) > tol).any(1)
    eksik_demo = demo["eksik"].to_numpy().astype(bool)
    print(f"2. Eksik saatler    : demo {eksik_demo.sum()}, benzetim {eksik_sim.sum()} · "
          f"{(eksik_demo == eksik_sim).sum()}/{n} saatte aynı karar")
    adim = pl.read_csv(cikti / "adim.csv")
    beklenen = np.stack([np.maximum(adim["pod_cpu"].to_numpy() - adim["kap_cpu"].to_numpy() / meta["birim"]["cpu"], 0),
                         np.maximum(adim["pod_mem"].to_numpy() - adim["kap_mem"].to_numpy() / meta["birim"]["mem"], 0)], 1)
    olculen = adim.select("bekleyen_cpu", "bekleyen_mem").to_numpy()
    ayni = (np.abs(beklenen - olculen) < 1e-9).all(1)
    print(f"   adım düzeyi     : {ayni.sum()}/{len(ayni)} adımda kümede ölçülen bekleyen pod sayısı = "
          f"talep − kapasite hesabı")
    zaman_asimi = (adim["zaman_asimi"] > 0).sum()
    if zaman_asimi:
        print(f"   uyarı: {zaman_asimi} adımda küme zaman aşımında durulmadı (ölçüm o adımda güvenilmez)")

    # 3–4. Maliyet ve rapor ölçütleri
    o_demo, o_sim = olcutler(x_demo, y_demo, D, tipler, p), olcutler(x_sim, y_sim, D, tipler, p)
    print("\n3–4. Ölçütler (aynı pencere)        demo      benzetim")
    for k, ad in (("maliyet_gunluk", "Maliyet ($/gün)"), ("maliyet_calisma_gunluk", "  çalışma"),
                  ("maliyet_acma_gunluk", "  açma"), ("n_acma", "Açılan düğüm"),
                  ("eksik_saat", "Eksik saat"), ("eksik_saat_orani_%", "Eksik saat oranı (%)"),
                  ("olay_sure_max", "En uzun olay (sa)")):
        print(f"   {ad:<30}{o_demo[k]:>10.2f}{o_sim[k]:>12.2f}")
    tamam = not fark.any() and (eksik_demo == eksik_sim).all() \
        and abs(o_demo["maliyet_calisma_gunluk"] - o_sim["maliyet_calisma_gunluk"]) < 1e-6
    print(f"\nSONUÇ: {'GEÇTİ — demo benzetimi birebir üretiyor' if tamam else 'FARK VAR — yukarıya bak'}"
          f" (açma maliyeti farkı beklenen: kümede aynı saatteki kapatma + açma birbirini götürür)\n")


if __name__ == "__main__":
    main()
