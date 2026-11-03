"""Demonun ölçtüğünü raporun benzetimiyle karşılaştırır (her strateji grubu için ayrı).

Denetlenenler (aynı saatler için):
  1. Düğüm sayıları: kümede çalışan sunucu sayıları benzetimin x değerleriyle her saat aynı mı?
  2. Eksik kapasite: kümede yer bulamayan pod olan saatler, benzetimin "talep > kapasite" saatleriyle aynı mı?
     Ayrıca her 5 dk adımda ölçülen bekleyen pod sayısı = talep − kapasite hesabı mı?
  3. Maliyet ve raporun ölçütleri (maliyet, eksik saat oranı, en uzun olay).

Panelden senaryo (darbe, basamak, düşüş) verildiyse talep artık raporun gerçek verisi değildir; o andan
sonraki saatlerin karşılaştırılacağı bir benzetim yoktur. Bu saatler denetimden çıkarılır ve iki stratejinin
senaryo altındaki sonuçları yalnızca yan yana raporlanır.

Açma maliyeti notu: benzetim, plan ve tamponun açılışlarını ayrı sayar. Gerçek kümede aynı saatteki bir
kapatma ve açma birbirini götürür; bu yüzden demonun açma maliyeti benzetiminkine eşit ya da küçüktür.

Çalıştırma: .venv/bin/python -m demo.dogrula [--cikti demo/cikti]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from demo.cekirdek import ADIM_SAAT, PROJE_KOKU, ayar_oku, havuz_olustur
from src.common.config import yaml_oku, yol
from src.evaluate import plan_olcutleri
from src.ilp.model import Plan, param_oku, tipleri_oku

SIM_ADI = {"ilp": "ilp_{model}", "r2_uyumlu": "r2_hibrit_uyumlu_{model}"}
AD = {"ilp": "Yalnız ILP", "r2_uyumlu": "ILP + tampon (R2)"}
OLCUT_SATIRLARI = (("maliyet_gunluk", "Maliyet ($/gün)"), ("maliyet_calisma_gunluk", "  çalışma"),
                   ("maliyet_acma_gunluk", "  açma"), ("n_acma", "Açılan düğüm"), ("eksik_saat", "Eksik saat"),
                   ("eksik_saat_orani_%", "Eksik saat oranı (%)"), ("olay_sure_max", "En uzun olay (sa)"))


def olcutler(x: np.ndarray, y: np.ndarray, D: np.ndarray, tipler, p) -> dict[str, float]:
    o = plan_olcutleri(Plan(x, y, np.zeros_like(x), amac=float("nan")), D[:, 0], D[:, 1], tipler, p)
    gun = o["n_dilim"] / 24
    return {"maliyet_gunluk": o["maliyet"] / gun, "maliyet_calisma_gunluk": o["maliyet_calisma"] / gun,
            "maliyet_acma_gunluk": o["maliyet_acma"] / gun, "eksik_saat": o["eksik_saat"],
            "eksik_saat_orani_%": 100 * o["eksik_saat_orani"], "olay_sure_max": o["olay_sure_max"],
            "n_acma": o["n_acma"]}


def demo_dizileri(saat: pl.DataFrame, kisa: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([saat[f"x_{k}"].to_numpy() for k in kisa]).astype(float)
    y = np.stack([saat[f"acilan_{k}"].to_numpy() for k in kisa]).astype(float)
    D = saat.select("tepe_cpu", "tepe_mem").to_numpy()
    return x, y, D


def grup_dogrula(meta: dict, kol: dict, cikti, tipler, p, ilk_senaryo_slot: int | None) -> bool | None:
    """Bir grubun senaryodan önceki saatlerini benzetimle karşılaştırır. None: karşılaştırılacak saat yok."""
    h = havuz_olustur(meta["havuz"])
    kisa = [t.split(".")[0] for t in tipler.ad]
    saat = pl.read_csv(cikti / kol["grup"] / "saat.csv")
    adim = pl.read_csv(cikti / kol["grup"] / "adim.csv")
    if ilk_senaryo_slot is not None:
        saat = saat.filter(pl.col("slot") < ilk_senaryo_slot)
        adim = adim.filter(pl.col("slot") < ilk_senaryo_slot)
    print(f"\n━━ {AD[kol['strateji']]} (grup '{kol['grup']}') ━━")
    if saat.is_empty():
        print("   Senaryodan önce tamamlanmış saat yok; benzetimle karşılaştırılacak bir şey yok.")
        return None
    sim_dosya = (yol(yaml_oku("ilp")["tepkisel"]["dizin"]) / "sim" / h.veri_seti / h.ad
                 / f"{SIM_ADI[kol['strateji']].format(model=meta['model'])}.parquet")
    sim = pl.read_parquet(sim_dosya).filter(pl.col("slot").is_in(saat["slot"].to_list())).sort("slot")
    assert sim["slot"].to_list() == saat["slot"].to_list(), "benzetim bu saatleri kapsamıyor"
    n = saat.height
    print(f"   {n} saat (slot {saat['slot'][0]}–{saat['slot'][-1]}) · benzetim: {sim_dosya.name}")

    x_demo, y_demo, _ = demo_dizileri(saat, kisa)
    x_sim = np.stack([sim[f"x_{a}"].to_numpy() for a in tipler.ad]).astype(float)
    y_sim = np.stack([sim[f"y_{a}"].to_numpy() for a in tipler.ad]).astype(float)
    D = sim.select("cpu_peak", "mem_peak").to_numpy()
    tol = yaml_oku("ilp")["degerlendirme"]["tolerans"]

    fark = np.abs(x_demo - x_sim).sum(0) > 0
    print(f"   1. Düğüm sayıları : {n - fark.sum()}/{n} saatte birebir aynı"
          + ("" if not fark.any() else f"  ← farklı: {saat['slot'].to_numpy()[fark][:10].tolist()}"))
    kap_sim = np.stack([(tipler.C[:, None] * x_sim).sum(0), (tipler.M[:, None] * x_sim).sum(0)], axis=1)
    eksik_sim = ((D - kap_sim) > tol).any(1)
    eksik_demo = saat["eksik"].to_numpy().astype(bool)
    print(f"   2. Eksik saatler  : demo {eksik_demo.sum()}, benzetim {eksik_sim.sum()} · "
          f"{(eksik_demo == eksik_sim).sum()}/{n} saatte aynı karar")
    beklenen = np.stack([
        np.maximum(adim["pod_cpu"].to_numpy() - adim["kap_cpu"].to_numpy() / meta["birim"]["cpu"], 0),
        np.maximum(adim["pod_mem"].to_numpy() - adim["kap_mem"].to_numpy() / meta["birim"]["mem"], 0)], 1)
    ayni = (np.abs(beklenen - adim.select("bekleyen_cpu", "bekleyen_mem").to_numpy()) < 1e-9).all(1)
    print(f"      adım düzeyi    : {ayni.sum()}/{len(ayni)} adımda ölçülen bekleyen pod = talep − kapasite hesabı")
    if (adim["zaman_asimi"] > 0).any():
        print(f"      uyarı: {(adim['zaman_asimi'] > 0).sum()} adımda küme zaman aşımında durulmadı")

    o_demo, o_sim = olcutler(x_demo, y_demo, D, tipler, p), olcutler(x_sim, y_sim, D, tipler, p)
    print("   3. Ölçütler                      demo    benzetim")
    for k, ad in OLCUT_SATIRLARI:
        print(f"      {ad:<27}{o_demo[k]:>10.2f}{o_sim[k]:>12.2f}")
    tamam = bool(not fark.any() and (eksik_demo == eksik_sim).all() and ayni.all()
                 and abs(o_demo["maliyet_calisma_gunluk"] - o_sim["maliyet_calisma_gunluk"]) < 1e-6)
    print(f"   SONUÇ: {'GEÇTİ — benzetimi birebir üretiyor' if tamam else 'FARK VAR — yukarıya bak'}")
    return tamam


def senaryo_karsilastir(meta: dict, cikti, tipler, p, ilk_senaryo_slot: int) -> None:
    """Senaryodan sonraki saatlerde iki grubun sonuçlarını yan yana yazar (referans benzetim yok)."""
    kisa = [t.split(".")[0] for t in tipler.ad]
    sonuc = {}
    for kol in meta["kollar"]:
        saat = pl.read_csv(cikti / kol["grup"] / "saat.csv").filter(pl.col("slot") >= ilk_senaryo_slot)
        if saat.is_empty():
            return
        x, y, D = demo_dizileri(saat, kisa)
        sonuc[kol["strateji"]] = olcutler(x, y, D, tipler, p) | {"n": saat.height}
    gecmis = ", ".join(f"{g['ad']} (slot {g['bas'] // ADIM_SAAT})" for g in meta["senaryo"]["gecmis"])
    print(f"\n━━ Senaryo sonrası (referans benzetim yok): {gecmis} ━━")
    basliklar = list(sonuc)
    print("   " + " " * 27 + "".join(f"{AD[s]:>20}" for s in basliklar))
    for k, ad in OLCUT_SATIRLARI:
        print(f"   {ad:<27}" + "".join(f"{sonuc[s][k]:>20.2f}" for s in basliklar))


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo sonuçlarını benzetimle karşılaştırır")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()
    ayar = ayar_oku()
    cikti = PROJE_KOKU / (a.cikti or ayar["cikti"])
    meta = json.loads((cikti / "durum.json").read_text(encoding="utf-8"))
    tipler = tipleri_oku()
    p = param_oku(tipler)
    gecmis = meta.get("senaryo", {}).get("gecmis", [])
    ilk = min(g["bas"] for g in gecmis) // ADIM_SAAT if gecmis else None

    print(f"\nDemo: {meta['havuz']} · {meta['model']} · gruplar: {', '.join(k['grup'] for k in meta['kollar'])}")
    if ilk is not None:
        print(f"Senaryo slot {ilk}'de başladı: o saat ve sonrası benzetimle karşılaştırılmaz.")
    sonuclar = [grup_dogrula(meta, kol, cikti, tipler, p, ilk) for kol in meta["kollar"]]
    if ilk is not None:
        senaryo_karsilastir(meta, cikti, tipler, p, ilk)
    gecerli = [s for s in sonuclar if s is not None]
    print(f"\nGENEL SONUÇ: {'GEÇTİ' if gecerli and all(gecerli) else 'FARK VAR' if gecerli else 'KARŞILAŞTIRMA YOK'}\n")


if __name__ == "__main__":
    main()
