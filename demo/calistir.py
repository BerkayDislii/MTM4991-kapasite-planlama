"""Demoyu çalıştırır: gerçek talebi kümeye oynatır, ILP planını ve tamponu saat saat uygular, ölçer.

Çalıştırma (proje kökünden):
    .venv/bin/python -m demo.calistir                 # ayar.yaml ile canlı küme
    .venv/bin/python -m demo.calistir --kuru --hizli  # Kubernetes olmadan, beklemeden (hesap denetimi)
    .venv/bin/python -m demo.calistir --gun-sayisi 6 --hizli   # tüm test dönemi, doğrulama için

Döngü (her benzetim saati t):
  1. Gün başıysa: o günün tahmini → ILP (src/ilp/model.py) → 24 saatlik plan
  2. R2 tamponu: bir önceki saatin gözlenen tepe talebine göre m8i tamponu (src/baselines/tepkisel.py kuralı)
  3. Küme: düğüm sayılarını plan + tampona getir
  4. 12 adım (5 dk): gerçek talebi pod sayısına çevir → kümeye ver → yerleşmeyi bekle → bekleyen pod'ları ölç
Çıktılar: <cikti>/adim.csv (her 5 dk), <cikti>/saat.csv (her saat), <cikti>/durum.json (panel için)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from demo.cekirdek import ADIM_SAAT, PROJE_KOKU, Hazirlik, ayar_oku, hazirla, kapasite
from demo.kume import KubeKume, SahteKume

log = logging.getLogger("demo")


class Kayit:
    """CSV'lere satır satır yazar (panel dosyayı canlı okuyabilsin diye her satırda diske boşaltır)."""

    def __init__(self, dizin: Path, tip_ad: tuple[str, ...]):
        dizin.mkdir(parents=True, exist_ok=True)
        self.dizin = dizin
        kisa = [a.split(".")[0] for a in tip_ad]
        self.adim_sut = ["slot", "gun", "saat", "adim", "talep_cpu", "talep_mem", "pod_cpu", "pod_mem",
                         "kap_cpu", "kap_mem", "bekleyen_cpu", "bekleyen_mem", "bekleme_s", "zaman_asimi"]
        self.saat_sut = (["slot", "gun", "saat"] + [f"x_{k}" for k in kisa] + [f"plan_{k}" for k in kisa]
                         + ["tampon"] + [f"acilan_{k}" for k in kisa] + [f"kapanan_{k}" for k in kisa]
                         + ["kap_cpu", "kap_mem", "tepe_cpu", "tepe_mem", "eksik_cpu", "eksik_mem", "eksik",
                            "bekleyen_cpu_max", "bekleyen_mem_max", "maliyet_calisma", "maliyet_acma",
                            "maliyet_birikimli", "ilp_sure_ms", "plan_kayitliyla_ayni"])
        self.dosyalar = {}
        for ad, sut in (("adim", self.adim_sut), ("saat", self.saat_sut)):
            f = open(dizin / f"{ad}.csv", "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=sut)
            w.writeheader()
            self.dosyalar[ad] = (f, w)

    def yaz(self, ad: str, satir: dict) -> None:
        f, w = self.dosyalar[ad]
        w.writerow(satir)
        f.flush()

    def durum(self, d: dict) -> None:
        gecici = self.dizin / "durum.json.tmp"
        gecici.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        gecici.replace(self.dizin / "durum.json")

    def kapat(self) -> None:
        for f, _ in self.dosyalar.values():
            f.close()


def kosu(hz: Hazirlik, kume, hizli: bool, saat_saniye: float, zaman_asimi_s: float, cikti: Path) -> None:
    """Ana döngü."""
    tipler, p, birim = hz.tipler, hz.p, hz.birim
    kisa = [a.split(".")[0] for a in tipler.ad]
    kayit = Kayit(cikti, tipler.ad)
    meta = {"havuz": hz.h.ad, "model": hz.ayar["model"], "strateji": hz.ayar["strateji"],
            "bas": hz.bas, "son": hz.son, "birim": {"cpu": birim.cpu, "mem": birim.mem},
            "tipler": list(tipler.ad), "fiyat": tipler.c.tolist(),
            "cekirdek": tipler.C.tolist(), "bellek_gib": tipler.M.tolist(), "baslangic": time.time(), "bitti": False}
    kayit.durum(meta | {"slot": hz.bas, "mesaj": "küme hazırlanıyor"})

    # Başlangıç durumu: canlı günden bir önceki saatin sunucu sayıları (ısınmadan gelir; açılış sayılmaz)
    kume.temizle()
    kume.dugumleri_ayarla(hz.x_baslangic)
    log.info("Başlangıç düğümleri: %s", dict(zip(kisa, hz.x_baslangic.astype(int))))

    plan, bilgi = None, {}
    d_once = hz.tepe[hz.bas - 1]               # tamponun gördüğü "bir önceki saatin tepe talebi"
    birikimli = 0.0
    t = hz.bas - 1
    adim_sure = saat_saniye / ADIM_SAAT
    try:
        for t in range(hz.bas, hz.son):
            gun, i = divmod(t, 24)
            if i == 0 or plan is None:
                plan, bilgi = hz.ilp.gun_plani(gun)
                log.info("Gün %d planı: ILP %s ms, kayıtlı planla aynı: %s", gun,
                         f"{bilgi.get('sure_ms', float('nan')):.0f}", bilgi.get("kayitliyla_ayni", "-"))
            baz = plan.x[:, i].astype(int)
            hedef, b = baz.copy(), 0.0
            if hz.tampon is not None:
                b, _, _ = hz.tampon.adim(d_once, kapasite(baz, tipler, p))
                hedef[hz.olcek_tipi] += int(b)
            acilan, kapanan = kume.dugumleri_ayarla(hedef)
            x = kume.dugum_sayilari()
            kap = kapasite(x, tipler, p)

            gozlenen = np.zeros(2)
            bekleyen_max = np.zeros(2, int)
            for s in range(ADIM_SAAT):
                t0 = time.monotonic()
                talep = hz.adim5[ADIM_SAAT * t + s]
                gozlenen = np.maximum(gozlenen, talep)          # izleme sisteminin gördüğü saatlik tepe
                n_cpu, n_mem = birim.pod_sayisi(talep)
                kume.yuku_ayarla(n_cpu, n_mem)
                o = kume.yerlesmeyi_bekle(zaman_asimi_s)
                if o.zaman_asimi:
                    log.warning("slot %d adım %d: yerleşme %.0f s'de durulmadı", t, s, zaman_asimi_s)
                bekleyen_max = np.maximum(bekleyen_max, [o.bekleyen_cpu, o.bekleyen_mem])
                kayit.yaz("adim", {"slot": t, "gun": gun, "saat": i, "adim": s,
                                   "talep_cpu": round(talep[0], 3), "talep_mem": round(talep[1], 3),
                                   "pod_cpu": n_cpu, "pod_mem": n_mem, "kap_cpu": kap[0], "kap_mem": kap[1],
                                   "bekleyen_cpu": o.bekleyen_cpu, "bekleyen_mem": o.bekleyen_mem,
                                   "bekleme_s": round(o.bekleme_s, 3), "zaman_asimi": int(o.zaman_asimi)})
                if not hizli:
                    time.sleep(max(0.0, adim_sure - (time.monotonic() - t0)))

            calisma = float((tipler.c * x).sum())
            acma = float((p.s * acilan).sum())
            birikimli += calisma + acma
            eksik = bekleyen_max > 0
            satir = {"slot": t, "gun": gun, "saat": i}
            satir |= {f"x_{k}": int(x[j]) for j, k in enumerate(kisa)}
            satir |= {f"plan_{k}": int(baz[j]) for j, k in enumerate(kisa)}
            satir |= {"tampon": int(b)}
            satir |= {f"acilan_{k}": int(acilan[j]) for j, k in enumerate(kisa)}
            satir |= {f"kapanan_{k}": int(kapanan[j]) for j, k in enumerate(kisa)}
            satir |= {"kap_cpu": kap[0], "kap_mem": kap[1], "tepe_cpu": round(gozlenen[0], 3),
                      "tepe_mem": round(gozlenen[1], 3), "eksik_cpu": int(eksik[0]), "eksik_mem": int(eksik[1]),
                      "eksik": int(eksik.any()), "bekleyen_cpu_max": int(bekleyen_max[0]),
                      "bekleyen_mem_max": int(bekleyen_max[1]), "maliyet_calisma": round(calisma, 5),
                      "maliyet_acma": round(acma, 5), "maliyet_birikimli": round(birikimli, 5),
                      "ilp_sure_ms": round(bilgi.get("sure_ms", float("nan")), 1) if i == 0 else "",
                      "plan_kayitliyla_ayni": int(bilgi["kayitliyla_ayni"]) if i == 0 and "kayitliyla_ayni" in bilgi
                      else ""}
            kayit.yaz("saat", satir)
            kayit.durum(meta | {"slot": t + 1, "mesaj": f"gün {gun}, saat {i:02d}:00 tamamlandı"})
            d_once = gozlenen
            log.info("gün %d %02d:00 | düğüm %s tampon %d | eksik %s | maliyet %.0f $", gun, i,
                     "/".join(map(str, x)), b, "EVET" if eksik.any() else "hayır", birikimli)
    finally:
        kayit.durum(meta | {"slot": t + 1, "bitti": True, "mesaj": "bitti"})
        kayit.kapat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kubernetes kapasite planlama demosu", add_help=False)
    ap.add_argument("--yardim", "-h", action="help", help="bu yardımı göster")
    ap.add_argument("--kuru", action="store_true", help="Kubernetes olmadan bellek içi kümeyle çalış")
    ap.add_argument("--hizli", action="store_true", help="adımlar arasında bekleme (gerçek zaman) yok")
    ap.add_argument("--strateji", choices=["ilp", "r2_uyumlu"])
    ap.add_argument("--bas-gun", type=int)
    ap.add_argument("--gun-sayisi", type=int)
    ap.add_argument("--saat-saniye", type=float, help="1 benzetim saati kaç gerçek saniye")
    ap.add_argument("--kayitli-plan", action="store_true", help="ILP'yi canlı çözme, kayıtlı planı kullan")
    ap.add_argument("--cikti", help="çıktı dizini (proje köküne göreli)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    ayar = ayar_oku()
    for anahtar, deger in (("strateji", a.strateji), ("bas_gun", a.bas_gun), ("gun_sayisi", a.gun_sayisi),
                           ("cikti", a.cikti)):
        if deger is not None:
            ayar[anahtar] = deger
    if a.kayitli_plan:
        ayar["canli_ilp"] = False
    saat_saniye = a.saat_saniye or float(ayar["hiz"]["saat_saniye"])

    log.info("Hazırlık: %s, %s, strateji %s, gün %d–%d", ayar["havuz"], ayar["model"], ayar["strateji"],
             ayar["bas_gun"], ayar["bas_gun"] + ayar["gun_sayisi"] - 1)
    hz = hazirla(ayar)
    if a.kuru:
        kume = SahteKume(hz.tipler, hz.birim)
    else:
        kume = KubeKume(hz.tipler, hz.birim, ayar["kume"]["ad_alani"], hz.tipler.c)
    kosu(hz, kume, a.hizli or a.kuru, saat_saniye, float(ayar["kume"]["yerlesme_zaman_asimi_s"]),
         PROJE_KOKU / ayar["cikti"])
    log.info("Bitti. Doğrulama: .venv/bin/python -m demo.dogrula")


if __name__ == "__main__":
    main()
