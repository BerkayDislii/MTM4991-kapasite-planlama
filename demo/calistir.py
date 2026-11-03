"""Demoyu çalıştırır: gerçek talebi kümeye oynatır, ILP planını ve tamponu saat saat uygular, ölçer.

Çalıştırma (proje kökünden):
    .venv/bin/python -m demo.calistir                       # iki strateji yan yana (yalnız ILP | ILP + tampon)
    .venv/bin/python -m demo.calistir --strateji r2_uyumlu   # tek strateji
    .venv/bin/python -m demo.calistir --kuru --hizli         # Kubernetes olmadan, beklemeden (hesap denetimi)
    .venv/bin/python -m demo.calistir --gun-sayisi 6 --hizli # tüm test dönemi (doğrulama için)

Yan yana koşuda aynı kümede iki ayrı düğüm grubu vardır ("ilp" ve "r2"). İkisi de aynı talebi aynı anda
yaşar; tek fark, r2 grubunda planın üstüne tepkisel tampon eklenmesidir.

Döngü (her benzetim saati t, her grup için):
  1. Gün başıysa: o günün tahmini → ILP (src/ilp/model.py) → 24 saatlik plan
  2. Yalnız r2: bir önceki saatin gözlenen tepe talebine göre m8i tamponu (src/baselines/tepkisel.py kuralı)
  3. Küme: düğüm sayılarını plan (+ tampon) değerine getir
  4. 12 adım (5 dk): gerçek talep (+ varsa panelden gelen senaryo) → pod sayısı → kümeye ver → yerleşmeyi
     bekle → yer bulamayan pod'ları ölç
Çıktılar: <cikti>/<grup>/adim.csv, <cikti>/<grup>/saat.csv, <cikti>/durum.json; panel komutu <cikti>/komut.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from demo.cekirdek import ADIM_SAAT, PROJE_KOKU, Hazirlik, ayar_oku, hazirla, kapasite
from demo.kume import KubeKume, SahteKume, kumeyi_sifirla
from src.common.config import yaml_oku

log = logging.getLogger("demo")
GRUP = {"ilp": "ilp", "r2_uyumlu": "r2"}                   # strateji → grup (düğüm etiketi, çıktı klasörü)
SENARYOLAR = ("darbe", "basamak", "dusus")                  # configs/synthetic.yaml'daki çarpan senaryoları


# ------------------------------------------------------------------ senaryo (panelden gelen komut)

class Senaryo:
    """Panelin yazdığı komut.json'u her adımda okur ve talebi raporun sentetik senaryolarıyla bozar.

    Çarpan ve süre configs/synthetic.yaml'dan gelir (darbe ×2 3 saat; basamak ×1,5 ve düşüş ×0,5 koşu sonuna
    dek). "normal" komutu etkin senaryoyu bitirir. Bozulma hem işlemci hem bellek talebine uygulanır.
    Adım numarası mutlak 5 dk dizinidir (12·slot + adım).
    """

    def __init__(self, dizin: Path):
        cfg = yaml_oku("synthetic")["senaryolar"]
        self.tanim = {ad: cfg[ad] for ad in SENARYOLAR}
        self.dosya = dizin / "komut.json"
        self.dosya.unlink(missing_ok=True)                  # önceki koşunun komutu yeni koşuyu etkilemesin
        self.son_id = None
        self.etkin: dict | None = None
        self.gecmis: list[dict] = []

    def guncelle(self, adim_no: int) -> tuple[str, float]:
        try:
            komut = json.loads(self.dosya.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            komut = None
        if komut and komut.get("id") != self.son_id:
            self.son_id = komut.get("id")
            if self.etkin is not None:                     # yeni komut eskisini bitirir
                self.etkin["bit"] = adim_no
                self.etkin = None
            ad = komut.get("senaryo")
            if ad in self.tanim:
                sc = self.tanim[ad]
                bit = adim_no + ADIM_SAAT * int(sc["sure_saat"]) if sc.get("sure_saat") else None
                self.etkin = {"ad": ad, "carpan": float(sc["carpan"]), "bas": adim_no, "bit": bit}
                self.gecmis.append(self.etkin)
                log.info(">>> Senaryo başladı: %s ×%s", ad, sc["carpan"])
            else:
                log.info(">>> Senaryo bitti, talep normale döndü")
        if self.etkin is not None and self.etkin["bit"] is not None and adim_no >= self.etkin["bit"]:
            log.info(">>> Senaryo süresi doldu: %s", self.etkin["ad"])
            self.etkin = None
        return (self.etkin["ad"], self.etkin["carpan"]) if self.etkin else ("", 1.0)

    def durum(self) -> dict:
        return {"etkin": self.etkin["ad"] if self.etkin else "", "gecmis": self.gecmis}


# ------------------------------------------------------------------ kayıt

class Kayit:
    """Bir grubun CSV'lerine satır satır yazar (panel dosyayı canlı okuyabilsin diye her satırda boşaltır)."""

    def __init__(self, dizin: Path, tip_ad: tuple[str, ...]):
        dizin.mkdir(parents=True, exist_ok=True)
        kisa = [a.split(".")[0] for a in tip_ad]
        sutunlar = {
            "adim": ["slot", "gun", "saat", "adim", "senaryo", "carpan", "talep_cpu", "talep_mem", "pod_cpu",
                     "pod_mem", "kap_cpu", "kap_mem", "bekleyen_cpu", "bekleyen_mem", "bekleme_s", "zaman_asimi"],
            "saat": (["slot", "gun", "saat"] + [f"x_{k}" for k in kisa] + [f"plan_{k}" for k in kisa] + ["tampon"]
                     + [f"acilan_{k}" for k in kisa] + [f"kapanan_{k}" for k in kisa]
                     + ["kap_cpu", "kap_mem", "tepe_cpu", "tepe_mem", "eksik_cpu", "eksik_mem", "eksik",
                        "bekleyen_cpu_max", "bekleyen_mem_max", "maliyet_calisma", "maliyet_acma",
                        "maliyet_birikimli", "senaryo", "ilp_sure_ms", "plan_kayitliyla_ayni"]),
        }
        self.dosyalar = {}
        for ad, sut in sutunlar.items():
            f = open(dizin / f"{ad}.csv", "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=sut)
            w.writeheader()
            self.dosyalar[ad] = (f, w)

    def yaz(self, ad: str, satir: dict) -> None:
        f, w = self.dosyalar[ad]
        w.writerow(satir)
        f.flush()

    def kapat(self) -> None:
        for f, _ in self.dosyalar.values():
            f.close()


def durum_yaz(dizin: Path, d: dict) -> None:
    gecici = dizin / "durum.json.tmp"
    gecici.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    gecici.replace(dizin / "durum.json")


# ------------------------------------------------------------------ bir strateji grubu

@dataclass
class Kol:
    """Yan yana koşan stratejilerden biri: kendi planlayıcısı, tamponu, düğüm grubu ve kayıtları."""
    grup: str
    hz: Hazirlik
    kume: object
    kayit: Kayit
    plan: object = None
    bilgi: dict = field(default_factory=dict)
    d_once: np.ndarray | None = None
    birikimli: float = 0.0
    x: np.ndarray | None = None
    kap: np.ndarray | None = None
    baz: np.ndarray | None = None
    b: float = 0.0
    acilan: np.ndarray | None = None
    kapanan: np.ndarray | None = None
    gozlenen: np.ndarray | None = None
    bekleyen_max: np.ndarray | None = None
    senaryolu: bool = False

    def saat_basi(self, t: int) -> None:
        """Planı ve tamponu uygular, düğümleri ayarlar."""
        hz, gun, i = self.hz, *divmod(t, 24)
        if i == 0 or self.plan is None:
            self.plan, self.bilgi = hz.ilp.gun_plani(gun)
            log.info("[%s] gün %d planı: ILP %s ms, kayıtlı planla aynı: %s", self.grup, gun,
                     f"{self.bilgi.get('sure_ms', float('nan')):.0f}", self.bilgi.get("kayitliyla_ayni", "-"))
        self.baz = self.plan.x[:, i].astype(int)
        hedef, self.b = self.baz.copy(), 0.0
        if hz.tampon is not None:
            self.b, _, _ = hz.tampon.adim(self.d_once, kapasite(self.baz, hz.tipler, hz.p))
            hedef[hz.olcek_tipi] += int(self.b)
        self.acilan, self.kapanan = self.kume.dugumleri_ayarla(hedef)
        self.x = self.kume.dugum_sayilari()
        self.kap = kapasite(self.x, hz.tipler, hz.p)
        self.gozlenen, self.bekleyen_max, self.senaryolu = np.zeros(2), np.zeros(2, int), False

    def saat_sonu(self, t: int) -> None:
        hz, (gun, i) = self.hz, divmod(t, 24)
        tipler, p = hz.tipler, hz.p
        kisa = [a.split(".")[0] for a in tipler.ad]
        calisma, acma = float((tipler.c * self.x).sum()), float((p.s * self.acilan).sum())
        self.birikimli += calisma + acma
        eksik = self.bekleyen_max > 0
        satir = {"slot": t, "gun": gun, "saat": i}
        satir |= {f"x_{k}": int(self.x[j]) for j, k in enumerate(kisa)}
        satir |= {f"plan_{k}": int(self.baz[j]) for j, k in enumerate(kisa)}
        satir |= {"tampon": int(self.b)}
        satir |= {f"acilan_{k}": int(self.acilan[j]) for j, k in enumerate(kisa)}
        satir |= {f"kapanan_{k}": int(self.kapanan[j]) for j, k in enumerate(kisa)}
        satir |= {"kap_cpu": self.kap[0], "kap_mem": self.kap[1], "tepe_cpu": round(self.gozlenen[0], 3),
                  "tepe_mem": round(self.gozlenen[1], 3), "eksik_cpu": int(eksik[0]), "eksik_mem": int(eksik[1]),
                  "eksik": int(eksik.any()), "bekleyen_cpu_max": int(self.bekleyen_max[0]),
                  "bekleyen_mem_max": int(self.bekleyen_max[1]), "maliyet_calisma": round(calisma, 5),
                  "maliyet_acma": round(acma, 5), "maliyet_birikimli": round(self.birikimli, 5),
                  "senaryo": int(self.senaryolu),
                  "ilp_sure_ms": round(self.bilgi.get("sure_ms", float("nan")), 1) if i == 0 else "",
                  "plan_kayitliyla_ayni": int(self.bilgi["kayitliyla_ayni"])
                  if i == 0 and "kayitliyla_ayni" in self.bilgi else ""}
        self.kayit.yaz("saat", satir)
        self.d_once = self.gozlenen                         # tamponun bir sonraki saatte göreceği tepe
        log.info("[%s] gün %d %02d:00 | düğüm %s tampon %d | eksik %s | maliyet %.0f $", self.grup, gun, i,
                 "/".join(map(str, self.x)), self.b, "EVET" if eksik.any() else "hayır", self.birikimli)


# ------------------------------------------------------------------ ana döngü

def kosu(kollar: list[Kol], hizli: bool, saat_saniye: float, zaman_asimi_s: float, cikti: Path) -> None:
    hz0 = kollar[0].hz
    tipler, birim = hz0.tipler, hz0.birim
    senaryo = Senaryo(cikti)
    meta = {"havuz": hz0.h.ad, "model": hz0.ayar["model"],
            "kollar": [{"grup": k.grup, "strateji": k.hz.ayar["strateji"]} for k in kollar],
            "bas": hz0.bas, "son": hz0.son, "birim": {"cpu": birim.cpu, "mem": birim.mem},
            "tipler": list(tipler.ad), "fiyat": tipler.c.tolist(), "cekirdek": tipler.C.tolist(),
            "bellek_gib": tipler.M.tolist(), "baslangic": time.time(), "bitti": False}
    durum_yaz(cikti, meta | {"slot": hz0.bas, "mesaj": "küme hazırlanıyor", "senaryo": senaryo.durum()})

    for k in kollar:                                        # başlangıç: canlı günden önceki saatin durumu
        k.kume.temizle()
        k.kume.dugumleri_ayarla(k.hz.x_baslangic)
        k.d_once = k.hz.tepe[k.hz.bas - 1]
        log.info("[%s] başlangıç düğümleri: %s", k.grup, "/".join(map(str, k.hz.x_baslangic.astype(int))))

    adim_sure = saat_saniye / ADIM_SAAT
    t = hz0.bas - 1
    try:
        for t in range(hz0.bas, hz0.son):
            gun, i = divmod(t, 24)
            for k in kollar:
                k.saat_basi(t)
            for s in range(ADIM_SAAT):
                t0 = time.monotonic()
                sc_ad, carpan = senaryo.guncelle(ADIM_SAAT * t + s)
                talep = hz0.adim5[ADIM_SAAT * t + s] * carpan
                n_cpu, n_mem = birim.pod_sayisi(talep)
                for k in kollar:                            # önce hepsine yükü ver, sonra birlikte bekle
                    k.kume.yuku_ayarla(n_cpu, n_mem)
                for k in kollar:
                    o = k.kume.yerlesmeyi_bekle(zaman_asimi_s)
                    if o.zaman_asimi:
                        log.warning("[%s] slot %d adım %d: yerleşme %.0f s'de durulmadı", k.grup, t, s, zaman_asimi_s)
                    k.gozlenen = np.maximum(k.gozlenen, talep)      # izleme sisteminin gördüğü saatlik tepe
                    k.bekleyen_max = np.maximum(k.bekleyen_max, [o.bekleyen_cpu, o.bekleyen_mem])
                    k.senaryolu |= carpan != 1.0
                    k.kayit.yaz("adim", {
                        "slot": t, "gun": gun, "saat": i, "adim": s, "senaryo": sc_ad, "carpan": carpan,
                        "talep_cpu": round(talep[0], 3), "talep_mem": round(talep[1], 3), "pod_cpu": n_cpu,
                        "pod_mem": n_mem, "kap_cpu": k.kap[0], "kap_mem": k.kap[1],
                        "bekleyen_cpu": o.bekleyen_cpu, "bekleyen_mem": o.bekleyen_mem,
                        "bekleme_s": round(o.bekleme_s, 3), "zaman_asimi": int(o.zaman_asimi)})
                durum_yaz(cikti, meta | {"slot": t, "adim": s, "mesaj": f"gün {gun}, saat {i:02d}:{5 * s:02d}",
                                         "senaryo": senaryo.durum()})
                if not hizli:
                    time.sleep(max(0.0, adim_sure - (time.monotonic() - t0)))
            for k in kollar:
                k.saat_sonu(t)
    finally:
        durum_yaz(cikti, meta | {"slot": t + 1, "bitti": True, "mesaj": "bitti", "senaryo": senaryo.durum()})
        for k in kollar:
            k.kayit.kapat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kubernetes kapasite planlama demosu", add_help=False)
    ap.add_argument("--yardim", "-h", action="help", help="bu yardımı göster")
    ap.add_argument("--kuru", action="store_true", help="Kubernetes olmadan bellek içi kümeyle çalış")
    ap.add_argument("--hizli", action="store_true", help="adımlar arasında bekleme (gerçek zaman) yok")
    ap.add_argument("--strateji", choices=["ikisi", "ilp", "r2_uyumlu"],
                    help="ikisi: yalnız ILP ve ILP + tampon yan yana (varsayılan ayar.yaml)")
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
    stratejiler = ["ilp", "r2_uyumlu"] if ayar["strateji"] == "ikisi" else [ayar["strateji"]]
    cikti = PROJE_KOKU / ayar["cikti"]
    cikti.mkdir(parents=True, exist_ok=True)

    log.info("Hazırlık: %s, %s, %s, gün %d–%d", ayar["havuz"], ayar["model"], " + ".join(stratejiler),
             ayar["bas_gun"], ayar["bas_gun"] + ayar["gun_sayisi"] - 1)
    onek = ayar["kume"]["ad_alani_onek"]
    if not a.kuru:
        kumeyi_sifirla(onek)                               # önceki koşulardan kalanlar (eski sürüm dahil)
    kollar = []
    for st in stratejiler:
        hz = hazirla(ayar | {"strateji": st})
        grup = GRUP[st]
        kume = SahteKume(hz.tipler, hz.birim) if a.kuru else \
            KubeKume(hz.tipler, hz.birim, grup, f"{onek}-{grup}", hz.tipler.c)
        kollar.append(Kol(grup, hz, kume, Kayit(cikti / grup, hz.tipler.ad)))
    hizli = a.hizli or (a.kuru and a.saat_saniye is None)     # kuru koşu, hız verilmedikçe beklemez
    kosu(kollar, hizli, saat_saniye, float(ayar["kume"]["yerlesme_zaman_asimi_s"]), cikti)
    log.info("Bitti. Doğrulama: .venv/bin/python -m demo.dogrula")


if __name__ == "__main__":
    main()
