"""Demonun Kubernetes'ten bağımsız çekirdeği: veri, ILP planlayıcı ve R2 tamponu.

Buradaki her karar raporun benzetimiyle (src/ilp/solve.py, src/baselines/tepkisel.py) birebir aynı kuraldır;
fark yalnızca kararların bir döngüde, saat saat verilmesidir. Böylece canlı küme, benzetimin ürettiği
düğüm sayılarını aynen üretmelidir (demo/dogrula.py bunu denetler).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from src.baselines.tepkisel import EsikParam, esik_param_oku  # noqa: E402
from src.common.config import yaml_oku, yol  # noqa: E402
from src.forecast.ortak import Havuz  # noqa: E402
from src.ilp.model import IlpParam, Plan, SunucuTipleri, coz, dogrula, param_oku, tipleri_oku  # noqa: E402

ADIM_SAAT = 12                       # 1 saat = 12 × 5 dk


# ------------------------------------------------------------------ ayarlar ve veri

def ayar_oku(dosya: Path | None = None) -> dict:
    """demo/ayar.yaml'ı okur."""
    with open(dosya or Path(__file__).with_name("ayar.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def havuz_olustur(ad: str) -> Havuz:
    """Havuz tanımı (src.forecast.ortak.havuzlari_listele ile aynı bölme kuralı, yalnız Bitbrains)."""
    assert ad.startswith("bitbrains_"), "Demo şimdilik yalnız Bitbrains havuzlarını destekler"
    b = yaml_oku("preprocess")["bitbrains"]["split_days"]
    t, v, s = 24 * b["train"], 24 * (b["train"] + b["val"]), 24 * (b["train"] + b["val"] + b["test"])
    lb = yaml_oku("forecast")["datasets"]["bitbrains"]["lookback_hours"]
    return Havuz(ad, "bitbrains", {"train": (0, t), "val": (t, v), "test": (v, s)}, lb, ad)


def adim_serisi(h: Havuz) -> np.ndarray:
    """5 dakikalık havuz toplamı talebi: (n_adım, 2) [çekirdek, GiB]."""
    df = pl.read_parquet(yol(f"data/processed/pool5_{h.ad}.parquet")).sort("step")
    return df.select("cpu", "mem").to_numpy().astype(np.float64)


def saatlik_tepe(h: Havuz) -> np.ndarray:
    """Saatlik tepe talebi: (n_saat, 2) — raporun planlama talebi (pool_<havuz>.parquet)."""
    df = pl.read_parquet(yol(f"data/processed/pool_{h.ad}.parquet")).sort("slot")
    return df.select("cpu_peak", "mem_peak").to_numpy().astype(np.float64)


def tahmin_oku(h: Havuz, model: str, gun: int) -> np.ndarray:
    """Gün başı tahmini (24, 2) [cpu_hat, mem_hat] — src.ilp.solve.tahmin_oku ile aynı sözleşme."""
    kok = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"])
    df = pl.read_parquet(kok / h.veri_seti / h.ad / model / f"forecast_{gun}.parquet").sort("slot")
    assert df["slot"].to_list() == list(range(24 * gun, 24 * gun + 24)), f"forecast_{gun}: dilimler hatalı"
    return df.select("cpu_hat", "mem_hat").to_numpy().astype(np.float64)


def kayitli_plan(h: Havuz, model: str, gun: int, tipler: SunucuTipleri) -> Plan:
    """Raporun kaydettiği günlük ILP planı (outputs/ilp/plans/...)."""
    d = yol(yaml_oku("ilp")["outputs"]["dizin"]) / "plans" / h.veri_seti / h.ad / model
    df = pl.read_parquet(d / f"plan_{gun}.parquet").sort("slot")
    dizi = {v: np.stack([df[f"{v}_{a}"].to_numpy() for a in tipler.ad]).astype(float) for v in ("x", "y", "z")}
    return Plan(dizi["x"], dizi["y"], dizi["z"], amac=float("nan"))


def kapasite(x: np.ndarray, tipler: SunucuTipleri, p: IlpParam) -> np.ndarray:
    """Tip başına sunucu sayısından (K,) → [çekirdek, GiB] kapasite (α payı yok)."""
    return np.array([(p.rho_cpu * tipler.C * x).sum(), (p.rho_mem * tipler.M * x).sum()])


# ------------------------------------------------------------------ birim pod

@dataclass(frozen=True)
class Birim:
    """Talebin bölündüğü sabit pod boyutu."""
    cpu: float          # çekirdek
    mem: float          # GiB

    def pod_sayisi(self, talep: np.ndarray) -> tuple[int, int]:
        """[çekirdek, GiB] talebini karşılayan en az işlemci ve bellek pod'u sayısı (yukarı yuvarlama)."""
        return (math.ceil(talep[0] / self.cpu - 1e-9), math.ceil(talep[1] / self.mem - 1e-9))

    def dugum_pod_kapasitesi(self, tipler: SunucuTipleri) -> np.ndarray:
        """(K, 2): bir düğüme sığan işlemci ve bellek pod'u sayısı. Tam bölünme şartı denetlenir."""
        k = np.stack([tipler.C / self.cpu, tipler.M / self.mem], axis=1)
        assert np.allclose(k, np.round(k)), "Birim pod boyutu sunucu kapasitelerine tam bölünmeli"
        return np.round(k).astype(int)


# ------------------------------------------------------------------ R2 tamponu (adım adım)

class Tampon:
    """src.baselines.tepkisel.esik_simule döngüsünün adım adım hâli (aynı kural, aynı durum değişkenleri).

    Her saat başında bir kez çağrılır: girdi bir önceki saatin gözlenen tepe talebi ve bu saatin plan
    kapasitesidir; çıktı tampondaki ölçek tipi (m8i) sunucu sayısıdır. Plan tabanın altına inilmez.
    """

    def __init__(self, tipler: SunucuTipleri, p: IlpParam, e: EsikParam):
        self.e, self.k, self.L = e, e.tip, int(p.L[e.tip])
        self.birim = np.array([p.rho_cpu * tipler.C[e.tip], p.rho_mem * tipler.M[e.tip]])
        self.b_once, self.kap_once, self.son_degisim = 0.0, np.zeros(2), -10 ** 9
        self.yb: list[float] = []         # tamponun saat saat açılışları ((5) için)
        self.t = 0

    def adim(self, d_once: np.ndarray, kap_baz: np.ndarray) -> tuple[float, float, float]:
        """Bir saatlik karar. Döndürür: (tampon, açılan, kapanan)."""
        e, t = self.e, self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(self.kap_once > 0, d_once / self.kap_once, np.where(d_once > 0, np.inf, 0.0)).max()
        gerek = max(0.0, math.ceil(((d_once / e.hedef - kap_baz) / self.birim).max() - 1e-9))
        L_alt = sum(self.yb[max(0, t - self.L + 1): t])
        if u > e.ust:
            yeni = max(self.b_once, gerek)
        elif u < e.alt and t - self.son_degisim >= e.soguma:
            yeni = min(self.b_once, gerek)
        else:
            yeni = self.b_once
        yeni = max(yeni, L_alt)
        acilan, kapanan = max(yeni - self.b_once, 0), max(self.b_once - yeni, 0)
        if yeni != self.b_once:
            self.son_degisim = t
        self.yb.append(acilan)
        self.b_once, self.kap_once = yeni, kap_baz + yeni * self.birim
        self.t += 1
        return yeni, acilan, kapanan


# ------------------------------------------------------------------ ILP planlayıcı (gün gün)

class IlpPlanlayici:
    """Her gün başında o günün tahminiyle (1)–(6)'yı çözer; durum (x0, devralınan açılışlar) günden güne geçer."""

    def __init__(self, h: Havuz, model: str, tipler: SunucuTipleri, p: IlpParam, canli: bool):
        self.h, self.model, self.tipler, self.p, self.canli = h, model, tipler, p, canli
        cfg = yaml_oku("ilp")["cozucu"]
        self.cozucu, self.zaman_siniri = cfg["ad"], cfg["zaman_siniri_s"]
        self.x0, self.y_onceki = np.zeros(tipler.K), None

    def gun_plani(self, gun: int, canli: bool | None = None) -> tuple[Plan, dict]:
        """Günün planı ve bilgi sözlüğü (çözüm süresi, kayıtlı planla aynı mı)."""
        canli = self.canli if canli is None else canli
        kayitli = kayitli_plan(self.h, self.model, gun, self.tipler)
        bilgi: dict = {"gun": gun, "canli": canli}
        if canli:
            Dh = tahmin_oku(self.h, self.model, gun)
            plan = coz(Dh[:, 0], Dh[:, 1], self.tipler, self.p, self.x0, self.y_onceki,
                       self.cozucu, self.zaman_siniri)
            ihlal = dogrula(plan, Dh[:, 0], Dh[:, 1], self.tipler, self.p, self.x0, self.y_onceki)
            assert not ihlal, f"gün {gun}: doğrulayıcı ihlali {ihlal[:3]}"
            bilgi |= {"sure_ms": float(plan.bilgi.get("sure_ms", float("nan"))), "amac": plan.amac,
                      "kayitliyla_ayni": bool(np.array_equal(plan.x, kayitli.x))}
        else:
            plan = kayitli
        self.x0, self.y_onceki = plan.x[:, -1].copy(), plan.y.copy()
        return plan, bilgi


# ------------------------------------------------------------------ hazırlık

@dataclass
class Hazirlik:
    """Bir koşunun tüm sabit girdileri ve ısınma sonrası durum."""
    ayar: dict
    h: Havuz
    tipler: SunucuTipleri
    p: IlpParam
    birim: Birim
    adim5: np.ndarray            # (n_adım, 2)
    tepe: np.ndarray             # (n_saat, 2)
    ilp: IlpPlanlayici
    tampon: Tampon | None
    bas: int                     # ilk canlı saat (dilim)
    son: int                     # son canlı saat + 1
    x_baslangic: np.ndarray      # bas−1 saatindeki toplam sunucu sayısı (K,)
    olcek_tipi: int


def hazirla(ayar: dict) -> Hazirlik:
    """Verileri yükler, tutarlılığı denetler ve ısınma günlerini (doğrulama dönemi başından canlı güne kadar)
    çevrimdışı yeniden oynatarak ILP ve tampon durumunu raporun benzetimiyle aynı noktaya getirir."""
    h = havuz_olustur(ayar["havuz"])
    tipler = tipleri_oku()
    p = param_oku(tipler)
    birim = Birim(float(ayar["birim"]["cpu"]), float(ayar["birim"]["mem_gib"]))
    birim.dugum_pod_kapasitesi(tipler)
    adim5, tepe = adim_serisi(h), saatlik_tepe(h)
    n = len(tepe)
    assert len(adim5) >= ADIM_SAAT * n
    # Saatlik tepe = o saatin 12 adımının en büyüğü (rapordaki tanım); demo adımları bu seriden oynatılır
    assert np.allclose(adim5[: ADIM_SAAT * n].reshape(n, ADIM_SAAT, 2).max(1), tepe), "adım/saat tutarsız"

    ilk_gun = h.sinir["val"][0] // 24                     # benzetimin soğuk başladığı gün (ısınma)
    test_son = h.sinir["test"][1] // 24
    bas_gun, gun_sayisi = int(ayar["bas_gun"]), int(ayar["gun_sayisi"])
    assert ilk_gun <= bas_gun and bas_gun + gun_sayisi <= test_son, \
        f"Canlı günler {ilk_gun}–{test_son - 1} aralığında olmalı"

    ilp = IlpPlanlayici(h, ayar["model"], tipler, p, bool(ayar["canli_ilp"]))
    e = esik_param_oku(tipler, uyumlu=True)
    tampon = Tampon(tipler, p, e) if ayar["strateji"] == "r2_uyumlu" else None
    assert ayar["strateji"] in ("ilp", "r2_uyumlu"), f"Bilinmeyen strateji: {ayar['strateji']}"

    x_son = np.zeros(tipler.K)
    for g in range(ilk_gun, bas_gun):                      # ısınma: kayıtlı planlarla, kümesiz
        plan, _ = ilp.gun_plani(g, canli=False)
        for i in range(24):
            t = 24 * g + i
            b = 0.0
            if tampon is not None:
                b, _, _ = tampon.adim(tepe[t - 1], kapasite(plan.x[:, i], tipler, p))
            x_son = plan.x[:, i].copy()
            x_son[e.tip] += b
    return Hazirlik(ayar, h, tipler, p, birim, adim5, tepe, ilp, tampon, 24 * bas_gun,
                    24 * (bas_gun + gun_sayisi), x_son, e.tip)
