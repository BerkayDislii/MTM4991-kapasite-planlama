"""Planların GERÇEK talep üzerinde değerlendirilmesi (MIMARI.md Bölüm 7).

Kapasite α payı olmadan hesaplanır: kap_t = Σ_k ρ C_k x_kt (CPU), Σ_k ρ M_k x_kt (bellek).
- Boşta kapasite oranı: ortalama max(kap − D, 0) / kap  (yetersiz dilimde boşta kapasite 0 sayılır)
- Yetersizlik: D > kap olan dilimler; karşılanmayan talep Σ max(D − kap, 0) (çekirdek-saat, GB-saat)
"""
import numpy as np

from src.ilp.model import IlpParam, Plan, SunucuTipleri


def plan_olcutleri(plan: Plan, D_cpu: np.ndarray, D_mem: np.ndarray, tipler: SunucuTipleri, p: IlpParam,
                   tol: float = 1e-9) -> dict[str, float]:
    """Tek bir planın (K, T) maliyet, boşta kapasite ve yetersizlik ölçütleri."""
    kap = {"cpu": (p.rho_cpu * tipler.C[:, None] * plan.x).sum(0),
           "mem": (p.rho_mem * tipler.M[:, None] * plan.x).sum(0)}
    D = {"cpu": np.asarray(D_cpu, float), "mem": np.asarray(D_mem, float)}
    calisma = float((tipler.c[:, None] * plan.x).sum())
    acma = float((p.s[:, None] * plan.y).sum())
    kapatma = float((p.d[:, None] * plan.z).sum())
    o: dict[str, float] = {"maliyet": calisma + acma + kapatma, "maliyet_calisma": calisma,
                           "maliyet_acma": acma, "maliyet_kapatma": kapatma,
                           "n_acma": float(plan.y.sum()), "n_dilim": float(len(D_cpu)),
                           "ort_sunucu": float(plan.x.sum(0).mean())}
    for k, ad in enumerate(tipler.ad):
        o[f"ort_{ad}"] = float(plan.x[k].mean())
    eksik_herhangi = np.zeros(len(D_cpu), bool)
    for r in ("cpu", "mem"):
        acik = D[r] - kap[r]
        eksik = acik > tol
        eksik_herhangi |= eksik
        with np.errstate(divide="ignore", invalid="ignore"):
            bos = np.where(kap[r] > 0, np.maximum(-acik, 0) / kap[r], 0.0)
        o[f"bos_orani_{r}"] = float(bos.mean())
        o[f"eksik_saat_{r}"] = float(eksik.sum())
        o[f"karsilanmayan_{r}"] = float(np.maximum(acik, 0).sum())
        # Karşılanmayan talebin toplam talebe oranı (%)
        o[f"karsilanmayan_{r}_yuzde"] = float(100 * np.maximum(acik, 0).sum() / D[r].sum())
    o["eksik_saat"] = float(eksik_herhangi.sum())
    o["eksik_saat_orani"] = float(eksik_herhangi.mean())
    return o | olay_olcutleri(eksik_herhangi)


def olay_olcutleri(eksik: np.ndarray) -> dict[str, float]:
    """Tepki gecikmesi (M6, kullanıcı kararı): art arda eksik kapasiteli dilimler bir "olay"dır.

    Olay süresi, açığın kaç dilimde kapatıldığını gösterir. Döndürür: olay sayısı, ortalama ve en uzun süre (dilim).
    """
    e = np.concatenate([[False], np.asarray(eksik, bool), [False]]).astype(np.int8)
    fark = np.diff(e)
    sureler = np.flatnonzero(fark == -1) - np.flatnonzero(fark == 1)
    return {"n_olay": float(sureler.size),
            "olay_sure_ort": float(sureler.mean()) if sureler.size else 0.0,
            "olay_sure_max": float(sureler.max()) if sureler.size else 0.0}
