"""M7b: sentetik senaryolar (MIMARI.md Bölüm 8; kullanıcı kararı 2026-11-01). Gerçek sonuçlardan AYRI.

Bozulma yalnız test dilimlerine, 5 dk havuz toplamı serisine uygulanır; saatlik tepe 5 dk serisinin saatlik
maksimumu olarak yeniden hesaplanır (M3/M4 tanımıyla aynı). Modeller bozulmamış eğitim+doğrulama verisiyle
bir kez eğitilir ve her senaryoda parametreleri sabit tutularak test günleri için walk-forward tahmin üretir:
  - P0 (son değer, mevsimsel naif): bozulmuş seri üzerinde doğrudan
  - P1 SARIMA: M4'te seçilen derece (outputs/models/sarima_secim.csv), eğitim+doğrulamaya yeniden uydurma
  - P2 LSTM: M4 aşama 2 (eğitim+doğrulama ölçekleyicisi, en iyi epoch sayısı outputs/models/lstm_egitim.csv)
Doğrulama günleri bozulmadığı için ısınma tahminleri M4 dosyalarından kopyalanır. Ardından ILP (kayan pencere),
kahin, R0, R1, R2 (iki varyant) merkez parametrelerle (ilp.yaml model) M6 çekirdeğiyle değerlendirilir.

Çıktılar (outputs/synthetic/):
  forecasts/<senaryo>/<dataset>/<pool>/<model>/forecast_<gün>.parquet
  sim/<senaryo>/<dataset>/<pool>/<strateji>[_<model>].parquet   saatlik kapasite ve bozulmuş gerçek talep
  senaryolar.csv          senaryo × veri seti başlangıç dilimleri
  tahmin_olcutleri.csv    senaryo × havuz × model test tahmin ölçütleri (MAE, eksik tahmin oranı, ...)
  kontrol_tutarlilik.csv  kontrol senaryosu tahminlerinin M4 test tahminlerinden en büyük farkı
  havuz_ozet.csv          senaryo × havuz × strateji × model
  tablo_ozet.csv, tablo_ozet_metin.csv   senaryo × grup × strateji × model, havuzlar üzerinden ortalama ± std

Çalıştırma: .venv/bin/python -m src.sentetik
"""
import ast
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from src.baselines.tepkisel import (KAHIN, esik_param_oku, havuz_deneyi, ozetle, regret_ekle, simulasyon_gunleri)
from src.common.config import yaml_oku, yol
from src.forecast.naif import mevsimsel_naif, son_deger
from src.forecast.ortak import Havuz, havuzlari_listele, mase_olcegi, olcutler, saatlik_seri, tahmin_yaz
from src.forecast.veri import bes_dk_seri
from src.ilp.model import param_oku, tipleri_oku
from src.ilp.solve import kayan_pencere_plani

log = logging.getLogger(__name__)
VERI_SETLERI: tuple[str, ...] = ("bitbrains", "alibaba")


# ---------------------------------------------------------------- senaryo üretimi

def baslangic_sec(sc: dict[str, Any], test: tuple[int, int], rng: np.random.Generator) -> int | None:
    """Bozulmanın başlangıç dilimi (test içinde). Kayma ve kontrol için None (tüm test dönemi)."""
    if sc["tur"] != "carpan":
        return None
    a, b = test
    etki = sc["sure_saat"] if sc["sure_saat"] is not None else sc["min_etki_saat"]
    assert b - a >= etki, "Test dönemi bozulma süresinden kısa"
    return int(rng.integers(a, b - etki + 1))


def bozulmus_seri(x5: np.ndarray, sc: dict[str, Any], test: tuple[int, int], bas: int | None,
                  k: int) -> tuple[np.ndarray, np.ndarray]:
    """5 dk serisine senaryoyu uygular; (x5', saatlik tepe y') döndürür. Test dışı dilimler değişmez."""
    x = x5.copy()
    a, b = test
    if sc["tur"] == "carpan":
        son = b if sc["sure_saat"] is None else min(b, bas + sc["sure_saat"])
        x[bas * k: son * k] *= sc["carpan"]
    elif sc["tur"] == "kayma":
        s = sc["kayma_saat"]
        assert a - s >= 0
        # Test dilimi t'nin değeri t − s dilimindekiyle değiştirilir (yalnız geçmiş gerçek veri kullanılır)
        x[a * k: b * k] = x5[(a - s) * k: (b - s) * k]
    else:
        assert sc["tur"] == "yok", f"Bilinmeyen senaryo türü: {sc['tur']}"
    y = x.reshape(-1, k, 2).max(axis=1)
    return x, y


def senaryo_tablosu(cfg: dict[str, Any], havuzlar: list[Havuz]) -> dict[tuple[str, str], int | None]:
    """(senaryo, veri seti) → başlangıç dilimi; aynı veri setinin havuzlarında ortak (tohumlu)."""
    ornek = {h.veri_seti: h for h in havuzlar}
    out = {}
    for i, (ad, sc) in enumerate(cfg["senaryolar"].items()):
        for j, vs in enumerate(VERI_SETLERI):
            rng = np.random.default_rng([cfg["tohum"], i, j])
            out[(ad, vs)] = baslangic_sec(sc, ornek[vs].sinir["test"], rng)
    return out


# ---------------------------------------------------------------- tahminler (test günleri)

def sarima_tahmin(h: Havuz, j: int, seriler: dict[str, np.ndarray], secim: dict[str, Any]) -> dict[str, np.ndarray]:
    """M4 SARIMA son modeli (seçilen derece, eğitim+doğrulamaya uydurma) ile senaryo başına test tahmini (n_gun, 24)."""
    from src.forecast.sarima import gun_ileri, uydur
    y0 = saatlik_seri(h)[:, j]
    t_son, v_son = h.sinir["train"][1], h.sinir["val"][1]
    olcek = float(y0[:t_son].mean())
    res, _ = uydur(y0[:v_son] / olcek, ast.literal_eval(secim["order"]), ast.literal_eval(secim["seasonal_order"]),
                   yaml_oku("forecast")["sarima"]["maxiter"])
    assert res is not None, f"{h.ad}/{j}: SARIMA son modeli uydurulamadı"
    gunler = h.degerlendirme_gunleri["test"]
    return {s: gun_ileri(res, y[:, j] / olcek, gunler) * olcek for s, y in seriler.items()}


def lstm_tahmin(h: Havuz, x5_ler: dict[str, np.ndarray], seriler: dict[str, np.ndarray], tohum: int,
                epoch: int) -> dict[str, np.ndarray]:
    """M4 LSTM aşama 2 (eğitim+doğrulama, sabit epoch) bir kez eğitilir; senaryo başına test tahmini (n_gun, 24, 2)."""
    from src.forecast.lstm import Olcekleyici, egit, ornekler, tahmin_et
    cfg = yaml_oku("forecast")["lstm"]
    k = yaml_oku("preprocess")["grid"]["steps_per_slot"]
    y0, x0 = saatlik_seri(h), bes_dk_seri(h)
    L, v_son = h.lookback, h.sinir["val"][1]
    sc = Olcekleyici(x0[:v_son * k])               # yalnız bozulmamış eğitim+doğrulama
    Xtv, Ytv = ornekler(sc.donustur(x0), sc.donustur(y0), list(range(L, v_son - 24 + 1)), L, k)
    model, _ = egit(Xtv, Ytv, None, None, cfg, tohum, epoch_sayisi=epoch)
    kok = [24 * g for g in h.degerlendirme_gunleri["test"]]
    return {s: sc.geri(tahmin_et(model, ornekler(sc.donustur(x5_ler[s]), sc.donustur(seriler[s]), kok, L, k)[0], cfg))
            for s in seriler}


def isinma_kopyala(h: Havuz, model: str, kok: Path) -> None:
    """Doğrulama (ısınma) günlerinin M4 tahmin dosyalarını senaryo dizinine kopyalar (bozulma yok)."""
    kaynak = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) / h.veri_seti / h.ad / model
    hedef = kok / h.veri_seti / h.ad / model
    hedef.mkdir(parents=True, exist_ok=True)
    for g in h.degerlendirme_gunleri[yaml_oku("ilp")["kayan_pencere"]["isinma_bolumu"]]:
        shutil.copy2(kaynak / f"forecast_{g}.parquet", hedef / f"forecast_{g}.parquet")


def m4_test_tahmini(h: Havuz, model: str) -> np.ndarray:
    """M4 test tahminleri (n_gun, 24, 2) — kontrol senaryosu karşılaştırması için."""
    d = yol(yaml_oku("forecast")["outputs"]["forecasts_dir"]) / h.veri_seti / h.ad / model
    return np.stack([pl.read_parquet(d / f"forecast_{g}.parquet").sort("slot").select("cpu_hat", "mem_hat").to_numpy()
                     for g in h.degerlendirme_gunleri["test"]])


# ---------------------------------------------------------------- karar değerlendirmesi

def senaryo_havuz(senaryo: str, bas: int | None, h: Havuz, y: np.ndarray, tahmin_kok: Path) -> list[dict]:
    """Senaryo × havuz: ILP planları senaryo tahminlerinden, M6 çekirdeği bozulmuş gerçek talep üzerinde."""
    cfg = yaml_oku("ilp")
    tipler = tipleri_oku()
    p = param_oku(tipler)
    planlar = {m: kayan_pencere_plani(h, m, y, tipler, p, tahmin_kok)
               for m in [*cfg["tepkisel"]["hibrit_modeller"], KAHIN]}
    assert all(q.x.shape[1] == 24 * len(simulasyon_gunleri(h)) for q in planlar.values())
    sim = yol(yaml_oku("synthetic")["dizin"]) / "sim" / senaryo / h.veri_seti / h.ad
    return havuz_deneyi(h, y, tipler, p, esik_param_oku(tipler), esik_param_oku(tipler, uyumlu=True), planlar,
                        sim_dizin=sim, ek={"senaryo": senaryo, "baslangic_dilim": bas})


def main() -> None:
    """Senaryoları üretir, tahminleri ve kararları değerlendirir, tabloları yazar."""
    scfg, icfg = yaml_oku("synthetic"), yaml_oku("ilp")
    k = yaml_oku("preprocess")["grid"]["steps_per_slot"]
    cikti = yol(scfg["dizin"])
    cikti.mkdir(parents=True, exist_ok=True)
    havuzlar = havuzlari_listele()
    senaryolar = list(scfg["senaryolar"])
    bas = senaryo_tablosu(scfg, havuzlar)
    pl.DataFrame([{"senaryo": s, "veri_seti": vs, "baslangic_dilim": b, **scfg["senaryolar"][s]}
                  for (s, vs), b in bas.items()], infer_schema_length=None).write_csv(cikti / "senaryolar.csv")

    # Senaryo serileri (havuz → senaryo → (x5', y'))
    seri: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for h in havuzlar:
        x5 = bes_dk_seri(h)
        seri[h.ad] = {s: bozulmus_seri(x5, scfg["senaryolar"][s], h.sinir["test"], bas[(s, h.veri_seti)], k)
                      for s in senaryolar}
        x_k, y_k = seri[h.ad]["kontrol"]
        assert np.array_equal(y_k, saatlik_seri(h)), f"{h.ad}: kontrol serisi saatlik tepeyle aynı değil"
        for s in senaryolar:   # test dışı dilimler değişmemeli (sızıntı / ısınma tutarlılığı)
            assert np.array_equal(seri[h.ad][s][1][:h.sinir["test"][0]], y_k[:h.sinir["test"][0]])

    # ---- Tahminler: tahmin[(havuz, model)][senaryo] = (n_gun_test, 24, 2)
    tahmin: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for h in havuzlar:
        gunler = h.degerlendirme_gunleri["test"]
        for model, f in (("son_deger", son_deger), ("mevsimsel_naif", mevsimsel_naif)):
            tahmin[(h.ad, model)] = {s: f(seri[h.ad][s][1], gunler) for s in senaryolar}
    secim = {(r["havuz"], r["kaynak"]): r for r in pl.read_csv(yol("outputs/models/sarima_secim.csv")).to_dicts()}
    sar = Parallel(n_jobs=icfg["cozucu"]["n_jobs"])(
        delayed(sarima_tahmin)(h, j, {s: seri[h.ad][s][1] for s in senaryolar}, secim[(h.ad, kay)])
        for h in havuzlar for j, kay in enumerate(("cpu", "mem")))
    for i, h in enumerate(havuzlar):
        tahmin[(h.ad, "sarima")] = {s: np.stack([sar[2 * i][s], sar[2 * i + 1][s]], axis=-1) for s in senaryolar}
    log.info("P0 ve SARIMA tahminleri tamam")
    egitim = {(r["havuz"], r["tohum"]): r["en_iyi_epoch"] for r in pl.read_csv(yol("outputs/models/lstm_egitim.csv")).to_dicts()}
    for h in havuzlar:
        for tohum in yaml_oku("forecast")["lstm"]["seeds"]:
            tahmin[(h.ad, f"lstm_s{tohum}")] = lstm_tahmin(h, {s: seri[h.ad][s][0] for s in senaryolar},
                                                           {s: seri[h.ad][s][1] for s in senaryolar},
                                                           tohum, egitim[(h.ad, tohum)])
        log.info("%s: LSTM tahminleri tamam", h.ad)

    # ---- Tahmin dosyaları, ölçütler ve kontrol tutarlılığı
    modeller = icfg["tepkisel"]["hibrit_modeller"]
    olcut, kontrol = [], []
    for h in havuzlar:
        olcek = mase_olcegi(saatlik_seri(h), h.sinir["train"][1])
        gunler = h.degerlendirme_gunleri["test"]
        for m in modeller:
            m4 = m4_test_tahmini(h, m)
            fark = np.abs(tahmin[(h.ad, m)]["kontrol"] - m4)
            kontrol.append({"havuz": h.ad, "model": m, "max_mutlak_fark": float(fark.max()),
                            "max_goreli_fark": float((fark / np.maximum(np.abs(m4), 1e-9)).max())})
            for s in senaryolar:
                kok = cikti / "forecasts" / s
                isinma_kopyala(h, m, kok)
                tahmin_yaz(h, m, gunler, tahmin[(h.ad, m)][s], kok)
                gercek = np.stack([seri[h.ad][s][1][24 * g: 24 * g + 24] for g in gunler])
                for o in olcutler(gercek, tahmin[(h.ad, m)][s], olcek):
                    olcut.append({"senaryo": s, "veri_seti": h.veri_seti, "havuz": h.ad, "r": h.r, "model": m, **o})
    pl.DataFrame(olcut, infer_schema_length=None).write_csv(cikti / "tahmin_olcutleri.csv")
    pl.DataFrame(kontrol).write_csv(cikti / "kontrol_tutarlilik.csv")

    # ---- Kararlar
    sonuc = Parallel(n_jobs=icfg["cozucu"]["n_jobs"])(
        delayed(senaryo_havuz)(s, bas[(s, h.veri_seti)], h, seri[h.ad][s][1], cikti / "forecasts" / s)
        for s in senaryolar for h in havuzlar)
    df = regret_ekle(pl.DataFrame([r for x in sonuc for r in x], infer_schema_length=None), ["senaryo", "havuz"])
    df.write_csv(cikti / "havuz_ozet.csv")
    ozet, metin = ozetle(df, ["senaryo"])
    ozet.write_csv(cikti / "tablo_ozet.csv")
    metin.write_csv(cikti / "tablo_ozet_metin.csv")
    log.info("M7b tamam: %d satır", df.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
