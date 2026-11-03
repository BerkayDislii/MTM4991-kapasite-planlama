"""LSTM girdisi için 5 dk çözünürlüklü havuz toplamı serileri (MIMARI.md Bölüm 5.2).

Saatlik pool_<ad>.parquet ile AYNI kurallarla (src.preprocess.havuz_bes_dk_toplam, aynı dilim kökeni) üretilir ve
saatlik tepenin 5 dk serisinden yeniden hesaplanınca birebir tuttuğu assert ile doğrulanır.

Çıktı: data/processed/pool5_<ad>.parquet  (step, cpu, mem)  — step = dilim·12 + adım, dilim 0'dan başlar.

Çalıştırma: .venv/bin/python -m src.forecast.veri
"""
import json
import logging

import numpy as np
import polars as pl

from src.common.config import yaml_oku, yol
from src.forecast.ortak import Havuz, havuzlari_listele, saatlik_seri
from src.preprocess import havuz_bes_dk_toplam

log = logging.getLogger(__name__)


def bes_dk_yolu(ad: str):
    """data/processed/pool5_<ad>.parquet yolu."""
    return yol(f"data/processed/pool5_{ad}.parquet")


def bes_dk_seri(h: Havuz) -> np.ndarray:
    """(n_dilim·12, 2) dizisi: cpu, mem 5 dk havuz toplamı (bölme dışı fazla adımlar atılmış)."""
    df = pl.read_parquet(bes_dk_yolu(h.ad)).sort("step")
    n = h.sinir["test"][1] * yaml_oku("preprocess")["grid"]["steps_per_slot"]
    return df.select("cpu", "mem").to_numpy()[:n].astype(np.float64)


def main() -> None:
    """Tüm havuzların 5 dk serilerini yazar ve saatlik tepe ile tutarlılığını doğrular."""
    pcfg = yaml_oku("preprocess")
    adim_s, k = pcfg["grid"]["step_seconds"], pcfg["grid"]["steps_per_slot"]
    uyeler = json.loads(yol("data/processed/pool_members.json").read_text(encoding="utf-8"))
    veri: dict[str, pl.DataFrame] = {}
    for h in havuzlari_listele():
        if h.veri_seti == "bitbrains":
            ds = uyeler[h.uye_anahtari]["veri_seti"]
            if ds not in veri:
                veri[ds] = pl.read_parquet(yol(f"data/interim/bitbrains_{ds}.parquet"))
            df = veri[ds]
            # Dilim kökeni veri seti düzeyinde ortak (src.preprocess ile aynı)
            c0, c1 = int(df["cell"].min()), int(df["cell"].max())
        else:
            if "alibaba" not in veri:
                c0a = pcfg["alibaba"]["start_day"] * 86400 // adim_s
                veri["alibaba"] = (pl.read_parquet(yol("data/interim/alibaba.parquet"))
                                   .filter(pl.col("cell") >= c0a).rename({"mem_gb_r1": "mem_gb"}))
            df = veri["alibaba"]
            c0, c1 = pcfg["alibaba"]["start_day"] * 86400 // adim_s, int(df["cell"].max())
        d5 = havuz_bes_dk_toplam(df, uyeler[h.uye_anahtari]["uyeler"], c0, c1)
        n = h.sinir["test"][1] * k
        x = d5.select("cpu_cores", "mem_gb").to_numpy()[:n]
        if h.r is not None:
            x[:, 1] *= h.r                       # M_ref = cpu_num × r (bellek doğrusal ölçeklenir)
        # Doğrulama: 5 dk serisinden saatlik tepe = pool_<ad>.parquet tepesi
        tepe = x.reshape(-1, k, 2).max(axis=1)
        assert np.allclose(tepe, saatlik_seri(h), rtol=1e-9, atol=1e-6), f"{h.ad}: 5 dk ↔ saatlik tepe uyuşmuyor"
        pl.DataFrame({"step": np.arange(n, dtype=np.int64), "cpu": x[:, 0], "mem": x[:, 1]}).write_parquet(bes_dk_yolu(h.ad))
        log.info("%s: %d adım yazıldı (saatlik tepe doğrulandı)", h.ad, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
