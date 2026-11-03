"""P0 taban modeller (MIMARI.md Bölüm 5.2): mevsimsel naif ve gün-ileri son değer.

- Mevsimsel naif   : ŷ[o+h] = y[o+h-24], h = 0..23 (önceki günün aynı saati)
- Gün-ileri son değer: ŷ[o+h] = y[o-1]   (gün başındaki son gözlenen tepe, 24 dilim boyunca sabit)

Parametresiz oldukları için eğitim yoktur; yalnızca köken öncesi veri kullanılır.

Çalıştırma: .venv/bin/python -m src.forecast.naif
"""
import logging

import numpy as np
import polars as pl

from src.forecast.ortak import (Havuz, bolum_olcutleri, havuzlari_listele, modeller_dizini, saatlik_seri,
                                tahminleri_kaydet)

log = logging.getLogger(__name__)


def mevsimsel_naif(y: np.ndarray, gunler: list[int], m: int = 24, ufuk: int = 24) -> np.ndarray:
    """Gün kökenlerinde mevsimsel naif tahmin: (n_gun, ufuk, 2). ufuk <= m olmalı (yalnız geçmiş kullanılır)."""
    assert ufuk <= m
    return np.stack([y[24 * g - m: 24 * g - m + ufuk] for g in gunler])


def son_deger(y: np.ndarray, gunler: list[int], ufuk: int = 24) -> np.ndarray:
    """Gün kökenlerinde son gözlenen değerin ufuk boyunca tekrarı: (n_gun, ufuk, 2)."""
    return np.stack([np.repeat(y[24 * g - 1][None, :], ufuk, axis=0) for g in gunler])


def havuz_calistir(h: Havuz) -> list[dict]:
    """Bir havuz için iki taban modelin tüm bölüm tahminlerini ve ölçütlerini üretir."""
    y = saatlik_seri(h)
    satirlar = []
    for model, f in (("mevsimsel_naif", mevsimsel_naif), ("son_deger", son_deger)):
        tahminler = {b: f(y, g) for b, g in h.degerlendirme_gunleri.items()}
        tahminleri_kaydet(h, model, tahminler)
        satirlar += bolum_olcutleri(h, y, tahminler, model)
    return satirlar


def main() -> None:
    """Tüm havuzlarda P0 tabanlarını çalıştırır, outputs/models/metrikler_naif.csv yazar."""
    satirlar = [s for h in havuzlari_listele() for s in havuz_calistir(h)]
    pl.DataFrame(satirlar, infer_schema_length=None).write_csv(modeller_dizini() / "metrikler_naif.csv")
    log.info("P0 taban modeller: %d ölçüt satırı yazıldı", len(satirlar))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
