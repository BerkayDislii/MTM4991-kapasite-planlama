"""M4 ölçüt tablolarını birleştirir ve havuzlar (LSTM'de havuz × tohum) üzerinden ortalama ± std özetler.

Girdi : outputs/models/metrikler_{naif,sarima,lstm}.csv
Çıktı : outputs/models/tablo_metrikler.csv   (tüm satırlar)
        outputs/models/tablo_ozet.csv        (grup × model × bölüm × kaynak, ortalama ve std sütunları)
        outputs/models/tablo_ozet_metin.csv  ("ort ± std" biçimli, rapor için)

Çalıştırma: .venv/bin/python -m src.forecast.ozet
"""
import logging

import polars as pl

from src.common.config import yaml_oku
from src.forecast.ortak import modeller_dizini

log = logging.getLogger(__name__)

OLCUTLER: list[str] = ["mae", "rmse", "mape", "mase", "eksik_tahmin_orani"]
MODEL_SIRASI: list[str] = ["son_deger", "mevsimsel_naif", "sarima", "lstm"]
BOLUM_SIRASI: list[str] = ["train", "val", "test"]


def grup_esleme() -> pl.DataFrame:
    """Bitbrains havuz -> özet grubu tablosu (forecast.yaml summary_groups). Alibaba grupları r'den türetilir."""
    gruplar = yaml_oku("forecast")["summary_groups"]
    return pl.DataFrame([{"havuz": p, "grup": g} for g, ps in gruplar.items() for p in ps])


def main() -> None:
    """Tabloları üretir."""
    md = modeller_dizini()
    tum = pl.concat([pl.read_csv(md / f"metrikler_{m}.csv", schema_overrides={"r": pl.Float64, "tohum": pl.Int64})
                     for m in ("naif", "sarima", "lstm")], how="diagonal")
    tum.write_csv(md / "tablo_metrikler.csv")
    esleme = grup_esleme()
    tum = tum.join(esleme, on="havuz", how="left").with_columns(
        pl.when(pl.col("veri_seti") == "alibaba")
          .then(pl.format("alibaba_r{}", pl.col("r").cast(pl.Int64)))
          .otherwise(pl.col("grup")).alias("grup"))
    assert tum["grup"].null_count() == 0, "Gruplanmamış havuz var"
    ozet = (tum.group_by("grup", "model", "bolum", "kaynak")
               .agg([pl.len().alias("n_satir"), pl.col("havuz").n_unique().alias("n_havuz")]
                    + [pl.col(c).mean().alias(f"{c}_ort") for c in OLCUTLER]
                    + [pl.col(c).std().alias(f"{c}_std") for c in OLCUTLER])
               .with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(MODEL_SIRASI)}).alias("_m"),
                             pl.col("bolum").replace_strict({b: i for i, b in enumerate(BOLUM_SIRASI)}).alias("_b"))
               .sort("grup", "kaynak", "_b", "_m").drop("_m", "_b"))
    ozet.write_csv(md / "tablo_ozet.csv")

    def bicim(c: str, basamak: int) -> pl.Expr:
        # Tek havuzlu gruplarda std yoktur (yalnız LSTM'de tohumlar arası std)
        return (pl.when(pl.col(f"{c}_std").is_null()).then(pl.col(f"{c}_ort").round(basamak).cast(pl.String))
                  .otherwise(pl.format("{} ± {}", pl.col(f"{c}_ort").round(basamak), pl.col(f"{c}_std").round(basamak)))
                  .alias(c))
    metin = ozet.select("grup", "kaynak", "bolum", "model", "n_havuz", "n_satir",
                        bicim("mae", 2), bicim("rmse", 2), bicim("mape", 2), bicim("mase", 3),
                        bicim("eksik_tahmin_orani", 3))
    metin.write_csv(md / "tablo_ozet_metin.csv")
    log.info("Özet tablolar yazıldı: %d satır", metin.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
