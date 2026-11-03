"""AWS EC2 on-demand Linux fiyatlarını resmi Price List (bulk) API'sinden çeker (MIMARI.md Bölüm 6.4).

Fiyatlar bellekten yazılmaz: bölgenin index.csv dosyası (~280 MB) diske kaydedilmeden akış olarak okunur,
yalnızca aday satırlar (c/m/r aileleri, Linux, paylaşımlı kiracılık, ön yüklü yazılım yok) tutulur.

Çıktılar:
  data/raw/aws/ec2_ondemand_<bölge>_<yayın tarihi>.csv  (aday satırların anlık görüntüsü)
  configs/server_types.yaml                             (ilp.yaml'da seçilen 3 tip, tarihli)

Çalıştırma: .venv/bin/python -m src.ilp.fiyat_cek
"""
import csv
import io
import logging
import re
import urllib.request
from datetime import date

import polars as pl
import yaml

from src.common.config import yaml_oku, yol

log = logging.getLogger(__name__)

URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{bolge}/index.csv"
# Aday tip deseni: hesaplama (c) / genel amaçlı (m) / bellek optimize (r) aileleri
ADAY = re.compile(r"^[cmr]\d+[a-z]*\.(large|xlarge|2xlarge|4xlarge)$")
# Satır süzgeci: on-demand, Linux, paylaşımlı, lisanssız, ön yüklü yazılım yok, kullanılan kapasite
SUZGEC = {"TermType": "OnDemand", "Operating System": "Linux", "Tenancy": "Shared",
          "Pre Installed S/W": "NA", "CapacityStatus": "Used", "License Model": "No License required",
          "Product Family": "Compute Instance", "Unit": "Hrs"}
TUT = ["Instance Type", "Instance Family", "vCPU", "Memory", "Physical Processor", "PricePerUnit", "Currency",
       "EffectiveDate", "SKU", "Current Generation"]


def adaylari_cek(bolge: str) -> tuple[pl.DataFrame, str]:
    """index.csv'yi akış olarak okur; aday satırları ve yayın tarihini döndürür."""
    with urllib.request.urlopen(URL.format(bolge=bolge), timeout=120) as yanit:
        akis = io.TextIOWrapper(yanit, encoding="utf-8")
        ust = {}
        for _ in range(5):                       # 5 satırlık üst bilgi (FormatVersion ... OfferCode)
            k, v = next(csv.reader([akis.readline()]))
            ust[k] = v
        satirlar = [{c: r[c] for c in TUT} for r in csv.DictReader(akis)
                    if ADAY.match(r["Instance Type"]) and all(r[k] == v for k, v in SUZGEC.items())]
    assert satirlar, "Aday satır bulunamadı (şema değişmiş olabilir)"
    return pl.DataFrame(satirlar), ust["Publication Date"]


def main() -> None:
    """Adayları kaydeder, ilp.yaml'daki tipleri doğrular ve server_types.yaml yazar."""
    cfg = yaml_oku("ilp")["sunucu"]
    df, yayin = adaylari_cek(cfg["bolge"])
    d = yol("data/raw/aws")
    d.mkdir(parents=True, exist_ok=True)
    df.write_csv(d / f"ec2_ondemand_{cfg['bolge']}_{yayin[:10]}.csv")
    log.info("%d aday satır kaydedildi (yayın: %s)", df.height, yayin)

    tipler = []
    for t in cfg["tipler"]:
        s = df.filter(pl.col("Instance Type") == t)
        assert s.height == 1, f"{t}: fiyat listesinde {s.height} satır (beklenen 1)"
        r = s.row(0, named=True)
        assert r["Currency"] == "USD"
        bellek = float(r["Memory"].replace(" GiB", ""))
        tipler.append({"ad": t, "aile": r["Instance Family"], "vcpu": int(r["vCPU"]), "bellek_gib": bellek,
                       "gb_per_vcpu": bellek / int(r["vCPU"]), "saatlik_usd": float(r["PricePerUnit"]),
                       "islemci": r["Physical Processor"], "sku": r["SKU"], "gecerlilik": r["EffectiveDate"]})
    nesiller = {re.sub(r"^[cmr]", "", t["ad"].split(".")[0]) for t in tipler}
    boyutlar = {t["ad"].split(".")[1] for t in tipler}
    assert len(nesiller) == 1 and len(boyutlar) == 1, "Bölüm 6.4: tipler aynı nesil ve aynı boyutta olmalı"
    cikti = {"kaynak": URL.format(bolge=cfg["bolge"]), "bolge": cfg["bolge"], "yayin_tarihi": yayin,
             "cekilme_tarihi": date.today().isoformat(), "fiyat": "on-demand, Linux, paylaşımlı, USD/saat",
             "tipler": tipler}
    with open(yol("configs/server_types.yaml"), "w", encoding="utf-8") as f:
        f.write("# OTOMATİK ÜRETİLDİ: src/ilp/fiyat_cek.py (elle düzenleme). MIMARI.md Bölüm 6.4.\n")
        yaml.safe_dump(cikti, f, allow_unicode=True, sort_keys=False)
    for t in tipler:
        log.info("%s: %d vCPU, %.0f GiB, %.4f USD/sa", t["ad"], t["vcpu"], t["bellek_gib"], t["saatlik_usd"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
