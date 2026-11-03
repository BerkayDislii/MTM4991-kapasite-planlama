"""M8 rapor yardımcıları testleri: Türkçe sayı biçimi ve 'ort ± std' (MIMARI.md Bölüm 13)."""
import polars as pl

from src.rapor import ort_std, sayi


def test_sayi_ondalik_virgul() -> None:
    assert sayi(12.345, 1) == "12,3" and sayi(0.2, 3) == "0,200" and sayi(None, 2) == "—"
    assert sayi(float("nan"), 1) == "—"


def test_ort_std_tek_satirda_yalniz_ortalama() -> None:
    df = pl.DataFrame({"a_ort": [0.5, 0.25], "a_std": [0.1, None], "n_satir": [3, 1]})
    assert ort_std(df, "a", 1, 100) == ["50,0 ± 10,0", "25,0"]
