"""M0: Donanım/ortam doğrulaması (MIMARI.md Bölüm 1 ve Bölüm 9).

Çalıştırma: .venv/bin/python -m pytest tests/test_m0_env.py -v -s
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Testlerde rastgelelik sabit tohumla üretilir
TOHUM: int = 42


def test_cuda_kullanilabilir() -> None:
    """PyTorch GPU'yu görüyor mu ve AMP ile basit bir işlem yapabiliyor mu?"""
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() False döndü"
    cihaz = torch.device("cuda")
    ozellik = torch.cuda.get_device_properties(cihaz)
    print(f"\nGPU: {ozellik.name}, VRAM: {ozellik.total_memory / 1024**3:.2f} GiB, "
          f"torch {torch.__version__}, CUDA {torch.version.cuda}")

    # AMP (torch.autocast) altında küçük bir matris çarpımı
    torch.manual_seed(TOHUM)
    a = torch.randn(256, 256, device=cihaz)
    b = torch.randn(256, 256, device=cihaz)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        c = a @ b
    assert c.dtype == torch.float16
    assert torch.isfinite(c).all()


def test_xgboost_gpu() -> None:
    """XGBoost device='cuda', tree_method='hist' ile eğitilebiliyor mu?"""
    import xgboost as xgb

    rng = np.random.default_rng(TOHUM)
    X = rng.normal(size=(500, 5))
    y = X @ np.arange(1, 6) + rng.normal(scale=0.1, size=500)
    model = xgb.XGBRegressor(n_estimators=20, device="cuda", tree_method="hist",
                             random_state=TOHUM)
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_ortools_scip_backend() -> None:
    """pywraplp SCIP çözücüsü mevcut mu ve bilinen optimumu buluyor mu?

    max 3x + 2y,  x + y <= 4,  x + 3y <= 6,  x <= 3,  x, y tamsayı >= 0
    Elle hesaplanan optimum: x=3, y=1, amaç=11.
    """
    from ortools.linear_solver import pywraplp

    cozucu = pywraplp.Solver.CreateSolver("SCIP")
    assert cozucu is not None, "SCIP backend bulunamadı"
    x = cozucu.IntVar(0, 3, "x")
    y = cozucu.IntVar(0, cozucu.infinity(), "y")
    cozucu.Add(x + y <= 4)
    cozucu.Add(x + 3 * y <= 6)
    cozucu.Maximize(3 * x + 2 * y)
    durum = cozucu.Solve()
    assert durum == pywraplp.Solver.OPTIMAL
    assert cozucu.Objective().Value() == pytest.approx(11.0)
    assert (x.solution_value(), y.solution_value()) == (pytest.approx(3), pytest.approx(1))
    print(f"\nSCIP sürümü: {cozucu.SolverVersion()}")


def test_ortools_cpsat() -> None:
    """CP-SAT (doğrulama çözücüsü) aynı örnekte aynı optimumu veriyor mu?"""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x = model.NewIntVar(0, 3, "x")
    y = model.NewIntVar(0, 10, "y")
    model.Add(x + y <= 4)
    model.Add(x + 3 * y <= 6)
    model.Maximize(3 * x + 2 * y)
    cozucu = cp_model.CpSolver()
    assert cozucu.Solve(model) == cp_model.OPTIMAL
    assert cozucu.ObjectiveValue() == pytest.approx(11.0)


def test_statsmodels_sarimax() -> None:
    """Mevsim=24 SARIMAX küçük sentetik seride kurulup tahmin üretiyor mu?

    Ayrıca walk-forward için gereken append(refit=False) API'si doğrulanır (Bölüm 5.1).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(TOHUM)
    n = 24 * 10  # 10 günlük saatlik seri
    t = np.arange(n)
    seri = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=n)
    egitim, yeni = seri[: 24 * 9], seri[24 * 9:]

    sonuc = SARIMAX(egitim, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24),
                    trend="c").fit(disp=False)
    tahmin = sonuc.forecast(steps=24)
    assert tahmin.shape == (24,)
    assert np.isfinite(tahmin).all()

    # Parametreleri sabit tutup yeni gözlemlerle durumu güncelle
    guncel = sonuc.append(yeni, refit=False)
    np.testing.assert_allclose(guncel.params, sonuc.params)
    assert guncel.forecast(steps=24).shape == (24,)
    mae = np.mean(np.abs(tahmin - yeni))
    print(f"\nSARIMAX 24 saatlik tahmin MAE (sentetik): {mae:.3f}")


def test_parquet_okuma_yazma(tmp_path: Path) -> None:
    """pyarrow ile yazılan Parquet pandas ve polars ile aynı okunuyor mu?"""
    import polars as pl

    df = pd.DataFrame({
        "slot": np.arange(24, dtype=np.int64),
        "cpu_peak": np.linspace(1.0, 2.0, 24),
        "mem_peak": np.linspace(3.0, 4.0, 24),
    })
    yol = tmp_path / "pool_test.parquet"
    df.to_parquet(yol, engine="pyarrow", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(yol), df)
    pd.testing.assert_frame_equal(pl.read_parquet(yol).to_pandas(), df)
