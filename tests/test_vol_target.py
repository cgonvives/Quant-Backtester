"""Tests de la extensión de volatility targeting (TODO 6.1).

Solo se ejecutan los de la extensión:  python -m pytest -m extension
"""

import math
import statistics

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.backtest import apply_vol_target, run_backtest
from tests.helpers import bdays, gbm

pytestmark = pytest.mark.extension


def _returns_with_nan_first_row(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.to_numpy()
    values = np.vstack([np.full((1, p.shape[1]), np.nan), p[1:] / p[:-1] - 1])
    return pd.DataFrame(values, index=prices.index, columns=prices.columns)


@pytest.fixture
def setup():
    prices = gbm(n_days=2000, n_assets=2, sigma=0.30, seed=8)
    returns = _returns_with_nan_first_row(prices)
    positions = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    return prices, returns, positions


def test_apply_vol_target_hits_target(setup):
    _, returns, positions = setup
    scaled = apply_vol_target(positions, returns, target_vol=0.10, lookback=60, max_leverage=5.0)
    pnl = (scaled * returns).iloc[100:]
    realized = pnl.std() * np.sqrt(252)
    np.testing.assert_allclose(realized.to_numpy(), 0.10, rtol=0.10)


def test_apply_vol_target_exact_scale():
    """Escala exacta día a día: σ muestral de los `lookback` rendimientos ANTERIORES, sin desplazar las posiciones."""
    r = [np.nan, 0.01, -0.02, 0.015, 0.0, -0.01, 0.02, 0.005, -0.015, 0.01, 0.0, 0.02]
    idx = bdays(periods=len(r))
    returns = pd.DataFrame({"A": r}, index=idx)
    positions = pd.DataFrame({"A": [1.0, -1.0] * (len(r) // 2)}, index=idx)
    lookback = 5
    scaled = apply_vol_target(positions, returns, target_vol=0.10, lookback=lookback, max_leverage=100.0)
    for t in range(lookback + 1, len(r)):
        sigma = statistics.stdev(r[t - lookback : t]) * math.sqrt(252)  # r_{t-5} ... r_{t-1}
        expected = positions["A"].iloc[t] * 0.10 / sigma
        assert scaled["A"].iloc[t] == pytest.approx(expected, rel=1e-9), f"fila {t}"


def test_apply_vol_target_warmup_is_zero(setup):
    _, returns, positions = setup
    lookback = 60
    scaled = apply_vol_target(positions, returns, lookback=lookback)
    # Rendimientos válidos desde la fila 1 → primera σ con 60 datos en la fila 60 → se usa al día siguiente
    assert (scaled.iloc[: lookback + 1] == 0).all().all()
    assert (scaled.iloc[lookback + 1] > 0).all()


def test_apply_vol_target_uses_only_past_returns(setup):
    _, returns, positions = setup
    t = 500
    altered = returns.copy()
    altered.iloc[t] = altered.iloc[t] * 10
    base = apply_vol_target(positions, returns)
    moved = apply_vol_target(positions, altered)
    assert_frame_equal(moved.iloc[: t + 1], base.iloc[: t + 1], obj="la escala del día t no puede usar r_t")
    assert not moved.iloc[t + 1].equals(base.iloc[t + 1])


def test_apply_vol_target_caps_leverage():
    idx = bdays(periods=300)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"CALM": rng.normal(0, 0.0005, 300), "FLAT": 0.0}, index=idx)
    returns.iloc[0] = np.nan
    positions = pd.DataFrame(1.0, index=idx, columns=returns.columns)
    scaled = apply_vol_target(positions, returns, target_vol=0.10, lookback=20, max_leverage=2.0)
    assert (scaled.iloc[21:] == 2.0).all().all(), "Volatilidad muy baja (o nula) → escala = max_leverage."


def test_apply_vol_target_keeps_direction(setup):
    _, returns, positions = setup
    short = -positions
    scaled = apply_vol_target(short, returns)
    assert (scaled <= 0).all().all()


def test_run_backtest_with_vol_target(setup):
    prices, _, positions = setup
    result = run_backtest(prices, positions, vol_target=0.10)
    assert result.positions.iloc[100:].ne(1.0).any().any()
    assert result.equity.notna().all()
