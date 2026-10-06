"""Tests de src/metrics.py (TODO 4.1 – 4.9).

Los valores esperados se calculan con el módulo `statistics` de Python o a mano,
de forma independiente de pandas.

Checkpoint:  python -m pytest tests/test_metrics.py
"""

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    DrawdownInfo,
    annual_turnover,
    annualized_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    hit_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from tests.helpers import bdays, series

R = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02]
SQRT252 = math.sqrt(252)


@pytest.fixture
def r():
    return series(R)


# ---------------------------------------------------------------------------
# TODO 4.1 · annualized_volatility
# ---------------------------------------------------------------------------
def test_annualized_volatility_known_value(r):
    assert annualized_volatility(r) == pytest.approx(statistics.stdev(R) * SQRT252)


def test_annualized_volatility_uses_sample_std():
    # stdev muestral de [0.01, -0.01] = 0.01414..., la poblacional sería 0.01
    assert annualized_volatility(series([0.01, -0.01])) == pytest.approx(math.sqrt(2) * 0.01 * SQRT252)


# ---------------------------------------------------------------------------
# TODO 4.2 · cagr
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_days, total_growth, expected",
    [(252, 1.10, 0.10), (504, 1.21, 0.10), (126, 1.05, 1.05**2 - 1)],
    ids=["1 año", "2 años", "medio año"],
)
def test_cagr_known_values(n_days, total_growth, expected):
    daily = total_growth ** (1 / n_days) - 1
    returns = pd.Series(daily, index=bdays(periods=n_days))
    assert cagr(returns) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TODO 4.3 · sharpe_ratio
# ---------------------------------------------------------------------------
def test_sharpe_ratio_known_value(r):
    assert sharpe_ratio(r) == pytest.approx(statistics.mean(R) / statistics.stdev(R) * SQRT252)


def test_sharpe_ratio_geometric_risk_free(r):
    rf = 0.05
    rf_daily = (1 + rf) ** (1 / 252) - 1
    expected = (statistics.mean(R) - rf_daily) / statistics.stdev(R) * SQRT252
    assert sharpe_ratio(r, rf=rf) == pytest.approx(expected, rel=1e-9)


def test_sharpe_ratio_constant_series_is_nan():
    assert np.isnan(sharpe_ratio(series([0.0] * 10)))


def test_sharpe_ratio_ignores_nan(r):
    with_nan = pd.concat([pd.Series([np.nan], index=bdays("2019-12-31", periods=1)), r])
    assert sharpe_ratio(with_nan) == pytest.approx(sharpe_ratio(r))


# ---------------------------------------------------------------------------
# TODO 4.4 · sortino_ratio
# ---------------------------------------------------------------------------
def test_sortino_ratio_known_value(r):
    downside = math.sqrt(sum(x * x for x in R if x < 0) / len(R))  # n = TODOS los días
    assert sortino_ratio(r) == pytest.approx(statistics.mean(R) / downside * SQRT252)


def test_sortino_ratio_with_risk_free(r):
    rf_daily = 1.03 ** (1 / 252) - 1
    excess = [x - rf_daily for x in R]
    downside = math.sqrt(sum(x * x for x in excess if x < 0) / len(excess))
    assert sortino_ratio(r, rf=0.03) == pytest.approx(statistics.mean(excess) / downside * SQRT252, rel=1e-9)


def test_sortino_ratio_without_losses_is_nan():
    assert np.isnan(sortino_ratio(series([0.01, 0.02, 0.0])))


# ---------------------------------------------------------------------------
# TODO 4.5 · drawdown_series
# ---------------------------------------------------------------------------
def test_drawdown_series_known_values():
    dd = drawdown_series(series([0.10, -0.10, 0.05, 0.10]))
    # equity: 1.1, 0.99, 1.0395, 1.14345   máximo previo: 1.1, 1.1, 1.1, 1.14345
    assert dd.tolist() == pytest.approx([0.0, -0.10, -0.055, 0.0])


def test_drawdown_series_initial_capital_is_a_peak():
    dd = drawdown_series(series([-0.05, 0.02]))
    assert dd.tolist() == pytest.approx([-0.05, 0.95 * 1.02 - 1])


def test_drawdown_series_non_positive_and_same_index(r):
    dd = drawdown_series(r)
    assert dd.index.equals(r.index)
    assert (dd <= 0).all()


# ---------------------------------------------------------------------------
# TODO 4.6 · max_drawdown
# ---------------------------------------------------------------------------
def test_max_drawdown_depth_and_dates():
    returns = series([0.10, -0.10, -0.10, 0.05, 0.30, -0.05])
    # equity: 1.1, 0.99, 0.891, 0.93555, 1.216215, 1.155404
    info = max_drawdown(returns)
    d = returns.index
    assert isinstance(info, DrawdownInfo)
    assert info.depth == pytest.approx(0.891 / 1.1 - 1)
    assert info.peak == d[0]
    assert info.trough == d[2]
    assert info.recovery == d[4]


def test_max_drawdown_from_initial_capital_has_no_peak_date():
    returns = series([-0.10, 0.05, 0.20])
    info = max_drawdown(returns)
    assert info.depth == pytest.approx(-0.10)
    assert info.peak is None, "La caída empieza desde el capital inicial: no hay fecha de pico."
    assert info.trough == returns.index[0]
    assert info.recovery == returns.index[2]


def test_max_drawdown_not_recovered():
    returns = series([0.10, -0.20, 0.05])
    info = max_drawdown(returns)
    assert info.depth == pytest.approx(-0.20)
    assert info.recovery is None


def test_max_drawdown_without_losses():
    info = max_drawdown(series([0.01, 0.02, 0.0]))
    assert info.depth == 0
    assert info.peak is None and info.trough is None and info.recovery is None


# ---------------------------------------------------------------------------
# TODO 4.7 · calmar_ratio
# ---------------------------------------------------------------------------
def test_calmar_ratio_known_value():
    values = [0.10, -0.10, -0.10, 0.05, 0.30, -0.05]
    growth = math.prod(1 + x for x in values)
    expected_cagr = growth ** (252 / len(values)) - 1
    assert calmar_ratio(series(values)) == pytest.approx(expected_cagr / (1 - 0.891 / 1.1))


def test_calmar_ratio_without_drawdown_is_nan():
    assert np.isnan(calmar_ratio(series([0.01, 0.02])))


# ---------------------------------------------------------------------------
# TODO 4.8 · hit_ratio
# ---------------------------------------------------------------------------
def test_hit_ratio_excludes_flat_days():
    assert hit_ratio(series([0.01, 0.0, -0.01, 0.02, 0.0])) == pytest.approx(2 / 3)


def test_hit_ratio_all_flat_is_nan():
    assert np.isnan(hit_ratio(series([0.0, 0.0])))


# ---------------------------------------------------------------------------
# TODO 4.9 · annual_turnover
# ---------------------------------------------------------------------------
def test_annual_turnover_known_value():
    turnover = pd.Series(0.0, index=bdays(periods=504))
    turnover.iloc[[0, 100, 200, 300]] = [1.0, 2.0, 3.0, 4.0]  # 10 en 2 años
    assert annual_turnover(turnover) == pytest.approx(5.0)
