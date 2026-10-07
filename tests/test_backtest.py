"""Tests del motor, src/backtest.py (TODO 2.1 – 2.6), incluidos los dos tests del README.

Checkpoint:  python -m pytest tests/test_backtest.py
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.backtest import (
    BacktestResult,
    asset_pnl,
    compute_costs,
    compute_turnover,
    equity_curve,
    run_backtest,
    signals_to_positions,
    to_portfolio,
)
from tests.helpers import frame, series

NAN = np.nan


# ---------------------------------------------------------------------------
# TODO 2.1 · signals_to_positions
# ---------------------------------------------------------------------------
def test_signals_to_positions_shifts_one_day():
    signals = frame({"A": [1, 1, 0, -1, -1], "B": [0, 1, 1, 1, 0]})
    expected = frame({"A": [0, 1, 1, 0, -1], "B": [0, 0, 1, 1, 1]})
    assert_frame_equal(signals_to_positions(signals), expected)


def test_signals_to_positions_lag_two():
    signals = frame({"A": [1, -1, 1, 1]})
    expected = frame({"A": [0, 0, 1, -1]})
    assert_frame_equal(signals_to_positions(signals, lag=2), expected)


def test_signals_to_positions_rejects_lag_zero():
    with pytest.raises(ValueError):
        signals_to_positions(frame({"A": [1, 0, 1]}), lag=0)


def test_signals_to_positions_never_uses_future_signals(gbm_prices):
    rng = np.random.default_rng(0)
    signals = pd.DataFrame(
        rng.choice([-1.0, 0.0, 1.0], size=gbm_prices.shape), index=gbm_prices.index, columns=gbm_prices.columns
    )
    k = 500
    altered = signals.copy()
    altered.iloc[k:] = -altered.iloc[k:]
    # Cambiar las señales desde el día k no puede cambiar ninguna posición hasta el día k (incluido)
    assert_frame_equal(signals_to_positions(altered).iloc[: k + 1], signals_to_positions(signals).iloc[: k + 1])


def test_signals_to_positions_no_nan_same_shape(gbm_prices):
    signals = pd.DataFrame(1.0, index=gbm_prices.index, columns=gbm_prices.columns)
    positions = signals_to_positions(signals)
    assert positions.shape == signals.shape
    assert positions.notna().all().all()


# ---------------------------------------------------------------------------
# TODO 2.2 · compute_turnover
# ---------------------------------------------------------------------------
def test_compute_turnover_known_values():
    positions = frame({"A": [1, 1, -1, 0, 0.5], "B": [0, 0, 0, 1, 1]})
    expected = frame({"A": [1, 0, 2, 1, 0.5], "B": [0, 0, 0, 1, 0]})
    assert_frame_equal(compute_turnover(positions), expected)


def test_compute_turnover_first_day_counts_as_entry():
    positions = frame({"A": [1, 1, 1]})
    assert compute_turnover(positions)["A"].tolist() == [1, 0, 0], (
        "Se parte de estar fuera (w_{-1} = 0): tener posición el primer día es haber operado."
    )


def test_compute_turnover_flip_counts_two():
    positions = frame({"A": [0, 1, -1]})
    assert compute_turnover(positions)["A"].iloc[2] == 2


def test_compute_turnover_no_nan(gbm_prices):
    positions = pd.DataFrame(1.0, index=gbm_prices.index, columns=gbm_prices.columns)
    assert compute_turnover(positions).notna().all().all()


# ---------------------------------------------------------------------------
# TODO 2.3 · compute_costs
# ---------------------------------------------------------------------------
def test_compute_costs_default_is_10_bps_per_unit_of_turnover():
    turnover = frame({"A": [1, 2, 0], "B": [0.5, 0, 1]})
    expected = frame({"A": [0.001, 0.002, 0.0], "B": [0.0005, 0.0, 0.001]})
    assert_frame_equal(compute_costs(turnover), expected)


def test_compute_costs_custom_bps():
    turnover = frame({"A": [1.0]})
    assert compute_costs(turnover, commission_bps=2, slippage_bps=3)["A"].iloc[0] == pytest.approx(0.0005)


def test_compute_costs_zero_bps():
    turnover = frame({"A": [1.0, 2.0]})
    assert (compute_costs(turnover, 0, 0) == 0).all().all()


# ---------------------------------------------------------------------------
# TODO 2.4 · asset_pnl
# ---------------------------------------------------------------------------
def test_asset_pnl_multiplies_without_shifting():
    positions = frame({"A": [0, 1, 1, -1]})
    returns = frame({"A": [NAN, 0.02, -0.01, 0.03]})
    expected = frame({"A": [0.0, 0.02, -0.01, -0.03]})
    assert_frame_equal(asset_pnl(positions, returns), expected)


def test_asset_pnl_nan_returns_count_as_zero():
    positions = frame({"A": [1, 1, 1], "B": [1, 1, 1]})
    returns = frame({"A": [NAN, NAN, 0.01], "B": [NAN, 0.02, NAN]})
    pnl = asset_pnl(positions, returns)
    assert pnl.notna().all().all()
    assert pnl["A"].tolist() == pytest.approx([0, 0, 0.01])
    assert pnl["B"].tolist() == pytest.approx([0, 0.02, 0])


# ---------------------------------------------------------------------------
# TODO 2.5 · to_portfolio
# ---------------------------------------------------------------------------
def test_to_portfolio_divides_by_total_number_of_assets():
    values = frame(
        {
            "A": [0.02, 0.0, NAN],
            "B": [0.0, 0.0, 0.03],
            "C": [0.01, 0.0, 0.0],
            "D": [0.0, 0.0, 0.0],
        }
    )
    out = to_portfolio(values)
    assert isinstance(out, pd.Series)
    assert out.tolist() == pytest.approx([0.03 / 4, 0.0, 0.03 / 4]), (
        "Se divide entre los 4 activos, aunque alguno esté fuera o tenga NaN."
    )


def test_to_portfolio_keeps_dates(gbm_prices):
    assert to_portfolio(gbm_prices).index.equals(gbm_prices.index)


# ---------------------------------------------------------------------------
# TODO 2.6 · equity_curve
# ---------------------------------------------------------------------------
def test_equity_curve_known_values():
    returns = series([0.10, -0.10, 0.0, 0.05])
    assert equity_curve(returns).tolist() == pytest.approx([1.1, 0.99, 0.99, 1.0395])


def test_equity_curve_initial_capital():
    returns = series([0.10, -0.10])
    assert equity_curve(returns, initial=100).tolist() == pytest.approx([110.0, 99.0])


def test_equity_curve_same_index():
    returns = series([0.01, 0.02, 0.03])
    assert equity_curve(returns).index.equals(returns.index)


# ---------------------------------------------------------------------------
# Motor completo (run_backtest). Necesita los TODO 1.1 y 2.1 – 2.6.
# ---------------------------------------------------------------------------
def _ones(prices, value=1.0):
    return pd.DataFrame(value, index=prices.index, columns=prices.columns)


def test_readme_constant_signal_one_is_buy_and_hold_minus_initial_cost(gbm_prices):
    """README: señal constante en 1 = buy-and-hold menos un coste inicial."""
    prices = gbm_prices[["A0"]]
    result = run_backtest(prices, _ones(prices))

    p = prices["A0"].to_numpy()
    first_return = p[1] / p[0] - 1
    # Día 0: aún no hay posición. Día 1: se entra (turnover 1 → coste 10 bps).
    assert result.returns.iloc[0] == 0
    assert result.costs.iloc[1] == pytest.approx(0.001)
    assert result.returns.iloc[1] == pytest.approx(first_return - 0.001)
    # Desde el día 2, nada de costes: el rendimiento es exactamente el del activo
    asset_returns = p[2:] / p[1:-1] - 1
    np.testing.assert_allclose(result.returns.iloc[2:].to_numpy(), asset_returns, rtol=1e-12)
    # Y la equity final es la del buy-and-hold, salvo el coste del primer día
    buy_and_hold = p[-1] / p[0]
    assert result.equity.iloc[-1] == pytest.approx(buy_and_hold * (1 + first_return - 0.001) / (1 + first_return))


def test_readme_constant_signal_zero_gives_zero_return(gbm_prices):
    """README: señal constante en 0 = rendimiento cero."""
    result = run_backtest(gbm_prices, _ones(gbm_prices, 0.0))
    assert (result.returns == 0).all()
    assert (result.costs == 0).all()
    assert (result.turnover == 0).all()
    assert (result.equity == 1).all()


def test_lookahead_oracle_cannot_win(gbm_prices):
    """Señal = signo del rendimiento de HOY (información legítima al cierre).

    Con el shift bien hecho, en un paseo aleatorio no da ventaja. Sin shift, la posición
    de hoy "conocería" el rendimiento de hoy y la equity explotaría.
    """
    p = gbm_prices.to_numpy()
    today = np.vstack([np.zeros((1, p.shape[1])), p[1:] / p[:-1] - 1])
    signals = pd.DataFrame(np.sign(today), index=gbm_prices.index, columns=gbm_prices.columns)
    result = run_backtest(gbm_prices, signals, commission_bps=0, slippage_bps=0)
    assert result.equity.iloc[-1] < 3, "Demasiado bueno para ser verdad: ¿falta el shift?"


def test_run_backtest_flip_costs(trend_prices):
    prices = trend_prices[["UP"]]
    signals = pd.DataFrame({"UP": [1.0, 1.0, -1.0, -1.0] + [0.0] * (len(prices) - 4)}, index=prices.index)
    result = run_backtest(prices, signals)
    # posiciones 0, 1, 1, -1, -1, 0 ...  → turnover 0, 1, 0, 2, 0, 1
    assert result.positions["UP"].iloc[:6].tolist() == [0, 1, 1, -1, -1, 0]
    assert result.turnover.iloc[:6].tolist() == pytest.approx([0, 1, 0, 2, 0, 1])
    assert result.costs.iloc[:6].tolist() == pytest.approx([0, 0.001, 0, 0.002, 0, 0.001])


def test_run_backtest_net_is_gross_minus_costs(gbm_prices):
    rng = np.random.default_rng(1)
    signals = pd.DataFrame(
        rng.choice([-1.0, 0.0, 1.0], size=gbm_prices.shape), index=gbm_prices.index, columns=gbm_prices.columns
    )
    result = run_backtest(gbm_prices, signals)
    assert isinstance(result, BacktestResult)
    assert_series_equal(result.returns, result.gross_returns - result.costs, check_names=False)
    assert (result.costs >= 0).all()
    assert result.costs.sum() > 0


def test_run_backtest_portfolio_of_identical_assets_equals_single_asset(gbm_prices):
    single = gbm_prices[["A0"]]
    double = pd.concat([single, single.rename(columns={"A0": "A0bis"})], axis=1)
    rng = np.random.default_rng(2)
    s = rng.choice([0.0, 1.0], size=len(single))
    r1 = run_backtest(single, pd.DataFrame({"A0": s}, index=single.index))
    r2 = run_backtest(double, pd.DataFrame({"A0": s, "A0bis": s}, index=single.index))
    assert_series_equal(r1.returns, r2.returns)


def test_backtest_result_slice_rebases_equity(gbm_prices):
    result = run_backtest(gbm_prices, _ones(gbm_prices))
    start = gbm_prices.index[300]
    part = result.slice(start, gbm_prices.index[600])
    assert part.returns.index[0] == start
    assert part.equity.iloc[0] == pytest.approx(1 + part.returns.iloc[0])
    assert len(part.positions) == 301


def test_backtest_result_cumulative_turnover(trend_prices):
    prices = trend_prices[["UP"]]
    signals = pd.DataFrame({"UP": [1.0, -1.0] + [-1.0] * (len(prices) - 2)}, index=prices.index)
    result = run_backtest(prices, signals)
    assert result.cumulative_turnover.iloc[-1] == pytest.approx(3.0)
