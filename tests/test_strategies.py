"""Tests de src/strategies.py (TODO 3.1 – 3.3).

Checkpoint:  python -m pytest tests/test_strategies.py
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.strategies import _month_end_dates, bollinger_mean_reversion, momentum_12m, sma_crossover
from tests.helpers import bdays, bollinger_reference, frame, gbm

CASES = [
    pytest.param(sma_crossover, {"fast": 5, "slow": 20}, id="sma_crossover"),
    pytest.param(sma_crossover, {"fast": 5, "slow": 20, "allow_short": True}, id="sma_crossover-short"),
    pytest.param(momentum_12m, {}, id="momentum_12m"),
    pytest.param(momentum_12m, {"skip_months": 1, "allow_short": True}, id="momentum_12m-short"),
    pytest.param(bollinger_mean_reversion, {"window": 10, "n_std": 1.5}, id="bollinger_mean_reversion"),
    pytest.param(
        bollinger_mean_reversion, {"window": 10, "n_std": 1.5, "allow_short": True}, id="bollinger_mean_reversion-short"
    ),
]


@pytest.fixture(scope="module")
def long_prices():
    """4 años: suficiente historia para el momentum de 12 meses."""
    return gbm(n_days=1050, n_assets=3, seed=11)


# ---------------------------------------------------------------------------
# Propiedades comunes: formato y causalidad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy, params", CASES)
def test_strategy_output_format(strategy, params, long_prices):
    signals = strategy(long_prices, **params)  # el decorador @strategy ya valida forma, NaN y valores
    assert set(np.unique(signals.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert (signals != 0).any().any(), "La estrategia no opera nunca: sospechoso."


@pytest.mark.parametrize("strategy, params", CASES)
def test_strategy_is_causal_truncating_the_future(strategy, params, long_prices):
    full = strategy(long_prices, **params)
    for k in (400, 523, 777):
        part = strategy(long_prices.iloc[:k], **params)
        assert_frame_equal(part, full.iloc[:k], obj=f"señales con datos hasta la fila {k}")


@pytest.mark.parametrize("strategy, params", CASES)
def test_strategy_is_causal_perturbing_the_future(strategy, params, long_prices):
    k = 600
    altered = long_prices.copy()
    altered.iloc[k:] = altered.iloc[k:] * np.random.default_rng(3).uniform(0.5, 1.5, size=altered.iloc[k:].shape)
    assert_frame_equal(
        strategy(altered, **params).iloc[:k],
        strategy(long_prices, **params).iloc[:k],
        obj="cambiar precios FUTUROS no debe cambiar señales PASADAS",
    )


# ---------------------------------------------------------------------------
# TODO 3.1 · sma_crossover
# ---------------------------------------------------------------------------
def test_sma_crossover_hand_example():
    prices = frame({"A": [1, 2, 3, 2, 1, 1, 2, 3]})
    # media(2): -   1.5  2.5  2.5  1.5  1.0  1.5  2.5
    # media(3): -   -    2.0  2.33 2.0  1.33 1.33 2.0
    long_only = sma_crossover(prices, fast=2, slow=3)
    assert long_only["A"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    long_short = sma_crossover(prices, fast=2, slow=3, allow_short=True)
    assert long_short["A"].tolist() == [0, 0, 1, 1, -1, -1, 1, 1]


def test_sma_crossover_trend_starts_when_slow_average_exists(trend_prices):
    slow = 50
    signals = sma_crossover(trend_prices, fast=10, slow=slow, allow_short=True)
    up, down = signals["UP"], signals["DOWN"]
    assert (up.iloc[: slow - 1] == 0).all(), "Sin `slow` precios no hay media lenta: señal 0 (min_periods)."
    assert (up.iloc[slow - 1 :] == 1).all(), "La señal empieza el mismo día en que existe la media lenta (sin shift)."
    assert (down.iloc[slow - 1 :] == -1).all()


def test_sma_crossover_long_only_never_short(trend_prices):
    signals = sma_crossover(trend_prices, fast=10, slow=50)
    assert (signals["DOWN"] == 0).all()


def test_sma_crossover_rejects_fast_not_below_slow(trend_prices):
    with pytest.raises(ValueError):
        sma_crossover(trend_prices, fast=50, slow=50)


# ---------------------------------------------------------------------------
# TODO 3.2 · momentum_12m
# ---------------------------------------------------------------------------
@pytest.fixture
def rise_then_crash():
    """Sube en línea recta hasta el 31-01-2011 y desde el 1-02-2011 se desploma a 50."""
    idx = bdays("2010-01-01", end="2011-12-31")
    price = np.where(idx <= "2011-01-31", 100 + 0.1 * np.arange(len(idx)), 50.0)
    return pd.DataFrame({"A": price}, index=idx)


def test_momentum_12m_changes_only_at_month_ends(long_prices):
    signals = momentum_12m(long_prices, allow_short=True)
    changes = signals.index[(signals.diff().fillna(0) != 0).any(axis=1)]
    month_ends = set(_month_end_dates(long_prices.index))
    assert len(changes) > 0
    assert set(changes) <= month_ends, "La señal solo puede cambiar en un fin de mes (rebalanceo mensual)."


def test_momentum_12m_warmup_and_first_signal(rise_then_crash):
    signals = momentum_12m(rise_then_crash)["A"]
    first = pd.Timestamp("2011-01-31")  # 13.º fin de mes: el primero con 12 meses de historia
    assert (signals.loc[: first - pd.Timedelta(days=1)] == 0).all()
    assert (signals.loc[first:"2011-02-25"] == 1).all()


def test_momentum_12m_after_crash(rise_then_crash):
    signals = momentum_12m(rise_then_crash, allow_short=True)["A"]
    # El 28-02-2011 el precio (50) está por debajo del de hace 12 meses → corto
    assert (signals.loc["2011-02-28":] == -1).all()
    assert (momentum_12m(rise_then_crash)["A"].loc["2011-02-28":] == 0).all()


def test_momentum_12m_skip_months(rise_then_crash):
    signals = momentum_12m(rise_then_crash, skip_months=1)["A"]
    # 12-1: el 28-02-2011 mira P(ene-2011)/P(feb-2010) − 1 > 0 → aún largo
    assert (signals.loc["2011-01-31":"2011-03-30"] == 1).all()
    # El 31-03-2011 ya mira P(feb-2011) = 50 → fuera
    assert (signals.loc["2011-03-31":] == 0).all()


# ---------------------------------------------------------------------------
# TODO 3.3 · bollinger_mean_reversion
# ---------------------------------------------------------------------------
HAND_CLOSE = [10, 10, 11, 10, 7, 8, 8.6, 11, 12, 10.5, 10, 14, 13, 11, 10]
# window=3, n_std=1. Fila 4: entra largo (7 < 7.25). Fila 5: SIGUE largo (8 no está bajo la banda,
# pero tampoco ha llegado a la media 8.33). Fila 6: sale (8.6 >= 7.87). Ver docs/03 para la tabla completa.
HAND_LONG = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
HAND_LONG_SHORT = [0, 0, -1, 0, 1, 1, 0, -1, -1, 0, 0, -1, -1, 1, 1]


def test_bollinger_mean_reversion_hand_example():
    prices = frame({"A": HAND_CLOSE})
    signals = bollinger_mean_reversion(prices, window=3, n_std=1.0)
    assert signals["A"].tolist() == HAND_LONG


def test_bollinger_mean_reversion_hand_example_with_shorts():
    prices = frame({"A": HAND_CLOSE})
    signals = bollinger_mean_reversion(prices, window=3, n_std=1.0, allow_short=True)
    assert signals["A"].tolist() == HAND_LONG_SHORT


@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("window, n_std", [(20, 2.0), (10, 1.0)])
def test_bollinger_mean_reversion_matches_day_by_day_reference(ou_prices, window, n_std, allow_short):
    signals = bollinger_mean_reversion(ou_prices, window=window, n_std=n_std, allow_short=allow_short)
    for col in ou_prices.columns:
        expected = bollinger_reference(ou_prices[col], window, n_std, allow_short)
        assert_series_equal(signals[col], expected, check_names=False)


def test_bollinger_mean_reversion_is_stateful(ou_prices):
    window, n_std = 20, 2.0
    signals = bollinger_mean_reversion(ou_prices, window=window, n_std=n_std)
    mid = ou_prices.rolling(window).mean()
    lower = mid - n_std * ou_prices.rolling(window).std()
    holding = (signals == 1) & (ou_prices >= lower)
    assert holding.to_numpy().any(), (
        "Nunca se mantiene una posición por encima de la banda inferior: "
        "¿la señal es solo 'close < banda inferior'? Falta el estado."
    )
