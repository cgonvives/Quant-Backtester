"""Verificación cruzada de las métricas propias contra quantstats (README, día 2).

Se salta entero si quantstats no está instalado.

Checkpoint:  python -m pytest tests/test_quantstats_check.py
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import compare_with_quantstats
from tests.helpers import bdays

pytest.importorskip("quantstats")


@pytest.fixture
def returns():
    rng = np.random.default_rng(5)
    values = rng.normal(0.0004, 0.012, size=1000)
    return pd.Series(values, index=bdays(periods=1000))


@pytest.mark.parametrize("rf", [0.0, 0.03])
def test_compare_with_quantstats_matches(returns, rf):
    table = compare_with_quantstats(returns, rf=rf)
    assert table["diferencia"].abs().max() < 1e-10, f"\n{table}"


def test_compare_with_quantstats_first_day_loss():
    # Caso límite del drawdown: se pierde desde el primer día (el capital inicial es el pico)
    returns = pd.Series([-0.05, 0.01, -0.02, 0.03, 0.01], index=bdays(periods=5))
    table = compare_with_quantstats(returns)
    assert table["diferencia"].abs().max() < 1e-10, f"\n{table}"


def test_compare_with_quantstats_on_backtest(gbm_prices):
    from src.backtest import run_backtest
    from src.strategies import sma_crossover

    result = run_backtest(gbm_prices, sma_crossover(gbm_prices, fast=10, slow=50))
    table = compare_with_quantstats(result)
    assert table["diferencia"].abs().max() < 1e-10, f"\n{table}"
