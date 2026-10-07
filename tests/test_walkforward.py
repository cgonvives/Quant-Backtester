"""Tests de src/walkforward.py (TODO 5.1 – 5.4).

Checkpoint:  python -m pytest tests/test_walkforward.py
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.metrics import sharpe_ratio
from src.strategies import bollinger_mean_reversion, sma_crossover
from src.walkforward import (
    GridSearchResult,
    WalkForwardResult,
    evaluate_params,
    expand_grid,
    generate_windows,
    grid_search,
    walk_forward,
)
from tests.helpers import bdays, constant_level

TS = pd.Timestamp


# ---------------------------------------------------------------------------
# TODO 5.1 · generate_windows
# ---------------------------------------------------------------------------
def test_generate_windows_count_and_first_last():
    index = bdays("2010-01-01", end="2020-12-31")
    windows = generate_windows(index, train_years=3, test_years=1, step_years=1)
    assert len(windows) == 8  # tramos de test: 2013, 2014, ..., 2020
    first, last = windows[0], windows[-1]
    assert (first.train_start, first.train_end) == (TS("2010-01-01"), TS("2013-01-01"))
    assert (first.test_start, first.test_end) == (TS("2013-01-01"), TS("2014-01-01"))
    assert (last.test_start, last.test_end) == (TS("2020-01-01"), TS("2021-01-01"))


def test_generate_windows_contiguous_and_disjoint():
    windows = generate_windows(bdays("2010-01-01", end="2020-12-31"))
    for w in windows:
        assert w.train_end == w.test_start, "El test empieza justo donde acaba el train (intervalos semiabiertos)."
    for prev, nxt in zip(windows, windows[1:]):
        assert prev.test_end == nxt.test_start, "Con step = test, los tramos de test se tocan sin solaparse."


def test_generate_windows_step_two_years():
    windows = generate_windows(bdays("2010-01-01", end="2020-12-31"), 3, 1, 2)
    assert [w.test_start.year for w in windows] == [2013, 2015, 2017, 2019]


def test_generate_windows_includes_partial_last_window():
    windows = generate_windows(bdays("2010-01-01", end="2020-06-30"))
    assert windows[-1].test_start == TS("2020-01-01")


def test_generate_windows_too_short_is_empty():
    assert generate_windows(bdays("2010-01-01", end="2012-06-30")) == []


def test_generate_windows_anchored_offsets():
    # Si se encadenan sumas sobre la ventana anterior, el 29-02 se "pierde" para siempre
    windows = generate_windows(bdays("2012-02-29", end="2020-12-31"), 1, 1, 1)
    assert windows[4].train_start == TS("2016-02-29")


def test_generate_windows_rejects_overlapping_tests():
    with pytest.raises(ValueError):
        generate_windows(bdays("2010-01-01", end="2020-12-31"), 3, 2, 1)


# ---------------------------------------------------------------------------
# TODO 5.2 · evaluate_params
# ---------------------------------------------------------------------------
START, END = TS("2013-01-01"), TS("2014-01-02")  # END es día hábil: comprueba que es exclusivo
PARAMS = {"fast": 20, "slow": 100}


def test_evaluate_params_uses_history_for_warmup_and_excludes_end(regime_prices):
    score = evaluate_params(regime_prices, sma_crossover, PARAMS, START, END, metric="sharpe")
    # Una estrategia causal da las mismas señales con todos los datos que con los anteriores a END,
    # así que la puntuación correcta es la del backtest completo, recortado a [START, END).
    full = run_backtest(regime_prices, sma_crossover(regime_prices, **PARAMS))
    idx = full.returns.index
    expected = sharpe_ratio(full.returns[(idx >= START) & (idx < END)])
    assert score == pytest.approx(expected, rel=1e-10)


def test_evaluate_params_never_sees_the_future(regime_prices):
    seen = []

    def spy(prices, **params):
        seen.append(prices.index.max())
        return sma_crossover(prices, **params)

    evaluate_params(regime_prices, spy, PARAMS, START, END)
    assert seen and max(seen) < END, "La estrategia ha recibido precios del día END o posteriores."


def test_evaluate_params_metric_choice(regime_prices):
    sharpe = evaluate_params(regime_prices, sma_crossover, PARAMS, START, END, metric="sharpe")
    cagr = evaluate_params(regime_prices, sma_crossover, PARAMS, START, END, metric="cagr")
    assert sharpe != cagr


# ---------------------------------------------------------------------------
# TODO 5.3 · grid_search
# ---------------------------------------------------------------------------
UP_START, UP_END = TS("2010-06-01"), TS("2012-06-01")  # tramo alcista en regime_prices


def test_grid_search_orders_best_first_nan_last(regime_prices):
    grid = [{"level": 0}, {"level": -1}, {"level": 1}]
    search = grid_search(regime_prices, constant_level, grid, UP_START, UP_END)
    assert isinstance(search, GridSearchResult)
    assert search.best_params == {"level": 1}
    assert list(search.table.index) == [2, 1, 0], "Índice = posición en la rejilla; orden de mejor a peor."
    assert np.isnan(search.table["score"].iloc[-1]), "Siempre fuera → Sharpe NaN → al final."
    assert search.best_score == pytest.approx(search.table["score"].iloc[0])
    assert search.metric == "sharpe"


def test_grid_search_table_columns(regime_prices):
    grid = expand_grid({"fast": [10, 20], "slow": [100, 150]})
    search = grid_search(regime_prices, sma_crossover, grid, START, END)
    assert list(search.table.columns) == ["fast", "slow", "score"]
    assert len(search.table) == 4


def test_grid_search_scores_match_evaluate_params(regime_prices):
    grid = expand_grid({"fast": [10, 20], "slow": [100, 150]})
    search = grid_search(regime_prices, sma_crossover, grid, START, END)
    for position, row in search.table.iterrows():
        expected = evaluate_params(regime_prices, sma_crossover, grid[position], START, END)
        assert row["score"] == pytest.approx(expected)


def test_grid_search_ties_keep_grid_order(regime_prices):
    # 40 combinaciones con solo dos puntuaciones distintas: una ordenación no estable las desordena
    grid = [{"level": level, "tag": k} for k, level in enumerate([1, -1] * 20)]
    search = grid_search(regime_prices, constant_level, grid, UP_START, UP_END)
    assert search.best_params["tag"] == 0
    assert list(search.table.index) == list(range(0, 40, 2)) + list(range(1, 40, 2)), (
        "Con empates, el orden de la rejilla se conserva (ordenación estable)."
    )


def test_grid_search_returns_original_dicts_with_original_types(ou_prices):
    grid = expand_grid({"window": [10, 20], "n_std": [1.0, 2.0]})
    search = grid_search(ou_prices, bollinger_mean_reversion, grid, TS("2011-01-01"), TS("2013-01-01"))
    assert search.best_params in grid
    assert type(search.best_params["window"]) is int, (
        "best_params debe ser el dict original: si lo reconstruyes desde una fila de la tabla, "
        "window pasa a ser float y rolling() falla."
    )


# ---------------------------------------------------------------------------
# TODO 5.4 · walk_forward
# ---------------------------------------------------------------------------
LEVELS = [{"level": 1}, {"level": -1}]


@pytest.fixture
def wf_levels(regime_prices):
    return walk_forward(regime_prices, constant_level, LEVELS)


def test_walk_forward_structure(wf_levels, regime_prices):
    assert isinstance(wf_levels, WalkForwardResult)
    assert len(wf_levels.windows) == 8
    assert len(wf_levels.searches) == 8
    returns = wf_levels.result.returns
    expected_index = regime_prices.index[regime_prices.index >= TS("2013-01-01")]
    assert returns.index.equals(expected_index), "Fuera de muestra = desde el primer test hasta el final, sin duplicados."


def test_walk_forward_selects_with_train_data(wf_levels, regime_prices):
    for window, search in zip(wf_levels.windows, wf_levels.searches):
        expected = grid_search(regime_prices, constant_level, LEVELS, window.train_start, window.train_end)
        assert search.best_params == expected.best_params


def test_walk_forward_applies_selected_params_in_each_test(wf_levels):
    positions = wf_levels.result.positions
    for window, search in zip(wf_levels.windows, wf_levels.searches):
        in_test = positions.loc[window.test_mask(positions.index)]
        assert (in_test == search.best_params["level"]).all().all()


def test_walk_forward_charges_costs_when_params_change(wf_levels):
    """Se encadenan POSICIONES: el cambio de parámetros entre ventanas paga costes."""
    levels = [s.best_params["level"] for s in wf_levels.searches]
    assert len(set(levels)) == 2, f"Este escenario debería alternar largo y corto: {levels}"
    turnover = wf_levels.result.turnover
    assert turnover.iloc[0] == pytest.approx(1.0), "El primer día fuera de muestra se entra desde 0."
    for i in range(1, len(levels)):
        window = wf_levels.windows[i]
        first_day = turnover.index[window.test_mask(turnover.index)][0]
        assert turnover.loc[first_day] == pytest.approx(abs(levels[i] - levels[i - 1])), (
            f"Turnover del {first_day:%Y-%m-%d} al pasar de {levels[i - 1]} a {levels[i]}."
        )


def test_walk_forward_selection_is_not_affected_by_future_data(regime_prices):
    altered = regime_prices.copy()
    cut = TS("2017-01-01")
    altered.loc[altered.index >= cut] = altered.loc[altered.index >= cut].iloc[::-1].to_numpy()
    grid = expand_grid({"fast": [10, 50], "slow": [100, 200]})
    original = walk_forward(regime_prices, sma_crossover, grid)
    perturbed = walk_forward(altered, sma_crossover, grid)
    for w, a, b in zip(original.windows, original.searches, perturbed.searches):
        if w.train_end <= cut:
            assert a.best_params == b.best_params
    before = original.result.returns.index < cut
    np.testing.assert_allclose(
        original.result.returns[before].to_numpy(), perturbed.result.returns[before].to_numpy()
    )


def test_walk_forward_with_real_strategy_runs(regime_prices):
    grid = expand_grid({"fast": [10, 20, 50], "slow": [100, 200]}, lambda p: p["fast"] < p["slow"])
    result = walk_forward(regime_prices, sma_crossover, grid)
    selections = result.selections
    assert len(selections) == len(result.windows)
    assert {"fast", "slow", "train_sharpe"} <= set(selections.columns)
    assert result.result.equity.notna().all()
