"""Tests de la "fontanería": el código que ya viene hecho. Deben pasar desde el primer día.

Si alguno falla, el problema no está en tus TODO: revisa la instalación (requirements.txt).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import config, metrics, plotting
from src.backtest import _validate_inputs
from src.data import _extract_close, load_parquet, save_parquet
from src.metrics import DrawdownInfo, format_summary, summary_table, to_markdown_table
from src.strategies import STRATEGIES, _month_end_dates, buy_and_hold, get_strategy, strategy
from src.walkforward import GridSearchResult, Window, WalkForwardResult, expand_grid, heatmap_table
from tests.helpers import bdays, frame

TS = pd.Timestamp


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_config_values():
    assert len(config.TICKERS) == len(set(config.TICKERS)) == 10
    assert config.COMMISSION_BPS == 5 and config.SLIPPAGE_BPS == 5
    assert config.TRADING_DAYS == 252
    assert (config.WF_TRAIN_YEARS, config.WF_TEST_YEARS, config.WF_STEP_YEARS) == (3, 1, 1)
    assert config.PRICES_FILE.is_relative_to(config.ROOT)


def test_config_grids_refer_to_existing_strategies():
    for name in [*config.PARAM_GRIDS, *config.STRATEGY_DEFAULTS]:
        assert name in STRATEGIES


def test_config_sma_constraint_discards_fast_not_below_slow():
    grid = expand_grid(config.PARAM_GRIDS["sma_crossover"], config.PARAM_CONSTRAINTS["sma_crossover"])
    assert grid and all(p["fast"] < p["slow"] for p in grid)


# ---------------------------------------------------------------------------
# data: descarga y almacenamiento
# ---------------------------------------------------------------------------
def _fake_download(columns):
    idx = bdays(periods=3)
    return pd.DataFrame(np.arange(3 * len(columns), dtype=float).reshape(3, -1), index=idx, columns=columns)


def test_extract_close_field_ticker_multiindex():
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "QQQ"]])
    close = _extract_close(_fake_download(cols), ["SPY", "QQQ"])
    assert list(close.columns) == ["SPY", "QQQ"]


def test_extract_close_ticker_field_multiindex():
    cols = pd.MultiIndex.from_product([["SPY", "QQQ"], ["Open", "Close"]])
    close = _extract_close(_fake_download(cols), ["SPY", "QQQ"])
    assert list(close.columns) == ["SPY", "QQQ"]


def test_extract_close_flat_single_ticker():
    close = _extract_close(_fake_download(["Open", "Close"]), ["SPY"])
    assert list(close.columns) == ["SPY"]


def test_parquet_roundtrip(tmp_path, gbm_prices):
    path = save_parquet(gbm_prices, tmp_path / "sub" / "prices.parquet")
    assert_frame_equal(load_parquet(path), gbm_prices, check_freq=False)


def test_load_parquet_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.data"):
        load_parquet(tmp_path / "nada.parquet")


# ---------------------------------------------------------------------------
# strategies: decorador, registro, fines de mes
# ---------------------------------------------------------------------------
def test_registry_has_all_strategies():
    assert {"buy_and_hold", "sma_crossover", "momentum_12m", "bollinger_mean_reversion"} <= set(STRATEGIES)
    assert get_strategy("buy_and_hold") is buy_and_hold
    with pytest.raises(KeyError):
        get_strategy("no_existe")


def test_buy_and_hold_is_all_ones(gbm_prices):
    signals = buy_and_hold(gbm_prices)
    assert signals.shape == gbm_prices.shape
    assert (signals == 1.0).all().all()


@pytest.mark.parametrize(
    "bad_output, error",
    [
        (lambda p: p.iloc[1:] * 0, ValueError),                       # filas de menos
        (lambda p: p * np.nan, ValueError),                           # NaN
        (lambda p: p * 0 + 2, ValueError),                            # valores no válidos
        (lambda p: (p * 0).iloc[:, ::-1], ValueError),                # columnas en otro orden
        (lambda p: p.iloc[:, 0] * 0, TypeError),                      # Series en vez de DataFrame
    ],
    ids=["filas", "nan", "valores", "columnas", "tipo"],
)
def test_strategy_decorator_rejects_bad_signals(bad_output, error, gbm_prices):
    @strategy
    def broken(prices):
        return bad_output(prices)

    try:
        with pytest.raises(error):
            broken(gbm_prices)
    finally:
        STRATEGIES.pop("broken", None)


def test_strategy_decorator_casts_to_float(gbm_prices):
    @strategy
    def ints(prices):
        return pd.DataFrame(1, index=prices.index, columns=prices.columns)

    try:
        assert all(dtype == np.float64 for dtype in ints(gbm_prices).dtypes)
    finally:
        STRATEGIES.pop("ints", None)


def test_strategy_decorator_rejects_unsorted_prices(gbm_prices):
    with pytest.raises(ValueError):
        buy_and_hold(gbm_prices.iloc[::-1])


def test_month_end_dates():
    idx = bdays("2020-01-01", end="2020-03-17")
    ends = _month_end_dates(idx)
    assert list(ends) == [TS("2020-01-31"), TS("2020-02-28")], "Marzo no ha terminado: no cuenta."
    full = bdays("2020-01-01", end="2020-03-31")
    assert _month_end_dates(full)[-1] == TS("2020-03-31")


# ---------------------------------------------------------------------------
# backtest: validación de entradas
# ---------------------------------------------------------------------------
def test_validate_inputs_rejects_mismatches(gbm_prices):
    ones = pd.DataFrame(1.0, index=gbm_prices.index, columns=gbm_prices.columns)
    _validate_inputs(gbm_prices, ones)
    with pytest.raises(ValueError):
        _validate_inputs(gbm_prices, ones.iloc[1:])
    with pytest.raises(ValueError):
        _validate_inputs(gbm_prices, ones[ones.columns[::-1]])
    with_nan = ones.copy()
    with_nan.iloc[5, 0] = np.nan
    with pytest.raises(ValueError):
        _validate_inputs(gbm_prices, with_nan)
    with pytest.raises(TypeError):
        _validate_inputs(gbm_prices, ones.iloc[:, 0])


# ---------------------------------------------------------------------------
# walkforward: estructuras
# ---------------------------------------------------------------------------
def test_expand_grid_product_constraint_and_types():
    grid = expand_grid({"window": [10, 20], "n_std": [1.0, 2.0]})
    assert grid == [
        {"window": 10, "n_std": 1.0},
        {"window": 10, "n_std": 2.0},
        {"window": 20, "n_std": 1.0},
        {"window": 20, "n_std": 2.0},
    ]
    assert type(grid[0]["window"]) is int
    assert expand_grid({"a": [1, 2, 3]}, lambda p: p["a"] != 2) == [{"a": 1}, {"a": 3}]


def test_window_masks_are_half_open():
    w = Window(TS("2020-01-01"), TS("2020-01-03"), TS("2020-01-03"), TS("2020-01-07"))
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    assert w.train_mask(idx).tolist() == [True, True, False, False, False]
    assert w.test_mask(idx).tolist() == [False, False, True, True, False]


def _fake_search():
    table = pd.DataFrame(
        {"fast": [20, 10, 20, 10], "slow": [200, 200, 100, 100], "score": [0.9, 0.5, 0.1, np.nan]},
        index=pd.Index([3, 1, 2, 0], name="combo"),
    )
    return GridSearchResult(table=table, best_params={"fast": 20, "slow": 200}, best_score=0.9, metric="sharpe")


def test_heatmap_table():
    heat = heatmap_table(_fake_search(), "fast", "slow")
    assert list(heat.index) == [10, 20] and list(heat.columns) == [100, 200]
    assert heat.loc[20, 200] == 0.9
    assert np.isnan(heat.loc[10, 100])


def test_walkforward_selections_table():
    w = Window(TS("2010-01-01"), TS("2013-01-01"), TS("2013-01-01"), TS("2014-01-01"))
    result = WalkForwardResult(result=None, windows=[w], searches=[_fake_search()])
    row = result.selections.iloc[0]
    assert row["fast"] == 20 and row["slow"] == 200 and row["train_sharpe"] == 0.9


# ---------------------------------------------------------------------------
# metrics: tablas
# ---------------------------------------------------------------------------
def test_drawdown_info_helpers():
    info = DrawdownInfo(-0.2, TS("2020-01-01"), TS("2020-02-01"), TS("2020-03-01"))
    assert info.recovered and info.duration == pd.Timedelta(days=60)
    assert "-20.0%" in str(info)
    assert DrawdownInfo(-0.1, None, TS("2020-01-01"), None).duration is None


def test_summary_table_tolerates_pending_metrics(monkeypatch):
    def pending(*args, **kwargs):
        raise NotImplementedError("TODO 4.3 · sharpe_ratio")

    monkeypatch.setattr(metrics, "sharpe_ratio", pending)
    returns = pd.Series([0.01, -0.01, 0.02], index=bdays(periods=3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = summary_table({"x": returns})
    assert list(table.columns) == metrics.SUMMARY_COLUMNS
    assert np.isnan(table.loc["x", "Sharpe"])
    assert any("TODO 4.3" in str(w.message) for w in caught)


def test_format_summary_and_markdown():
    table = pd.DataFrame(
        [[0.0712, 0.153, 0.81, 1.2, -0.337, 0.21, 0.52, 3.04]],
        index=pd.Index(["Buy & Hold"], name="Estrategia"),
        columns=metrics.SUMMARY_COLUMNS,
    )
    formatted = format_summary(table)
    assert formatted.loc["Buy & Hold", "CAGR"] == "7.1%"
    assert formatted.loc["Buy & Hold", "Sharpe"] == "0.81"
    assert formatted.loc["Buy & Hold", "Turnover anual"] == "3.0×"
    md = to_markdown_table(table)
    assert md.splitlines()[0].startswith("| Estrategia | CAGR")
    assert "| Buy & Hold | 7.1% |" in md


# ---------------------------------------------------------------------------
# plotting (solo comprobamos que dibuja sin errores)
# ---------------------------------------------------------------------------
def test_plot_sharpe_heatmap_and_windows():
    heat = heatmap_table(_fake_search(), "fast", "slow")
    ax = plotting.plot_sharpe_heatmap(heat, best=(20, 200))
    assert ax.get_title(loc="left")
    w = [Window(TS("2010-01-01"), TS("2013-01-01"), TS("2013-01-01"), TS("2014-01-01"))]
    plotting.plot_wf_windows(w)
    plt.close("all")


def test_plot_cost_sensitivity():
    table = pd.DataFrame({"a": [1.0, 0.5, 0.0], "b": [0.8, 0.7, 0.6]}, index=[0, 10, 20])
    plotting.plot_cost_sensitivity(table)
    plt.close("all")


def test_colors_never_cycle():
    assert plotting.colors_for(["a", "b"]) == {"a": plotting.PALETTE[0], "b": plotting.PALETTE[1]}
    with pytest.raises(ValueError):
        plotting.colors_for([str(i) for i in range(9)])


def test_plot_param_stability():
    selections = pd.DataFrame(
        {"test_start": pd.to_datetime(["2013-01-01", "2014-01-01"]), "fast": [10, 20], "slow": [100, 200]}
    )
    plotting.plot_param_stability(selections, ["fast", "slow"])
    plt.close("all")
