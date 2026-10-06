"""Tests de src/data.py: rendimientos (TODO 1.1, 1.2) y limpieza (TODO 1.3).

Checkpoint:  python -m pytest tests/test_data.py
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.data import clean_prices, log_returns, simple_returns
from tests.helpers import bdays, frame

NAN = np.nan


# ---------------------------------------------------------------------------
# TODO 1.1 · simple_returns
# ---------------------------------------------------------------------------
def test_simple_returns_known_values():
    prices = frame({"A": [100, 110, 99, 99], "B": [50, 25, 50, 55]})
    expected = frame({"A": [NAN, 0.10, -0.10, 0.0], "B": [NAN, -0.50, 1.00, 0.10]})
    assert_frame_equal(simple_returns(prices), expected)


def test_simple_returns_first_row_nan_same_shape(gbm_prices):
    r = simple_returns(gbm_prices)
    assert r.index.equals(gbm_prices.index)
    assert list(r.columns) == list(gbm_prices.columns)
    assert r.iloc[0].isna().all(), "La primera fila debe ser NaN: no hay día anterior."
    assert r.iloc[1:].notna().all().all()


def test_simple_returns_does_not_fill_gaps():
    prices = frame({"A": [100, 110, NAN, 121, 133.1]})
    r = simple_returns(prices)["A"]
    assert np.isnan(r.iloc[2]), "Sin precio hoy no hay rendimiento."
    assert np.isnan(r.iloc[3]), "Sin precio ayer no hay rendimiento (nada de rellenar en silencio)."
    assert r.iloc[4] == pytest.approx(0.10)


def test_simple_returns_works_with_series():
    s = pd.Series([100.0, 105.0, 102.9], index=bdays(periods=3))
    r = simple_returns(s)
    assert isinstance(r, pd.Series)
    assert r.iloc[1:].tolist() == pytest.approx([0.05, -0.02])


def test_simple_returns_does_not_modify_input(gbm_prices):
    before = gbm_prices.copy()
    simple_returns(gbm_prices)
    assert_frame_equal(gbm_prices, before)


# ---------------------------------------------------------------------------
# TODO 1.2 · log_returns
# ---------------------------------------------------------------------------
def test_log_returns_known_values():
    prices = frame({"A": [100, 110, 121, 60.5]})
    expected = frame({"A": [NAN, np.log(1.1), np.log(1.1), np.log(0.5)]})
    assert_frame_equal(log_returns(prices), expected)


def test_log_returns_sum_is_log_of_total_growth(gbm_prices):
    total = log_returns(gbm_prices).sum()
    expected = np.log(gbm_prices.iloc[-1] / gbm_prices.iloc[0])
    assert_series_equal(total, expected, check_names=False)


def test_log_returns_consistent_with_simple_returns(gbm_prices):
    # exp(ℓ) − 1 = r
    assert_frame_equal(np.expm1(log_returns(gbm_prices)), simple_returns(gbm_prices))


def test_log_returns_first_row_nan_and_no_fill():
    prices = frame({"A": [100, NAN, 110, 121]})
    r = log_returns(prices)["A"]
    assert r.iloc[:3].isna().all()
    assert r.iloc[3] == pytest.approx(np.log(1.1))


# ---------------------------------------------------------------------------
# TODO 1.3 · clean_prices
# ---------------------------------------------------------------------------
def test_clean_prices_drops_duplicates_keep_last_and_sorts():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-02"])
    prices = pd.DataFrame({"A": [3.0, 1.0, 2.0, 2.5]}, index=idx)
    out = clean_prices(prices)
    assert list(out.index) == list(pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert out["A"].tolist() == [1.0, 2.5, 3.0], "Con fechas repetidas se queda la ÚLTIMA aparición."


def test_clean_prices_non_positive_prices_are_gaps():
    prices = frame({"A": [10, 11, 0, 12, -1, 13], "B": [1, 2, 3, 4, 5, 6]})
    out = clean_prices(prices)
    assert out["A"].tolist() == [10, 11, 11, 12, 12, 13]


def test_clean_prices_ffill_respects_limit():
    prices = frame({"A": [10] + [NAN] * 7 + [20], "B": list(range(1, 10))})
    out = clean_prices(prices, max_ffill=5)
    a = out["A"]
    assert a.iloc[:6].tolist() == [10.0] * 6, "Los primeros 5 huecos se rellenan con el último precio."
    assert a.iloc[6:8].isna().all(), "A partir del 6.º día seguido el hueco se queda en NaN."
    assert a.iloc[8] == 20


def test_clean_prices_never_backfills_and_trims_to_common_start():
    prices = frame({"A": [NAN, NAN, 10, 11, 12], "B": [5, 6, 7, 8, 9]})
    out = clean_prices(prices)
    assert out.index[0] == prices.index[2], "Debe empezar el primer día en que TODOS tienen precio."
    assert out["A"].tolist() == [10, 11, 12]
    assert out["B"].tolist() == [7, 8, 9]


def test_clean_prices_drops_rows_where_all_assets_are_nan():
    prices = frame({"A": [10, NAN, 11], "B": [5, NAN, 6]})
    out = clean_prices(prices)
    assert len(out) == 2, "Una fila sin ningún precio no es un día de mercado: se elimina, no se rellena."
    assert prices.index[1] not in out.index


def test_clean_prices_returns_floats():
    prices = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=bdays(periods=3))
    out = clean_prices(prices)
    assert all(dtype == np.float64 for dtype in out.dtypes)


def test_clean_prices_does_not_modify_input():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"A": [1.0, -1.0, 2.0, NAN], "B": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    before = prices.copy()
    clean_prices(prices)
    assert_frame_equal(prices, before)


def test_clean_prices_clean_data_is_unchanged(gbm_prices):
    assert_frame_equal(clean_prices(gbm_prices), gbm_prices)
