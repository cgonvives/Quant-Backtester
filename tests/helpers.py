"""Datos sintéticos (sin red y reproducibles) y utilidades para los tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def bdays(start: str = "2010-01-01", periods: int | None = None, end: str | None = None) -> pd.DatetimeIndex:
    """Días hábiles SIN `freq`, igual que el índice que devuelve yfinance."""
    idx = pd.bdate_range(start=start, end=end, periods=periods)
    return pd.DatetimeIndex(idx.to_numpy(), name="Date")


def frame(columns: dict[str, list], start: str = "2020-01-01") -> pd.DataFrame:
    """DataFrame de floats con índice de días hábiles a partir de `start`."""
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=bdays(start, n), dtype=float)


def series(values: list, start: str = "2020-01-01", name: str | None = None) -> pd.Series:
    return pd.Series(values, index=bdays(start, len(values)), dtype=float, name=name)


def gbm(
    n_days: int = 1260,
    n_assets: int = 3,
    mu: float = 0.08,
    sigma: float = 0.20,
    start: str = "2010-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Movimiento browniano geométrico: rendimientos independientes, sin memoria."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(n_days, n_assets))
    shocks[0] = 0.0
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=bdays(start, n_days), columns=[f"A{i}" for i in range(n_assets)])


def ou(
    n_days: int = 1500,
    n_assets: int = 2,
    theta: float = 0.10,
    sigma: float = 0.015,
    start: str = "2010-01-01",
    seed: int = 7,
) -> pd.DataFrame:
    """Ornstein-Uhlenbeck en log-precio: el precio vuelve a su media (ideal para Bollinger)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_days, n_assets))
    eps = rng.standard_normal((n_days, n_assets))
    for t in range(1, n_days):
        x[t] = (1 - theta) * x[t - 1] + sigma * eps[t]
    return pd.DataFrame(100 * np.exp(x), index=bdays(start, n_days), columns=[f"OU{i}" for i in range(n_assets)])


def trend(n_days: int = 300, slope: float = 0.5, start: str = "2010-01-01") -> pd.DataFrame:
    """Dos activos deterministas: uno sube en línea recta y el otro baja."""
    t = np.arange(n_days, dtype=float)
    return pd.DataFrame(
        {"UP": 100 + slope * t, "DOWN": 300 - slope * t}, index=bdays(start, n_days)
    )


# Signo de la tendencia de cada año en regimes(): +1 alcista, -1 bajista
REGIME_SIGNS = {
    2010: 1, 2011: 1, 2012: 1, 2013: -1, 2014: -1, 2015: -1,
    2016: -1, 2017: 1, 2018: 1, 2019: 1, 2020: 1,
}


def regimes(n_assets: int = 2, drift: float = 0.25, sigma: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """2010-2020 con un régimen por año natural (alcista o bajista), según REGIME_SIGNS."""
    idx = bdays("2010-01-01", end="2020-12-31")
    signs = np.array([REGIME_SIGNS[d.year] for d in idx], dtype=float)
    rng = np.random.default_rng(seed)
    shocks = signs[:, None] * drift / TRADING_DAYS + sigma / np.sqrt(TRADING_DAYS) * rng.standard_normal(
        (len(idx), n_assets)
    )
    shocks[0] = 0.0
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"R{i}" for i in range(n_assets)])


def bollinger_reference(close: pd.Series, window: int, n_std: float, allow_short: bool = False) -> pd.Series:
    """Bollinger con estado, escrito como un bucle día a día (lento pero obvio).

    Sirve de referencia para comprobar la versión vectorizada.
    """
    values = close.to_numpy(dtype=float)
    long_state, short_state = 0.0, 0.0
    out = np.zeros(len(values))
    for t in range(len(values)):
        if t >= window - 1:
            segment = values[t - window + 1 : t + 1]
            mid = segment.mean()
            std = segment.std(ddof=1)
            lower, upper = mid - n_std * std, mid + n_std * std
            price = values[t]
            if price < lower:
                long_state = 1.0
            elif price >= mid:
                long_state = 0.0
            if allow_short:
                if price > upper:
                    short_state = -1.0
                elif price <= mid:
                    short_state = 0.0
        out[t] = long_state + short_state
    return pd.Series(out, index=close.index, name=close.name)


def constant_level(prices: pd.DataFrame, level: float = 1.0, tag: str | None = None) -> pd.DataFrame:
    """Estrategia de juguete: siempre la misma señal. `tag` solo sirve para crear empates."""
    return pd.DataFrame(float(level), index=prices.index, columns=prices.columns)
