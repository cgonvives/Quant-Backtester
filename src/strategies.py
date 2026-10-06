"""Estrategias: funciones  precios → señales.

Formato común (lo hace cumplir el decorador @strategy):
- Entrada: DataFrame de precios de cierre limpios (índice = fechas, columnas = activos)
  y los parámetros como argumentos con nombre.
- Salida: DataFrame con EXACTAMENTE el mismo índice y columnas, valores -1, 0 o 1, sin NaN.
  1 = largo, -1 = corto, 0 = fuera.
- La señal del día t solo puede usar precios hasta t (incluido).
- Una estrategia NUNCA hace shift: el desfase de un día lo pone el motor
  (backtest.signals_to_positions). Así el motor es agnóstico y el shift vive en un único sitio.
"""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np
import pandas as pd

StrategyFn = Callable[..., pd.DataFrame]

# Registro nombre → estrategia, para recorrerlas en los notebooks y en el walk-forward
STRATEGIES: dict[str, StrategyFn] = {}

_VALID_SIGNALS = {-1.0, 0.0, 1.0}


# ---------------------------------------------------------------------------
# Decorador y utilidades (hecho)
# ---------------------------------------------------------------------------
def strategy(func: StrategyFn) -> StrategyFn:
    """Registra una estrategia y valida su entrada y su salida.

    Si tu estrategia devuelve algo con otra forma, con NaN o con valores raros, el
    decorador lanza un error que explica qué falla, en vez de dejar que el motor
    calcule algo sin sentido.
    """

    @functools.wraps(func)
    def wrapper(prices: pd.DataFrame, *args, **params) -> pd.DataFrame:
        _check_prices(prices, func.__name__)
        signals = func(prices, *args, **params)
        return _check_signals(signals, prices, func.__name__)

    STRATEGIES[func.__name__] = wrapper
    return wrapper


def _check_prices(prices: pd.DataFrame, name: str) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"{name}: los precios deben ser un DataFrame, no {type(prices).__name__}.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: el índice de los precios debe ser un DatetimeIndex.")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError(f"{name}: las fechas deben estar ordenadas y sin repetir (usa clean_prices).")
    if prices.empty:
        raise ValueError(f"{name}: la tabla de precios está vacía.")


def _check_signals(signals, prices: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(signals, pd.DataFrame):
        raise TypeError(
            f"{name} debe devolver un DataFrame (fechas × activos), no {type(signals).__name__}."
        )
    if not signals.index.equals(prices.index):
        raise ValueError(
            f"{name} debe devolver las mismas fechas que los precios "
            f"({len(signals)} filas frente a {len(prices)}). ¿Has hecho dropna() o un resample?"
        )
    if list(signals.columns) != list(prices.columns):
        raise ValueError(f"{name} debe devolver las mismas columnas que los precios y en el mismo orden.")
    values = signals.to_numpy(dtype=float)
    n_nan = int(np.isnan(values).sum())
    if n_nan:
        raise ValueError(
            f"{name} devolvió {n_nan} NaN. Durante el calentamiento (cuando aún no hay "
            "datos suficientes) la señal debe ser 0."
        )
    bad = set(np.unique(values)) - _VALID_SIGNALS
    if bad:
        raise ValueError(f"{name} devolvió valores no válidos {sorted(bad)}: solo se admiten -1, 0 y 1.")
    return signals.astype(float)


def _month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Último día con datos de cada mes, dentro de `index`.

    El último día del índice solo cuenta como fin de mes si es el último día hábil del mes
    (si los datos acaban a mitad de mes, ese mes aún no ha cerrado).
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return index
    months = index.to_period("M")
    is_month_end = np.append(months[1:] != months[:-1], False)
    if pd.offsets.BMonthEnd().is_on_offset(index[-1]):
        is_month_end[-1] = True
    return index[is_month_end]


def get_strategy(name: str) -> StrategyFn:
    """Devuelve una estrategia del registro por su nombre."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"No hay ninguna estrategia '{name}'. Disponibles: {sorted(STRATEGIES)}") from None


# ---------------------------------------------------------------------------
# Ejemplo resuelto del formato
# ---------------------------------------------------------------------------
@strategy
def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """Siempre largo en todos los activos: la referencia contra la que comparar."""
    return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


# ---------------------------------------------------------------------------
# Momentum (TODO 3.1 y 3.2)
# ---------------------------------------------------------------------------
@strategy
def sma_crossover(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200, allow_short: bool = False
) -> pd.DataFrame:
    """Cruce de medias móviles simples.

    - Media rápida y lenta con rolling(n, min_periods=n): sin n precios, la media es NaN.
    - Señal 1 si media rápida > media lenta; si no, 0.
    - Con allow_short=True: -1 si media rápida < media lenta.
    - Mientras alguna de las medias sea NaN (calentamiento), la señal es 0.
    """
    if not 0 < fast < slow:
        raise ValueError(f"Se necesita 0 < fast < slow (recibido fast={fast}, slow={slow}).")
    # TODO 3.1 · sma_crossover
    # Objetivo: dos medias móviles por columna y una comparación entre ellas.
    # Pista: comparar con NaN da False, y (False).astype(float) da 0.0.
    raise NotImplementedError("TODO 3.1 · sma_crossover — ver docs/03_Estrategias.pdf")


@strategy
def momentum_12m(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Momentum de serie temporal con rebalanceo mensual.

    - La señal solo se calcula en los fines de mes (_month_end_dates).
    - En cada fin de mes m:  mom = P(m − skip_months) / P(m − lookback_months) − 1,
      contando en fines de mes. Con skip_months=1 se ignora el último mes (momentum 12-1).
    - Señal 1 si mom > 0; si no, 0. Con allow_short=True: -1 si mom < 0.
    - Entre dos fines de mes se mantiene la última señal. Antes del primer fin de mes
      con historia suficiente, 0.
    """
    if not 0 <= skip_months < lookback_months:
        raise ValueError(
            f"Se necesita 0 <= skip_months < lookback_months "
            f"(recibido {skip_months} y {lookback_months})."
        )
    # TODO 3.2 · momentum_12m
    # Objetivo: señales en fechas de fin de mes y luego "estirarlas" a todos los días.
    # Pasos: 1) precios de fin de mes con prices.loc[_month_end_dates(prices.index)]
    #        2) shift() en esa tabla mensual = desplazarse meses, no días
    #        3) reindex al índice diario + ffill + 0 donde aún no hay señal
    raise NotImplementedError("TODO 3.2 · momentum_12m — ver docs/03_Estrategias.pdf")


# ---------------------------------------------------------------------------
# Mean reversion (TODO 3.3)
# ---------------------------------------------------------------------------
@strategy
def bollinger_mean_reversion(
    prices: pd.DataFrame, window: int = 20, n_std: float = 2.0, allow_short: bool = False
) -> pd.DataFrame:
    """Reversión a la media con bandas de Bollinger. Es una estrategia CON ESTADO.

    - media = rolling(window, min_periods=window).mean();  σ = rolling(...).std() (ddof=1)
    - banda inferior = media − n_std·σ;  banda superior = media + n_std·σ
    - Largo: se ENTRA cuando close < banda inferior y se SALE cuando close >= media.
      Entre medias se mantiene lo que hubiera (por eso "con estado").
    - Corto (solo con allow_short=True), simétrico: se entra cuando close > banda superior
      y se sale cuando close <= media.
    - Antes de la primera entrada y durante el calentamiento, 0.
    """
    if window < 2 or n_std <= 0:
        raise ValueError(f"Se necesita window >= 2 y n_std > 0 (recibido {window} y {n_std}).")
    # TODO 3.3 · bollinger_mean_reversion
    # Objetivo: una máquina de estados (fuera ↔ dentro) SIN bucles.
    # Truco: tabla de NaN → poner 1 donde hay entrada y 0 donde hay salida → ffill() → fillna(0).
    #        Los NaN que quedan en medio "heredan" el último estado.
    # Con cortos: haz la pata larga y la corta por separado y súmalas.
    raise NotImplementedError("TODO 3.3 · bollinger_mean_reversion — ver docs/03_Estrategias.pdf")
