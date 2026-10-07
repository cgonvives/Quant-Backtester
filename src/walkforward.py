"""Validación walk-forward: optimizar en el pasado, aplicar en el futuro, repetir.

    |──── train (3 años) ────|─ test (1 año) ─|
             |──── train (3 años) ────|─ test (1 año) ─|
                      |──── train (3 años) ────|─ test (1 año) ─|      ...

En cada ventana se elige el mejor parámetro SOLO con datos de train y se aplica en test.
La concatenación de los tramos de test es la única curva "honesta": ninguna de sus
decisiones usó datos que no se conocían en su momento.

Convenciones (las comprueban los tests):
- Ventanas por años naturales con pd.DateOffset(years=...), ancladas en la primera fecha.
- Intervalos SEMIABIERTOS [inicio, fin): el fin de train es el inicio de test, y una fecha
  nunca está en dos tramos a la vez.
- step >= test: los tramos de test no se solapan.
- Al evaluar un tramo, la estrategia recibe TODA la historia anterior al fin del tramo
  (para que las medias móviles estén "calientes"), pero se puntúa solo dentro del tramo.
- Los tramos de test se encadenan como POSICIONES, no como rendimientos: así se cobran
  los costes de cambiar de parámetros al pasar de una ventana a la siguiente.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import config
from src.backtest import BacktestResult, backtest_positions, run_backtest, signals_to_positions
from src.metrics import METRICS


# ---------------------------------------------------------------------------
# Estructuras de datos (hecho)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """Una ventana walk-forward. Los fines son EXCLUSIVOS: [train_start, train_end)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.train_start) & (index < self.train_end)

    def test_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.test_start) & (index < self.test_end)

    def __str__(self) -> str:
        return (
            f"train [{self.train_start:%Y-%m-%d}, {self.train_end:%Y-%m-%d})  "
            f"test [{self.test_start:%Y-%m-%d}, {self.test_end:%Y-%m-%d})"
        )


@dataclass
class GridSearchResult:
    """Resultado de una búsqueda en rejilla.

    table: una fila por combinación, ORDENADA de mejor a peor puntuación (NaN al final,
           empates en el orden de la rejilla). Columnas: los parámetros + "score".
           Índice: la posición de la combinación en la rejilla original.
    best_params: el dict ORIGINAL de la rejilla con mejor puntuación (mismos tipos).
    """

    table: pd.DataFrame
    best_params: dict
    best_score: float
    metric: str


@dataclass
class WalkForwardResult:
    """Resultado de walk_forward: la curva fuera de muestra y lo elegido en cada ventana."""

    result: BacktestResult                 # backtest de los tramos de test encadenados
    windows: list[Window]
    searches: list[GridSearchResult]       # la búsqueda en train de cada ventana

    @property
    def selections(self) -> pd.DataFrame:
        """Una fila por ventana: fechas, parámetros elegidos y puntuación en train."""
        rows = []
        for window, search in zip(self.windows, self.searches):
            rows.append(
                {
                    "train_start": window.train_start,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    **search.best_params,
                    f"train_{search.metric}": search.best_score,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilidades (hecho)
# ---------------------------------------------------------------------------
def expand_grid(
    param_grid: Mapping[str, Sequence], constraint: Callable[[dict], bool] | None = None
) -> list[dict]:
    """{"fast": [10, 20], "slow": [100, 200]} → [{"fast": 10, "slow": 100}, ...] (producto cartesiano).

    constraint: función dict → bool para descartar combinaciones (p. ej. fast >= slow).
    """
    keys = list(param_grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(param_grid[k] for k in keys))]
    if constraint is not None:
        combos = [c for c in combos if constraint(c)]
    return combos


def heatmap_table(search: GridSearchResult, index: str, columns: str) -> pd.DataFrame:
    """Puntuaciones de una búsqueda con 2 parámetros en forma de tabla (para el mapa de calor)."""
    params = [c for c in search.table.columns if c != "score"]
    if sorted(params) != sorted([index, columns]):
        raise ValueError(f"heatmap_table necesita exactamente los parámetros {index} y {columns}; hay {params}.")
    return search.table.set_index([index, columns])["score"].unstack(columns).sort_index().sort_index(axis=1)


# ---------------------------------------------------------------------------
# TODO 5.1 – 5.4
# ---------------------------------------------------------------------------
def generate_windows(
    index: pd.DatetimeIndex,
    train_years: int = config.WF_TRAIN_YEARS,
    test_years: int = config.WF_TEST_YEARS,
    step_years: int = config.WF_STEP_YEARS,
) -> list[Window]:
    """Ventanas walk-forward sobre las fechas de `index`.

    - La ventana k (k = 0, 1, 2...) empieza en  index[0] + DateOffset(years = k·step_years).
    - train = [inicio, inicio + train_years);  test = [fin de train, fin de train + test_years).
    - Se generan ventanas mientras el inicio del test sea <= la última fecha de `index`
      (la última puede quedar incompleta: se usa lo que haya).
    - Si no cabe ninguna, lista vacía.
    """
    if min(train_years, test_years, step_years) < 1:
        raise ValueError("train_years, test_years y step_years deben ser >= 1.")
    if step_years < test_years:
        raise ValueError(
            "step_years < test_years: los tramos de test se solaparían y "
            "algunos días contarían dos veces."
        )
    index = pd.DatetimeIndex(index)
    # TODO 5.1 · generate_windows
    # Pista: calcula cada inicio desde index[0] (index[0] + DateOffset(years=k*step)) en lugar de
    #        ir sumando sobre la ventana anterior: así no se acumulan desajustes (29 de febrero).
    raise NotImplementedError("TODO 5.1 · generate_windows — ver docs/05_Walk_forward_y_sobreajuste.pdf")


def evaluate_params(
    prices: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    params: dict,
    start,
    end,
    metric: str = config.WF_METRIC,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
    lag: int = config.SIGNAL_LAG,
) -> float:
    """Puntuación de una combinación de parámetros en el tramo [start, end).

    1. La estrategia recibe los precios ANTERIORES a `end` (toda la historia previa, para
       el calentamiento, y nada del futuro).
    2. Backtest con run_backtest sobre esos mismos precios.
    3. Se puntúan solo los rendimientos con fecha en [start, end) con METRICS[metric].
    """
    if metric not in METRICS:
        raise KeyError(f"Métrica '{metric}' desconocida. Disponibles: {sorted(METRICS)}")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # TODO 5.2 · evaluate_params
    # Cuidado 1: si recortas los precios a [start, end) ANTES de calcular la señal, una media de
    #            200 días pasa los primeros 200 días del tramo sin señal. Eso no es lo que pasaría.
    # Cuidado 2: `end` es exclusivo.
    raise NotImplementedError("TODO 5.2 · evaluate_params — ver docs/05_Walk_forward_y_sobreajuste.pdf")


def grid_search(
    prices: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    grid: Sequence[dict],
    start,
    end,
    metric: str = config.WF_METRIC,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
    lag: int = config.SIGNAL_LAG,
) -> GridSearchResult:
    """Evalúa cada combinación de `grid` en [start, end) y las ordena de mejor a peor.

    - Mayor puntuación = mejor. Las puntuaciones NaN van al final.
    - Ordenación ESTABLE: con empates gana la que aparece antes en la rejilla.
    - best_params es el dict original de `grid` (no lo reconstruyas desde la tabla: los
      enteros se convertirían en float).
    - Ver el docstring de GridSearchResult para el formato de `table`.
    """
    grid = list(grid)
    if not grid:
        raise ValueError("La rejilla de parámetros está vacía.")
    # TODO 5.3 · grid_search
    # Pistas: evalúa con evaluate_params; ordena POSICIONES (0..n-1) con sorted(), que es estable,
    #         usando una clave que mande los NaN al final.
    raise NotImplementedError("TODO 5.3 · grid_search — ver docs/05_Walk_forward_y_sobreajuste.pdf")


def walk_forward(
    prices: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    grid: Sequence[dict],
    train_years: int = config.WF_TRAIN_YEARS,
    test_years: int = config.WF_TEST_YEARS,
    step_years: int = config.WF_STEP_YEARS,
    metric: str = config.WF_METRIC,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
    lag: int = config.SIGNAL_LAG,
) -> WalkForwardResult:
    """Walk-forward completo. Para cada ventana:

    1. grid_search en [train_start, train_end) → mejores parámetros.
    2. Señales con esos parámetros sobre los precios anteriores a test_end, y posiciones
       con signals_to_positions(..., lag).
    3. Quedarse con las posiciones de [test_start, test_end).

    Después: concatenar las posiciones de todos los tramos de test, ponerlas sobre el índice
    completo de `prices` (0 fuera de los tramos), backtestear con backtest_positions y
    devolver solo el periodo fuera de muestra (BacktestResult.slice desde la primera hasta
    la última fecha de test).
    """
    windows = generate_windows(prices.index, train_years, test_years, step_years)
    if not windows:
        raise ValueError(
            f"No cabe ninguna ventana de {train_years}+{test_years} años en los datos "
            f"({prices.index[0]:%Y-%m-%d} a {prices.index[-1]:%Y-%m-%d})."
        )
    costs = {"commission_bps": commission_bps, "slippage_bps": slippage_bps}
    # TODO 5.4 · walk_forward
    # Cuidado: si concatenas los RENDIMIENTOS de cada tramo, el cambio de posición entre el último
    #          día de un tramo y el primero del siguiente no paga costes. Encadena POSICIONES.
    # Devuelve: WalkForwardResult(result=..., windows=windows, searches=[...una por ventana...])
    raise NotImplementedError("TODO 5.4 · walk_forward — ver docs/05_Walk_forward_y_sobreajuste.pdf")
