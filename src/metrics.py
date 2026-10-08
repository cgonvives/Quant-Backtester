"""Métricas de rendimiento y riesgo, implementadas a mano y verificadas contra quantstats.

Todas reciben una Series de rendimientos DIARIOS (por ejemplo BacktestResult.returns).
El decorador @_metric limpia la entrada antes de llamar a tu código:
- admite una Series, un DataFrame de una columna o directamente un BacktestResult;
- elimina los NaN y convierte a float.
Así, dentro de cada TODO puedes suponer una Series limpia de floats.

Convenciones (las comprueban los tests y coinciden con quantstats):
- Desviaciones típicas muestrales (ddof = 1). Anualización con √252.
- CAGR por número de observaciones: un año = 252 rendimientos.
- El tipo libre de riesgo rf es ANUAL y se pasa a diario de forma geométrica.
- El drawdown cuenta el capital inicial (1) como primer pico.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src import config
from src.backtest import BacktestResult


# ---------------------------------------------------------------------------
# Utilidades (hecho)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DrawdownInfo:
    """Descripción del peor drawdown de una serie."""

    depth: float                     # caída máxima, negativa: -0.25 = -25 %
    peak: pd.Timestamp | None        # último máximo antes del valle (None = el capital inicial)
    trough: pd.Timestamp | None      # fecha del valle (None si no hubo ninguna caída)
    recovery: pd.Timestamp | None    # primera fecha que vuelve al pico (None = no se ha recuperado)

    @property
    def recovered(self) -> bool:
        return self.recovery is not None

    @property
    def duration(self) -> pd.Timedelta | None:
        """Tiempo desde el pico hasta la recuperación (None si falta alguna de las dos)."""
        if self.peak is None or self.recovery is None:
            return None
        return self.recovery - self.peak

    def __str__(self) -> str:
        def fmt(ts):
            return "—" if ts is None else f"{ts:%Y-%m-%d}"

        return (
            f"Max DD {self.depth:.1%}  pico {fmt(self.peak)}  valle {fmt(self.trough)}  "
            f"recuperación {fmt(self.recovery)}"
        )


def _as_series(x, source: str = "returns") -> pd.Series:
    """Series / DataFrame de una columna / BacktestResult → Series de floats sin NaN."""
    if isinstance(x, BacktestResult):
        x = getattr(x, source)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise TypeError(
                "Las métricas reciben UNA serie de rendimientos. Para varias, usa summary_table."
            )
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return x.astype(float).dropna()


def _metric(source: str = "returns"):
    """Decorador: limpia el primer argumento con _as_series antes de llamar a la métrica."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(x, *args, **kwargs):
            return func(_as_series(x, source), *args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Métricas (TODO 4.1 – 4.9)
# ---------------------------------------------------------------------------
@_metric()
def annualized_volatility(returns: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Volatilidad anualizada:  σ_diaria · √periods, con σ muestral (ddof = 1)."""
    # TODO 4.1 · annualized_volatility

    std_diaria = returns.std()
    annualized_vol = std_diaria * np.sqrt(periods)
    return annualized_vol


@_metric()
def cagr(returns: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Tasa de crecimiento anual compuesta:  (∏(1 + r))^(periods / n) − 1.

    n = número de rendimientos. Con 252 rendimientos diarios, el exponente es 1.
    """
    # TODO 4.2 · cagr
    # Pista: crecimiento total = producto de (1 + r); años = n / periods.

    n = len(returns)
    product = (1 + returns).prod()
    cagr = (product ** (periods / n)) - 1
    return cagr

@_metric()
def sharpe_ratio(
    returns: pd.Series, rf: float = config.RISK_FREE, periods: int = config.TRADING_DAYS
) -> float:
    """Ratio de Sharpe anualizado:  media(exceso) / σ(exceso) · √periods.

    - rf es ANUAL; el diario es  rf_d = (1 + rf)^(1/periods) − 1  y  exceso = r − rf_d.
    - σ muestral (ddof = 1). Si σ es 0 (serie constante), devuelve NaN.
    """
    # TODO 4.3 · sharpe_ratio
    # Cuidado: rf / 252 es una aproximación; aquí se pide la conversión geométrica.

    rf_d = ((1 + rf) ** (1 / periods)) - 1
    exceso = returns - rf_d
    media_exceso = exceso.mean()
    std_exceso = exceso.std()
    sharpe = media_exceso / std_exceso * np.sqrt(periods)
    return sharpe


@_metric()
def sortino_ratio(
    returns: pd.Series, rf: float = config.RISK_FREE, periods: int = config.TRADING_DAYS
) -> float:
    """Ratio de Sortino anualizado:  media(exceso) / σ_bajista · √periods.

    - exceso = r − rf_d (como en el Sharpe).
    - σ_bajista = √( Σ min(exceso, 0)² / n ), con n = número TOTAL de rendimientos
      (no solo los negativos). Es la convención de quantstats.
    - Si no hay ningún exceso negativo, devuelve NaN.
    """
    # TODO 4.4 · sortino_ratio

    n = len(returns)
    rf_d = ((1 + rf) ** (1 / periods)) - 1
    exceso = returns - rf_d
    media_exceso = exceso.mean()
    std_bajista_exceso = np.sqrt((np.minimum(exceso, 0) ** 2).sum() / n)
    if std_bajista_exceso == 0:
        return np.nan
    sortino = media_exceso / std_bajista_exceso * np.sqrt(periods)
    return sortino


@_metric()
def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown diario:  equity_t / máximo_previo_t − 1  (valores <= 0).

    - equity_t = ∏_{s ≤ t} (1 + r_s), partiendo de un capital de 1.
    - El capital inicial cuenta como pico: si el primer día se pierde un 5 %, el drawdown
      de ese día es -5 %, no 0.
    - Mismo índice que `returns`.
    """
    # TODO 4.5 · drawdown_series
    # Pista: cummax() da el máximo acumulado; ¿cómo haces que nunca sea menor que 1?

    equity_t = (1 + returns).cumprod()
    max_prev = (equity_t.cummax()).clip(lower=1)
    drawdown_series = equity_t / max_prev - 1
    return drawdown_series


@_metric()
def max_drawdown(returns: pd.Series) -> DrawdownInfo:
    """Peor drawdown, con sus fechas.

    - depth: el mínimo de drawdown_series (negativo). Si nunca hay caída: 0.0 y fechas None.
    - trough: fecha del mínimo (si se repite, la primera).
    - peak: última fecha ANTERIOR o igual al valle con drawdown 0. Si no hay ninguna,
      el pico era el capital inicial: None.
    - recovery: primera fecha POSTERIOR al valle con drawdown 0. Si no hay: None.
    """
    # TODO 4.6 · max_drawdown
    # Pista: idxmin(); y filtrar la serie antes y después del valle con .loc[:fecha] / .loc[fecha:].

    dd = drawdown_series(returns)
    profundidad = dd.min()
    if profundidad == 0:
        return DrawdownInfo(0.0, None, None, None)

    valle = dd.idxmin()
    picos = dd.loc[:valle][dd.loc[:valle] == 0]
    if not picos.empty:
        pico = picos.index[-1]
    else:
        pico = None

    recuperaciones = dd.loc[valle:][dd.loc[valle:] == 0]
    if not recuperaciones.empty:
        recuperacion = recuperaciones.index[0]
    else:
        recuperacion = None

    max_drawdown_info = DrawdownInfo(profundidad, pico, valle, recuperacion)
    return max_drawdown_info


@_metric()
def calmar_ratio(returns: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Ratio de Calmar:  CAGR / |max drawdown|. Si no hay drawdown, NaN."""
    # TODO 4.7 · calmar_ratio

    drawdown_info = max_drawdown(returns)
    max_drawdown_value = drawdown_info.depth
    if max_drawdown_value == 0:
        return np.nan

    cagr_value = cagr(returns, periods)
    calmar = cagr_value / np.abs(max_drawdown_value)
    return calmar


@_metric()
def hit_ratio(returns: pd.Series) -> float:
    """Proporción de días con ganancia, SIN contar los días planos (rendimiento exactamente 0).

    Si todos los días son planos, NaN.
    """
    # TODO 4.8 · hit_ratio
    # Pregunta: ¿por qué tiene sentido excluir los días en que la estrategia está fuera?

    no_planos = returns[returns != 0]
    if no_planos.empty:
        return np.nan
    
    n_plus = len(returns[returns > 0])
    n_minus = len(returns[returns < 0])

    hit = n_plus / len(no_planos)
    return hit


@_metric(source="turnover")
def annual_turnover(turnover: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Turnover anual medio:  Σ turnover / años, con años = n / periods.

    Recibe el turnover DIARIO de la cartera (BacktestResult.turnover). Un turnover anual
    de 4 significa que, en un año, se ha movido 4 veces el capital.
    """
    # TODO 4.9 · annual_turnover

    n = len(turnover)
    years = n / periods

    ann_turnover = turnover.sum() / years
    return ann_turnover


# ---------------------------------------------------------------------------
# Registro y tablas (hecho)
# ---------------------------------------------------------------------------
def _max_drawdown_depth(returns) -> float:
    return max_drawdown(returns).depth


# Métricas que devuelven un número, por nombre. El walk-forward MAXIMIZA la elegida, así que
# solo están las que cuanto más altas, mejor (el max drawdown es negativo: -10 % > -30 %).
METRICS: dict[str, Callable[[pd.Series], float]] = {
    "cagr": cagr,
    "sharpe": sharpe_ratio,
    "sortino": sortino_ratio,
    "max_drawdown": _max_drawdown_depth,
    "calmar": calmar_ratio,
    "hit_ratio": hit_ratio,
}

SUMMARY_COLUMNS = ["CAGR", "Vol.", "Sharpe", "Sortino", "Max DD", "Calmar", "Hit ratio", "Turnover anual"]
_PERCENT_COLUMNS = {"CAGR", "Vol.", "Max DD", "Hit ratio"}


def summary_table(
    results: Mapping[str, BacktestResult | pd.Series],
    rf: float = config.RISK_FREE,
    periods: int = config.TRADING_DAYS,
) -> pd.DataFrame:
    """Tabla comparativa (una fila por estrategia) con las columnas del README.

    Acepta BacktestResult o Series de rendimientos (sin turnover: esa columna queda NaN).
    Si alguna métrica es todavía un TODO, su columna sale NaN y se avisa, sin romper nada.
    """
    calculators = {
        "CAGR": lambda r: cagr(r, periods),
        "Vol.": lambda r: annualized_volatility(r, periods),
        "Sharpe": lambda r: sharpe_ratio(r, rf, periods),
        "Sortino": lambda r: sortino_ratio(r, rf, periods),
        "Max DD": lambda r: max_drawdown(r).depth,
        "Calmar": lambda r: calmar_ratio(r, periods),
        "Hit ratio": hit_ratio,
    }
    pending: dict[str, str] = {}
    rows = {}
    for name, res in results.items():
        returns = res.returns if isinstance(res, BacktestResult) else res
        row = {}
        for col, fn in calculators.items():
            try:
                row[col] = fn(returns)
            except NotImplementedError as exc:
                pending[col] = str(exc)
                row[col] = np.nan
        if isinstance(res, BacktestResult):
            try:
                row["Turnover anual"] = annual_turnover(res.turnover, periods)
            except NotImplementedError as exc:
                pending["Turnover anual"] = str(exc)
                row["Turnover anual"] = np.nan
        else:
            row["Turnover anual"] = np.nan
        rows[name] = row
    if pending:
        todos = sorted(set(pending.values()))
        warnings.warn("Métricas pendientes (salen NaN): " + "; ".join(todos), stacklevel=2)
    table = pd.DataFrame.from_dict(rows, orient="index", columns=SUMMARY_COLUMNS)
    table.index.name = "Estrategia"
    return table


def _format_value(col: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if col in _PERCENT_COLUMNS:
        return f"{value:.1%}"
    if col == "Turnover anual":
        return f"{value:.1f}×"
    return f"{value:.2f}"


def format_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Versión legible de summary_table: porcentajes, 2 decimales en ratios, turnover en ×."""
    return pd.DataFrame(
        {col: [_format_value(col, v) for v in table[col]] for col in table.columns},
        index=table.index,
    )


def to_markdown_table(table: pd.DataFrame) -> str:
    """Tabla en Markdown, lista para pegar en la sección de Resultados del README."""
    formatted = format_summary(table)
    header = "| " + " | ".join([table.index.name or "Estrategia", *formatted.columns]) + " |"
    sep = "|" + "---|" * (len(formatted.columns) + 1)
    lines = [header, sep]
    for name, row in formatted.iterrows():
        lines.append("| " + " | ".join([str(name), *row.tolist()]) + " |")
    return "\n".join(lines)


def compare_with_quantstats(
    returns, rf: float = config.RISK_FREE, periods: int = config.TRADING_DAYS
) -> pd.DataFrame:
    """Compara tus métricas con las de quantstats sobre la misma serie.

    Columnas: propia, quantstats, diferencia. Las diferencias deberían ser prácticamente 0
    (el test exige menos de 1e-10). El CAGR no se compara: las versiones antiguas de quantstats
    cuentan años de calendario (las recientes cuentan sesiones, como aquí).
    """
    try:
        import quantstats as qs
    except ImportError as exc:
        raise ImportError("Instala quantstats para comparar:  python -m pip install quantstats") from exc

    r = _as_series(returns)
    own = {
        "Sharpe": sharpe_ratio(r, rf, periods),
        "Sortino": sortino_ratio(r, rf, periods),
        "Volatilidad": annualized_volatility(r, periods),
        "Max DD": max_drawdown(r).depth,
    }
    theirs = {
        "Sharpe": qs.stats.sharpe(r, rf=rf, periods=periods),
        "Sortino": qs.stats.sortino(r, rf=rf, periods=periods),
        "Volatilidad": qs.stats.volatility(r, periods=periods),
        "Max DD": qs.stats.max_drawdown(r),
    }
    table = pd.DataFrame({"propia": own, "quantstats": pd.Series(theirs, dtype=float)})
    table["diferencia"] = table["propia"] - table["quantstats"]
    return table
