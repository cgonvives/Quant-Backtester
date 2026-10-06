"""Motor de backtesting vectorizado: señales → posiciones → P&L.

Flujo de run_backtest (las piezas con número son tus TODO):

    señales s ─(2.1 shift)─► posiciones w ─┬─(2.4)─► P&L por activo  w_t · r_t ──────┐
                                           └─(2.2)─► turnover |w_t − w_{t−1}| ─(2.3)─► costes
    (2.5) cartera = suma / N   ─►   neto = bruto − costes   ─►   (2.6) equity = ∏ (1 + neto)

Convenciones (todas las comprueban los tests):
- w_t = s_{t−lag}, con lag = 1 por defecto. La señal calculada con el cierre del día t−1
  se mantiene desde ese cierre hasta el del día t, y gana r_t = P_t / P_{t−1} − 1.
- Antes de la primera señal disponible la posición es 0 (se parte de estar fuera).
- turnover_t = |w_t − w_{t−1}|, con w_{−1} = 0: entrar el primer día también cuesta.
- coste_t = turnover_t · (comisión + slippage) / 10 000, cobrado el día t.
- Cartera equiponderada: suma de los N activos dividida entre N (siempre N, todos).
- neto = bruto − costes;  equity_t = ∏_{s ≤ t} (1 + neto_s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.data import simple_returns


# ---------------------------------------------------------------------------
# Resultado de un backtest (hecho)
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Todo lo que produce un backtest, alineado sobre el mismo índice de fechas."""

    returns: pd.Series              # retornos netos diarios de la cartera
    gross_returns: pd.Series        # retornos antes de costes
    costs: pd.Series                # costes diarios de la cartera (en tanto por uno)
    equity: pd.Series               # curva de capital, partiendo de 1
    turnover: pd.Series             # turnover diario de la cartera (suma / N)
    positions: pd.DataFrame         # posiciones por activo
    asset_turnover: pd.DataFrame    # turnover diario por activo
    asset_returns: pd.DataFrame     # rendimientos simples de los activos

    @property
    def cumulative_turnover(self) -> pd.Series:
        """Turnover acumulado de la cartera."""
        return self.turnover.cumsum()

    def slice(self, start=None, end=None) -> BacktestResult:
        """Sub-periodo [start, end] (ambos incluidos, como .loc). La equity vuelve a partir de 1."""
        window = slice(start, end)
        returns = self.returns.loc[window]
        return BacktestResult(
            returns=returns,
            gross_returns=self.gross_returns.loc[window],
            costs=self.costs.loc[window],
            equity=equity_curve(returns).rename("equity"),
            turnover=self.turnover.loc[window],
            positions=self.positions.loc[window],
            asset_turnover=self.asset_turnover.loc[window],
            asset_returns=self.asset_returns.loc[window],
        )

    def __repr__(self) -> str:
        idx = self.returns.index
        if len(idx) == 0:
            return "BacktestResult(vacío)"
        return (
            f"BacktestResult({idx[0]:%Y-%m-%d} a {idx[-1]:%Y-%m-%d}, {len(idx)} días, "
            f"{self.positions.shape[1]} activos, equity final {self.equity.iloc[-1]:.3f})"
        )


def _validate_inputs(prices: pd.DataFrame, other: pd.DataFrame, name: str = "signals") -> None:
    """Comprueba que precios y señales/posiciones encajan antes de backtestear."""
    for label, df in (("prices", prices), (name, other)):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{label} debe ser un DataFrame (fechas × activos), no {type(df).__name__}.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"El índice de {label} debe ser un DatetimeIndex.")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError(f"El índice de {label} debe estar ordenado y sin fechas repetidas.")
    if not prices.index.equals(other.index):
        raise ValueError(f"prices y {name} deben tener exactamente las mismas fechas.")
    if list(prices.columns) != list(other.columns):
        raise ValueError(f"prices y {name} deben tener las mismas columnas y en el mismo orden.")
    values = other.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contiene NaN o infinitos: rellena el periodo de calentamiento con 0."
        )


# ---------------------------------------------------------------------------
# Orquestadores (hecho)
# ---------------------------------------------------------------------------
def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
    lag: int = config.SIGNAL_LAG,
    vol_target: float | None = None,
) -> BacktestResult:
    """Backtest completo: señales → posiciones (shift) → P&L neto de costes.

    vol_target (extensión, TODO 6.1): si se indica, por ejemplo 0.10, escala las
    posiciones de cada activo para apuntar a esa volatilidad anual.
    """
    _validate_inputs(prices, signals, "signals")
    positions = signals_to_positions(signals, lag=lag)
    if vol_target is not None:
        positions = apply_vol_target(
            positions,
            simple_returns(prices),
            target_vol=vol_target,
            lookback=config.VOL_LOOKBACK,
            max_leverage=config.MAX_LEVERAGE,
        )
    return backtest_positions(prices, positions, commission_bps, slippage_bps)


def backtest_positions(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
) -> BacktestResult:
    """Backtest a partir de posiciones YA desplazadas (sin volver a hacer shift).

    Lo usan run_backtest, el walk-forward (que encadena posiciones de varias ventanas)
    y el volatility targeting.
    """
    _validate_inputs(prices, positions, "positions")
    asset_returns = simple_returns(prices)
    asset_turnover = compute_turnover(positions)
    asset_costs = compute_costs(asset_turnover, commission_bps, slippage_bps)

    gross = to_portfolio(asset_pnl(positions, asset_returns))
    costs = to_portfolio(asset_costs)
    turnover = to_portfolio(asset_turnover)
    net = gross - costs

    return BacktestResult(
        returns=net.rename("returns"),
        gross_returns=gross.rename("gross_returns"),
        costs=costs.rename("costs"),
        equity=equity_curve(net).rename("equity"),
        turnover=turnover.rename("turnover"),
        positions=positions,
        asset_turnover=asset_turnover,
        asset_returns=asset_returns,
    )


# ---------------------------------------------------------------------------
# Piezas del motor (TODO 2.1 – 2.6)
# ---------------------------------------------------------------------------
def signals_to_positions(signals: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Convierte señales en posiciones:  w_t = s_{t−lag}.

    - Las primeras `lag` filas, sin señal previa, valen 0 (se empieza fuera del mercado).
    - lag < 1 lanza ValueError: con lag = 0 la posición del día t usaría el cierre del
      propio día t, que es justo lo que intenta predecir (look-ahead bias).
    - Mismo índice y columnas que `signals`.
    """
    # TODO 2.1 · signals_to_positions
    # Objetivo: desplazar las señales `lag` días hacia el futuro y rellenar el hueco inicial con 0.
    # Pista: shift(n) con n > 0 mueve los valores hacia fechas posteriores.
    # Pregunta: ¿qué pasaría con el Sharpe de una estrategia que "ve" el cierre de hoy?
    raise NotImplementedError("TODO 2.1 · signals_to_positions — ver docs/02_Motor_de_backtesting.pdf")


def compute_turnover(positions: pd.DataFrame) -> pd.DataFrame:
    """Turnover por activo:  |w_t − w_{t−1}|, con w_{−1} = 0.

    - Pasar de +1 a −1 cuenta 2 (cierras el largo y abres el corto).
    - La primera fila es |w_0|: se parte de estar fuera, así que entrar también es operar.
    - Sin NaN en el resultado.
    """
    # TODO 2.2 · compute_turnover
    # Objetivo: cuánto cambia la posición de cada activo de un día al siguiente.
    # Pista: diff() deja NaN en la primera fila; ¿cuál debería ser su valor si w_{−1} = 0?
    #        (shift admite fill_value).
    raise NotImplementedError("TODO 2.2 · compute_turnover — ver docs/02_Motor_de_backtesting.pdf")


def compute_costs(
    turnover: pd.DataFrame,
    commission_bps: float = config.COMMISSION_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Costes por activo, en tanto por uno:  turnover · (comisión + slippage) / 10 000.

    1 punto básico (bp) = 0,01 % = 0,0001. Con 5 + 5 bps, un turnover de 1 cuesta 0,001.
    """
    # TODO 2.3 · compute_costs
    # Objetivo: pasar de bps a tanto por uno y multiplicar por el turnover.
    raise NotImplementedError("TODO 2.3 · compute_costs — ver docs/02_Motor_de_backtesting.pdf")


def asset_pnl(positions: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """P&L bruto por activo:  w_t · r_t.

    Los rendimientos NaN (el primer día o un hueco sin precio) cuentan como 0:
    sin precio no se puede ganar ni perder nada.
    """
    # TODO 2.4 · asset_pnl
    # Objetivo: posición de hoy × rendimiento de hoy, activo a activo.
    # Ojo: aquí NO se desplaza nada: el desfase ya lo puso signals_to_positions.
    raise NotImplementedError("TODO 2.4 · asset_pnl — ver docs/02_Motor_de_backtesting.pdf")


def to_portfolio(asset_values: pd.DataFrame) -> pd.Series:
    """Agrega por activo → cartera equiponderada: suma de las columnas / N.

    N es el número TOTAL de activos (columnas), estén invertidos o no ese día.
    Los NaN cuentan como 0. Sirve para el P&L, los costes y el turnover.
    """
    # TODO 2.5 · to_portfolio
    # Objetivo: una Series con un valor por fecha.
    # Cuidado: mean(axis=1) NO es lo mismo cuando hay NaN. ¿Por qué?
    raise NotImplementedError("TODO 2.5 · to_portfolio — ver docs/02_Motor_de_backtesting.pdf")


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Curva de capital:  equity_t = initial · ∏_{s ≤ t} (1 + r_s).

    Mismo índice que `returns` (el primer valor ya incluye el primer rendimiento).
    """
    # TODO 2.6 · equity_curve
    # Objetivo: capitalizar los rendimientos diarios.
    # Pista: el producto acumulado es cumprod().
    raise NotImplementedError("TODO 2.6 · equity_curve — ver docs/02_Motor_de_backtesting.pdf")


# ---------------------------------------------------------------------------
# Extensión: volatility targeting (TODO 6.1)
# ---------------------------------------------------------------------------
def apply_vol_target(
    positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: float = config.VOL_TARGET,
    lookback: int = config.VOL_LOOKBACK,
    max_leverage: float = config.MAX_LEVERAGE,
    periods: int = config.TRADING_DAYS,
) -> pd.DataFrame:
    """Escala cada posición para que el activo apunte a una volatilidad anual `target_vol`.

        σ̂_t = std(r_{t−lookback} … r_{t−1}) · √periods      (ddof = 1, SOLO datos hasta t−1)
        escala_t = min(target_vol / σ̂_t, max_leverage)
        w'_t = w_t · escala_t

    - Mientras no haya `lookback` rendimientos para estimar σ̂, la escala es 0.
    - Si σ̂ = 0, la escala es max_leverage.
    """
    # TODO 6.1 · apply_vol_target (extensión)
    # Objetivo: una escala por activo y día, multiplicada por las posiciones.
    # Pista: rolling(lookback, min_periods=lookback).std() incluye el día t;
    #        la posición del día t solo puede usar información hasta t−1.
    raise NotImplementedError("TODO 6.1 · apply_vol_target — ver docs/06_Volatility_targeting.pdf")
