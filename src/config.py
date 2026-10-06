"""Configuración central del proyecto: rutas, universo de activos y parámetros por defecto.

Todo lo que sea un "número mágico" vive aquí, para que los módulos y los notebooks
usen los mismos valores y se puedan cambiar en un único sitio.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas (siempre relativas a la raíz del repo, funcionen donde funcionen)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PRICES_FILE = RAW_DIR / "prices.parquet"
PRICES_META_FILE = RAW_DIR / "prices_meta.json"
DOCS_DIR = ROOT / "docs"

# ---------------------------------------------------------------------------
# Universo y periodo
# ---------------------------------------------------------------------------
# ETFs de distintas clases de activo + acciones grandes y líquidas.
# Ojo: elegirlos hoy introduce sesgo de supervivencia (ver README, Limitaciones).
TICKERS: list[str] = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "EFA",   # Desarrollados ex-EE. UU.
    "GLD",   # Oro
    "TLT",   # Bonos del Tesoro 20+ años
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
]
START = "2010-01-01"
END: str | None = None  # None = hasta hoy

# ---------------------------------------------------------------------------
# Mercado y costes
# ---------------------------------------------------------------------------
TRADING_DAYS = 252          # sesiones por año para anualizar
COMMISSION_BPS = 5.0        # comisión por unidad de turnover, en puntos básicos
SLIPPAGE_BPS = 5.0          # slippage por unidad de turnover, en puntos básicos
RISK_FREE = 0.0             # tipo libre de riesgo ANUAL (0.02 = 2 %)
SIGNAL_LAG = 1              # días entre la señal y la posición (shift)

# Limpieza de datos: máximo de días seguidos que se rellenan con el último precio
MAX_FFILL_DAYS = 5

# ---------------------------------------------------------------------------
# Estrategias: parámetros por defecto (los del README)
# ---------------------------------------------------------------------------
STRATEGY_DEFAULTS: dict[str, dict] = {
    "buy_and_hold": {},
    "sma_crossover": {"fast": 50, "slow": 200},
    "momentum_12m": {"lookback_months": 12, "skip_months": 0},
    "bollinger_mean_reversion": {"window": 20, "n_std": 2.0},
}

# ---------------------------------------------------------------------------
# Walk-forward: rejillas de búsqueda (reducidas a propósito) y ventanas
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "sma_crossover": {
        "fast": [10, 20, 50, 100],
        "slow": [50, 100, 150, 200, 250],
    },
    "bollinger_mean_reversion": {
        "window": [10, 20, 40, 60],
        "n_std": [1.0, 1.5, 2.0, 2.5],
    },
}


def _fast_below_slow(params: dict) -> bool:
    return params["fast"] < params["slow"]


# Combinaciones que no tienen sentido y se descartan al expandir la rejilla
PARAM_CONSTRAINTS = {
    "sma_crossover": _fast_below_slow,
}

WF_TRAIN_YEARS = 3
WF_TEST_YEARS = 1
WF_STEP_YEARS = 1
WF_METRIC = "sharpe"        # clave de metrics.METRICS que se maximiza en train

# ---------------------------------------------------------------------------
# Extensión: volatility targeting
# ---------------------------------------------------------------------------
VOL_TARGET = 0.10           # volatilidad anual objetivo por activo
VOL_LOOKBACK = 60           # días para estimar la volatilidad realizada
MAX_LEVERAGE = 2.0          # tope del factor de escala
