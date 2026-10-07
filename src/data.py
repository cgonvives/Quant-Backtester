"""Datos: descarga con yfinance, guardado en parquet, limpieza y rendimientos.

Uso desde la terminal (en la raíz del repo, con el venv activado):

    python -m src.data                                   # TICKERS de config.py desde START
    python -m src.data --tickers SPY QQQ --start 2015-01-01

La descarga guarda los precios "en bruto" en data/raw/prices.parquet. La limpieza
(TODO 1.3) se aplica al cargar, con load_prices(clean=True): así puedes cambiar la
limpieza sin volver a descargar nada.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Descarga y almacenamiento (hecho)
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str] | None = None,
    start: str = config.START,
    end: str | None = config.END,
    retries: int = 3,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Descarga precios de cierre AJUSTADOS (dividendos y splits) de Yahoo Finance.

    Devuelve un DataFrame con índice de fechas (DatetimeIndex sin zona horaria) y una
    columna por ticker. Son precios en bruto: puede haber huecos (NaN). Los tickers
    sin ningún dato se descartan con un aviso.
    """
    import yfinance as yf  # import perezoso: el resto del proyecto funciona sin red

    tickers = list(tickers or config.TICKERS)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw is not None and not raw.empty:
                break
            last_error = RuntimeError("yfinance devolvió una tabla vacía")
        except Exception as exc:  # errores de red, límites de Yahoo...
            last_error = exc
        if attempt < retries:
            time.sleep(pause * attempt)
    else:
        raise RuntimeError(
            f"No se pudieron descargar los precios tras {retries} intentos: {last_error}"
        )

    close = _extract_close(raw, tickers)
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.rename("Date")
    close = close.sort_index().astype(float)

    empty = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if empty:
        warnings.warn(
            f"Sin datos para {', '.join(empty)}: se descartan. "
            "Revisa el ticker o vuelve a intentarlo más tarde.",
            stacklevel=2,
        )
    return close[[t for t in tickers if t not in empty]]


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Saca la tabla de cierres (fechas × tickers) de lo que devuelve yf.download.

    Según la versión y el número de tickers, yfinance devuelve columnas planas
    ("Open", "Close"...) o un MultiIndex (campo, ticker) o (ticker, campo).
    """
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close = raw.xs("Close", axis=1, level=level)
                break
        else:
            raise KeyError("La descarga de yfinance no tiene columna 'Close'.")
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise KeyError("La descarga de yfinance no tiene columna 'Close'.")

    close = close.copy()
    close.columns = [str(c) for c in close.columns]
    return close


_PARQUET_HINT = (
    "No se puede usar parquet. Si el error dice 'Control de aplicaciones', Windows (Smart App "
    "Control) está bloqueando una DLL de pyarrow: instala una versión anterior con  "
    'python -m pip install "pyarrow<25"  (ver requirements.txt).'
)


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Guarda un DataFrame en parquet, creando la carpeta si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except ImportError as exc:
        raise ImportError(f"{_PARQUET_HINT}\nError original: {exc}") from exc
    return path


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Lee un parquet guardado con save_parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Descarga los datos primero con:  python -m src.data"
        )
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(f"{_PARQUET_HINT}\nError original: {exc}") from exc


def load_prices(
    path: Path | str = config.PRICES_FILE,
    clean: bool = True,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Carga los precios guardados y, si clean=True, los limpia con clean_prices (TODO 1.3).

    Mientras no hayas hecho el TODO 1.3, usa clean=False.
    """
    prices = load_parquet(path)
    if tickers is not None:
        prices = prices[list(tickers)]
    if clean:
        prices = clean_prices(prices)
    return prices


# ---------------------------------------------------------------------------
# Rendimientos (TODO 1.1 y 1.2)
# ---------------------------------------------------------------------------
def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendimientos simples diarios:  r_t = P_t / P_{t-1} - 1.

    Convenciones (las comprueban los tests):
    - Mismo índice y columnas que `prices`. La primera fila es NaN (no hay día anterior).
    - Si falta el precio de hoy o el de ayer, el rendimiento es NaN: no se rellena nada.
    - Debe funcionar igual con un DataFrame que con una Series.
    """
    # TODO 1.1 · simple_returns
    # Objetivo: el rendimiento de cada día frente al anterior, en todas las columnas a la vez.
    # Fórmula: r_t = P_t / P_{t-1} - 1
    # Pista: "el precio de ayer", para toda la tabla de golpe, es prices.shift(1).
    # Cuidado: en pandas < 3, pct_change() rellenaba los huecos por defecto (fill_method="pad").
    # Escríbelo a partir de la fórmula: así sabes exactamente qué pasa con los NaN.
    r_t = prices / prices.shift(1) - 1
    return r_t


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendimientos logarítmicos diarios:  ℓ_t = ln(P_t / P_{t-1}).

    Mismas convenciones que simple_returns (primera fila NaN, sin rellenos).
    Propiedad útil: la suma de los ℓ_t de un periodo es ln(P_final / P_inicial).
    """
    # TODO 1.2 · log_returns
    # Objetivo: igual que simple_returns, pero con logaritmos.
    # Pista: numpy aplica np.log elemento a elemento sobre un DataFrame y conserva índice y columnas.
    l_t = np.log(prices / prices.shift(1))
    return l_t


# ---------------------------------------------------------------------------
# Limpieza (TODO 1.3)
# ---------------------------------------------------------------------------
def clean_prices(prices: pd.DataFrame, max_ffill: int = config.MAX_FFILL_DAYS) -> pd.DataFrame:
    """Limpia una tabla de precios en bruto. Pasos, en este orden:

    1. Eliminar fechas duplicadas quedándose con la ÚLTIMA aparición, y ordenar por fecha.
    2. Convertir a float y marcar como NaN los precios <= 0 (son errores de datos).
    3. Eliminar las filas en las que TODOS los activos son NaN.
    4. Rellenar huecos hacia delante con el último precio, como mucho `max_ffill`
       días seguidos (si el hueco es más largo, el resto se queda en NaN).
       Nunca hacia atrás (bfill): eso metería precios del futuro en el pasado.
    5. Recortar el inicio a la primera fecha en la que TODOS los activos tienen precio.

    No modifica `prices`: devuelve una tabla nueva.
    """
    # TODO 1.3 · clean_prices
    # Objetivo: los 5 pasos del docstring, en ese orden.
    # Pistas: index.duplicated(keep=...), sort_index(), mask(), dropna(how=...),
    # ffill(limit=...) y first_valid_index() (la "fecha común" es la más tardía de las primeras).
    duplicated = prices.index.duplicated(keep="last")
    prices_new = prices[~duplicated].sort_index().astype(float)
    cond_negs = (prices_new <= 0)
    prices_new = prices_new.mask(cond=cond_negs).dropna(how="all")
    prices_new = prices_new.ffill(limit=max_ffill)
    first_dates = (prices_new[col].first_valid_index() for col in prices_new.columns)
    common_start = max(first_dates)
    prices_new = prices_new.loc[common_start:]

    return prices_new

# ---------------------------------------------------------------------------
# Punto de entrada:  python -m src.data
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Descarga precios ajustados con yfinance y los guarda en data/raw/."
    )
    parser.add_argument("--tickers", nargs="+", default=config.TICKERS)
    parser.add_argument("--start", default=config.START)
    parser.add_argument("--end", default=config.END)
    args = parser.parse_args(argv)

    print(f"Descargando {len(args.tickers)} tickers desde {args.start}...")
    prices = download_prices(args.tickers, args.start, args.end)
    path = save_parquet(prices, config.PRICES_FILE)

    import yfinance as yf

    meta = {
        "tickers": list(prices.columns),
        "start": args.start,
        "end": args.end,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(prices)),
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "first_valid": {c: str(prices[c].first_valid_index().date()) for c in prices.columns},
        "missing_values": {c: int(prices[c].isna().sum()) for c in prices.columns},
        "yfinance": yf.__version__,
        "auto_adjust": True,
    }
    config.PRICES_META_FILE.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Guardado en {path.relative_to(config.ROOT)}")
    print(f"  {meta['rows']} sesiones, de {meta['first_date']} a {meta['last_date']}")
    summary = pd.DataFrame(
        {"primer dato": meta["first_valid"], "NaN": meta["missing_values"]}
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
