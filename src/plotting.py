"""Gráficos para los notebooks (matplotlib). No tiene TODO: úsalo tal cual.

Reglas de estilo que sigue el módulo:
- Colores categóricos en orden fijo (nunca se reciclan: más de 8 series → varios gráficos).
- Una sola escala vertical por gráfico (nada de doble eje).
- Leyenda siempre que haya 2 o más series; el texto va en tinta neutra, no en el color de la serie.
- Mapa de calor: escala divergente centrada en 0 (rojo < 0 < azul) si hay valores de los dos
  signos; de un solo color si todos tienen el mismo signo.

Las funciones aceptan un dict {nombre: BacktestResult o Series de rendimientos} y devuelven
el Axes (o la Figure), por si quieres retocar algo. Cualquier función de dibujo aplica el
estilo del proyecto a toda la sesión (apply_style).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter

from src.backtest import BacktestResult, equity_curve
from src.metrics import drawdown_series

# Paleta categórica (orden fijo) y tintas neutras
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL_FILL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#b8302f", "#e34948", NEUTRAL_FILL, "#3987e5", "#184f95"]
)
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQUENTIAL_RED = LinearSegmentedColormap.from_list("red", ["#7d1f1e", "#b8302f", "#e34948", "#f2a5a4", "#fbe3e2"])

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.dpi": 110,
}


def apply_style() -> None:
    """Aplica el estilo del proyecto a todos los gráficos de la sesión."""
    # Se cambia de forma global a propósito: con plt.rc_context, si el primer gráfico de la sesión
    # se crea dentro del contexto, al salir se deshace la activación del backend de Jupyter y los
    # gráficos siguientes dejan de mostrarse.
    plt.rcParams.update(_RC)


def colors_for(names: Sequence[str]) -> dict[str, str]:
    """Asigna los colores de la paleta en orden fijo. Más de 8 series: mejor varios gráficos."""
    names = list(names)
    if len(names) > len(PALETTE):
        raise ValueError(
            f"{len(names)} series son demasiadas para un solo gráfico (máximo {len(PALETTE)}): "
            "agrupa las menos importantes o usa varios gráficos."
        )
    return dict(zip(names, PALETTE))


def _returns_of(item) -> pd.Series:
    if isinstance(item, BacktestResult):
        return item.returns
    if isinstance(item, pd.Series):
        return item
    raise TypeError(f"Se esperaba un BacktestResult o una Series de rendimientos, no {type(item).__name__}.")


def _new_ax(ax, figsize=(10, 4.5)):
    apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# ---------------------------------------------------------------------------
# Curvas
# ---------------------------------------------------------------------------
def plot_equity(
    results: Mapping[str, BacktestResult | pd.Series],
    log: bool = True,
    title: str = "Curva de capital",
    ax=None,
):
    """Equity de varias estrategias (escala log por defecto: compara rentabilidades, no niveles)."""
    ax = _new_ax(ax)
    colors = colors_for(results)
    for name, item in results.items():
        equity = item.equity if isinstance(item, BacktestResult) else equity_curve(_returns_of(item))
        ax.plot(equity.index, equity.to_numpy(), color=colors[name], label=f"{name}  ({equity.iloc[-1]:.2f}×)")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_title(title)
    ax.set_ylabel("capital (inicio = 1)")
    if len(results) > 1:
        ax.legend(loc="upper left")
    return ax


def plot_drawdowns(
    results: Mapping[str, BacktestResult | pd.Series],
    title: str = "Drawdown",
    ax=None,
):
    """Drawdown de cada estrategia (necesita el TODO 4.5)."""
    ax = _new_ax(ax, figsize=(10, 3.5))
    colors = colors_for(results)
    for name, item in results.items():
        dd = drawdown_series(_returns_of(item))
        ax.plot(dd.index, dd.to_numpy(), color=colors[name], label=f"{name}  (mín. {dd.min():.1%})")
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title)
    if len(results) > 1:
        ax.legend(loc="lower left")
    return ax


def plot_equity_and_drawdown(results: Mapping[str, BacktestResult | pd.Series], log: bool = True):
    """Equity arriba y drawdown abajo, con el eje de fechas compartido."""
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    plot_equity(results, log=log, ax=top)
    plot_drawdowns(results, ax=bottom)
    legend = bottom.get_legend()
    if legend is not None:
        legend.remove()  # la leyenda de arriba ya identifica las series
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Costes
# ---------------------------------------------------------------------------
def plot_cost_sensitivity(table: pd.DataFrame, metric_label: str = "Sharpe", ax=None):
    """Una línea por estrategia: métrica (eje y) frente a coste total en bps (eje x).

    table: índice = coste en bps, columnas = estrategias.
    """
    ax = _new_ax(ax)
    colors = colors_for(table.columns)
    for name in table.columns:
        ax.plot(table.index, table[name].to_numpy(), color=colors[name], marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlabel("coste por unidad de turnover (comisión + slippage, bps)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} según los costes de transacción")
    if table.shape[1] > 1:
        ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def plot_sharpe_heatmap(
    table: pd.DataFrame,
    title: str = "Sharpe por combinación de parámetros",
    best: tuple | None = None,
    ax=None,
):
    """Mapa de calor (p. ej. de walkforward.heatmap_table).

    Si hay valores positivos y negativos, escala divergente centrada en 0 (rojo < 0 < azul).
    Si todos tienen el mismo signo, escala de un solo color para distinguir mejor la magnitud.
    best: (fila, columna) de la celda a recuadrar, por ejemplo la elegida en train.
    """
    ax = _new_ax(ax, figsize=(1.1 * table.shape[1] + 2.5, 0.75 * table.shape[0] + 1.5))
    ax.grid(False)
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
    if low < 0 < high:
        bound = max(abs(low), abs(high))
        cmap, norm = DIVERGING, TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    else:
        cmap = SEQUENTIAL_BLUE if low >= 0 else SEQUENTIAL_RED
        norm = plt.Normalize(vmin=low, vmax=max(high, low + 1e-9))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")

    def on_dark_cell(v: float) -> bool:
        r, g, b, _ = cmap(norm(v))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45

    def label(v):
        return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)

    ax.set_xticks(range(table.shape[1]), [label(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [label(r) for r in table.index])
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if on_dark_cell(v) else INK)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
    if best is not None:
        i = list(table.index).index(best[0])
        j = list(table.columns).index(best[1])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    cbar = plt.colorbar(image, ax=ax, shrink=0.85)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    ax.set_title(title)
    return ax


def plot_wf_windows(windows, ax=None):
    """Diagrama de las ventanas walk-forward: train y test de cada una."""
    ax = _new_ax(ax, figsize=(10, 0.35 * len(windows) + 1.2))
    ax.grid(False)
    ax.grid(True, axis="x")
    train_color, test_color = PALETTE[0], PALETTE[1]
    for k, w in enumerate(windows):
        ax.barh(k, w.train_end - w.train_start, left=w.train_start, height=0.55, color=train_color,
                label="train" if k == 0 else None)
        ax.barh(k, w.test_end - w.test_start, left=w.test_start, height=0.55, color=test_color,
                label="test" if k == 0 else None)
    ax.set_yticks(range(len(windows)), [f"ventana {k + 1}" for k in range(len(windows))])
    ax.invert_yaxis()
    ax.set_title("Ventanas walk-forward")
    ax.legend(loc="upper right", ncols=2)  # las ventanas bajan en diagonal: esa esquina está libre
    return ax


def plot_param_stability(selections: pd.DataFrame, params: Sequence[str]):
    """Parámetro elegido en cada ventana (un gráfico por parámetro, mismo eje de fechas)."""
    apply_style()
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 2.4 * len(params)), sharex=True, squeeze=False)
    for ax, param in zip(axes[:, 0], params):
        ax.step(selections["test_start"], selections[param], where="post", color=PALETTE[0])
        ax.plot(selections["test_start"], selections[param], "o", color=PALETTE[0],
                markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=7)
        ax.set_ylabel(param)
        ax.set_yticks(sorted(selections[param].unique()))
    axes[0, 0].set_title("Parámetros elegidos en cada ventana (fecha = inicio del test)")
    fig.tight_layout()
    return fig
