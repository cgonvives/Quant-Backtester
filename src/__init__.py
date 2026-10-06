"""Quant-Backtester: motor de backtesting vectorizado en pandas, construido desde cero.

Módulos, en el orden en que se construyen:
    config       parámetros y rutas
    data         descarga, limpieza y rendimientos            (TODO 1.x)
    backtest     señales → posiciones → P&L                   (TODO 2.x, 6.1)
    strategies   momentum y reversión a la media              (TODO 3.x)
    metrics      Sharpe, Sortino, drawdown...                 (TODO 4.x)
    walkforward  validación fuera de muestra                  (TODO 5.x)
    plotting     gráficos para los notebooks

Guía: docs/00_Guia_paso_a_paso.pdf
"""
