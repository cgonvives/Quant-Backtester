# Progreso

Marca cada casilla cuando su checkpoint esté en verde. Guía completa: `docs/00_Guia_paso_a_paso.pdf`.

## Preparación

- [X] Entorno creado e instalado (`python -m pip install -r requirements.txt`)
- [X] Datos descargados (`python -m src.data`)
- [X] Primera ejecución de `python -m pytest`: solo *passed* y *skipped*, ningún fallo

## Día 1 · Datos y motor (rama `feature/datos`)

- [X] 1.1 `simple_returns`
- [X] 1.2 `log_returns`
- [X] 1.3 `clean_prices`
- [X] 2.1 `signals_to_positions`
- [X] 2.2 `compute_turnover`
- [X] 2.3 `compute_costs`
- [X] 2.4 `asset_pnl`
- [X] 2.5 `to_portfolio`
- [X] 2.6 `equity_curve`
- [X] `python -m pytest tests/test_data.py tests/test_backtest.py` en verde
- [X] Notebook `01_exploracion` completo y preguntas respondidas
- [X] Merge en `main`

## Día 2 · Estrategias y métricas (rama `feature/estrategias`)

- [X] 3.1 `sma_crossover`
- [X] 3.2 `momentum_12m`
- [X] 3.3 `bollinger_mean_reversion`
- [X] 4.1 `annualized_volatility`
- [X] 4.2 `cagr`
- [X] 4.3 `sharpe_ratio`
- [X] 4.4 `sortino_ratio`
- [X] 4.5 `drawdown_series`
- [X] 4.6 `max_drawdown`
- [X] 4.7 `calmar_ratio`
- [X] 4.8 `hit_ratio`
- [X] 4.9 `annual_turnover`
- [X] `python -m pytest tests/test_strategies.py tests/test_metrics.py tests/test_quantstats_check.py` en verde
- [X] Notebook `02_estrategias` completo (tabla con y sin costes, sensibilidad a costes)
- [X] Merge en `main`

## Día 3 · Walk-forward y documentación (rama `feature/walk-forward`)

- [X] 5.1 `generate_windows`
- [X] 5.2 `evaluate_params`
- [X] 5.3 `grid_search`
- [X] 5.4 `walk_forward`
- [X] `python -m pytest tests/test_walkforward.py` en verde
- [X] Notebook `03_walkforward` completo (tres curvas, mapas de calor, Bollinger)
- [X] README: tabla de resultados y `docs/img/equity.png`
- [X] README: limitaciones con lo aprendido
- [X] `python -m pytest --todo-fail` sin fallos
- [X] Merge en `main`

## Extensión

- [X] 6.1 `apply_vol_target` (`python -m pytest -m extension`)
- [X] Sección 7 del notebook 02 (volatility targeting)
