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
- [ ] 2.1 `signals_to_positions`
- [ ] 2.2 `compute_turnover`
- [ ] 2.3 `compute_costs`
- [ ] 2.4 `asset_pnl`
- [ ] 2.5 `to_portfolio`
- [ ] 2.6 `equity_curve`
- [ ] `python -m pytest tests/test_data.py tests/test_backtest.py` en verde
- [ ] Notebook `01_exploracion` completo y preguntas respondidas
- [ ] Merge en `main`

## Día 2 · Estrategias y métricas (rama `feature/estrategias`)

- [ ] 3.1 `sma_crossover`
- [ ] 3.2 `momentum_12m`
- [ ] 3.3 `bollinger_mean_reversion`
- [ ] 4.1 `annualized_volatility`
- [ ] 4.2 `cagr`
- [ ] 4.3 `sharpe_ratio`
- [ ] 4.4 `sortino_ratio`
- [ ] 4.5 `drawdown_series`
- [ ] 4.6 `max_drawdown`
- [ ] 4.7 `calmar_ratio`
- [ ] 4.8 `hit_ratio`
- [ ] 4.9 `annual_turnover`
- [ ] `python -m pytest tests/test_strategies.py tests/test_metrics.py tests/test_quantstats_check.py` en verde
- [ ] Notebook `02_estrategias` completo (tabla con y sin costes, sensibilidad a costes)
- [ ] Merge en `main`

## Día 3 · Walk-forward y documentación (rama `feature/walk-forward`)

- [ ] 5.1 `generate_windows`
- [ ] 5.2 `evaluate_params`
- [ ] 5.3 `grid_search`
- [ ] 5.4 `walk_forward`
- [ ] `python -m pytest tests/test_walkforward.py` en verde
- [ ] Notebook `03_walkforward` completo (tres curvas, mapas de calor, Bollinger)
- [ ] README: tabla de resultados y `docs/img/equity.png`
- [ ] README: limitaciones con lo aprendido
- [ ] `python -m pytest --todo-fail` sin fallos
- [ ] Merge en `main`

## Extensión

- [ ] 6.1 `apply_vol_target` (`python -m pytest -m extension`)
- [ ] Sección 7 del notebook 02 (volatility targeting)
