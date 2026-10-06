# Quant-Backtester

Motor de backtesting vectorizado en pandas, construido desde cero (sin `backtrader` ni `vectorbt`) para entender la mecánica completa: señales → posiciones → rendimientos, con costes de transacción, slippage y validación *walk-forward*.

Se prueban dos familias de estrategias clásicas —momentum (cruce de medias) y mean reversion (bandas de Bollinger)— sobre un universo pequeño de acciones y ETFs líquidos descargados con `yfinance`.

> **Objetivo del proyecto:** no encontrar "la estrategia ganadora", sino mostrar con rigor por qué la mayoría de estrategias que lucen bien in-sample no sobreviven fuera de muestra.

---

## Resultados

*(Pendiente: tabla comparativa y gráfico de equity de las estrategias vs. buy-and-hold, con y sin costes, y curva walk-forward.)*

| Estrategia | CAGR | Vol. | Sharpe | Sortino | Max DD | Calmar | Turnover anual |
|---|---|---|---|---|---|---|---|
| Buy & Hold | | | | | | | |
| Momentum (cruce MM) | | | | | | | |
| Momentum 12m | | | | | | | |
| Mean reversion (Bollinger) | | | | | | | |
| Walk-forward | | | | | | | |

---

## Estructura del repositorio

```
quant-backtester/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py          # descarga y limpieza (yfinance)
│   ├── strategies.py    # señales: momentum, bollinger
│   ├── backtest.py      # motor: señales → posiciones → P&L
│   ├── metrics.py       # sharpe, sortino, drawdown, calmar
│   └── walkforward.py   # validación out-of-sample
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_estrategias.ipynb
│   └── 03_walkforward.ipynb
└── tests/
    └── test_backtest.py
```

---

## Plan de desarrollo (3 días)

### Día 1 — Datos y motor

**`data.py`**
- Descarga de 8-10 tickers líquidos (SPY, QQQ, GLD, TLT, AAPL, MSFT…) desde 2010 hasta hoy.
- Precios ajustados, retornos logarítmicos y simples.
- Guardado en parquet para no depender de `yfinance` en cada ejecución.

**`backtest.py`** — el núcleo, vectorizado en pandas
- Entrada: DataFrame de precios y DataFrame de señales (-1, 0, 1) por activo.
- Posiciones = señal desplazada un día (`shift(1)`): se opera al cierre de mañana con la información de hoy. Es el error número uno de los backtests caseros y está documentado explícitamente.
- Retorno bruto = posición × retorno del activo.
- Costes: `turnover = |posición_t − posición_{t−1}|`; coste = turnover × (comisión en bps + slippage en bps). Valores por defecto: 5 bps + 5 bps.
- Salida: retornos netos, curva de equity, posiciones y turnover acumulado.

**`tests/test_backtest.py`**
- Señal constante en 1 debe reproducir el buy-and-hold menos un coste inicial.
- Señal constante en 0 debe dar retorno cero.

### Día 2 — Estrategias y métricas

**`strategies.py`**
- Momentum: cruce de medias móviles (p. ej. 50/200) y variante de momentum de 12 meses con rebalanceo mensual.
- Mean reversion: bandas de Bollinger (20 días, 2σ); largo bajo la banda inferior, salida en la media.
- Todas las estrategias devuelven señales en el mismo formato, de modo que el motor es agnóstico.

**`metrics.py`**
- CAGR, volatilidad anualizada, Sharpe, Sortino, max drawdown con fechas, Calmar, hit ratio, turnover anual.
- Sharpe implementado a mano y verificado contra `quantstats`.

**`notebooks/02_estrategias.ipynb`**
- Tabla comparativa de las estrategias frente a buy-and-hold, con y sin costes.
- Gráficos de equity y drawdown.
- Análisis del impacto de los costes en estrategias de alto turnover (mean reversion).

### Día 3 — Walk-forward y documentación

**`walkforward.py`**
- Ventanas rodantes: 3 años de entrenamiento, 1 año de test, paso anual.
- En cada ventana, grid search reducido de parámetros (ventanas de medias, σ de Bollinger) en entrenamiento; aplicación del mejor en test.
- Concatenación de los tramos de test: la única curva de equity "honesta".

**`notebooks/03_walkforward.ipynb`**
- Comparación de tres curvas: parámetros fijos, parámetros optimizados in-sample sobre todo el histórico (sobreajuste) y walk-forward.
- Mapa de calor de Sharpe por combinación de parámetros, para mostrar que las zonas "buenas" son estrechas e inestables.

**README**
- Resultados en la primera pantalla (tabla y gráfico).
- Instrucciones de ejecución.
- Decisiones de diseño: el `shift`, el modelo de costes.
- Sección de limitaciones.

### Extensiones (si sobra tiempo)
- *Volatility targeting*: escalar la posición para fijar una volatilidad objetivo del 10 %.
- Tests para el módulo de walk-forward.

---

## Instalación y ejecución

```bash
git clone https://github.com/<usuario>/quant-backtester.git
cd quant-backtester
pip install -r requirements.txt
python -m src.data          # descarga y guarda los datos
pytest                      # ejecuta los tests
```

Después, abre los notebooks en orden (`01` → `02` → `03`).

**Dependencias principales:** `pandas`, `numpy`, `yfinance`, `matplotlib`, `pyarrow`, `pytest`, `quantstats` (solo para verificación de métricas).

---

## Decisiones de diseño

- **Sin look-ahead bias:** las señales se desplazan un día antes de convertirse en posiciones.
- **Costes explícitos:** comisión y slippage en puntos básicos sobre el turnover; se pueden anular para comparar con el resultado bruto.
- **Motor agnóstico:** cualquier estrategia que devuelva un DataFrame de señales (-1, 0, 1) se puede backtestear sin tocar el motor.
- **Validación temporal:** el walk-forward es el único resultado que se considera representativo; las cifras in-sample se muestran solo como contraste.

---

## Limitaciones

- Sesgo de supervivencia: el universo de tickers se elige en el presente.
- Datos diarios al cierre; no se modela intradía ni gaps de apertura.
- Sin modelo de impacto de mercado ni de liquidez.
- Posiciones cortas sin coste de préstamo ni restricciones realistas.
- Sin apalancamiento ni gestión de margen.
