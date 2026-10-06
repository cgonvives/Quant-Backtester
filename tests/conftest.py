"""Configuración de pytest: fixtures sintéticas y el "autocorrector" de los TODO.

Un test que llega a un TODO sin implementar (NotImplementedError) se marca como
SKIPPED "TODO pendiente: ...", no como fallo: así ves "N passed, M skipped" mientras
avanzas, en lugar de un muro rojo. Al final se imprime la lista de TODO pendientes.

    python -m pytest                     # todo
    python -m pytest -k simple_returns   # solo los tests de una función
    python -m pytest --todo-fail         # los TODO pendientes cuentan como fallos
"""

from __future__ import annotations

import re
from collections import Counter

import matplotlib
import pytest

from tests import helpers

matplotlib.use("Agg")  # los tests de gráficos no abren ventanas

_PENDING = pytest.StashKey[Counter]()
_TODO_ID = re.compile(r"TODO (\d+)\.(\d+)")


def pytest_addoption(parser):
    parser.addoption(
        "--todo-fail",
        action="store_true",
        default=False,
        help="Cuenta los TODO pendientes (NotImplementedError) como fallos en vez de como skipped.",
    )


def pytest_configure(config):
    config.stash[_PENDING] = Counter()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    excinfo = call.excinfo
    if (
        excinfo is not None
        and excinfo.errisinstance(NotImplementedError)
        and not item.config.getoption("--todo-fail")
    ):
        message = str(excinfo.value) or "NotImplementedError"
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] + 1, f"TODO pendiente: {message}")
        item.config.stash[_PENDING][message] += 1
    return report


def _todo_order(message: str) -> tuple[int, int]:
    match = _TODO_ID.search(message)
    return (int(match.group(1)), int(match.group(2))) if match else (99, 99)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    pending = config.stash.get(_PENDING, Counter())
    if not pending:
        return
    terminalreporter.section("TODO pendientes")
    for message in sorted(pending, key=_todo_order):
        n = pending[message]
        terminalreporter.write_line(f"  {message}   ({n} test{'s' if n != 1 else ''} esperando)")
    terminalreporter.write_line(
        "Ve a por el primero de la lista. Detalle de un test:  python -m pytest -k <nombre> -rs"
    )


# ---------------------------------------------------------------------------
# Fixtures: precios sintéticos (semillas fijas, sin red)
# ---------------------------------------------------------------------------
@pytest.fixture
def gbm_prices():
    """3 activos, 5 años de paseo aleatorio geométrico."""
    return helpers.gbm()


@pytest.fixture
def ou_prices():
    """2 activos que revierten a la media."""
    return helpers.ou()


@pytest.fixture
def trend_prices():
    """UP sube en línea recta; DOWN baja en línea recta."""
    return helpers.trend()


@pytest.fixture
def regime_prices():
    """2010-2020 con un régimen alcista o bajista por año (helpers.REGIME_SIGNS)."""
    return helpers.regimes()
