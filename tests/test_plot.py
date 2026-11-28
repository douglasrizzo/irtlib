"""Tests for what every module of irtlib.plot shares: matplotlib as the optional plot extra, and the checks of input."""

import pkgutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.axes import Axes

import irtlib
from irtlib import ItemBank, plot

# Items that every plot of items takes, and each such plot, given its data and waiting for its keyword options.
_ITEMS = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0]]))
_PLOTS: dict[str, Callable[..., Axes]] = {
  "item_curve": partial(plot.item_curve, _ITEMS, 0),
  "bank_information": partial(plot.bank_information, _ITEMS),
  "standard_error": partial(plot.standard_error, _ITEMS),
  "expected_score": partial(plot.expected_score, _ITEMS),
  "reliability": partial(plot.reliability, _ITEMS),
  "relative_efficiency": partial(plot.relative_efficiency, _ITEMS, _ITEMS),
}


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
  """Run ``code`` in a fresh interpreter, whose module cache holds nothing the test session imported."""
  return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_importing_irtlib_loads_no_plotting_module() -> None:
  """Importing irtlib and every module of it other than irtlib.plot loads neither irtlib.plot nor matplotlib.

  The modules are found when the test runs, private ones included, so a module added later is checked too. Fails if any
  of them imports matplotlib, even indirectly, so that ``import irtlib`` would break for users who did not install the
  plot extra, or pay matplotlib's import time without drawing anything.
  """
  result = _run_python(
    "import importlib, pkgutil, sys\n"
    "import irtlib\n"
    "modules = [m.name for m in pkgutil.iter_modules(irtlib.__path__, 'irtlib.') if m.name != 'irtlib.plot']\n"
    "assert len(modules) > 10, modules\n"
    "for name in modules:\n"
    "  importlib.import_module(name)\n"
    "print(sorted(name for name in sys.modules if name.partition('.')[0] in {'matplotlib', 'mpl_toolkits'}"
    " or name == 'irtlib.plot'))\n"
  )

  assert (result.returncode, result.stdout) == (0, "[]\n"), result.stderr


def test_only_matplotlib_contract_names_every_module_but_plot() -> None:
  """The import-linter contract that keeps matplotlib in irtlib.plot names every other module of irtlib.

  The contract checks only the modules it lists, so a module added without being listed could import matplotlib and
  pass the gate. Fails if the list misses a module of irtlib, or names one that no longer exists.
  """
  contracts = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
    "importlinter"
  ]["contracts"]
  (contract,) = [contract for contract in contracts if contract["id"] == "plot-only-matplotlib"]
  modules = {module.name for module in pkgutil.iter_modules(irtlib.__path__, "irtlib.")} - {"irtlib.plot"}

  assert set(contract["source_modules"]) == modules


def test_plot_without_matplotlib_names_the_extra() -> None:
  """Without matplotlib, importing irtlib.plot raises an ImportError that says to install irtlib[plot].

  Fails if the user gets matplotlib's bare ModuleNotFoundError, or an error that drops the original one.
  """
  result = _run_python("import sys\nsys.modules['matplotlib'] = None\nimport irtlib.plot\n")

  assert result.returncode == 1
  assert "The above exception was the direct cause of the following exception" in result.stderr
  assert result.stderr.splitlines()[-1] == (
    "ImportError: irtlib.plot needs matplotlib; install it with: pip install 'irtlib[plot]'"
  )


@pytest.mark.parametrize("draw", _PLOTS.values(), ids=_PLOTS.keys())
def test_plots_draw_on_the_given_axes_with_the_title_and_label(draw: Callable[..., Axes]) -> None:
  """With ax, a plot draws on that axes and returns it, opening no figure; its title and curve carry title and label.

  Fails if a plot opens a figure of its own when given axes, draws on a second axes such as a twin with another y
  scale, drops the title, or leaves its curve without the label, so that a legend of several cannot tell them apart.
  """
  figure, ax = plt.subplots()

  drawn = draw(ax=ax, title="A title", label="A label")

  assert drawn is ax
  assert plt.get_fignums() == [figure.number]
  assert figure.axes == [ax]
  assert ax.get_title() == "A title"
  assert ax.get_lines()[0].get_label() == "A label"


@pytest.mark.parametrize("draw", _PLOTS.values(), ids=_PLOTS.keys())
@pytest.mark.parametrize(
  ("theta", "error", "message"),
  [
    (0.5, ValueError, r"theta has shape \(\); a curve is drawn at a one-dimensional array of at least two abilities"),
    ([], ValueError, r"theta has shape \(0,\)"),
    ([0.5], ValueError, r"theta has shape \(1,\)"),
    ([[0.0, 1.0]], ValueError, r"theta has shape \(1, 2\)"),
    ([0.0, np.nan, 1.0], ValueError, r"theta\[1\] is nan; abilities must be finite numbers"),
    ([0.0, np.inf], ValueError, r"theta\[1\] is inf; abilities must be finite numbers"),
    ([0.0, 1.0, 1.0], ValueError, r"theta\[2\] is 1.0, not above theta\[1\] = 1.0; abilities must increase"),
    ([1.0, 0.0], ValueError, r"theta\[1\] is 0.0, not above theta\[0\] = 1.0; abilities must increase"),
    ([[0.0, 1.0], [2.0]], ValueError, r"theta has a ragged shape; a curve is drawn at a one-dimensional array"),
    ([False, True], TypeError, "theta holds bool values; abilities must be real numbers"),
    ([0.0, True, 2.0], TypeError, r"theta\[1\] is True; abilities must be real numbers, not bools"),
    ([0, 1j], TypeError, "theta holds complex128 values; abilities must be real numbers"),
    (np.array([0.0, 1.0], dtype=object), TypeError, "theta holds object values; abilities must be real numbers"),
    (["-1", "1"], TypeError, "theta holds <U2 values; abilities must be real numbers"),
  ],
  ids=[
    "scalar",
    "empty",
    "one",
    "matrix",
    "nan",
    "inf",
    "repeated",
    "decreasing",
    "ragged",
    "bools",
    "bool-among-numbers",
    "complex",
    "objects",
    "strings",
  ],
)
def test_plots_reject_an_ability_grid_that_draws_no_curve_before_drawing(
  draw: Callable[..., Axes], theta: npt.ArrayLike, error: type[Exception], message: str
) -> None:
  """Abilities that are not a line's x values, at least two, finite and increasing, raise before any figure exists.

  Fails if a plot draws a single point that shows no curve, a curve with a gap at a NaN, a line that doubles back on
  itself through abilities out of order, or flags read as abilities 0 and 1, alone or among numbers; if complex or
  object values pass; if a ragged list raises NumPy's message, which names no argument; or if a plot leaves an empty
  figure open behind the error.
  """
  with pytest.raises(error, match=message):
    draw(theta=theta)
  assert plt.get_fignums() == []
