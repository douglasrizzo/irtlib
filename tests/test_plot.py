"""Tests for matplotlib as the optional plot extra, which every module of irtlib.plot shares."""

import inspect
import pkgutil
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import irtlib
import irtlib.plot


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


def _figure_prefixes(docstring: str) -> list[str | None]:
  """Return the ``:filename-prefix:`` of each ``.. plot::`` directive in a docstring, None for one that sets none."""
  options = re.findall(r"^\.\. plot::\n((?:[ ]+:.*\n)*)", docstring, flags=re.MULTILINE)
  return [match.group(1) if (match := re.search(r":filename-prefix: (\S+)", block)) else None for block in options]


@pytest.mark.parametrize("name", irtlib.plot.__all__)
def test_every_plot_figure_is_saved_under_its_functions_name(name: str) -> None:
  """Each plot function's docstring draws a figure, saved under the function's name.

  The rule: a docstring with one ``.. plot::`` figure sets ``:filename-prefix:`` to the function's name; one with
  several sets each to the name, a hyphen and a suffix, all different. The plot directive numbers the figures of each
  reST block from 1, and the API page renders each section in a block of its own, so figures without a prefix in two
  sections share one file, and the page shows one plot's figure under another's without a warning. Fails if a plot
  function has no figure, or a figure has no prefix, another function's, or one its docstring repeats.
  """
  plot = getattr(irtlib.plot, name)
  prefixes = _figure_prefixes(inspect.getdoc(plot) or "")

  if inspect.isfunction(plot):
    assert prefixes, f"{name} draws no figure in its docstring"
  if len(prefixes) == 1:
    assert prefixes == [name]
  else:
    assert all(prefix is not None and prefix.startswith(f"{name}-") for prefix in prefixes), prefixes
    assert len(set(prefixes)) == len(prefixes), prefixes


@pytest.mark.parametrize(
  ("docstring", "prefixes"),
  [
    ("Draw.\n\n.. plot::\n    :caption: A figure.\n\n    plot()\n", [None]),
    ("Draw.\n\n.. plot::\n    :filename-prefix: a\n    :caption: A.\n\n.. plot::\n    :caption: B.\n", ["a", None]),
    ("Draw.\n", []),
  ],
  ids=["no-prefix", "two-figures", "no-figure"],
)
def test_figure_prefixes_reads_each_directives_prefix(docstring: str, prefixes: list[str | None]) -> None:
  """The prefix reader finds every ``.. plot::`` directive and its prefix, or None where it has none.

  Fails if a directive without a prefix is skipped, which would let the rule above pass a figure that has none.
  """
  assert _figure_prefixes(docstring) == prefixes
