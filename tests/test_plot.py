"""Tests for matplotlib as the optional plot extra, which every module of irtlib.plot shares."""

import pkgutil
import subprocess
import sys
import tomllib
from pathlib import Path

import irtlib


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
