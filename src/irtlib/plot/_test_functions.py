"""Plots of a test against ability: its information, standard error, expected score, reliability and efficiency."""

import math

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.markers import TICKUP

from .. import irt
from ..bank import ItemBank
from ..typing import FloatArray
from ._common import ability_grid, as_bank, prepare_axes, reference_line

_THETA = numpy.linspace(-4, 4, 161)
"""The abilities a test's curves are drawn at when the caller gives none."""
_THETA.setflags(write=False)
_INTERVAL_BOUNDS = 2
"""The bounds an interval of abilities has: a lower and an upper one."""


def bank_information(  # noqa: PLR0913 - the items, their abilities and two marks, and every plot's ax, title, label
  items: ItemBank | FloatArray,
  *,
  theta: npt.ArrayLike | None = None,
  locations: bool = False,
  interval: tuple[float, float] | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the test information of a set of items against ability: the sum of the items' information.

  To compare item sets, such as the forms of a test, draw each on one axes with its ``label``. The two options mark
  where the items measure: ``locations`` ticks each item's location along the bottom of the axes, and ``interval``
  shades the information over an interval of abilities and labels the shading, for a legend, with that information and
  its share of the test's information over the whole ability scale.

  .. plot::
      :filename-prefix: bank_information
      :caption: The information of a 20-item test and of each of its halves.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import bank_information

      items = ItemBank.generate(20, "3PL", rng=11)
      ax = bank_information(items, label="All 20 items")
      bank_information(items.items[:10], ax=ax, label="Items 1 to 10")
      bank_information(items.items[10:], ax=ax, label="Items 11 to 20")
      ax.legend()
      plt.show()

  .. plot::
      :filename-prefix: bank_information-interval
      :caption: The information of the same test, each item's location ticked, and its share between -1 and 1.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import bank_information

      items = ItemBank.generate(20, "3PL", rng=11)
      ax = bank_information(items, locations=True, interval=(-1, 1))
      ax.legend(loc="upper left")
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix; a subset of a bank's items is a subset of its matrix's rows.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which draws it from -4
      to 4.
  locations : bool, optional
      Whether to tick each item's location, the ability at which it gives the most information (its difficulty for a
      one- or two-parameter item), in the curve's colour, for the items located within ``theta``. The ticks are
      labelled "Item locations", after ``label`` when one is given. Default is False.
  interval : tuple of float or None, optional
      The lower and upper bound of the abilities to shade the information over, possibly infinite. The shading keeps
      within ``theta``, and its label gives the information over the whole interval even where none of it is drawn.
      Default is None, which shades nothing.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``theta`` or ``interval`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``theta`` is not a one-dimensional array of at least two finite and
      increasing abilities, or ``interval`` is not a pair whose lower bound is below its upper bound, before any figure
      is created.
  """
  bank = as_bank(items)
  grid = _grid(theta)
  bounds = None if interval is None else _bounds(interval)
  ax = prepare_axes(ax, title)
  line = _draw(ax, grid, irt.total_information(grid, bank.items), ylabel=r"Information $I(\theta)$", label=label)
  if locations:
    # Within the curve's abilities only: a tick beyond them would stretch the axes past the abilities drawn.
    shown = bank.max_info_thetas[(bank.max_info_thetas >= grid[0]) & (bank.max_info_thetas <= grid[-1])]
    # x in data units and y in axes units, so the ticks stand on the bottom of the axes whatever their y limits, and a
    # legend shows a tick, where a collection of lines would show a horizontal line like the curve's.
    ax.plot(
      shown,
      numpy.zeros(shown.size),
      linestyle="none",
      marker=TICKUP,
      color=line.get_color(),
      transform=ax.get_xaxis_transform(),
      label="Item locations" if label is None else f"{label}: item locations",
    )
  if bounds is not None:
    _shade(ax, grid, bank.items, bounds, line)
  return ax


def standard_error(
  items: ItemBank | FloatArray,
  *,
  theta: npt.ArrayLike | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the standard error of measurement of a set of items against ability: one over the root of the information.

  The standard error is in units of ability. Where the items give no information it is infinite, and the curve leaves
  those abilities out. The information and the standard error of one test go on two axes that share the ability axis,
  one call each, as the example shows.

  .. plot::
      :filename-prefix: standard_error
      :caption: The information of a 20-item test and of its first ten items above, their standard errors below.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import bank_information, standard_error

      items = ItemBank.generate(20, "3PL", rng=11)
      fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
      for rows, label in [(items.items, "All 20 items"), (items.items[:10], "Items 1 to 10")]:
          bank_information(rows, ax=top, label=label)
          standard_error(rows, ax=bottom, label=label)
      top.legend()
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix; a subset of a bank's items is a subset of its matrix's rows.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which draws it from -4
      to 4.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, or ``theta`` is not a one-dimensional array of at least two finite and
      increasing abilities, before any figure is created.
  """
  bank = as_bank(items)
  grid = _grid(theta)
  ax = prepare_axes(ax, title)
  information = irt.total_information(grid, bank.items)
  errors = numpy.full_like(information, numpy.nan)
  # NaN, which a line leaves out, rather than the division by zero's warning and an infinite value.
  numpy.divide(1, numpy.sqrt(information), out=errors, where=information > 0)
  _draw(ax, grid, errors, ylabel=r"Standard error $1/\sqrt{I(\theta)}$", label=label)
  return ax


def expected_score(
  items: ItemBank | FloatArray,
  *,
  theta: npt.ArrayLike | None = None,
  proportion: bool = False,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the expected number of correct responses to a set of items against ability: the test characteristic curve.

  The expected number correct is the sum of the items' characteristic curves. With ``proportion``, the curve is the
  expected proportion correct, the relative score of :cite:t:`Wright1979` (p. 132), on an axis from 0 to 1.

  .. plot::
      :filename-prefix: expected_score
      :caption: The expected number correct on a 20-item test.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import expected_score

      expected_score(ItemBank.generate(20, "3PL", rng=11))
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix; a subset of a bank's items is a subset of its matrix's rows.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which draws it from -4
      to 4.
  proportion : bool, optional
      Whether to draw the expected proportion correct rather than the number. Default is False.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, or ``theta`` is not a one-dimensional array of at least two finite and
      increasing abilities, before any figure is created.
  """
  bank = as_bank(items)
  grid = _grid(theta)
  ax = prepare_axes(ax, title)
  scores = irt.icc(grid, bank.items).sum(axis=1)
  if proportion:
    _draw(ax, grid, scores / bank.n_items, ylabel="Expected proportion correct", label=label)
    ax.set_ylim(0, 1)
  else:
    _draw(ax, grid, scores, ylabel="Expected number correct", label=label)
  return ax


def reliability(
  items: ItemBank | FloatArray,
  *,
  theta: npt.ArrayLike | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the reliability of a set of items against ability, for abilities distributed N(0, 1).

  The reliability at each ability is :py:func:`irtlib.irt.reliability`'s, on an axis from 0 to 1: values close to 1 mean
  a precise test there.

  .. plot::
      :filename-prefix: reliability
      :caption: The reliability of a 20-item test and of its first ten items.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import reliability

      items = ItemBank.generate(20, "3PL", rng=11)
      ax = reliability(items, label="All 20 items")
      reliability(items.items[:10], ax=ax, label="Items 1 to 10")
      ax.legend()
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix; a subset of a bank's items is a subset of its matrix's rows.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which draws it from -4
      to 4.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, or ``theta`` is not a one-dimensional array of at least two finite and
      increasing abilities, before any figure is created.
  """
  bank = as_bank(items)
  grid = _grid(theta)
  ax = prepare_axes(ax, title)
  values = numpy.array([irt.reliability(float(ability), bank.items) for ability in grid])
  _draw(ax, grid, values, ylabel="Reliability", label=label)
  ax.set_ylim(0, 1)
  return ax


def relative_efficiency(
  items: ItemBank | FloatArray,
  reference: ItemBank | FloatArray,
  *,
  theta: npt.ArrayLike | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the relative efficiency of a set of items over a reference set against ability: their information's ratio.

  The relative efficiency (:cite:t:`Wright1982`, p. 177) is the information of ``items`` over that of ``reference``
  at each ability. Above the dashed line at 1, ``items`` measure more precisely. Where ``reference`` gives no
  information, the curve leaves those abilities out.

  .. plot::
      :filename-prefix: relative_efficiency
      :caption: The relative efficiency of the first ten items of a 20-item test over its last ten.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import relative_efficiency

      items = ItemBank.generate(20, "3PL", rng=11)
      relative_efficiency(items.items[:10], items.items[10:])
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items whose efficiency is drawn, as a bank or as an item matrix.
  reference : irtlib.ItemBank or numpy.ndarray
      The items they are compared with, as a bank or as an item matrix.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which draws it from -4
      to 4.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` or ``reference`` is not a valid item matrix, or ``theta`` is not a one-dimensional array of at least
      two finite and increasing abilities, before any figure is created.
  """
  bank, reference_bank = as_bank(items), as_bank(reference)
  grid = _grid(theta)
  ax = prepare_axes(ax, title)
  information = irt.total_information(grid, bank.items)
  reference_information = irt.total_information(grid, reference_bank.items)
  ratio = numpy.full_like(information, numpy.nan)
  # NaN, which a line leaves out, rather than the division by zero's warning and an infinite value.
  numpy.divide(information, reference_information, out=ratio, where=reference_information > 0)
  _draw(ax, grid, ratio, ylabel="Relative efficiency", label=label)
  reference_line(ax, 1.0)
  return ax


def _grid(theta: npt.ArrayLike | None) -> FloatArray:
  """Return the abilities to draw at: the caller's, once checked, or the default ones."""
  return _THETA if theta is None else ability_grid(theta)


def _bounds(interval: tuple[float, float]) -> tuple[float, float]:
  """Return an interval of abilities as two floats, after checking that its lower bound is below its upper bound."""
  values = numpy.asarray(interval)
  # Booleans would be read as abilities 0 and 1.
  if values.dtype.kind not in "iuf":
    msg = f"interval holds {values.dtype} values; its bounds must be real numbers"
    raise TypeError(msg)
  if values.shape != (_INTERVAL_BOUNDS,):
    msg = f"interval has shape {values.shape}; it must be a pair of bounds, (lower, upper)"
    raise ValueError(msg)
  lower, upper = (float(bound) for bound in values)
  # Negated, so that a NaN bound fails it too.
  if not lower < upper:
    msg = f"interval is ({lower}, {upper}); its lower bound must be below its upper bound"
    raise ValueError(msg)
  return lower, upper


def _draw(ax: Axes, grid: FloatArray, values: FloatArray, *, ylabel: str, label: str | None) -> Line2D:
  """Draw ``values`` against the abilities ``grid`` in the next colour of the axes, and name both axes."""
  (line,) = ax.plot(grid, values, label=label)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel(ylabel)
  return line


def _shade(ax: Axes, grid: FloatArray, items: FloatArray, bounds: tuple[float, float], line: Line2D) -> None:
  """Shade the test information over ``bounds``, within ``grid``, labelled with that information and its share."""
  lower, upper = bounds
  part = float(irt.info_integral(lower, upper, items).sum())
  # The share is of the information over the whole ability scale, as mirt's areainfo integrates it (R/areainfo.R:66),
  # although its manual (p. 7) says -10 to 10: a flat item, a = 0.3 and b = 0, has 9.5% of its information beyond them.
  total = float(irt.info_integral(-math.inf, math.inf, items).sum())
  low, high = max(lower, grid[0]), min(upper, grid[-1])
  # The shading starts and ends at the bounds themselves, not at the abilities of grid nearest to them.
  inside = grid[(grid > low) & (grid < high)]
  shaded = numpy.concatenate(([low], inside, [high])) if low < high else numpy.empty(0)
  ax.fill_between(
    shaded,
    irt.total_information(shaded, items),
    color=line.get_color(),
    alpha=0.2,
    linewidth=0,
    label=f"Information in [{lower:g}, {upper:g}]: {part:.3g} of {total:.3g} ({part / total:.1%})",
  )
