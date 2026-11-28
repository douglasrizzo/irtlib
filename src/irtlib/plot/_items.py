"""Plots of items: their characteristic and information curves."""

from enum import StrEnum
from typing import assert_never

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes

from .. import irt
from ..bank import ItemBank
from ..typing import FloatArray
from ._common import ability_grid, as_bank, item_position, prepare_axes

_HALF_WIDTH = 4.0
"""How far the curves of one item span on each side of its difficulty, in units of ability."""
_POINTS = 161
"""The abilities each curve is evaluated at."""


class Curve(StrEnum):
  """The curves :py:func:`item_curve` draws.

  Attributes
  ----------
  ICC
      The item characteristic curve, the probability of a correct response against ability.
  IIC
      The item information curve, the item's information against ability.
  """

  ICC = "icc"
  IIC = "iic"


def item_curve(  # noqa: PLR0913 - items and item, the curve, its abilities and two marks, every plot's ax, title, label
  items: ItemBank | FloatArray,
  item: int,
  *,
  kind: Curve | str = Curve.ICC,
  theta: npt.ArrayLike | None = None,
  max_info: bool = True,
  parameters: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot one item's characteristic curve or its information curve.

  The curve spans four units of ability on each side of the item's difficulty, and the information curve also four on
  each side of its peak, unless ``theta`` gives the abilities. It is labelled with the item's name for a legend. A box
  in the upper corner over the curve's lower end gives the item's name and parameters, to three significant digits.
  Both curves of an item go on two axes that share the ability axis, one call each, as the first example shows.

  .. plot::
      :filename-prefix: item_curve
      :caption: One item's characteristic curve above and its information curve below. The dot marks the peak.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import item_curve

      items = ItemBank.generate(1, "3PL", rng=7)
      fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
      item_curve(items, 0, ax=top)
      item_curve(items, 0, kind="iic", parameters=False, ax=bottom)
      plt.show()

  .. plot::
      :filename-prefix: item_curve-items
      :caption: The characteristic curves of four items above and their information curves below, on one range.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.plot import item_curve

      items = ItemBank.generate(4, "3PL", rng=2)
      theta = numpy.linspace(-4, 4, 161)
      fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
      for item in range(items.n_items):
          item_curve(items, item, theta=theta, parameters=False, ax=top)
          item_curve(items, item, kind="iic", theta=theta, max_info=False, parameters=False, ax=bottom)
      top.legend()
      plt.show()

  .. plot::
      :filename-prefix: item_curve-recovered
      :caption: The characteristic curve of a three-parameter item, and the curve of the two-parameter model calibrated
          from 1,000 examinees' responses to it and nine other items.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.mml import MMLCalibrator
      from irtlib.plot import item_curve

      generating = ItemBank.generate(10, "3PL", rng=3)
      abilities = numpy.random.default_rng(4).normal(size=1000)
      calibration = MMLCalibrator("2PL").calibrate(draw_responses(abilities, generating.items, rng=5))
      theta = numpy.linspace(-4, 4, 161)
      ax = item_curve(generating, 9, theta=theta, parameters=False, label="Generating, 3PL")
      item_curve(calibration.items, 9, theta=theta, parameters=False, ax=ax, label="Calibrated, 2PL")
      ax.legend()
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank, whose names label them, or as an item matrix, whose items are named by position.
  item : int
      The item's position in ``items``; ``ItemBank.index(name)`` gives the position of a named item.
  kind : Curve or str, optional
      The curve: ``Curve.ICC`` or ``"icc"`` for the characteristic curve, the default, ``Curve.IIC`` or ``"iic"`` for
      the information curve.
  theta : array-like or None, optional
      The abilities to draw the curve at, at least two, finite and increasing. Default is None, which spans the item as
      described above.
  max_info : bool, optional
      Whether to mark the ability at which the item gives the most information with a dot, on the information curve
      only, and when ``theta`` is given only if that ability lies within it, its ends included. Default is True.
  parameters : bool, optional
      Whether to write the item's name and parameters in a box. Default is True.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which labels it with the item's name.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If ``item`` is not an integer, or ``theta`` holds values that are not real numbers, bools included, before any
      figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``item`` is not a position in it, ``kind`` is not a :py:class:`Curve`
      or one of its values, or ``theta`` is not a one-dimensional array of at least two finite and increasing
      abilities, before any figure is created.
  """
  items = as_bank(items)
  item = item_position("item", item, items.n_items)
  kind = Curve(kind)
  row = items.items[[item]]
  peak = float(irt.max_info_theta(row)[0])
  if theta is None:
    # The information of a flat item with a guessing parameter peaks far from its difficulty, so its curve spans both.
    centres = (items.difficulty[item],) if kind is Curve.ICC else (items.difficulty[item], peak)
    grid = numpy.linspace(min(centres) - _HALF_WIDTH, max(centres) + _HALF_WIDTH, _POINTS)
  else:
    grid = ability_grid(theta)
  label = items.names[item] if label is None else label
  ax = prepare_axes(ax, title)

  match kind:
    case Curve.ICC:
      values = irt.icc(grid, row)[:, 0]
      ax.plot(grid, values, label=label)
      ax.set_ylabel(r"$P(\theta)$")
    case Curve.IIC:
      values = irt.information(grid, row)[:, 0]
      (line,) = ax.plot(grid, values, label=label)
      ax.set_ylabel(r"$I(\theta)$")
      # A dot beyond the end of the curve would mark a peak the curve does not show.
      if max_info and grid[0] <= peak <= grid[-1]:
        ax.plot(peak, irt.information(peak, row)[0], "o", color=line.get_color(), label="Maximum information")
    case _:
      assert_never(kind)

  ax.set_xlabel(r"$\theta$")
  if parameters:
    _annotate(ax, items.names[item], row[0], left=values[0] <= values[-1])
  return ax


def _annotate(ax: Axes, name: str, parameters: FloatArray, *, left: bool) -> None:
  """Write the item's name and parameters, to three significant digits, in a box in an upper corner of the axes.

  The corner is the left one with ``left`` and the right one otherwise, the one over the curve's lower end: a curve
  keeps clear of it while it falls toward that end, as a characteristic curve does toward the left and an information
  curve toward the side away from its peak. The box takes the axes' face and edge colours, so it follows the caller's
  style.
  """
  a, b, c, d = parameters
  # The parameters are mathtext, so a dollar sign in the name would pair up with theirs.
  shown = name.replace("$", r"\$")
  ax.annotate(
    f"{shown}\n$a = {a:.3g}$\n$b = {b:.3g}$\n$c = {c:.3g}$\n$d = {d:.3g}$",
    bbox={"facecolor": ax.get_facecolor(), "edgecolor": ax.spines["left"].get_edgecolor()},
    xy=(0.02, 0.97) if left else (0.98, 0.97),
    xycoords="axes fraction",
    ha="left" if left else "right",
    va="top",
  )
