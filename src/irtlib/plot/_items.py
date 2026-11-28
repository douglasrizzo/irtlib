"""Plots of items: their characteristic and information curves."""

from enum import StrEnum
from typing import assert_never

import numpy
from matplotlib.axes import Axes

from .. import irt
from ..bank import ItemBank
from ..typing import FloatArray
from ._common import HALF_WIDTH, POINTS, as_bank, item_position, prepare_axes, text_box


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


def item_curve(  # noqa: PLR0913 - the items and the item, the curve and its two marks, and every plot's ax, title, label
  items: ItemBank | FloatArray,
  item: int,
  *,
  kind: Curve | str = Curve.ICC,
  max_info: bool = True,
  parameters: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot one item's characteristic curve or its information curve.

  The curve spans four units of ability on each side of the item's difficulty, and the information curve also four on
  each side of its peak. It is labelled with the item's name for a legend. A box in the upper left corner gives the
  item's name and parameters, to three significant digits. Both curves of an item go on two axes that share the
  ability axis, one call each, as the example shows.

  .. plot::
      :caption: One item's characteristic curve above and its information curve below. The dot marks the peak.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from irtlib.plot import item_curve

      items = ItemBank.generate(1, "3PL", rng=7)
      fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
      item_curve(items, 0, ax=top)
      item_curve(items, 0, kind="iic", parameters=False, ax=bottom)
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
  max_info : bool, optional
      Whether to mark the ability at which the item gives the most information with a dot, on the information curve
      only. Default is True.
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
      If ``item`` is not an integer, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``item`` is not a position in it, or ``kind`` is not a
      :py:class:`Curve` or one of its values, before any figure is created.
  """
  items = as_bank(items)
  item = item_position("item", item, items.n_items)
  kind = Curve(kind)
  row = items.items[[item]]
  # The information of a flat item with a guessing parameter peaks far from its difficulty, so its curve spans both.
  peak = float(irt.max_info_theta(row)[0])
  centres = (items.difficulty[item],) if kind is Curve.ICC else (items.difficulty[item], peak)
  theta = numpy.linspace(min(centres) - HALF_WIDTH, max(centres) + HALF_WIDTH, POINTS)
  label = items.names[item] if label is None else label
  ax = prepare_axes(ax, title)

  match kind:
    case Curve.ICC:
      ax.plot(theta, irt.icc(theta, row)[:, 0], label=label)
      ax.set_ylabel(r"$P(\theta)$")
    case Curve.IIC:
      (line,) = ax.plot(theta, irt.information(theta, row)[:, 0], label=label)
      ax.set_ylabel(r"$I(\theta)$")
      if max_info:
        ax.plot(peak, irt.information(peak, row)[0], "o", color=line.get_color(), label="Maximum information")
    case _:
      assert_never(kind)

  ax.set_xlabel(r"$\theta$")
  if parameters:
    _annotate(ax, items.names[item], row[0])
  return ax


def _annotate(ax: Axes, name: str, parameters: FloatArray) -> None:
  """Write the item's name and parameters, to three significant digits, in a box in the upper left corner of the axes.

  The curves of typical items keep clear of that corner: at the low end of the ability range the characteristic curve
  is near its lower asymptote and the information curve near zero.
  """
  a, b, c, d = parameters
  # The parameters are mathtext, so a dollar sign in the name would pair up with theirs.
  shown = name.replace("$", r"\$")
  text_box(ax, f"{shown}\n$a = {a:.3g}$\n$b = {b:.3g}$\n$c = {c:.3g}$\n$d = {d:.3g}$")
