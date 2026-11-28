"""What the plot modules share: the axes to draw on, the checks of their arguments, and the marks they draw alike."""

import operator
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.typing import ColorType

from ..bank import ItemBank
from ..typing import FloatArray, IntArray

HALF_WIDTH = 4.0
"""How far an item's curves span on each side of its difficulty, in units of ability."""
POINTS = 161
"""The abilities each curve is evaluated at."""
_MAX_NAMED_TICKS = 30
"""The most items whose names :py:func:`label_items` writes on the ticks."""


def prepare_axes(ax: Axes | None, title: str | None) -> Axes:
  """Return ``ax``, or the axes of a new figure, with ``title`` set when one is given.

  Parameters
  ----------
  ax : matplotlib.axes.Axes or None
      The axes the caller passed, or None for the axes of a new figure.
  title : str or None
      The title to set, or None to set none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes to draw on.
  """
  if ax is None:
    _, ax = plt.subplots()
  if title is not None:
    ax.set_title(title)
  return ax


def as_bank(items: ItemBank | FloatArray) -> ItemBank:
  """Return the items as an item bank: ``items`` itself, or a bank of the matrix with generated names.

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      An item bank, or an item matrix as :py:class:`irtlib.ItemBank` reads one.

  Returns
  -------
  irtlib.ItemBank
      The items, with their names.
  """
  return items if isinstance(items, ItemBank) else ItemBank(numpy.asarray(items, dtype=float))


def item_position(label: str, value: object, n_items: int) -> int:
  """Return an item's position in a bank as an int, after checking that it is an integer in the bank's range.

  Parameters
  ----------
  label : str
      The argument's name, for the message.
  value : object
      The position the caller passed.
  n_items : int
      The number of items in the bank.

  Returns
  -------
  int
      The position.

  Raises
  ------
  TypeError
      If ``value`` is not an integer, a bool included.
  ValueError
      If ``value`` is negative, or not below ``n_items``.
  """
  return _position(label, value, n_items, "item")


def examinee_position(label: str, value: object, n_examinees: int) -> int:
  """Return an examinee's row in a response matrix as an int, after checking that it is an integer in its range.

  Parameters
  ----------
  label : str
      The argument's name, for the message.
  value : object
      The row the caller passed.
  n_examinees : int
      The number of rows of the response matrix.

  Returns
  -------
  int
      The row.

  Raises
  ------
  TypeError
      If ``value`` is not an integer, a bool included.
  ValueError
      If ``value`` is negative, or not below ``n_examinees``.
  """
  return _position(label, value, n_examinees, "examinee")


def _position(label: str, value: object, size: int, kind: str) -> int:
  """Return a position among ``size`` items or examinees as an int, after checking that it is an integer in range."""
  # bool is an int to operator.index, but a flag passed by mistake is not a position.
  if isinstance(value, (bool, numpy.bool_)):
    msg = f"{label} is {value!r}; {kind} positions must be integers, not bools"
    raise TypeError(msg)
  try:
    position = operator.index(value)  # ty: ignore[invalid-argument-type]
  except TypeError:
    # NumPy would index with a float such as 3.7 as position 3.
    msg = f"{label} is {value!r}; {kind} positions must be integers"
    raise TypeError(msg) from None
  # NumPy would read a negative position as one counted from the end.
  if not 0 <= position < size:
    msg = f"{label} is {position}; {kind} positions must be between 0 and {size - 1}"
    raise ValueError(msg)
  return position


def abilities(theta: npt.ArrayLike, n_examinees: int) -> FloatArray:
  """Return the abilities as a new float array, after checking that they are one real number per examinee.

  Parameters
  ----------
  theta : array_like
      The abilities the caller passed, one per examinee.
  n_examinees : int
      The number of examinees.

  Returns
  -------
  numpy.ndarray
      A float copy of the abilities, which may hold infinities and NaN for the caller to check where it needs them.

  Raises
  ------
  TypeError
      If ``theta`` holds anything but integers and floats, Booleans included.
  ValueError
      If ``theta`` does not hold one ability per examinee.
  """
  values = numpy.asarray(theta)
  # Booleans would be read as abilities 0 and 1, and None as NaN.
  if values.dtype.kind not in "iuf":
    msg = f"theta holds {values.dtype} values; abilities must be real numbers"
    raise TypeError(msg)
  if values.shape != (n_examinees,):
    msg = f"theta must hold one ability per examinee, {n_examinees}, got shape {values.shape}"
    raise ValueError(msg)
  return values.astype(float)


def check_level(level: float) -> float:
  """Return the confidence level of an interval, after checking that it lies strictly between 0 and 1.

  Parameters
  ----------
  level : float
      The level the caller passed, such as 0.95 for 95% intervals.

  Returns
  -------
  float
      The level.

  Raises
  ------
  ValueError
      If ``level`` is not strictly between 0 and 1, or is NaN.
  """
  if not 0 < level < 1:
    msg = f"level is {level}; a confidence level must lie strictly between 0 and 1"
    raise ValueError(msg)
  return level


def reference_style(ax: Axes) -> tuple[ColorType, float]:
  """Return the colour and width of the axes' spines, for the reference lines a plot draws behind its data.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The axes the lines go on, whose style they follow.

  Returns
  -------
  tuple[matplotlib.typing.ColorType, float]
      The colour and the width of a line.
  """
  spine = ax.spines["left"]
  return spine.get_edgecolor(), spine.get_linewidth()


def text_box(ax: Axes, text: str, *, lower_right: bool = False) -> None:
  """Write ``text`` in a box in a corner of the axes, in the axes' face and edge colours.

  The box takes its colours from the axes, so it follows the caller's style.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The axes to write on.
  text : str
      The text, which may hold mathtext between dollar signs.
  lower_right : bool, optional
      Whether the box goes in the lower right corner. Default is False, which puts it in the upper left one.
  """
  ax.annotate(
    text,
    bbox={"facecolor": ax.get_facecolor(), "edgecolor": ax.spines["left"].get_edgecolor()},
    xy=(0.98, 0.03) if lower_right else (0.02, 0.97),
    xycoords="axes fraction",
    ha="right" if lower_right else "left",
    va="bottom" if lower_right else "top",
    multialignment="left",
  )


def name_points(ax: Axes, x: FloatArray, y: FloatArray, names: Sequence[str]) -> None:
  """Write each name to the right of its point, as plain text, so that a dollar sign in a name prints as itself.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The axes the points are on.
  x, y : numpy.ndarray
      The points' coordinates.
  names : Sequence[str]
      The name of each point.
  """
  for x_value, y_value, name in zip(x, y, names, strict=True):
    ax.annotate(name, (x_value, y_value), xytext=(4, 0), textcoords="offset points", va="center", parse_math=False)


def label_items(ax: Axes, positions: IntArray, names: Sequence[str]) -> None:
  """Tick the x axis at ``positions`` with the ``names`` of the items there, or at whole numbers if there are many.

  Names are written as given, vertically so that long ones fit. With more than 30 names, matplotlib picks the ticks
  among the whole numbers, which also holds in a view zoomed in to a few positions.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The axes whose x axis is ticked.
  positions : numpy.ndarray
      The x position of each item.
  names : Sequence[str]
      The name of each item.
  """
  if len(names) <= _MAX_NAMED_TICKS:
    ax.set_xticks(positions, names, rotation=90, parse_math=False)
  else:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
