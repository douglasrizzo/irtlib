"""What every plot module shares: the axes to draw on, the items to draw and the abilities to draw them at."""

import operator

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from matplotlib.axes import Axes

from ..bank import ItemBank
from ..typing import FloatArray

_MIN_ABILITIES = 2
"""The fewest abilities a curve is drawn at: a line needs two points."""


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


def abilities(theta: npt.ArrayLike) -> FloatArray:
  """Return the abilities a curve is drawn at as a float array, after checking that they can be drawn as one.

  Parameters
  ----------
  theta : array-like
      The abilities the caller passed.

  Returns
  -------
  numpy.ndarray
      The abilities, a new one-dimensional float array.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, bools included.
  ValueError
      If ``theta`` is ragged or not one-dimensional, holds fewer than two abilities, or holds one that is not finite or
      not above the one before it.
  """
  try:
    values = numpy.asarray(theta)
  except ValueError:
    # NumPy builds no array from sequences of different lengths, and its own message names no argument.
    msg = "theta has a ragged shape; a curve is drawn at a one-dimensional array of at least two abilities"
    raise ValueError(msg) from None
  # Booleans would be read as abilities 0 and 1, and None as NaN.
  if values.dtype.kind not in "iuf":
    msg = f"theta holds {values.dtype} values; abilities must be real numbers"
    raise TypeError(msg)
  if values.ndim != 1 or values.size < _MIN_ABILITIES:
    msg = f"theta has shape {values.shape}; a curve is drawn at a one-dimensional array of at least two abilities"
    raise ValueError(msg)
  # A list that mixes bools with numbers becomes an array of numbers, so its entries are checked as given.
  for position, entry in enumerate(numpy.asarray(theta, dtype=object)):
    if isinstance(entry, (bool, numpy.bool_)):
      msg = f"theta[{position}] is {entry!r}; abilities must be real numbers, not bools"
      raise TypeError(msg)
  values = values.astype(float)
  if not numpy.isfinite(values).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(values))[0])
    msg = f"theta[{position}] is {values[position].item()!r}; abilities must be finite numbers"
    raise ValueError(msg)
  # A line through abilities out of order doubles back on itself.
  steps = numpy.diff(values)
  if (steps <= 0).any():
    position = int(numpy.flatnonzero(steps <= 0)[0]) + 1
    msg = (
      f"theta[{position}] is {values[position].item()!r}, not above theta[{position - 1}] = "
      f"{values[position - 1].item()!r}; abilities must increase"
    )
    raise ValueError(msg)
  return values


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
  # bool is an int to operator.index, but a flag passed by mistake is not an item.
  if isinstance(value, (bool, numpy.bool_)):
    msg = f"{label} is {value!r}; item positions must be integers, not bools"
    raise TypeError(msg)
  try:
    position = operator.index(value)  # ty: ignore[invalid-argument-type]
  except TypeError:
    # NumPy would index the bank with a float such as 3.7 as item 3.
    msg = f"{label} is {value!r}; item positions must be integers"
    raise TypeError(msg) from None
  # NumPy would read a negative position as one counted from the end of the bank.
  if not 0 <= position < n_items:
    msg = f"{label} is {position}; item positions must be between 0 and {n_items - 1}"
    raise ValueError(msg)
  return position
