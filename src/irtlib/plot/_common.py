"""What every plot module shares: the axes to draw on and the items to draw."""

import operator

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes

from ..bank import ItemBank
from ..typing import FloatArray


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
