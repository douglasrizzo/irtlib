"""Plots that screen a response matrix before any calibration: its numbers correct and its items' rest score curves."""

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from .._responses import response_matrix
from ..bank import _generated_names
from ..typing import FloatArray
from ._common import item_position, prepare_axes

_BAR_WIDTH = 0.8
"""The width of a bar at each number correct, which leaves a gap between neighbouring numbers."""


def number_correct_histogram(
  responses: npt.ArrayLike, *, ax: Axes | None = None, title: str | None = None, label: str | None = None
) -> Axes:
  """Plot how many examinees got each number correct, from 0 to the number of items.

  An examinee's number correct counts the items the examinee answered correctly, so a missing response adds nothing.
  Every number from 0 to the number of items has a bar, empty or not, and the x axis ticks whole numbers.

  .. plot::
      :filename-prefix: number_correct_histogram
      :caption: The numbers correct of 1,000 examinees on a 20-item test.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.plot import number_correct_histogram

      items = ItemBank.generate(20, "3PL", rng=0)
      theta = numpy.random.default_rng(1).normal(0, 1, 1000)
      number_correct_histogram(draw_responses(theta, items.items, rng=2))
      plt.show()

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The bars' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the bars are on.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, before any figure is created.
  ValueError
      If ``responses`` is not a matrix with at least one row, or a response is not 0, 1 or NaN, before any figure is
      created.
  """
  matrix = response_matrix(responses)
  if matrix.shape[0] == 0:
    msg = "responses has no rows; pass the responses of at least one examinee"
    raise ValueError(msg)
  n_items = matrix.shape[1]
  counts = numpy.bincount(_correct(matrix).sum(axis=1).astype(int), minlength=n_items + 1)
  ax = prepare_axes(ax, title)
  ax.bar(numpy.arange(n_items + 1), counts, _BAR_WIDTH, label=label)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel("Number correct")
  ax.set_ylabel("Examinees")
  return ax


def rest_score_curve(
  responses: npt.ArrayLike,
  item: int,
  *,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot an item's proportion correct at each rest score, the number correct on the other items.

  Only the examinees who answered the item count. An examinee's rest score counts the other items the examinee
  answered correctly, so a missing response elsewhere adds nothing to it. Each rest score that some examinee has is a
  point, and the points are joined in order; the x axis ticks whole numbers.

  .. plot::
      :filename-prefix: rest_score_curve
      :caption: An item with a guessing parameter of 0.25, on a 15-item test taken by 3,000 examinees.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.plot import rest_score_curve

      items = ItemBank.generate(15, "2PL", rng=0).items.copy()
      items[0, 2] = 0.25
      theta = numpy.random.default_rng(1).normal(0, 1, 3000)
      rest_score_curve(draw_responses(theta, items, rng=2), 0)
      plt.show()

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  item : int
      The item's position, its column in ``responses``.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which names the item by its position, as a bank built from
      the item matrix does.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, or ``item`` is not an integer, before any figure is
      created.
  ValueError
      If ``responses`` is not a matrix, a response is not 0, 1 or NaN, ``item`` is not a column of it, or no examinee
      answered the item, before any figure is created.
  """
  matrix = response_matrix(responses)
  n_items = matrix.shape[1]
  item = item_position("item", item, n_items)
  answered = ~numpy.isnan(matrix[:, item])
  if not answered.any():
    msg = f"no examinee answered item {item}; its proportion correct is undefined at every rest score"
    raise ValueError(msg)
  correct = _correct(matrix)
  rest = (correct.sum(axis=1) - correct[:, item])[answered].astype(int)
  examinees = numpy.bincount(rest, minlength=n_items)
  right = numpy.bincount(rest, weights=correct[answered, item], minlength=n_items)
  scores = numpy.flatnonzero(examinees)
  ax = prepare_axes(ax, title)
  ax.plot(
    scores,
    right[scores] / examinees[scores],
    "o-",
    label=_generated_names(n_items)[item] if label is None else label,
  )
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel("Rest score")
  ax.set_ylabel("Proportion correct")
  return ax


def _correct(matrix: FloatArray) -> FloatArray:
  """Return the response matrix with each missing response as 0, so that sums over it count the correct ones."""
  return numpy.where(numpy.isnan(matrix), 0.0, matrix)
