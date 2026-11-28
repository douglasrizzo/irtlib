"""Classical item statistics, which screen responses before a calibration: proportion correct and item-rest correlation.

Each function takes a response matrix, one row per examinee and one column per item, holding 1, 0 or NaN for a
missing response, and returns one value per item, computed over the examinees who answered that item.
"""

import numpy
import numpy.typing as npt

from ._responses import response_matrix
from .typing import FloatArray


def proportion_correct(responses: npt.ArrayLike) -> FloatArray:
  """Return each item's proportion correct, over the examinees who answered it.

  An item nobody answered gets NaN.

  The :doc:`/methods/classical` page gives the method and its sources.

  Parameters
  ----------
  responses : array_like
      The response matrix: one row per examinee, one column per item, 1 or 0, and NaN for a missing response.

  Returns
  -------
  numpy.ndarray
      The proportion correct of each item.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses are not a matrix, or a response is not 0, 1 or NaN.

  Examples
  --------
  >>> import numpy
  >>> proportion_correct(numpy.array([[1, 0], [1, numpy.nan], [0, 1]]))
  array([0.66666667, 0.5       ])
  """
  matrix = response_matrix(responses)
  answered = ~numpy.isnan(matrix)
  # an item nobody answered divides 0 by 0, which leaves it NaN
  with numpy.errstate(invalid="ignore"):
    return numpy.sum(matrix, axis=0, where=answered) / answered.sum(axis=0)


def item_rest_correlation(responses: npt.ArrayLike) -> FloatArray:
  """Return each item's correlation with the rest score, the number of the other items an examinee got correct.

  An examinee's rest score counts the other items the examinee answered, and each item's correlation runs over the
  examinees who answered it. It is NaN where fewer than two of them answered the item, or where the item or the rest
  score takes one value among them.

  The :doc:`/methods/classical` page gives the method and its sources.

  Parameters
  ----------
  responses : array_like
      The response matrix: one row per examinee, one column per item, 1 or 0, and NaN for a missing response.

  Returns
  -------
  numpy.ndarray
      The item-rest correlation of each item.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses are not a matrix, or a response is not 0, 1 or NaN.

  Examples
  --------
  >>> import numpy
  >>> responses = numpy.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, numpy.nan]])
  >>> item_rest_correlation(responses).round(4)
  array([0.3273, 0.3273, 0.5222])
  """
  matrix = response_matrix(responses)
  answered = ~numpy.isnan(matrix)
  scores = numpy.where(answered, matrix, 0.0)
  rest = scores.sum(axis=1, keepdims=True) - scores
  n_answered = answered.sum(axis=0)
  # an item nobody answered divides 0 by 0, which leaves its means NaN
  with numpy.errstate(invalid="ignore"):
    item_deviation = scores - numpy.sum(scores, axis=0, where=answered) / n_answered
    rest_deviation = rest - numpy.sum(rest, axis=0, where=answered) / n_answered
  covariance = numpy.sum(item_deviation * rest_deviation, axis=0, where=answered)
  spread = numpy.sum(item_deviation**2, axis=0, where=answered) * numpy.sum(rest_deviation**2, axis=0, where=answered)
  # a constant item or rest score has no correlation
  return numpy.divide(covariance, numpy.sqrt(spread), out=numpy.full(spread.shape, numpy.nan), where=spread > 0)
