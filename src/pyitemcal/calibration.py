"""The response matrix and the result that every calibrator shares (ADR-0006).

A response matrix has one row per examinee and one column per item, and holds 1 for a correct response, 0 for an
incorrect one and NaN where the examinee did not answer the item.
"""

from dataclasses import dataclass

import numpy
from catsim import ItemBank
from catsim.typing import FloatArray

_RESPONSE_NDIM = 2


class ConvergenceWarning(RuntimeWarning):
  """A calibration ran out of cycles before its stopping rule was met."""


@dataclass(frozen=True, slots=True)
class Calibration:
  """The result of a calibration.

  Attributes
  ----------
  items : catsim.ItemBank
      The estimated items, ``[a, b, c, d]`` per row, in the order of the response matrix's columns.
  standard_errors : numpy.ndarray or None
      The standard errors of the estimates, in the shape of ``items.items``, NaN for a parameter not estimated;
      None, with a warning, when the information matrix could not be inverted.
  log_likelihood : float
      The log-likelihood of the responses at the estimates.
  n_parameters : int
      The number of parameters estimated, a parameter shared by every item counted once.
  cycles : int
      The cycles the calibration ran.
  converged : bool
      Whether the stopping rule was met; if not, the calibration ran out of cycles and warned with
      :py:class:`ConvergenceWarning`.
  change : float
      The stopping rule's final value: the largest change of an estimated parameter in the last cycle.
  """

  items: ItemBank
  standard_errors: FloatArray | None
  log_likelihood: float
  n_parameters: int
  cycles: int
  converged: bool
  change: float


def validate_responses(responses: FloatArray) -> FloatArray:
  """Return a response matrix as a float array, after checking its shape, its values and that every item is answered.

  An examinee who answered no item is accepted.

  Parameters
  ----------
  responses : numpy.ndarray
      The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN.

  Returns
  -------
  numpy.ndarray
      The responses as a float array of the same shape.

  Raises
  ------
  ValueError
      If the matrix is not two-dimensional or is empty, holds a value other than 0, 1 and NaN, or has an item no
      examinee answered.

  Examples
  --------
  >>> import numpy
  >>> validate_responses(numpy.array([[1, 0], [numpy.nan, 1]]))
  array([[ 1.,  0.],
         [nan,  1.]])
  """
  matrix = numpy.asarray(responses, dtype=float)
  if matrix.ndim != _RESPONSE_NDIM or matrix.size == 0:
    msg = f"responses must be a non-empty matrix of examinees by items, got shape {matrix.shape}"
    raise ValueError(msg)
  answered = ~numpy.isnan(matrix)
  if not numpy.all((matrix[answered] == 0) | (matrix[answered] == 1)):
    msg = "responses must hold 1, 0 or NaN for a missing response"
    raise ValueError(msg)
  unanswered = numpy.flatnonzero(~answered.any(axis=0))
  if unanswered.size:
    msg = f"no examinee answered the items in columns {unanswered.tolist()}"
    raise ValueError(msg)
  return matrix
