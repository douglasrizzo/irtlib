"""The response matrix and the result that every calibrator shares (ADR-0006).

A response matrix has one row per examinee and one column per item, and holds 1 for a correct response, 0 for an
incorrect one and NaN where the examinee did not answer the item.
"""

import warnings
from dataclasses import dataclass

import numpy
import numpy.typing as npt

from ._responses import response_matrix
from .bank import ItemBank, _generated_names, _listed, _violations
from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .quadrature import QuadratureGrid
from .typing import FloatArray, IntArray

_GRID_NODES = 61


class ConvergenceWarning(RuntimeWarning):
  """A calibration ran out of cycles before its stopping rule was met."""


@dataclass(frozen=True, slots=True)
class Calibration:
  """The result of a calibration.

  Attributes
  ----------
  estimates : numpy.ndarray
      The estimates ``[a, b, c, d]`` of every item, one row per column of the response matrix, whether or not they are
      valid for an item bank.
  items : irtlib.ItemBank or None
      The items whose estimates are valid in an item bank, every parameter finite with ``a > 0`` and
      ``0 <= c < d <= 1``, each named ``item_`` and its column, as in a bank of every item; None if no item's
      estimates are. The calibrator warns with a ``RuntimeWarning`` naming the items it leaves out, such as a
      miskeyed item, whose slope is not positive.
  standard_errors : numpy.ndarray or None
      The standard errors of the estimates, in the shape of ``estimates``, NaN for a parameter not estimated;
      None, with a warning, when the information matrix could not be inverted.
  log_likelihood : float
      The log-likelihood of the responses at the estimates.
  n_parameters : int
      The number of parameters estimated, a parameter shared by every item counted once.
  item_parameters : numpy.ndarray
      The number of parameters estimated for each item, a parameter shared by every item counted for each.
  cycles : int
      The cycles the calibration ran.
  converged : bool
      Whether the stopping rule was met; if not, the calibration ran out of cycles and warned with
      :py:class:`ConvergenceWarning`.
  change : float
      The stopping rule's final value: the largest change of an estimated parameter in the last cycle.
  grid : irtlib.quadrature.QuadratureGrid
      The nodes abilities were integrated over, which item fit uses too.
  """

  estimates: FloatArray
  items: ItemBank | None
  standard_errors: FloatArray | None
  log_likelihood: float
  n_parameters: int
  item_parameters: IntArray
  cycles: int
  converged: bool
  change: float
  grid: QuadratureGrid


def validate_responses(responses: npt.ArrayLike) -> FloatArray:
  """Return a response matrix as a new float array, after checking its type, its shape, its values and its items.

  The matrix must have at least one examinee and one item, and every item must be answered by some examinee; an
  examinee who answered no item is accepted. Booleans and integers are read as 1 and 0.

  Parameters
  ----------
  responses : array_like
      The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN.

  Returns
  -------
  numpy.ndarray
      A float copy of the responses, of the same shape.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
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
  matrix = response_matrix(responses)
  if matrix.size == 0:
    msg = f"responses must hold at least one examinee and one item, got shape {matrix.shape}"
    raise ValueError(msg)
  unanswered = numpy.flatnonzero(numpy.isnan(matrix).all(axis=0))
  if unanswered.size:
    msg = f"no examinee answered the items in columns {unanswered.tolist()}"
    raise ValueError(msg)
  return matrix


def default_grid() -> QuadratureGrid:
  """Return the grid abilities are integrated over by default (ADR-0007).

  Returns
  -------
  irtlib.quadrature.QuadratureGrid
      61 equally spaced nodes on [-6, 6], with equal weights.

  Examples
  --------
  >>> grid = default_grid()
  >>> grid.nodes.size, float(grid.nodes[0]), float(grid.nodes[-1])
  (61, -6.0, 6.0)
  """
  return QuadratureGrid.uniform(_GRID_NODES, THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)


def _valid_items(estimates: FloatArray) -> ItemBank | None:
  """Return a calibration's bank, the estimated items valid in one, warning from a calibrator about the others.

  Every calibrator builds the ``items`` of its :py:class:`Calibration` here, so the warning points at the caller of
  the calibrator's method.

  Parameters
  ----------
  estimates : numpy.ndarray
      The estimates ``[a, b, c, d]`` of every item.

  Returns
  -------
  irtlib.ItemBank or None
      The items valid in a bank, each with the name it would have in a bank of every estimate, ``item_`` and its
      row; None if no item is valid.
  """
  names = tuple(_generated_names(estimates.shape[0]))
  violations = _violations(estimates)
  invalid = numpy.zeros(estimates.shape[0], dtype=bool)
  for _, rows in violations:
    invalid[rows] = True
  if violations:
    rules = "; ".join(f"{rule} for {_listed(rows, names)}" for rule, rows in violations)
    msg = f"the calibration's bank leaves out the items whose estimates are not valid in an item bank: {rules}"
    if numpy.any(estimates[:, 0] <= 0):
      msg += (
        "; a slope that is not positive means the chance of a correct answer falls as ability rises, as for a "
        "miskeyed item: reverse its scoring or remove it"
      )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
  valid = numpy.flatnonzero(~invalid)
  if not valid.size:
    return None
  return ItemBank(estimates[valid], names=[names[item] for item in valid])
