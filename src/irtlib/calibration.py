"""The response matrix and the result that every calibrator shares (ADR-0006).

A response matrix has one row per examinee and one column per item, and holds 1 for a correct response, 0 for an
incorrect one and NaN where the examinee did not answer the item.
"""

import warnings
from dataclasses import dataclass
from enum import StrEnum

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


class Likelihood(StrEnum):
  """The likelihood a calibration maximizes, whose logarithm at the estimates it reports.

  ``MARGINAL`` is the likelihood of the item parameters with abilities integrated out over a grid, which marginal
  maximum likelihood maximizes; ``CONDITIONAL`` the likelihood of the responses given each examinee's number
  correct, which conditional maximum likelihood maximizes. A member equals its name in lower case.

  Examples
  --------
  >>> Likelihood("conditional") is Likelihood.CONDITIONAL
  True
  """

  MARGINAL = "marginal"
  CONDITIONAL = "conditional"


@dataclass(frozen=True, slots=True)
class Calibration:
  """The result of a calibration, by any method.

  Attributes
  ----------
  estimates : numpy.ndarray
      The estimates ``[a, b, c, d]`` of every item, one row per column of the response matrix, whether or not they are
      valid for an item bank; NaN for a parameter of an item the method gives no estimate.
  items : irtlib.ItemBank or None
      The items whose estimates are valid in an item bank, every parameter finite with ``a > 0`` and
      ``0 <= c < d <= 1``, each named ``item_`` and its column, as in a bank of every item; None if no item's
      estimates are. The calibrator warns with a ``RuntimeWarning`` naming the items it leaves out, such as a
      miskeyed item, whose slope is not positive.
  standard_errors : numpy.ndarray or None
      The standard errors of the estimates, in the shape of ``estimates``, NaN for a parameter not estimated;
      None, with a warning, when the information matrix could not be inverted.
  log_likelihood : float
      The logarithm of the likelihood named by ``likelihood``, at the estimates; with priors, without the log-prior.
  likelihood : Likelihood
      The likelihood the calibration maximizes, given as a member or its name. Model comparison and S-X² take only
      marginal ones.
  n_parameters : int
      The number of parameters estimated, a parameter shared by every item counted once.
  item_parameters : numpy.ndarray
      The number of parameters estimated for each item, a parameter shared by every item counted for each.
  cycles : int
      The cycles of its algorithm the calibration ran, such as EM cycles or Newton-Raphson iterations.
  converged : bool
      Whether the stopping rule was met; if not, the calibration ran out of cycles or could not go on, and warned with
      :py:class:`ConvergenceWarning`.
  change : float
      The stopping rule's final value: the largest change of an estimated parameter in the last cycle; for
      Newton-Raphson, in the cycle's full step, before any halving.
  grid : irtlib.quadrature.QuadratureGrid or None
      The nodes abilities were integrated over, which item fit uses too: set when the likelihood is marginal, and None
      otherwise, since no other likelihood integrates over abilities.

  Raises
  ------
  ValueError
      If ``likelihood`` names no :py:class:`Likelihood`, or ``grid`` is set for a likelihood that is not marginal, or
      missing for a marginal one.
  """

  estimates: FloatArray
  items: ItemBank | None
  standard_errors: FloatArray | None
  log_likelihood: float
  likelihood: Likelihood
  n_parameters: int
  item_parameters: IntArray
  cycles: int
  converged: bool
  change: float
  grid: QuadratureGrid | None

  def __post_init__(self) -> None:
    """Read the likelihood as a :py:class:`Likelihood`, and check that a grid comes exactly with a marginal one."""
    # a member equals its name in lower case, so "marginal" is read as Likelihood.MARGINAL
    object.__setattr__(self, "likelihood", Likelihood(self.likelihood))
    if (self.grid is not None) != (self.likelihood is Likelihood.MARGINAL):
      msg = f"a calibration has a grid exactly when its likelihood is marginal; this one's is {self.likelihood}"
      raise ValueError(msg)


def _check_marginal(calibration: Calibration, statistic: str) -> QuadratureGrid:
  """Return the grid of a calibration, after checking that its likelihood is the marginal one a statistic needs.

  Model comparison and item fit call it on every calibration they take, so that none compares a log-likelihood of
  another kind with a marginal one, or integrates over a grid the calibration never used.

  Parameters
  ----------
  calibration : Calibration
      The calibration the statistic is computed from.
  statistic : str
      The statistic's name, for the error.

  Returns
  -------
  irtlib.quadrature.QuadratureGrid
      The nodes the calibration integrated abilities over.

  Raises
  ------
  ValueError
      If the calibration's likelihood is not marginal, so that it has no grid and its log-likelihood compares with
      no marginal one.
  """
  if calibration.grid is None:
    msg = (
      f"{statistic} needs a calibration of the marginal likelihood, which integrates abilities over a grid; this "
      f"calibration's likelihood is {calibration.likelihood}"
    )
    raise ValueError(msg)
  return calibration.grid


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
