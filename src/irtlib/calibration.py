"""The response matrix and the result that every calibrator shares (ADR-0006).

A response matrix has one row per examinee and one column per item, and holds 1 for a correct response, 0 for an
incorrect one and NaN where the examinee did not answer the item. Conditional and joint maximum likelihood share the
leaving out of the examinees and items that have no finite estimate, documented with :py:mod:`irtlib.cml` and
:py:mod:`irtlib.jml`.
"""

import warnings
from dataclasses import dataclass
from enum import StrEnum

import numpy
import numpy.typing as npt

from ._responses import response_matrix
from .ability import AbilityEstimates
from .bank import ItemBank, _generated_names, _listed, _violations
from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .quadrature import QuadratureGrid
from .typing import BoolArray, FloatArray, IntArray

_GRID_NODES = 61


class ConvergenceWarning(RuntimeWarning):
  """A calibration ran out of cycles before its stopping rule was met."""


class Likelihood(StrEnum):
  """The likelihood a calibration maximizes, whose logarithm at the estimates it reports.

  ``MARGINAL`` is the likelihood of the item parameters with abilities integrated out over a grid, which marginal
  maximum likelihood maximizes; ``CONDITIONAL`` the likelihood of the responses given each examinee's number
  correct, which conditional maximum likelihood maximizes; ``JOINT`` the likelihood of the responses given every
  examinee's ability, which joint maximum likelihood maximizes over the abilities and the item parameters together. A
  member equals its name in lower case.

  Examples
  --------
  >>> Likelihood("conditional") is Likelihood.CONDITIONAL
  True
  """

  MARGINAL = "marginal"
  CONDITIONAL = "conditional"
  JOINT = "joint"


@dataclass(frozen=True, slots=True)
class Calibration:
  """The result of a calibration, by any method.

  Parameters
  ----------
  estimates : numpy.ndarray
      The estimates ``[a, b, c, d]`` of every item, one row per column of the response matrix, whether or not they are
      valid for an item bank; NaN for a parameter of an item the method gives no estimate.
  items : irtlib.ItemBank or None
      The items whose estimates are valid in an item bank, every parameter finite with ``a > 0`` and
      ``0 <= c < d <= 1``, each named ``item_`` and its column, as in a bank of every item; None if no item's
      estimates are. The calibrator warns with a ``RuntimeWarning`` naming the items it leaves out, such as a
      miskeyed item, whose slope is not positive.
  covariances : numpy.ndarray or None
      The covariance of each item's estimates, one 4 x 4 matrix per row of ``estimates``, its rows and columns in the
      order ``a, b, c, d``, NaN in those of a parameter not estimated, and symmetric. A parameter shared by every
      item, such as the 1PL's slope, appears in each item's matrix, with its variance and its covariances with that
      item's estimates. Only the variances are checked, so a matrix may not be positive definite. None, with a
      warning, when the information matrix could not be inverted or gives a variance that is not finite and positive.
  log_likelihood : float
      The logarithm of the likelihood named by ``likelihood``, at the estimates, before any correction the method makes
      to them once the likelihood is maximized; with priors, without the log-prior.
  log_likelihoods : numpy.ndarray
      The same logarithm at the end of each cycle of its algorithm the calibration ran, such as EM cycles,
      Newton-Raphson iterations, or the item and ability steps of joint maximum likelihood; the last equal to
      ``log_likelihood``, and with priors without the log-prior.
  likelihood : Likelihood
      The likelihood the calibration maximizes, given as a member or its name. Model comparison and S-X² take only
      marginal ones.
  n_parameters : int
      The number of free parameters estimated: a parameter shared by every item counted once, and for the joint
      likelihood the ability of every examinee kept, less one for each constraint that fixes the scale.
  item_parameters : numpy.ndarray
      The number of parameters estimated for each item, a parameter shared by every item counted for each.
  converged : bool
      Whether the stopping rule was met; if not, the calibration ran out of cycles or could not go on, and warned with
      :py:class:`ConvergenceWarning`.
  change : float
      The stopping rule's final value: the largest change of an estimated item parameter in the last cycle; for
      Newton-Raphson, in the cycle's full step, before any halving.
  grid : irtlib.quadrature.QuadratureGrid or None
      The nodes abilities were integrated over, which item fit uses too: set when the likelihood is marginal, and None
      otherwise, since no other likelihood integrates over abilities.
  abilities : irtlib.ability.AbilityEstimates or None
      The abilities estimated with the items, one per row of the response matrix, at the estimates before any
      correction: set when the likelihood is joint, and None otherwise, since no other likelihood estimates them.

  Raises
  ------
  ValueError
      If ``likelihood`` names no :py:class:`Likelihood`; ``grid`` is set for a likelihood that is not marginal, or
      missing for a marginal one; or ``abilities`` is set for a likelihood that is not joint, or missing for a joint
      one.
  """

  estimates: FloatArray
  items: ItemBank | None
  covariances: FloatArray | None
  log_likelihood: float
  log_likelihoods: FloatArray
  likelihood: Likelihood
  n_parameters: int
  item_parameters: IntArray
  converged: bool
  change: float
  grid: QuadratureGrid | None
  abilities: AbilityEstimates | None

  def __post_init__(self) -> None:
    """Read the likelihood as a :py:class:`Likelihood`, and check that the grid and the abilities match it."""
    # a member equals its name in lower case, so "marginal" is read as Likelihood.MARGINAL
    object.__setattr__(self, "likelihood", Likelihood(self.likelihood))
    if (self.grid is not None) != (self.likelihood is Likelihood.MARGINAL):
      msg = f"a calibration has a grid exactly when its likelihood is marginal; this one's is {self.likelihood}"
      raise ValueError(msg)
    if (self.abilities is not None) != (self.likelihood is Likelihood.JOINT):
      msg = f"a calibration has abilities exactly when its likelihood is joint; this one's is {self.likelihood}"
      raise ValueError(msg)

  @property
  def standard_errors(self) -> FloatArray | None:
    """The standard errors of the estimates, the square roots of the diagonals of ``covariances``.

    Returns
    -------
    numpy.ndarray or None
        One row per row of ``estimates``, NaN for a parameter not estimated; None when ``covariances`` is None.
    """
    if self.covariances is None:
      return None
    return numpy.sqrt(numpy.diagonal(self.covariances, axis1=1, axis2=2))

  @property
  def cycles(self) -> int:
    """The cycles of its algorithm the calibration ran, one per value of ``log_likelihoods``.

    Returns
    -------
    int
        The number of cycles.
    """
    return self.log_likelihoods.size


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
      f"for {statistic}, a calibration must be of the marginal likelihood, which integrates abilities over a grid; "
      f"this one's likelihood is {calibration.likelihood}"
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


def _complete_responses(calibration: Calibration, responses: npt.ArrayLike, statistic: str, reason: str) -> FloatArray:
  """Return the responses as a float matrix, after checking that they are complete and match the calibration's items.

  Parameters
  ----------
  calibration : Calibration
      The calibration the statistic is computed from.
  responses : array_like
      The response matrix: one row per examinee, one column per item.
  statistic : str
      The statistic's name, for the errors.
  reason : str
      What the error for a missing response says after naming the statistic: why it needs complete responses, or
      what to do instead.

  Returns
  -------
  numpy.ndarray
      A float copy of the responses.

  Raises
  ------
  ValueError
      If the responses fail :py:func:`validate_responses`, have a missing response, or do not have one column per
      item of the calibration.
  """
  matrix = validate_responses(responses)
  if numpy.isnan(matrix).any():
    msg = f"complete responses are needed for {statistic}; {reason}"
    raise ValueError(msg)
  n_items = calibration.estimates.shape[0]
  if matrix.shape[1] != n_items:
    msg = f"the responses have {matrix.shape[1]} items, the calibration {n_items}"
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


def _edited(matrix: FloatArray, likelihood: Likelihood) -> tuple[FloatArray, BoolArray, BoolArray]:
  """Return the responses of the examinees and items kept, and which they are, once those with no estimate are left out.

  Examinees with no item or every item correct, and items answered correctly by every examinee or by none, are left
  out, repeatedly until neither is left.

  Parameters
  ----------
  matrix : numpy.ndarray
      The complete response matrix.
  likelihood : Likelihood
      The likelihood the calibration maximizes, for the error.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
      The responses of the examinees and items kept, a mask of the examinees kept among the rows, and a mask of the
      items kept among the columns.

  Raises
  ------
  ValueError
      If no item is kept.
  """
  examinees, items = _kept(matrix)
  # _kept leaves no item rather than one, whose every examinee would have no item or every item correct
  if not items.any():
    msg = (
      f"no item has a {likelihood} estimate: once the examinees with no item or every item correct are left out, "
      "every item is answered correctly by every examinee or by none"
    )
    raise ValueError(msg)
  return matrix[numpy.ix_(examinees, items)], examinees, items


def _reject_separation(kept: FloatArray, items: BoolArray, likelihood: Likelihood) -> None:
  """Raise an error naming the split of the responses kept, if one leaves the Rasch model's likelihood no maximum.

  Parameters
  ----------
  kept : numpy.ndarray
      The responses of the examinees and items kept, as :py:func:`_edited` returns them.
  items : numpy.ndarray
      The mask of the items kept among the columns of the responses.
  likelihood : Likelihood
      The likelihood the calibration maximizes, for the error.

  Raises
  ------
  ValueError
      If the responses split into two groups of items that the examinees with at most some number correct answered
      none of, and the others all of.
  """
  separation = _separation(kept)
  if separation is None:
    return
  columns = numpy.flatnonzero(items)
  score, easier, harder = separation
  msg = (
    f"no difficulty has a finite {likelihood} estimate: the examinees with at most {score} correct answered none of "
    f"the items in columns {columns[harder].tolist()} correctly, and those with more answered all of the items in "
    f"columns {columns[easier].tolist()}, so the responses fix only that the first group is harder than the second"
  )
  raise ValueError(msg)


def _kept(matrix: FloatArray) -> tuple[BoolArray, BoolArray]:
  """Return the examinees and items kept: none with no item or every item correct, none answered by all or none."""
  examinees = numpy.ones(matrix.shape[0], dtype=bool)
  items = numpy.ones(matrix.shape[1], dtype=bool)
  while True:
    scores = matrix[:, items].sum(axis=1)
    kept_examinees = examinees & (scores > 0) & (scores < items.sum())
    totals = matrix[kept_examinees].sum(axis=0)
    kept_items = items & (totals > 0) & (totals < kept_examinees.sum())
    if numpy.array_equal(kept_examinees, examinees) and numpy.array_equal(kept_items, items):
      return examinees, items
    examinees, items = kept_examinees, kept_items


def _separation(matrix: FloatArray) -> tuple[int, BoolArray, BoolArray] | None:
  """Return a split of the responses that leaves the Rasch model's conditional and joint likelihoods no maximum, if any.

  Haberman's Theorem 10, and his Theorem 1, which coincides with it on responses with no examinee who has no item or
  every item correct and no item answered by every examinee or by none: a number correct and a cut of the item totals
  such that the examinees with at most that many correct answered no item below the cut, and those with more answered
  every item above it.

  Parameters
  ----------
  matrix : numpy.ndarray
      The complete responses of the examinees and items kept.

  Returns
  -------
  tuple[int, numpy.ndarray, numpy.ndarray] or None
      The number correct, the items above the cut and the items below it, as masks of the columns; None if no split
      exists, and so the conditional and joint maximum likelihood estimates do.
  """
  n_items = matrix.shape[1]
  scores = matrix.sum(axis=1)[:, numpy.newaxis]
  totals = matrix.sum(axis=0)
  # each item's smallest number correct among the examinees who answered it correctly, and largest among those who
  # did not: no examinee with at most s correct answered an item right exactly when its smallest exceeds s
  lowest_right = numpy.where(matrix == 1, scores, numpy.inf).min(axis=0)
  highest_wrong = numpy.where(matrix == 0, scores, -numpy.inf).max(axis=0)
  # the theorem's a, a number correct plus one half, lies strictly between 1 and M - 1, and its b, an item total plus
  # one half, below the largest total; with no examinee left out, it lies between n_0 = 0 and n - n_0 = n as well
  for score in range(1, n_items - 1):
    for cut in numpy.unique(totals)[:-1] + 0.5:
      harder, easier = totals < cut, totals > cut
      # Theorem 10's fifth condition holds whenever the first four do, since every examinee kept is low or high
      if numpy.all(lowest_right[harder] > score) and numpy.all(highest_wrong[easier] <= score):
        return score, easier, harder
  return None
