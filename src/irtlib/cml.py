r"""Conditional maximum likelihood calibration of the Rasch model.

In the Rasch model an examinee's number of correct responses is a sufficient statistic for its ability, so the
likelihood of the responses given each examinee's number correct holds the item difficulties alone, and "does not
depend on the distribution function" of ability :cite:p:`Haberman2004` (p. 35). Its maximizers, the conditional
maximum likelihood (CML) estimates, are consistent as the number of examinees grows (:cite:t:`Andersen1970`, p. 286,
Theorem 1), though "not in general ... efficient" (p. 284).

Write :math:`\varepsilon_i = e^{-b_i}` for item :math:`i`'s easiness, the item parameter of Gustafsson's
multiplicative form of the model (:cite:t:`Gustafsson1980`, p. 379, Equation 1), :math:`s_i` for its number of
correct responses and :math:`n_r` for the number of examinees with :math:`r` correct, out of :math:`M` items. The
conditional likelihood is :math:`\prod_i \varepsilon_i^{s_i} / \prod_{r=1}^{M-1} \gamma_r^{n_r}`, where the
elementary symmetric function :math:`\gamma_r` is the sum, over every set of :math:`r` items, of the product of their
easiness values (p. 379, Equations 5 and 6). Its logarithm, the conditional log-likelihood, is
:math:`-\sum_i s_i b_i - \sum_r n_r \log \gamma_r`; the marginal log-likelihood adds to it the log-likelihood of the
numbers correct (:cite:t:`Haberman2004`, pp. 35-36), so the two are not compared (ADR-0006).

For an examinee with no item or every item correct, "both sides are equal to one" in the conditional likelihood
(:cite:t:`Engelen1987`, p. 11), and items answered correctly by every examinee or by none have no estimate
(:cite:t:`Gustafsson1980`, p. 379): both are left out, the examinees first and then the items, repeatedly until
neither is left, since leaving an item out can leave an examinee with no item or every item correct. The items left
out get no estimate.

Even then the conditional likelihood can lack a maximum. It has none exactly when, for some number correct between 1
and :math:`M - 2` and some cut of the item totals, the examinees with at most that many correct answered no item below
the cut and the others every item above it (:cite:t:`Haberman2004`, pp. 37-38, Theorem 10, whose bounds on the item
totals, :math:`n_0` and :math:`n - n_0`, are 0 and :math:`n` once the examinees are left out). The likelihood then rises
without bound as the two groups of items move apart, so that under the sum to zero no difficulty has a finite
estimate, and calibration raises an error naming the groups.

The estimates solve the likelihood equations

.. math::

   s_i = \sum_{r=1}^{M-1} n_r \pi_{ri}, \qquad \pi_{ri} = \frac{\varepsilon_i \gamma_{r-1}^{(i)}}{\gamma_r},

where :math:`\gamma_{r-1}^{(i)}`, the derivative of :math:`\gamma_r` with respect to :math:`\varepsilon_i`, is the
symmetric function of order :math:`r - 1` of every item but :math:`i` (:cite:t:`Gustafsson1980`, p. 380, Equation 7),
and :math:`\pi_{ri}` is the probability of a correct response to item :math:`i` given :math:`r` correct
(:cite:t:`Haberman2004`, pp. 15-16). The difficulties are identified by summing to zero (ADR-0008), as
:cite:t:`Andersen1972` (p. 45) identifies them; :cite:t:`Gustafsson1980` (p. 380) sets one easiness to 1 instead.

The equations are solved by Newton-Raphson (:cite:t:`Haberman2004`, p. 43). The gradient of the conditional
log-likelihood with respect to :math:`b_i` is :math:`\sum_r n_r \pi_{ri} - s_i`, and its Hessian is :math:`-V`, where
:math:`V_{ij} = \sum_{r=1}^{M-1} n_r v_{rij}` is the conditional covariance of the item scores given the numbers
correct (p. 39), with :math:`v_{rii} = \pi_{ri} - \pi_{ri}^2` and, for :math:`i \neq j`,
:math:`v_{rij} = \pi_{rij} - \pi_{ri} \pi_{rj}` (pp. 27-28, 43). Here
:math:`\pi_{rij} = \varepsilon_i \varepsilon_j \gamma_{r-2}^{(i,j)} / \gamma_r`, the probability of correct responses
to both items, where :math:`\gamma_{r-2}^{(i,j)}`, the second derivative of :math:`\gamma_r` with respect to
:math:`\varepsilon_i` and :math:`\varepsilon_j`, is the symmetric function of order :math:`r - 2` of every item but
:math:`i` and :math:`j` (:cite:t:`Gustafsson1980`, p. 380). :math:`V` has rank :math:`M - 1`
(:cite:t:`Haberman2004`, p. 39), so a step solves the system in its Moore-Penrose inverse :math:`V^+`: Haberman's
iteration sets the first difficulty to 0 through a matrix :math:`K` (pp. 26, 43), and for difficulties that sum to
zero, the centred parameters of p. 21, it reads :math:`b_{t+1} = b_t + V^+ (\sum_r n_r \pi_r - s)`. Each step is
halved, up to 30 times, until the conditional log-likelihood does not fall, which the sources do not do: from the
starting values below, full steps diverged on two items answered 11 to 1. The iterations start from
:cite:t:`Andersen1972`'s (p. 52, Equation 40) values, :math:`b_i = \log((N - s_i) / s_i)` centred to sum to zero,
computed over the :math:`N` examinees kept where Andersen counts every examinee. They stop when no difficulty changes
by more than the tolerance, :math:`10^{-5}` by default, in an iteration (:cite:t:`Haberman2004`, p. 44), counting
only a full step: a step that halving shrank below the tolerance can lie far from the maximum. Without extreme items,
Newton-Raphson usually takes 4 to 5 iterations (:cite:t:`Gustafsson1980`, p. 383).

The symmetric functions come from the summation algorithm, which adds one item at a time,
:math:`\gamma_r(\varepsilon_1, \dots, \varepsilon_t) = \gamma_r(\varepsilon_1, \dots, \varepsilon_{t-1}) +
\varepsilon_t \gamma_{r-1}(\varepsilon_1, \dots, \varepsilon_{t-1})` from :math:`\gamma_0 = 1`, with no subtraction,
so that it stays accurate on long tests (:cite:t:`Gustafsson1980`, p. 381, Equation 10). Run with an item's easiness,
or a pair's, set to zero, it gives the first and second derivatives (p. 382; :cite:t:`Baker1996`, pp. 182-183), so
that an iteration costs a number of operations growing as :math:`M^4` (p. 190). The recursion runs on the logarithms
of the functions, which departs from the sources: those of orders near :math:`M/2` grow with the test until they
reach "the limit set by the size of the floating point numbers" (:cite:t:`Gustafsson1980`, p. 383).

The standard error of each difficulty is the square root of the diagonal of :math:`V^+` at the estimates.
:cite:t:`Haberman2004` (p. 40) takes it from :math:`K V^- K^T` for the first difficulty set to 0; for difficulties
that sum to zero, centring that covariance gives :math:`V^+`. :cite:t:`Andersen1972` (p. 50) estimates the
covariance by the inverse of the information as well.

CML needs complete responses: for responses missing at random, "ordinary CML" "will likely fail"
(:cite:t:`Robitzsch2021`, p. 827), and :py:class:`irtlib.mml.MMLCalibrator` calibrates incomplete responses
(ADR-0006).
"""

import math
import warnings

import numpy

from .calibration import Calibration, ConvergenceWarning, Likelihood, _valid_items, validate_responses
from .typing import BoolArray, FloatArray

# The most halvings of a Newton-Raphson step before the iterations stop at the maximum.
_STEP_HALVINGS = 30


class CMLCalibrator:
  """Calibrate the Rasch model by conditional maximum likelihood.

  Each call to :py:meth:`calibrate` estimates the difficulties of the items of one complete response matrix, with
  every slope 1, from the likelihood of the responses given each examinee's number correct, whatever the
  distribution of ability. The difficulties sum to zero.

  Parameters
  ----------
  tolerance : float, optional
      The stopping rule: calibration stops when no difficulty changes by more than this in an iteration. Default is
      1e-5.
  max_cycles : int, optional
      The most iterations a calibration runs before it stops unconverged. Default is 100.

  Raises
  ------
  ValueError
      If ``tolerance`` is not positive, or ``max_cycles`` is less than 1.

  Examples
  --------
  >>> import numpy
  >>> responses = numpy.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
  >>> calibration = CMLCalibrator().calibrate(responses)
  >>> calibration.estimates[:, 1].round(4)  # b_2 - b_1 = log(2 / 1)
  array([-0.3466,  0.3466])
  """

  def __init__(self, *, tolerance: float = 1e-5, max_cycles: int = 100) -> None:
    if not tolerance > 0:
      msg = f"tolerance must be positive, got {tolerance}"
      raise ValueError(msg)
    if max_cycles < 1:
      msg = f"max_cycles must be at least 1, got {max_cycles}"
      raise ValueError(msg)
    self._tolerance = tolerance
    self._max_cycles = max_cycles

  def calibrate(self, responses: FloatArray) -> Calibration:
    """Estimate the difficulties of the items of a complete response matrix.

    Examinees with no item or every item correct change no estimate and are left out, as are items answered
    correctly by every examinee or by none, repeatedly until neither is left.

    Parameters
    ----------
    responses : numpy.ndarray
        The response matrix: one row per examinee, one column per item, holding 1 or 0.

    Returns
    -------
    Calibration
        The estimates ``[1, b, 0, 1]``, the standard errors of the difficulties and the conditional log-likelihood at
        them, of the examinees and items kept; the result has no grid. An item left out has a NaN difficulty and
        standard error, and the result's bank leaves it out, with a ``RuntimeWarning`` naming it. If the stopping
        rule, a full step within the tolerance, is not met within ``max_cycles``, or no halving of a step keeps the
        log-likelihood from falling, the result says so and a :py:class:`~irtlib.calibration.ConvergenceWarning` is
        issued; if the information matrix cannot be inverted at the estimates, the result has no standard errors and
        a ``RuntimeWarning`` is issued.

    Raises
    ------
    TypeError
        If the responses are neither Booleans, integers nor floats.
    ValueError
        If the responses fail :py:func:`irtlib.calibration.validate_responses` or have a missing response; no item
        is left once examinees and items are left out; the conditional likelihood has no maximum, which the error
        explains by the two groups of items the responses split into; or the information matrix is singular during
        the iterations.
    """
    matrix = validate_responses(responses)
    if numpy.isnan(matrix).any():
      msg = (
        "conditional maximum likelihood needs complete responses; irtlib.mml.MMLCalibrator calibrates responses "
        "with missing entries"
      )
      raise ValueError(msg)
    kept, items = _estimable(matrix)
    totals = kept.sum(axis=0)
    score_counts = numpy.bincount(kept.sum(axis=1).astype(int), minlength=totals.size + 1).astype(float)
    # Andersen's (1972) starting values: the log-odds of an incorrect response, centred
    start = numpy.log((kept.shape[0] - totals) / totals)
    difficulty, cycles, change, failure = _newton_raphson(
      start - start.mean(), totals, score_counts, tolerance=self._tolerance, max_cycles=self._max_cycles
    )
    if failure is not None:
      warnings.warn(f"calibration stopped after {cycles} iterations: {failure}", ConvergenceWarning, stacklevel=2)
    variances = _variances(_derivatives(difficulty, totals, score_counts)[1])
    if variances is None:
      warnings.warn(
        "the information matrix could not be inverted, so the calibration reports no standard errors; the responses "
        "may leave an item no finite estimate",
        RuntimeWarning,
        stacklevel=2,
      )
    estimates = numpy.tile([1.0, numpy.nan, 0.0, 1.0], (matrix.shape[1], 1))
    estimates[items, 1] = difficulty
    standard_errors = None
    if variances is not None:
      standard_errors = numpy.full(estimates.shape, numpy.nan)
      standard_errors[items, 1] = numpy.sqrt(variances)
    return Calibration(
      estimates=estimates,
      items=_valid_items(estimates),
      standard_errors=standard_errors,
      log_likelihood=_log_likelihood(difficulty, totals, score_counts),
      likelihood=Likelihood.CONDITIONAL,
      n_parameters=difficulty.size - 1,
      item_parameters=items.astype(int),
      cycles=cycles,
      converged=failure is None,
      change=change,
      grid=None,
    )


def _newton_raphson(
  difficulty: FloatArray, totals: FloatArray, score_counts: FloatArray, *, tolerance: float, max_cycles: int
) -> tuple[FloatArray, int, float, str | None]:
  """Run Newton-Raphson iterations, each step halved until the log-likelihood does not fall, until the rule is met.

  The rule is met when a full step, taken without halving, changes no difficulty by more than ``tolerance``.

  Returns
  -------
  tuple[numpy.ndarray, int, float, str or None]
      The difficulties; the iterations run; the largest change of a difficulty in the last iteration's full step,
      before any halving; and None if the rule was met, or else why the iterations stopped short of it.

  Raises
  ------
  ValueError
      If the information matrix is singular.
  """
  log_likelihood = _log_likelihood(difficulty, totals, score_counts)
  change, halvings = math.inf, 0
  for cycle in range(1, max_cycles + 1):
    gradient, information = _derivatives(difficulty, totals, score_counts)
    step = _solve(information, gradient)
    change = float(numpy.max(numpy.abs(step)))
    # Full steps from Andersen's starting values diverged on 8 of 391 random data sets with an estimate, each left
    # with two items, such as two items answered 11 to 1; with halving, all 391 converged, within 7 iterations
    # (measured 2026-11-27).
    for halvings in range(_STEP_HALVINGS + 1):
      trial = difficulty + step / 2**halvings
      trial_log_likelihood = _log_likelihood(trial, totals, score_counts)
      # accept a step that does not lower the log-likelihood beyond rounding
      if trial_log_likelihood >= log_likelihood - 1e-12 * abs(log_likelihood):
        break
    else:
      # stalled: a halved step is no evidence of a maximum, which a full step within the tolerance would be
      return (
        difficulty,
        cycle,
        change,
        f"halving the Newton-Raphson step {_STEP_HALVINGS} times did not keep the conditional log-likelihood from "
        f"falling, with a full step of {change:.3g}",
      )
    # the conditional log-likelihood is the same for difficulties shifted by a constant, so centring keeps it
    difficulty, log_likelihood = trial - trial.mean(), trial_log_likelihood
    # a halved step under the tolerance can lie far from the maximum, so only a full one meets the rule
    if halvings == 0 and change <= tolerance:
      return difficulty, cycle, change, None
  if halvings:
    failure = f"the last Newton-Raphson step was halved {halvings} times"
  else:
    failure = f"the last Newton-Raphson step changed a difficulty by {change:.3g}, above the tolerance {tolerance:g}"
  return difficulty, max_cycles, change, failure


def _estimable(matrix: FloatArray) -> tuple[FloatArray, BoolArray]:
  """Return the responses of the examinees and items kept, and the items kept, once a maximum is known to exist.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray]
      The responses of the examinees and items kept, and a mask of the items kept among the columns.

  Raises
  ------
  ValueError
      If no item is kept, or the conditional likelihood of the responses kept has no maximum.
  """
  examinees, items = _kept(matrix)
  # _kept leaves no item rather than one, whose every examinee would have no item or every item correct
  if not items.any():
    msg = (
      "no item has a conditional estimate: once the examinees with no item or every item correct are left out, "
      "every item is answered correctly by every examinee or by none"
    )
    raise ValueError(msg)
  kept = matrix[numpy.ix_(examinees, items)]
  separation = _separation(kept)
  if separation is not None:
    columns = numpy.flatnonzero(items)
    score, easier, harder = separation
    msg = (
      f"no difficulty has a finite conditional estimate: the examinees with at most {score} correct answered none of "
      f"the items in columns {columns[harder].tolist()} correctly, and those with more answered all of the items in "
      f"columns {columns[easier].tolist()}, so the responses fix only that the first group is harder than the second"
    )
    raise ValueError(msg)
  return kept, items


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
  """Return a split of the responses that leaves the conditional likelihood no maximum, or None if there is none.

  Haberman's Theorem 10 on responses with no examinee who has no item or every item correct and no item answered by
  every examinee or by none: a number correct and a cut of the item totals such that the examinees with at most that
  many correct answered no item below the cut, and those with more answered every item above it.

  Parameters
  ----------
  matrix : numpy.ndarray
      The complete responses of the examinees and items kept.

  Returns
  -------
  tuple[int, numpy.ndarray, numpy.ndarray] or None
      The number correct, the items above the cut and the items below it, as masks of the columns; None if no split
      exists, and so the conditional maximum likelihood estimate does.
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


def _log_symmetric_functions(log_easiness: FloatArray) -> FloatArray:
  """Return the logarithms of the elementary symmetric functions of every order, by the summation algorithm.

  Parameters
  ----------
  log_easiness : numpy.ndarray
      The logarithms of the easiness values, ``-b``, of the items on the last axis; ``-inf`` leaves an item out.

  Returns
  -------
  numpy.ndarray
      The logarithms of the symmetric functions of orders 0 to the number of items on the last axis, ``-inf`` for an
      order above the number of items not left out.
  """
  n_items = log_easiness.shape[-1]
  log_gamma = numpy.full((*log_easiness.shape[:-1], n_items + 1), -numpy.inf)
  log_gamma[..., 0] = 0.0
  for item in range(n_items):
    log_gamma[..., 1 : item + 2] = numpy.logaddexp(
      log_gamma[..., 1 : item + 2], log_easiness[..., item, numpy.newaxis] + log_gamma[..., : item + 1]
    )
  return log_gamma


def _log_likelihood(difficulty: FloatArray, totals: FloatArray, score_counts: FloatArray) -> float:
  """Return the conditional log-likelihood of the item totals and the counts of examinees by number correct."""
  log_gamma = _log_symmetric_functions(-difficulty)
  return float(-totals @ difficulty - score_counts[1:-1] @ log_gamma[1:-1])


def _derivatives(difficulty: FloatArray, totals: FloatArray, score_counts: FloatArray) -> tuple[FloatArray, FloatArray]:
  """Return the gradient of the conditional log-likelihood in the difficulties, and its information matrix V."""
  n_items = difficulty.size
  log_easiness = -difficulty
  first, second = numpy.triu_indices(n_items, 1)
  # every item; every item but one, in turn; every item but a pair, for each pair
  batch = numpy.tile(log_easiness, (1 + n_items + first.size, 1))
  rows = numpy.arange(n_items)
  batch[1 + rows, rows] = -numpy.inf
  pairs = 1 + n_items + numpy.arange(first.size)
  batch[pairs, first] = -numpy.inf
  batch[pairs, second] = -numpy.inf
  log_gamma = _log_symmetric_functions(batch)
  orders = numpy.arange(1, n_items)  # the numbers correct of the examinees kept
  full = log_gamma[0, orders]
  # the probability of a correct response to an item given r correct, items by orders
  single = numpy.exp(log_easiness[:, numpy.newaxis] + log_gamma[1 : 1 + n_items, orders - 1] - full)
  # the same for both items of a pair, from the functions of order r - 2, which is 0 for r = 1
  without_pair = numpy.concatenate([numpy.full((first.size, 1), -numpy.inf), log_gamma[1 + n_items :]], axis=1)
  both = numpy.exp((log_easiness[first] + log_easiness[second])[:, numpy.newaxis] + without_pair[:, orders - 1] - full)
  counts = score_counts[orders]
  information = numpy.zeros((n_items, n_items))
  information[rows, rows] = (single * (1 - single)) @ counts
  information[first, second] = information[second, first] = (both - single[first] * single[second]) @ counts
  return single @ counts - totals, information


def _centred_inverse(information: FloatArray) -> FloatArray:
  """Return the Moore-Penrose inverse of an information matrix whose null space is spanned by the vector of ones.

  Returns
  -------
  numpy.ndarray
      The inverse, which maps a vector summing to zero to the difficulties that sum to zero.

  Raises
  ------
  numpy.linalg.LinAlgError
      If the matrix has a null space beyond the vector of ones.
  """
  n_items = information.shape[0]
  # adding s 11^T / n gives the vector of ones the eigenvalue s, of the size of the others, and inverting it there
  # adds 11^T / (s n) to the inverse, which is taken off again
  scale = float(numpy.trace(information)) / n_items
  ones = numpy.full((n_items, n_items), 1.0 / n_items)
  return numpy.linalg.inv(information + scale * ones) - ones / scale


def _solve(information: FloatArray, gradient: FloatArray) -> FloatArray:
  """Return the Newton-Raphson step, the Moore-Penrose inverse of the information times the gradient.

  Returns
  -------
  numpy.ndarray
      The change of each difficulty.

  Raises
  ------
  ValueError
      If the information matrix is singular.
  """
  try:
    return _centred_inverse(information) @ gradient
  except numpy.linalg.LinAlgError as error:
    msg = "the information matrix of the difficulties is singular, so the items cannot be estimated"
    raise ValueError(msg) from error


def _variances(information: FloatArray) -> FloatArray | None:
  """Return the variances of the difficulties, or None if the information cannot be inverted or one is not positive."""
  try:
    variances = numpy.diag(_centred_inverse(information)).copy()
  except numpy.linalg.LinAlgError:
    return None
  if not numpy.all(numpy.isfinite(variances) & (variances > 0)):
    return None
  return variances
