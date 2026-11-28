"""Ability estimation: each examinee's ability, with its standard error, from responses to items of known parameters.

Each estimator takes a response matrix with one row per examinee and one column per item, holding 1, 0 or NaN where the
examinee did not answer the item, and the items, one row ``[a, b, c, d]`` per column, and returns an
:py:class:`AbilityEstimates`. :py:func:`number_correct_eap`, whose estimate depends on the number correct alone, takes
the items only and returns a :py:class:`NumberCorrectEstimates`, one entry per number correct.
"""

import functools
import math
import operator
import os
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy
import numpy.typing as npt
from scipy.integrate import IntegrationWarning, quad

from . import irt
from ._responses import responses_and_items
from ._search import (
  _SEARCH_POINTS,
  _BatchObjective,
  _Bounds,
  _brent,
  _brent_rows,
  _Derivatives,
  _Examinee,
  _maxima,
  _maximum,
  _PriorTerms,
  _Rows,
)
from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .quadrature import (
  GridPrior,
  LogPrior,
  QuadratureGrid,
  _NormalLogPrior,
  normal_log_prior,
  posterior,
  posterior_mean,
  posterior_variance,
  prior_masses,
)
from .typing import BoolArray, FloatArray, IntArray

# Step of the central difference that gives a prior's curvature at the posterior mode.
_CURVATURE_STEP = 1e-4
# Points on which Warm's modified score is scanned to bracket its roots: 0.1 apart on [-6, 6].
_SCAN_POINTS = 121
# How often the maximum likelihood search without bounds may move an end of its range out, doubling the distance each
# time: past 6 * 2**64, about 1e20, only items of a vanishing discrimination still change the likelihood.
_MAX_EXTENSIONS = 64
# The package's directory, whose frames a warning skips, so that it points at the caller's line.
_PACKAGE = str(Path(__file__).parent) + os.sep
# How a warning about one examinee begins: the examinee's row in the response matrix.
_ROW = "examinee in row {}: "
_ROW_PATTERN = re.compile(r"^examinee in row (\d+): ")
# Log-likelihoods closer than this, relative to their size, count as equal: the rounding of a sum over the items.
_RELATIVE_SLACK = 1e-12


@functools.cache
def _default_grid() -> QuadratureGrid:
  """Return the grid of EAP estimates without one, built once and never handed out."""
  return QuadratureGrid.uniform()


@functools.cache
def _default_prior() -> LogPrior:
  """Return the standard normal prior of EAP and MAP estimates without one, built once."""
  return normal_log_prior()


@dataclass(frozen=True, slots=True)
class AbilityEstimates:
  """Each examinee's ability estimate, its standard error, and the evaluations of the estimator's objective it took.

  Parameters
  ----------
  theta : numpy.ndarray
      The estimates, one per row of the response matrix.
  standard_errors : numpy.ndarray
      Their standard errors, by the estimator's formula.
  evaluations : numpy.ndarray
      For each examinee, the number of evaluations of the estimator's objective, the likelihood, the posterior or
      Warm's modified score, or of the objective's derivatives, each ability of a grid counted once.
  from_fallback : numpy.ndarray
      For each examinee, whether its estimate, standard error and evaluations are those of the fallback estimator of
      :py:func:`maximum_likelihood`; False everywhere for the other estimators.
  """

  theta: FloatArray
  standard_errors: FloatArray
  evaluations: IntArray
  from_fallback: BoolArray


def _check_tol(tol: float) -> float:
  """Return ``tol`` as a float, after checking that it is positive.

  Returns
  -------
  float
      The tolerance.

  Raises
  ------
  ValueError
      If ``tol`` is not positive, NaN included.
  """
  if not tol > 0:
    msg = f"tol must be positive, got {tol}"
    raise ValueError(msg)
  return float(tol)


def _check_bounds(bounds: _Bounds) -> _Bounds:
  """Return ``bounds`` as floats, after checking that they are finite with low < high.

  Returns
  -------
  tuple[float, float]
      The lower and the upper bound.

  Raises
  ------
  ValueError
      If either bound is not finite, or the lower bound is not below the upper one.
  """
  low, high = bounds
  if not (math.isfinite(low) and math.isfinite(high) and low < high):
    msg = f"bounds must be finite with low < high, got {bounds}"
    raise ValueError(msg)
  return float(low), float(high)


def _answered(matrix: FloatArray, items: FloatArray) -> Iterable[tuple[FloatArray, FloatArray]]:
  """Yield each examinee's responses and items, the items it did not answer left out."""
  for row in matrix:
    answered = ~numpy.isnan(row)
    yield row[answered], items[answered]


def _estimates(rows: Iterable[tuple[float, float, int]]) -> AbilityEstimates:
  """Return the estimates, standard errors and evaluation counts of the examinees, one tuple per examinee."""
  values = list(rows)
  return AbilityEstimates(
    theta=numpy.array([theta for theta, _, _ in values], dtype=float),
    standard_errors=numpy.array([se for _, se, _ in values], dtype=float),
    evaluations=numpy.array([evaluations for _, _, evaluations in values], dtype=int),
    from_fallback=numpy.zeros(len(values), dtype=bool),
  )


def expected_a_posteriori(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  grid: QuadratureGrid | None = None,
  log_prior: LogPrior | None = None,
  grid_prior: GridPrior | None = None,
) -> AbilityEstimates:
  """Return each examinee's EAP estimate, the mean of its posterior over a grid, with the posterior standard deviation.

  Every estimate is finite, whatever the responses; an examinee who answered no item gets the prior's mean and standard
  deviation on the grid. The evaluation count is the number of grid nodes.

  The :doc:`/methods/ability/eap` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, one row per column of ``responses``, with ``a > 0``
      and ``0 <= c < d <= 1``.
  grid : QuadratureGrid or None, optional
      The nodes the posterior is computed on. Default is ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6].
  log_prior : LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.
  grid_prior : GridPrior or None, optional
      A grid and prior evaluated once, in place of ``grid`` and ``log_prior``, for a caller that estimates with the
      same ones many times. Default is None.

  Returns
  -------
  AbilityEstimates
      The posterior means, the posterior standard deviations and the grid's size, one of each per examinee.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``grid_prior`` is given together with ``grid`` or ``log_prior``. Also if ``items`` is not an ``(n, 4)``
      matrix, ``responses`` is not a matrix with one column per item, a response is not 0, 1 or NaN, or the log-prior
      is NaN or +inf at a node or has no mass on the grid.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> estimates = expected_a_posteriori(numpy.array([[1, 0, numpy.nan], [1, 1, 1]]), items)
  >>> estimates.theta.round(4), estimates.standard_errors.round(4)
  (array([-0.2809,  0.8555]), array([0.7837, 0.828 ]))
  """
  if grid_prior is None:
    grid = _default_grid() if grid is None else grid
    nodes = grid.nodes
    post = posterior(
      irt.log_likelihood(nodes, responses, items), grid, _default_prior() if log_prior is None else log_prior
    )
  elif grid is None and log_prior is None:
    nodes = grid_prior.nodes
    post = grid_prior.posterior(irt.log_likelihood(nodes, responses, items))
  else:
    msg = "grid_prior replaces grid and log_prior: give either grid_prior or them"
    raise ValueError(msg)
  n_examinees = post.shape[0]
  return AbilityEstimates(
    theta=posterior_mean(post, nodes),
    standard_errors=numpy.sqrt(posterior_variance(post, nodes)),
    evaluations=numpy.full(n_examinees, nodes.size),
    from_fallback=numpy.zeros(n_examinees, dtype=bool),
  )


@dataclass(frozen=True, slots=True)
class NumberCorrectEstimates:
  """The EAP estimate of each number correct on a set of items, its standard error, and the number's probability.

  Entry ``k`` of each array is that of the number correct ``k``, from 0 to the number of items.

  Parameters
  ----------
  theta : numpy.ndarray
      The mean of the ability's posterior given each number correct.
  standard_errors : numpy.ndarray
      The posterior standard deviations.
  probability : numpy.ndarray
      The probability of each number correct for an examinee whose ability is drawn from the prior, summing to 1.
  """

  theta: FloatArray
  standard_errors: FloatArray
  probability: FloatArray


def number_correct_eap(
  items: FloatArray, *, grid: QuadratureGrid | None = None, log_prior: LogPrior | None = None
) -> NumberCorrectEstimates:
  """Return the EAP estimate of every number correct on the items, the mean of the posterior given the number alone.

  An examinee who answered every item, ``k`` of them correctly, has entry ``k``, whichever items those were. The
  probabilities of the numbers correct are the distribution of number correct the items and the prior predict.

  The :doc:`/methods/ability/number-correct-eap` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, with ``a > 0`` and ``0 <= c < d <= 1``.
  grid : QuadratureGrid or None, optional
      The nodes the posterior is computed on. Default is ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6].
  log_prior : LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.

  Returns
  -------
  NumberCorrectEstimates
      The posterior mean and standard deviation, and the probability, of each number correct from 0 to ``n``.

  Raises
  ------
  ValueError
      If ``items`` is not an ``(n, 4)`` matrix, or the log-prior is NaN or +inf at a node or has no mass on the grid.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> estimates = number_correct_eap(items)
  >>> estimates.theta.round(4), estimates.standard_errors.round(4)
  (array([-0.9932, -0.4077,  0.2652,  0.8555]), array([0.7846, 0.7986, 0.8274, 0.828 ]))
  >>> round(float(estimates.probability.sum()), 12)
  1.0
  """
  grid = _default_grid() if grid is None else grid
  masses = prior_masses(grid, _default_prior() if log_prior is None else log_prior)
  joint = irt.number_correct_probabilities(grid.nodes, items).T * masses  # numbers correct x nodes
  probability = joint.sum(axis=1)
  post = joint / probability[:, numpy.newaxis]
  return NumberCorrectEstimates(
    theta=posterior_mean(post, grid.nodes),
    standard_errors=numpy.sqrt(posterior_variance(post, grid.nodes)),
    probability=probability,
  )


def maximum_a_posteriori(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  log_prior: LogPrior | None = None,
  bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's MAP estimate, the mode of its posterior within the bounds, with its standard error.

  The estimate is the highest mode the search finds, inside the prior's support, and finite whatever the responses; an
  examinee who answered no item gets the prior's mode. The evaluation count is the number of evaluations of the
  posterior or of its derivatives.

  The :doc:`/methods/ability/map` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, one row per column of ``responses``, with ``a > 0``
      and ``0 <= c < d <= 1``.
  log_prior : LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.
  bounds : tuple[float, float], optional
      The range the mode is searched in. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``, [-6, 6].
  tol : float, optional
      The absolute tolerance of the refinement, in the ability scale. Default is 1e-6.

  Returns
  -------
  AbilityEstimates
      The modes, their standard errors, ``inf`` where the information and the prior's curvature give none, and the
      evaluation counts.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``tol`` is not positive, ``bounds`` is not a finite interval, ``items`` is not an ``(n, 4)`` matrix,
      ``responses`` is not a matrix with one column per item, a response is not 0, 1 or NaN, or the log-posterior is
      not finite at any search point.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> estimates = maximum_a_posteriori(numpy.array([[1, 0, numpy.nan], [1, 1, 1]]), items)
  >>> estimates.theta.round(4), estimates.standard_errors.round(4)
  (array([-0.2728,  0.823 ]), array([0.8029, 0.8002]))
  """
  log_prior = _default_prior() if log_prior is None else log_prior
  bounds = _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  prior = _normal_prior_terms(log_prior)
  # one examinee keeps the search of one row, which a step of an adaptive test calls and which has less to set up
  if prior is not None and matrix.shape[0] > 1:
    return _posterior_modes(matrix, items, prior, bounds, tol)
  return _estimates(
    _posterior_mode(row_responses, row_items, log_prior, bounds, tol, prior=prior)
    for row_responses, row_items in _answered(matrix, items)
  )


def _posterior_mode(
  responses: FloatArray,
  items: FloatArray,
  log_prior: LogPrior,
  bounds: _Bounds,
  tol: float,
  *,
  prior: _PriorTerms | None,
) -> tuple[float, float, int]:
  """Return one examinee's posterior mode, its standard error and the posterior evaluations it took.

  Returns
  -------
  tuple[float, float, int]
      The mode, its standard error and the number of evaluations of the posterior or of its derivatives.

  Raises
  ------
  ValueError
      If the log-posterior is not finite at any search point.
  """
  examinee = _Examinee.of(responses, items)
  derivatives = _posterior_derivatives(examinee, prior)
  theta, _, evaluations = _maximum(
    lambda theta: examinee.log_likelihood(theta) + log_prior(theta),
    bounds,
    tol,
    derivatives,
    concave=derivatives is not None and examinee.concave,
    n_points=_SEARCH_POINTS if derivatives is None else examinee.search_points,
  )
  h = _CURVATURE_STEP
  below, at, above = log_prior(numpy.array([theta - h, theta, theta + h]))
  curvature = float((above - 2.0 * at + below) / h**2)
  precision = irt.total_information(theta, items) - (curvature if math.isfinite(curvature) else 0.0)
  se = 1.0 / math.sqrt(precision) if precision > 0.0 and math.isfinite(precision) else math.inf
  return theta, se, evaluations


def _posterior_modes(
  matrix: FloatArray, items: FloatArray, prior: _PriorTerms, bounds: _Bounds, tol: float
) -> AbilityEstimates:
  """Return what :py:func:`_posterior_mode` returns for each row, every row searched at once."""
  batch = _Rows.of(matrix, items)
  rows = numpy.arange(matrix.shape[0])
  theta, _, evaluations = _maxima(_BatchObjective(batch, prior), rows, bounds, tol)
  h = _CURVATURE_STEP
  below, at, above = prior.log_density(theta - h), prior.log_density(theta), prior.log_density(theta + h)
  curvature = (above - 2.0 * at + below) / h**2
  precision = batch.total_information(theta, rows) - numpy.where(numpy.isfinite(curvature), curvature, 0.0)
  standard_errors = numpy.full(rows.size, math.inf)
  positive = (precision > 0.0) & numpy.isfinite(precision)
  standard_errors[positive] = 1.0 / numpy.sqrt(precision[positive])
  return AbilityEstimates(theta, standard_errors, evaluations, numpy.zeros(rows.size, dtype=bool))


def _normal_prior_terms(log_prior: LogPrior) -> _PriorTerms | None:
  """Return a normal prior's terms for the searches' derivatives; None for any other prior.

  A normal prior whose precision, 1 / sd^2, is not a finite positive float, from a standard deviation near the floats'
  limits, also gives None, leaving its posterior to the search without derivatives.

  Returns
  -------
  _PriorTerms or None
      The prior's log-density, mean and precision, or None.
  """
  if not isinstance(log_prior, _NormalLogPrior):
    return None
  inverse = 1.0 / log_prior.sd
  precision = inverse * inverse
  if not 0.0 < precision < math.inf:
    return None
  return _PriorTerms(log_prior, log_prior.mean, precision)


def _posterior_derivatives(examinee: _Examinee, prior: _PriorTerms | None) -> _Derivatives | None:
  """Return the log-posterior's first and second derivatives at one ability, under a normal prior's terms, or None.

  Returns
  -------
  Callable or None
      The derivatives as a function of one ability, or None without a normal prior.
  """
  if prior is None:
    return None

  def derivatives(theta: float) -> tuple[float, float]:
    slope, curvature = examinee.derivatives(theta)
    return slope - prior.precision * (theta - prior.mean), curvature - prior.precision

  return derivatives


def maximum_likelihood(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  bounds: tuple[float, float] | None = None,
  fallback: Callable[[FloatArray, FloatArray], AbilityEstimates] | None = None,
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's maximum likelihood estimate, with its standard error.

  Without ``bounds`` the estimate maximizes the likelihood over the whole ability scale, infinities included: it is
  ``inf`` or ``-inf`` where the likelihood comes closest to its supremum only in that limit, as it does for responses
  all correct or all incorrect, and its standard error is then ``inf``. With ``bounds`` it maximizes the likelihood
  within them, and is finite. An examinee who answered no item has no estimate, NaN, and standard error ``inf``. A
  ``fallback`` replaces every estimate that is not finite, infinite or NaN.

  The :doc:`/methods/ability/ml` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, one row per column of ``responses``, with ``a > 0``
      and ``0 <= c < d <= 1``.
  bounds : tuple[float, float] or None, optional
      The range the estimate is searched in; responses all correct or all incorrect give the bound they point to.
      Default is None, the whole ability scale.
  fallback : Callable or None, optional
      An estimator called as ``fallback(responses, items)``, once, on the rows whose estimate is not finite, such as
      ``functools.partial(expected_a_posteriori, log_prior=normal_log_prior(0.0, 1.0))``; its estimates, standard
      errors and evaluation counts replace theirs, and ``from_fallback`` marks them. Default is None, which keeps them.
  tol : float, optional
      The absolute tolerance of the search, in the ability scale. Default is 1e-6.

  Returns
  -------
  AbilityEstimates
      The estimates, their standard errors, and the number of evaluations of each examinee's likelihood or of its
      derivatives.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If both ``bounds`` and ``fallback`` are given, ``bounds`` is not a finite interval, ``tol`` is not positive,
      ``items`` is not an ``(n, 4)`` matrix, ``responses`` is not a matrix with one column per item, a response is not
      0, 1 or NaN, or the fallback does not return one estimate per row it is given.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> responses = numpy.array([[1, 0, numpy.nan], [1, 1, 1]])
  >>> maximum_likelihood(responses, items).theta.round(4)
  array([-0.6463,     inf])
  >>> maximum_likelihood(responses, items, bounds=(-4.0, 4.0)).theta.round(4)
  array([-0.6463,  4.    ])
  >>> estimates = maximum_likelihood(responses, items, fallback=expected_a_posteriori)
  >>> estimates.theta.round(4), estimates.from_fallback
  (array([-0.6463,  0.8555]), array([False,  True]))
  """
  if bounds is not None and fallback is not None:
    msg = "bounds and fallback cannot be given together: with bounds every estimate is finite"
    raise ValueError(msg)
  bounds = None if bounds is None else _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  # one examinee keeps the search of one row, which a step of an adaptive test calls and which has less to set up
  estimates = (
    _likelihood_maxima(matrix, items, bounds, tol)
    if matrix.shape[0] > 1
    else _estimates(
      _likelihood_maximum(row_responses, row_items, bounds, tol)
      for row_responses, row_items in _answered(matrix, items)
    )
  )
  replaced = ~numpy.isfinite(estimates.theta)
  if fallback is None or not replaced.any():
    return estimates
  substitute = _fallback_estimates(fallback, matrix, items, replaced)
  if substitute.theta.shape != (int(replaced.sum()),):
    msg = f"the fallback returned {substitute.theta.shape} estimates for {int(replaced.sum())} rows"
    raise ValueError(msg)
  theta, standard_errors = estimates.theta.copy(), estimates.standard_errors.copy()
  evaluations = estimates.evaluations.copy()
  theta[replaced], standard_errors[replaced] = substitute.theta, substitute.standard_errors
  evaluations[replaced] = substitute.evaluations
  return AbilityEstimates(theta, standard_errors, evaluations, from_fallback=replaced)


def _fallback_estimates(
  fallback: Callable[[FloatArray, FloatArray], AbilityEstimates], matrix: FloatArray, items: FloatArray, rows: BoolArray
) -> AbilityEstimates:
  """Return the fallback's estimates of the given rows, its warnings re-issued with the rows of the whole matrix."""
  positions = numpy.flatnonzero(rows)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    substitute = fallback(matrix[rows], items)
  for warning in caught:
    message = _ROW_PATTERN.sub(lambda match: _ROW.format(positions[int(match.group(1))]), str(warning.message))
    warnings.warn(message, warning.category, skip_file_prefixes=(_PACKAGE,))
  return substitute


def _likelihood_maximum(
  responses: FloatArray, items: FloatArray, bounds: _Bounds | None, tol: float
) -> tuple[float, float, int]:
  """Return one examinee's maximum likelihood estimate, its standard error and the likelihood evaluations it took."""
  if not responses.size:
    # every ability maximizes the likelihood of no response, which gives no information
    return math.nan, math.inf, 0
  n_correct = int(numpy.count_nonzero(responses))
  if n_correct in {0, responses.size}:
    # the likelihood of responses all correct rises with the ability, and that of responses all incorrect falls
    all_correct = n_correct == responses.size
    if bounds is None:
      return (math.inf if all_correct else -math.inf), math.inf, 0
    theta = bounds[1] if all_correct else bounds[0]
    return theta, irt.see(theta, items), 0
  examinee = _Examinee.of(responses, items)
  if bounds is None:
    theta, evaluations = _supremum(examinee, tol)
  else:
    theta, _, evaluations = _maximum(
      examinee.log_likelihood,
      bounds,
      tol,
      examinee.derivatives,
      concave=examinee.concave,
      n_points=examinee.search_points,
    )
  return theta, irt.see(theta, items) if math.isfinite(theta) else math.inf, evaluations


def _likelihood_maxima(matrix: FloatArray, items: FloatArray, bounds: _Bounds | None, tol: float) -> AbilityEstimates:
  """Return what :py:func:`_likelihood_maximum` returns for each row, every row searched at once."""
  batch = _Rows.of(matrix, items)
  n_answered, n_correct = batch.answered.sum(axis=1), batch.correct.sum(axis=1)
  theta = numpy.full(matrix.shape[0], math.nan)
  evaluations = numpy.zeros(matrix.shape[0], dtype=int)
  all_correct = (n_answered > 0) & (n_correct == n_answered)
  all_incorrect = (n_answered > 0) & (n_correct == 0)
  # the likelihood of responses all correct rises with the ability, and that of responses all incorrect falls
  low, high = (-math.inf, math.inf) if bounds is None else bounds
  theta[all_correct], theta[all_incorrect] = high, low
  mixed = numpy.flatnonzero((n_correct > 0) & (n_correct < n_answered))
  if mixed.size:
    search = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED) if bounds is None else bounds
    found, value, used = _maxima(_BatchObjective(batch), mixed, search, tol)
    theta[mixed], evaluations[mixed] = found, used
    if bounds is None:
      _settle_ends(batch, mixed, value, theta, evaluations, tol=tol)
  return AbilityEstimates(
    theta, _inverse_root_information(batch, theta), evaluations, numpy.zeros(theta.size, dtype=bool)
  )


def _inverse_root_information(batch: _Rows, theta: FloatArray) -> FloatArray:
  """Return what :py:func:`irtlib.irt.see` gives for each row at its estimate where finite, and ``inf`` elsewhere."""
  standard_errors = numpy.full(theta.size, math.inf)
  finite = numpy.flatnonzero(numpy.isfinite(theta))
  information = batch.total_information(theta[finite], finite)
  variance = numpy.divide(1.0, information, out=numpy.full(finite.size, math.inf), where=information.astype(bool))
  standard_errors[finite] = numpy.sqrt(variance)
  return standard_errors


def _settle_ends(
  batch: _Rows, rows: IntArray, value: FloatArray, theta: FloatArray, evaluations: IntArray, *, tol: float
) -> None:
  """Settle each row's maximum over the extended real line, as :py:func:`_supremum` does, in ``theta``.

  ``value`` is each row's log-likelihood at its maximum on [-6, 6]. A row whose range ends leave no room for a larger
  log-likelihood beyond them is settled at once, at an infinity or at that maximum; the others search on as
  :py:func:`_supremum` does, one row at a time, which also sets their ``evaluations``.
  """
  lower_ceiling, lower_limit, upper_ceiling, upper_limit = _end_sums(batch, rows)
  # max takes the lower of two equal limits, as _supremum's does
  upper = upper_limit > lower_limit
  largest = numpy.where(upper, upper_limit, lower_limit)
  target = numpy.maximum(value, largest)
  slack = _RELATIVE_SLACK * numpy.maximum(1.0, numpy.abs(target))
  unresolved = (lower_ceiling > target + slack) | (upper_ceiling > target + slack)
  infinite = ~unresolved & (largest >= value - slack)
  theta[rows[infinite]] = numpy.where(upper[infinite], math.inf, -math.inf)
  # few rows need a wider range, mostly those of a near-flat likelihood, so they are searched on one at a time
  for row in rows[unresolved].tolist():
    kept = batch.answered[row]
    responses = numpy.where(batch.correct[row, kept], 1.0, 0.0)
    theta[row], evaluations[row] = _supremum(_Examinee.of(responses, batch.items[kept]), tol)


def _end_sums(batch: _Rows, rows: IntArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return what :py:func:`_end` sums for each row at -6 and at 6: the ceilings and the limits there."""
  _, _, c, d = batch.items.T
  log_p, _ = irt.log_response_probabilities(THETA_MIN_EXTENDED, batch.items)
  _, log_q = irt.log_response_probabilities(THETA_MAX_EXTENDED, batch.items)
  correct = batch.correct[rows]
  # c = 0 and d = 1 give log(0) = -inf, a limit of 0
  with numpy.errstate(divide="ignore"):
    terms = (
      numpy.where(correct, log_p, numpy.log1p(-c)),
      numpy.where(correct, numpy.log(c), numpy.log1p(-c)),
      numpy.where(correct, numpy.log(d), log_q),
      numpy.where(correct, numpy.log(d), numpy.log1p(-d)),
    )
  lower_ceiling, lower_limit, upper_ceiling, upper_limit = (
    numpy.where(batch.answered[rows], term, 0.0).sum(axis=-1) for term in terms
  )
  return lower_ceiling, lower_limit, upper_ceiling, upper_limit


@dataclass(frozen=True, slots=True)
class _End:
  """One end of the search range, with the largest log-likelihood beyond it and the log-likelihood's limit there."""

  direction: float
  """-1 for the lower end, 1 for the upper one."""
  theta: float
  """The ability at the end."""
  ceiling: float
  """The largest log-likelihood any ability beyond the end can have."""
  limit: float
  """The log-likelihood's limit as the ability goes to infinity in the end's direction."""


def _end(theta: float, direction: float, responses: FloatArray, items: FloatArray) -> _End:
  """Return an end of the search range: the largest log-likelihood any ability beyond it can have, and the limit."""
  log_p, log_q = irt.log_response_probabilities(theta, items)
  _, _, c, d = items.T
  correct = responses == 1
  # c = 0 and d = 1 give log(0) = -inf, a limit of 0
  with numpy.errstate(divide="ignore"):
    if direction < 0:
      ceiling = numpy.where(correct, log_p, numpy.log1p(-c))
      limit = numpy.where(correct, numpy.log(c), numpy.log1p(-c))
    else:
      ceiling = numpy.where(correct, numpy.log(d), log_q)
      limit = numpy.where(correct, numpy.log(d), numpy.log1p(-d))
  return _End(direction, theta, float(ceiling.sum()), float(limit.sum()))


def _supremum(examinee: _Examinee, tol: float) -> tuple[float, int]:
  """Return the ability that maximizes one examinee's likelihood over the extended real line, and the evaluations."""

  def search(bounds: _Bounds) -> tuple[float, float, int]:
    return _maximum(
      examinee.log_likelihood,
      bounds,
      tol,
      examinee.derivatives,
      concave=examinee.concave,
      n_points=examinee.search_points,
    )

  low, high = THETA_MIN_EXTENDED, THETA_MAX_EXTENDED
  theta, value, evaluations = search((low, high))
  step = high - low
  for _ in range(_MAX_EXTENSIONS):
    ends = [_end(low, -1.0, examinee.responses, examinee.items), _end(high, 1.0, examinee.responses, examinee.items)]
    largest = max(ends, key=operator.attrgetter("limit"))
    target = max(value, largest.limit)
    slack = _RELATIVE_SLACK * max(1.0, abs(target))
    unresolved = [end for end in ends if end.ceiling > target + slack]
    if not unresolved:
      if largest.limit >= value - slack:
        return math.copysign(math.inf, largest.direction), evaluations
      return theta, evaluations
    for end in unresolved:
      stretch = (end.theta - step, end.theta) if end.direction < 0 else (end.theta, end.theta + step)
      low, high = min(low, stretch[0]), max(high, stretch[1])
      found, found_value, used = search(stretch)
      evaluations += used
      if found_value > value:
        theta, value = found, found_value
    step *= 2
  warnings.warn(
    f"the maximum likelihood search reached [{low:.3g}, {high:.3g}] without settling whether a larger likelihood lies "
    "beyond; the estimate is the best ability found",
    RuntimeWarning,
    skip_file_prefixes=(_PACKAGE,),
  )
  return theta, evaluations


def weighted_likelihood(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's weighted likelihood estimate, Warm's, with its standard error.

  The estimate is the root of Warm's modified score within the bounds; of several, the one of largest weighted
  likelihood, and of none, the bound the score points to. An examinee who answered no item has no estimate, NaN, and
  standard error ``inf``. The evaluation count is the number of modified-score evaluations. A ranking integral that does
  not converge warns with :py:class:`scipy.integrate.IntegrationWarning`.

  The :doc:`/methods/ability/wle` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, one row per column of ``responses``, with ``a > 0``
      and ``0 <= c < d <= 1``.
  bounds : tuple[float, float], optional
      The range the estimate is searched in. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``, [-6, 6].
  tol : float, optional
      The absolute tolerance of the root search, in the ability scale. Default is 1e-6.

  Returns
  -------
  AbilityEstimates
      The estimates, their standard errors, ``inf`` where the information is 0, and the evaluation counts.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``bounds`` is not a finite interval, ``tol`` is not positive, ``items`` is not an ``(n, 4)`` matrix,
      ``responses`` is not a matrix with one column per item, or a response is not 0, 1 or NaN.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> estimates = weighted_likelihood(numpy.array([[1, 0, numpy.nan], [1, 1, 1]]), items)
  >>> estimates.theta.round(4), estimates.standard_errors.round(4)
  (array([-0.5154,  1.8903]), array([1.4041, 1.9431]))
  """
  bounds = _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  if matrix.shape[0] > 1:
    return _weighted_likelihood_roots(matrix, items, bounds, tol)
  rows: list[tuple[float, float, int]] = []
  for row, (row_responses, row_items) in enumerate(_answered(matrix, items)):
    theta, se, evaluations, messages = _weighted_likelihood_root(row_responses, row_items, bounds, tol)
    rows.append((theta, se, evaluations))
    _warn_ranking(row, messages)
  return _estimates(rows)


def _warn_ranking(row: int, messages: list[str]) -> None:
  """Warn of each message from a row's ranking integrals, naming the row, from the line that called irtlib."""
  for message in messages:
    warnings.warn(_ROW.format(row) + message, IntegrationWarning, skip_file_prefixes=(_PACKAGE,))


def _weighted_likelihood_root(
  responses: FloatArray, items: FloatArray, bounds: _Bounds, tol: float
) -> tuple[float, float, int, list[str]]:
  """Return one examinee's WLE, its standard error, the score evaluations it took and the ranking's warnings."""
  if not responses.size:
    # every ability maximizes the weighted likelihood of no response, which gives no information
    return math.nan, math.inf, 0, []

  examinee = _Examinee.of(responses, items)
  score = examinee.modified_score_at
  low, high = bounds
  scan = numpy.linspace(low, high, _SCAN_POINTS)
  values = examinee.modified_score(scan)
  evaluations = scan.size
  candidates: list[float] = []
  for index in _brackets(values).tolist():
    ends, ends_values = scan[index : index + 2].tolist(), values[index : index + 2].tolist()
    root, used = _brent(score, ends[0], ends[1], ends_values[0], ends_values[1], tol=tol)
    candidates.append(root)
    evaluations += used
  candidates.extend(_outward_bounds(values, bounds))
  messages: list[str] = []
  if len(candidates) == 1:
    theta = candidates[0]
  else:
    theta, ranking_evaluations, messages = _most_likely(score, low, candidates)
    evaluations += ranking_evaluations
  return theta, irt.see(theta, items), evaluations, messages


def _brackets(values: FloatArray) -> IntArray:
  """Return the flat indices where the modified score at the scan's points turns from + to -, along its last axis.

  Each is a maximum of the weighted likelihood between a point and the next; a zero at the right is a valid end of
  Brent's bracket.

  Returns
  -------
  numpy.ndarray
      The index of each such point, flat over the other axes.
  """
  return numpy.flatnonzero((values[..., :-1] > 0.0) & (values[..., 1:] <= 0.0))


def _outward_bounds(values: FloatArray, bounds: _Bounds) -> list[float]:
  """Return the bounds that the modified score, at the scan's ends, points out of: maxima of the weighted likelihood."""
  return [bound for bound, outward in zip(bounds, (values[0] <= 0.0, values[-1] > 0.0), strict=True) if outward]


def _weighted_likelihood_roots(matrix: FloatArray, items: FloatArray, bounds: _Bounds, tol: float) -> AbilityEstimates:
  """Return what :py:func:`_weighted_likelihood_root` returns for each row, every row's scan and roots at once.

  The rows with several maxima, or none, are ranked one at a time, and those whose root search fails are searched
  alone, so that each raises or warns as it does alone, in the order of the rows.

  Returns
  -------
  AbilityEstimates
      Each row's estimate, standard error and evaluations of the modified score.
  """
  batch = _Rows.of(matrix, items)
  rows = numpy.flatnonzero(batch.answered.any(axis=1))
  scans = _Scans.of(batch, rows, bounds, tol)
  theta = numpy.full(matrix.shape[0], math.nan)
  evaluations = numpy.zeros(matrix.shape[0], dtype=int)
  evaluations[rows] = _SCAN_POINTS
  numpy.add.at(evaluations, rows[scans.owner], scans.used)
  settled = scans.settled
  theta[rows[settled]] = scans.only_candidate(bounds)[settled]
  for k in numpy.flatnonzero(~settled).tolist():
    row = int(rows[k])
    examinee = _Examinee.of(matrix[row, batch.answered[row]], items[batch.answered[row]])
    if scans.failed[scans.of_row(k)].any():
      theta[row], _, evaluations[row], messages = _weighted_likelihood_root(
        examinee.responses, examinee.items, bounds, tol
      )
    else:
      found, ranking_evaluations, messages = _most_likely(
        examinee.modified_score_at, bounds[0], scans.candidates(k, bounds)
      )
      theta[row] = found
      evaluations[row] += ranking_evaluations
    _warn_ranking(row, messages)
  return AbilityEstimates(
    theta, _inverse_root_information(batch, theta), evaluations, numpy.zeros(theta.size, dtype=bool)
  )


@dataclass(frozen=True, slots=True)
class _Scans:
  """Many rows' scans of Warm's modified score, and the roots Brent's method finds in their brackets."""

  values: FloatArray
  """The modified score of each row, by rows, at each of the scan's points."""
  owner: IntArray
  """Each bracket's row, as its position among the rows scanned: in the order of the rows, and within each, of the
  brackets."""
  roots: FloatArray
  """Each bracket's root."""
  used: IntArray
  """Each bracket's evaluations of the modified score, besides the two at its ends."""
  failed: BoolArray
  """Whether each bracket's search failed, where it raises for the row alone."""
  starts: IntArray
  """Where each row's brackets start, and after the last row's, where they end."""

  @classmethod
  def of(cls, batch: _Rows, rows: IntArray, bounds: _Bounds, tol: float) -> Self:
    """Return the scans of the given rows at the scan's points within the bounds, and the roots in their brackets."""
    scan = numpy.linspace(*bounds, _SCAN_POINTS)
    values = batch.modified_score_grid(scan, rows)
    owner, index = numpy.divmod(_brackets(values), scan.size - 1)
    roots, used, failed = _brent_rows(
      batch.modified_score,
      rows[owner],
      (scan[index], scan[index + 1]),
      (values[owner, index], values[owner, index + 1]),
      tol=tol,
    )
    return cls(values, owner, roots, used, failed, numpy.searchsorted(owner, numpy.arange(rows.size + 1)))

  def of_row(self, k: int) -> slice:
    """Return the brackets of the row at position ``k``, which are consecutive."""
    return slice(int(self.starts[k]), int(self.starts[k + 1]))

  @property
  def n_candidates(self) -> IntArray:
    """Return each row's number of maxima of the weighted likelihood: roots, and bounds the score points out of."""
    return numpy.diff(self.starts) + (self.values[:, 0] <= 0.0) + (self.values[:, -1] > 0.0)

  @property
  def settled(self) -> BoolArray:
    """Return whether each row has a single maximum, found without a failed search."""
    failed = numpy.zeros(self.values.shape[0], dtype=bool)
    failed[self.owner[self.failed]] = True
    return (self.n_candidates == 1) & ~failed

  def only_candidate(self, bounds: _Bounds) -> FloatArray:
    """Return the maximum of each row that has one: its root, else the bound the score points out of."""
    root = numpy.full(self.values.shape[0], math.nan)
    root[self.owner] = self.roots
    bound = numpy.where(self.values[:, 0] <= 0.0, bounds[0], bounds[1])
    return numpy.where(numpy.isnan(root), bound, root)

  def candidates(self, k: int, bounds: _Bounds) -> list[float]:
    """Return the maxima of the row at position ``k``, its roots then the bounds, in the order of the one-row search."""
    return self.roots[self.of_row(k)].tolist() + _outward_bounds(self.values[k], bounds)


def _most_likely(score: Callable[[float], float], low: float, candidates: list[float]) -> tuple[float, int, list[str]]:
  """Return the candidate of largest weighted likelihood, the score evaluations spent, and quad's warnings."""
  # The modified score is the derivative of the weighted log-likelihood, so its integral from the lower bound ranks the
  # candidates by weighted likelihood
  heights = []
  evaluations = 0
  messages = []
  for candidate in candidates:
    # with full_output, quad returns its convergence warning instead of raising it
    height, _, info, *message = quad(score, low, candidate, limit=200, full_output=1)
    heights.append(height)
    evaluations += info["neval"]
    messages.extend(message[:1])
  return candidates[int(numpy.argmax(heights))], evaluations, messages
