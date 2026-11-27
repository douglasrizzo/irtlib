r"""Ability estimation: each examinee's ability, with its standard error, from responses to items of known parameters.

Each estimator takes the items, one row ``[a, b, c, d]`` per item, and a response matrix with one row per examinee and
one column per item, holding 1, 0 or NaN where the examinee did not answer the item. It returns an
:py:class:`AbilityEstimates`: for each examinee an estimate, its standard error and the number of evaluations of the
estimator's objective. An examinee's estimate depends only on the items it answered, so its row gives the same
estimate alone as in any matrix. Below, :math:`L_j(\theta)` is the likelihood of examinee :math:`j`'s responses
:math:`u_{ij}`, the product of :math:`P_i^{u_{ij}} Q_i^{1 - u_{ij}}` over the items :math:`i` it answered, and
:math:`I(\theta)` the test information of those items, both as :py:mod:`irtlib.irt` defines them.

**Maximum likelihood.** The estimate maximizes :math:`L_j(\theta)`, and its standard error is
:math:`1 / \sqrt{I(\hat\theta)}` (:cite:t:`Magis2012`, p. 4, Equations 2 and 3). The likelihood of responses all
correct rises with the ability and that of responses all incorrect falls, so their estimates are :math:`+\infty` and
:math:`-\infty` (:cite:t:`Warm1989`, p. 442). As :math:`\theta \to -\infty` each :math:`P_i` tends to :math:`c_i`, and
as :math:`\theta \to +\infty` to :math:`d_i`, so the likelihood tends to :math:`\prod_i c_i^{u_{ij}} (1 - c_i)^{1 -
u_{ij}}` and to :math:`\prod_i d_i^{u_{ij}} (1 - d_i)^{1 - u_{ij}}`: under the 3PL and 4PL, responses that are neither
all correct nor all incorrect can also have their supremum at an infinity. By the maintainer's decision of 2026-11-27,
the maximum likelihood estimate is the maximizer over the extended real line: :math:`+\infty` or :math:`-\infty`
wherever the likelihood comes closest to its supremum only in that limit, with standard error :math:`\infty`, since the
information tends to 0 at either infinity. The same decision gives the caller two ways to finite numbers, never used
together:

- ``bounds``: the estimate maximizes the likelihood within them, and responses all correct or all incorrect give the
  bound they point to. Bounds are always set in practice (:cite:t:`Warm1989`, p. 442), and catR 3.17 sets them by
  default: its default ``constantPatt = NULL`` estimates such responses by maximum likelihood too
  (:cite:t:`Magis2022`, p. 115), which returns the end of its ``range`` that the score points to (source
  ``R/thetaEst.R:9-10, 100-109``).
- ``fallback``: another estimator, called once on the rows whose estimate is not finite, whose estimates, standard
  errors and evaluation counts replace theirs; :py:attr:`AbilityEstimates.from_fallback` marks those rows.

The search evaluates the log-likelihood at 201 equally spaced abilities of the range and refines the best between its
neighbours with SciPy's bounded search, ``scipy.optimize.minimize_scalar``, keeping the refinement only if no worse: the
search of the MAP estimate below. Without bounds the range starts as [-6, 6]. Beyond the lower end of the range, a
correct response's probability is at most its value at the end, since it rises with the ability, and an incorrect one's
at most :math:`1 - c_i`; beyond the upper end, a correct one's is at most :math:`d_i` and an incorrect one's at most its
value at the end. While these bounds leave room beyond an end for a log-likelihood above both the best found and the two
limits, that end moves out, twice as far each time, and the new stretch is searched the same way. The estimate is then
the infinity of the larger limit, :math:`-\infty` of two equal ones, if that limit is at least the best value found,
within a relative :math:`10^{-12}`, and the best ability found otherwise. After 64 moves of an end, which only items of
a vanishing discrimination need, the search stops at the best ability found and warns.

mirt 1.48 also gives :math:`+\infty` or :math:`-\infty` to responses all correct or all incorrect
(``R/fscores.internal.R:397-405``), with no standard error, NA, where irtlib's is :math:`\infty`. It searches the
other responses with R's ``nlm`` from the EAP estimate, up to an ability of :math:`\pm 20` (``R/fscores.R:305``;
``R/fscores.internal.R:687``), takes the standard error from ``nlm``'s numerical Hessian, the observed information
(``R/fscores.internal.R:1072-1101``), which is :math:`I` for items with :math:`c_i = 0` and :math:`d_i = 1` (see the
MAP estimate below), and turns an estimate whose standard error exceeds 30 into an infinity
(``R/fscores.internal.R:1094-1096``), which irtlib does not do.

**Maximum a posteriori.** The MAP, or Bayes modal, estimate maximizes the posterior, :math:`\log \pi(\theta) + \log
L_j(\theta)` for a prior density :math:`\pi` (:cite:t:`Magis2012`, p. 5, Equation 5; :cite:t:`Warm1989`, p. 430). For
a normal prior of variance :math:`\sigma^2` its standard error is :math:`1 / \sqrt{1 / \sigma^2 + I(\hat\theta)}`
(:cite:t:`Magis2012`, p. 5, Equation 6). For any prior, irtlib takes :math:`1 / \sqrt{I(\hat\theta) -
(\log\pi)''(\hat\theta)}`, with the prior's second derivative a central difference of step :math:`10^{-4}`: Equation 6
for a normal prior, and for a uniform one, inside its support, the maximum likelihood standard error, as catR 3.17
computes them (``R/semTheta.R:43-50, 82-89``). Where the difference is not finite, at the edge of a bounded prior's
support, the prior adds nothing; where the sum is not positive, the standard error is :math:`\infty`. The search
evaluates the log-posterior at 201 equally spaced abilities within the bounds. Three or more tied at the maximum make a
flat posterior, and the estimate is the middle one; otherwise the best is refined between its neighbours, and the
refinement is kept only if no worse, so the estimate stays inside the prior's support and is the highest of the modes
the points separate. An examinee who answered no item gets the prior's mode.

mirt 1.48 maximizes the same posterior with ``nlm`` and takes the standard error from ``nlm``'s numerical Hessian
(``R/fscores.internal.R:684-703, 1047-1070``): the observed information plus the prior's precision. When every answered
item has :math:`c_i = 0` and :math:`d_i = 1`, the observed information is :math:`I` exactly (:cite:t:`Warm1989`, p. 449,
whose model fixes :math:`d_i = 1`, p. 428, Equation 1), and the two standard errors agree.

**Expected a posteriori.** The EAP estimate is the mean of the examinee's posterior over the grid of
:py:mod:`irtlib.quadrature`, and its standard error the posterior standard deviation (:cite:t:`Bock1982`, p. 433,
Equations 4 and 5; :cite:t:`Magis2012`, pp. 5-6, Equations 10 and 11). It is finite whatever the responses, and an
examinee who answered no item gets the prior's mean and standard deviation on the grid. irtlib's default grid and prior
are mirt 1.48's: 121 nodes on [-6, 6] (``R/fscores.R:302, 318-319``; ``R/fscores.internal.R:231``) with the standard
normal density at the nodes, normalized (``R/fscores.internal.R:316-317``). mirt gives NA to an examinee who answered no
item.

**Weighted likelihood.** Warm's estimate is the root of the modified score equation

.. math::

   \sum_i \frac{(u_{ij} - P_i) P_i'}{P_i Q_i} + \frac{J}{2 I} = 0, \quad J = \sum_i \frac{P_i' P_i''}{P_i Q_i}

(:cite:t:`Warm1989`, p. 431, Equation 10; p. 430), the derivative of the log of the weighted likelihood
:math:`w(\theta) L_j(\theta)`, with :math:`\partial \log w / \partial\theta = J / (2 I)`, which the estimate maximizes
(p. 446, Equations A3 and A4). It removes the first-order bias of the maximum likelihood estimate (p. 427; p. 431,
Theorem). When every :math:`c_i = 0`, :math:`w = I^{1/2}` (p. 431), in Warm's model, which fixes :math:`d_i = 1`
(p. 428, Equation 1). The estimate's variance is asymptotically :math:`1 / I`, that of the maximum likelihood estimate
(p. 431), and its standard error is :math:`1 / \sqrt{I(\hat\theta)}`, catR 3.17's default (``R/semTheta.R:90-96``);
:cite:t:`Magis2012` (p. 6, Equation 14) give a corrected one, which irtlib does not use. The search evaluates the
modified score at 121 equally spaced abilities within the bounds. Each change of sign from positive to negative,
refined by Brent's method, ``scipy.optimize.brentq``, is a maximum of the weighted likelihood, and so is a bound the
score points out of. Of several, the estimate is the one of largest weighted likelihood, ranked by integrating the score
from the lower bound with ``scipy.integrate.quad``. Warm finds finite estimates for responses all correct or all
incorrect on his tests (p. 443, Table 1), and :cite:t:`Magis2012` (p. 11) propose the estimator for them; irtlib's
estimate lies within the bounds whatever the responses.

mirt 1.48 maximizes :math:`\log L_j + \tfrac12 \log I` instead (``R/fscores.internal.R:705-738``), Warm's estimate when
every answered item has :math:`c_i = 0` and :math:`d_i = 1`, and takes its standard error from ``nlm``'s numerical
Hessian of that weighted log-likelihood (``R/fscores.internal.R:1103-1119``).
"""

import math
import operator
import os
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy
import numpy.typing as npt
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, minimize_scalar

from . import irt
from ._responses import responses_and_items
from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .quadrature import LogPrior, QuadratureGrid, normal_log_prior, posterior, posterior_mean, posterior_variance
from .typing import BoolArray, FloatArray, IntArray

# Points on which the posterior mode and the likelihood maximum are located before they are refined: 0.06 apart on
# [-6, 6].
_SEARCH_POINTS = 201
# Equal maxima on this many search points mean a flat objective; two can flank a symmetric peak.
_FLAT_MAXIMUM_POINTS = 3
# Step of the central difference that gives a prior's curvature at the posterior mode.
_CURVATURE_STEP = 1e-4
# Points on which Warm's modified score is scanned to bracket its roots: 0.1 apart on [-6, 6].
_SCAN_POINTS = 121
# How often the maximum likelihood search without bounds may move an end of its range out, doubling the distance each
# time: past 6 * 2**64, about 1e20, only items of a vanishing discrimination still change the likelihood.
_MAX_EXTENSIONS = 64
# The package's directory, whose frames a warning skips, so that it points at the caller's line.
_PACKAGE = str(Path(__file__).parent) + os.sep
# Log-likelihoods closer than this, relative to their size, count as equal: the rounding of a sum over the items.
_RELATIVE_SLACK = 1e-12

type _Bounds = tuple[float, float]
type _Objective = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class AbilityEstimates:
  """Each examinee's ability estimate, its standard error, and the evaluations of the estimator's objective it took.

  Attributes
  ----------
  theta : numpy.ndarray
      The estimates, one per row of the response matrix.
  se : numpy.ndarray
      Their standard errors, by the estimator's formula.
  evaluations : numpy.ndarray
      For each examinee, how many times the estimator evaluated its objective, the likelihood, the posterior or Warm's
      modified score, counting each ability a grid holds.
  from_fallback : numpy.ndarray
      For each examinee, whether its estimate, standard error and evaluations are those of the fallback estimator of
      :py:func:`maximum_likelihood`; False everywhere for the other estimators.
  """

  theta: FloatArray
  se: FloatArray
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


def _answered(items: FloatArray, matrix: FloatArray) -> Iterable[tuple[FloatArray, FloatArray]]:
  """Yield each examinee's items and responses, the items it did not answer left out."""
  for row in matrix:
    answered = ~numpy.isnan(row)
    yield items[answered], row[answered]


def _estimates(rows: Iterable[tuple[float, float, int]]) -> AbilityEstimates:
  """Return the estimates, standard errors and evaluation counts of the examinees, one tuple per examinee."""
  values = list(rows)
  return AbilityEstimates(
    theta=numpy.array([theta for theta, _, _ in values], dtype=float),
    se=numpy.array([se for _, se, _ in values], dtype=float),
    evaluations=numpy.array([evaluations for _, _, evaluations in values], dtype=int),
    from_fallback=numpy.zeros(len(values), dtype=bool),
  )


def _log_likelihood(items: FloatArray, responses: FloatArray) -> _Objective:
  """Return one examinee's log-likelihood at each of an array of abilities, from the items it answered."""
  log_likelihood = irt.log_likelihood_function(responses[numpy.newaxis], items)
  return lambda theta: log_likelihood(theta)[0]


def _maximum(objective: _Objective, bounds: _Bounds, tol: float) -> tuple[float, float, int]:
  """Return the ability of largest objective within the bounds, the objective there and the evaluations taken.

  The best of equally spaced points, or the middle of three or more tied ones, refined between its neighbours with
  SciPy's bounded search and kept only if no worse.

  Returns
  -------
  tuple[float, float, int]
      The ability, the objective there, and the number of abilities the objective was evaluated at.

  Raises
  ------
  ValueError
      If the objective is not finite at any of the points.
  """
  nodes = numpy.linspace(bounds[0], bounds[1], _SEARCH_POINTS)
  values = objective(nodes)
  peak = float(numpy.max(values))
  # the log-likelihood is finite at every finite ability, so only a prior can leave no finite point
  if not math.isfinite(peak):
    msg = (
      f"the log-posterior is not finite at any of the {nodes.size} search points in {bounds}; the prior's support "
      "must contain at least one of them"
    )
    raise ValueError(msg)
  ties = numpy.flatnonzero(values == peak)
  best = int(ties[ties.size // 2])
  if ties.size >= _FLAT_MAXIMUM_POINTS:
    return float(nodes[best]), peak, nodes.size
  # Past a narrow prior's support the objective is -inf, and SciPy's parabolic step then subtracts infinities; the
  # result is checked against the best point below
  with numpy.errstate(invalid="ignore"):
    result = minimize_scalar(
      lambda theta: -float(objective(numpy.array([theta]))[0]),
      bounds=(nodes[max(best - 1, 0)], nodes[min(best + 1, nodes.size - 1)]),
      method="bounded",
      options={"xatol": tol},
    )
  evaluations = nodes.size + int(result.nfev)
  # A neighbour can lie outside a narrow prior's support, so the refinement is kept only if it did at least as well as
  # the best point; the estimate then always lies inside the support
  if -result.fun >= peak:
    return float(result.x), -float(result.fun), evaluations
  return float(nodes[best]), peak, evaluations


def expected_a_posteriori(
  items: FloatArray,
  responses: npt.ArrayLike,
  *,
  grid: QuadratureGrid | None = None,
  log_prior: LogPrior | None = None,
) -> AbilityEstimates:
  """Return each examinee's EAP estimate, the mean of its posterior over a grid, with the posterior standard deviation.

  Every estimate is finite, whatever the responses; an examinee who answered no item gets the prior's mean and standard
  deviation on the grid. The evaluation count is the number of grid nodes.

  Parameters
  ----------
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, with ``a > 0`` and ``0 <= c < d <= 1``.
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  grid : QuadratureGrid or None, optional
      The nodes the posterior is computed on. Default is ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6].
  log_prior : LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.

  Returns
  -------
  AbilityEstimates
      The posterior means, the posterior standard deviations and the grid's size, one of each per examinee.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``items`` is not an ``(n, 4)`` matrix, ``responses`` is not a matrix with one column per item, a response is
      not 0, 1 or NaN, or the log-prior is NaN or +inf at a node or has no mass on the grid.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
  >>> estimates = expected_a_posteriori(items, numpy.array([[1, 0, numpy.nan], [1, 1, 1]]))
  >>> estimates.theta.round(4), estimates.se.round(4)
  (array([-0.2809,  0.8555]), array([0.7837, 0.828 ]))
  """
  grid = QuadratureGrid.uniform() if grid is None else grid
  log_prior = normal_log_prior() if log_prior is None else log_prior
  post = posterior(irt.log_likelihood(grid.nodes, responses, items), grid, log_prior)
  n_examinees = post.shape[0]
  return AbilityEstimates(
    theta=posterior_mean(post, grid.nodes),
    se=numpy.sqrt(posterior_variance(post, grid.nodes)),
    evaluations=numpy.full(n_examinees, grid.nodes.size),
    from_fallback=numpy.zeros(n_examinees, dtype=bool),
  )


def maximum_a_posteriori(
  items: FloatArray,
  responses: npt.ArrayLike,
  *,
  log_prior: LogPrior | None = None,
  bounds: _Bounds = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's MAP estimate, the mode of its posterior within the bounds, with its standard error.

  The estimate is the highest mode the search finds, inside the prior's support, and finite whatever the responses; an
  examinee who answered no item gets the prior's mode. The evaluation count is the number of search points plus the
  refinement's posterior evaluations.

  Parameters
  ----------
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, with ``a > 0`` and ``0 <= c < d <= 1``.
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
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
  >>> estimates = maximum_a_posteriori(items, numpy.array([[1, 0, numpy.nan], [1, 1, 1]]))
  >>> estimates.theta.round(4), estimates.se.round(4)
  (array([-0.2728,  0.823 ]), array([0.8029, 0.8002]))
  """
  log_prior = normal_log_prior() if log_prior is None else log_prior
  bounds = _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  return _estimates(
    _posterior_mode(row_items, row_responses, log_prior, bounds, tol)
    for row_items, row_responses in _answered(items, matrix)
  )


def _posterior_mode(
  items: FloatArray, responses: FloatArray, log_prior: LogPrior, bounds: _Bounds, tol: float
) -> tuple[float, float, int]:
  """Return one examinee's posterior mode, its standard error and the posterior evaluations it took.

  Returns
  -------
  tuple[float, float, int]
      The mode, its standard error and the number of abilities the posterior was evaluated at.

  Raises
  ------
  ValueError
      If the log-posterior is not finite at any search point.
  """
  log_likelihood = _log_likelihood(items, responses)
  theta, _, evaluations = _maximum(lambda theta: log_likelihood(theta) + log_prior(theta), bounds, tol)
  h = _CURVATURE_STEP
  below, at, above = log_prior(numpy.array([theta - h, theta, theta + h]))
  curvature = float((above - 2.0 * at + below) / h**2)
  precision = irt.total_information(theta, items) - (curvature if math.isfinite(curvature) else 0.0)
  se = 1.0 / math.sqrt(precision) if precision > 0.0 and math.isfinite(precision) else math.inf
  return theta, se, evaluations


def maximum_likelihood(
  items: FloatArray,
  responses: npt.ArrayLike,
  *,
  bounds: _Bounds | None = None,
  fallback: Callable[[FloatArray, FloatArray], AbilityEstimates] | None = None,
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's maximum likelihood estimate, with its standard error.

  Without ``bounds`` the estimate maximizes the likelihood over the whole ability scale, infinities included: it is
  ``inf`` or ``-inf`` where the likelihood comes closest to its supremum only in that limit, as it does for responses
  all correct or all incorrect, and its standard error is then ``inf``. With ``bounds`` it maximizes the likelihood
  within them, and is finite. An examinee who answered no item has no estimate, NaN, and no standard error, NaN. A
  ``fallback`` replaces every estimate that is not finite, infinite or NaN.

  Parameters
  ----------
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, with ``a > 0`` and ``0 <= c < d <= 1``.
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  bounds : tuple[float, float] or None, optional
      The range the estimate is searched in; responses all correct or all incorrect give the bound they point to.
      Default is None, the whole ability scale.
  fallback : Callable or None, optional
      An estimator called as ``fallback(items, responses)``, once, on the rows whose estimate is not finite, such as
      ``functools.partial(expected_a_posteriori, log_prior=normal_log_prior(0.0, 1.0))``; its estimates, standard
      errors and evaluation counts replace theirs, and ``from_fallback`` marks them. Default is None, which keeps them.
  tol : float, optional
      The absolute tolerance of the search, in the ability scale. Default is 1e-6.

  Returns
  -------
  AbilityEstimates
      The estimates, their standard errors, and the number of likelihood evaluations each took.

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
  >>> maximum_likelihood(items, responses).theta.round(4)
  array([-0.6463,     inf])
  >>> maximum_likelihood(items, responses, bounds=(-4.0, 4.0)).theta.round(4)
  array([-0.6463,  4.    ])
  >>> estimates = maximum_likelihood(items, responses, fallback=expected_a_posteriori)
  >>> estimates.theta.round(4), estimates.from_fallback
  (array([-0.6463,  0.8555]), array([False,  True]))
  """
  if bounds is not None and fallback is not None:
    msg = "bounds and fallback cannot be given together: with bounds every estimate is finite"
    raise ValueError(msg)
  bounds = None if bounds is None else _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  estimates = _estimates(
    _likelihood_maximum(row_items, row_responses, bounds, tol) for row_items, row_responses in _answered(items, matrix)
  )
  replaced = ~numpy.isfinite(estimates.theta)
  if fallback is None or not replaced.any():
    return estimates
  substitute = fallback(items, matrix[replaced])
  if substitute.theta.shape != (int(replaced.sum()),):
    msg = f"the fallback returned {substitute.theta.shape} estimates for {int(replaced.sum())} rows"
    raise ValueError(msg)
  theta, se, evaluations = estimates.theta.copy(), estimates.se.copy(), estimates.evaluations.copy()
  theta[replaced], se[replaced], evaluations[replaced] = substitute.theta, substitute.se, substitute.evaluations
  return AbilityEstimates(theta=theta, se=se, evaluations=evaluations, from_fallback=replaced)


def _likelihood_maximum(
  items: FloatArray, responses: FloatArray, bounds: _Bounds | None, tol: float
) -> tuple[float, float, int]:
  """Return one examinee's maximum likelihood estimate, its standard error and the likelihood evaluations it took."""
  if not responses.size:
    return math.nan, math.nan, 0
  n_correct = int(numpy.count_nonzero(responses))
  if n_correct in {0, responses.size}:
    # the likelihood of responses all correct rises with the ability, and that of responses all incorrect falls
    all_correct = n_correct == responses.size
    if bounds is None:
      return (math.inf if all_correct else -math.inf), math.inf, 0
    theta = bounds[1] if all_correct else bounds[0]
    return theta, irt.see(theta, items), 0
  log_likelihood = _log_likelihood(items, responses)
  if bounds is None:
    theta, evaluations = _supremum(log_likelihood, items, responses, tol)
  else:
    theta, _, evaluations = _maximum(log_likelihood, bounds, tol)
  return theta, irt.see(theta, items) if math.isfinite(theta) else math.inf, evaluations


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


def _end(theta: float, direction: float, items: FloatArray, responses: FloatArray) -> _End:
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


def _supremum(log_likelihood: _Objective, items: FloatArray, responses: FloatArray, tol: float) -> tuple[float, int]:
  """Return the ability that maximizes one examinee's likelihood over the extended real line, and the evaluations."""
  low, high = THETA_MIN_EXTENDED, THETA_MAX_EXTENDED
  theta, value, evaluations = _maximum(log_likelihood, (low, high), tol)
  step = high - low
  for _ in range(_MAX_EXTENSIONS):
    ends = [_end(low, -1.0, items, responses), _end(high, 1.0, items, responses)]
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
      found, found_value, used = _maximum(log_likelihood, stretch, tol)
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
  items: FloatArray,
  responses: npt.ArrayLike,
  *,
  bounds: _Bounds = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's weighted likelihood estimate, Warm's, with its standard error.

  The estimate is the root of Warm's modified score within the bounds; of several, the one of largest weighted
  likelihood, and of none, the bound the score points to. An examinee who answered no item has no estimate, NaN, and no
  standard error, NaN. The evaluation count is the number of modified-score evaluations. A ranking integral that does
  not converge warns with :py:class:`scipy.integrate.IntegrationWarning`.

  Parameters
  ----------
  items : numpy.ndarray
      The items ``[a, b, c, d]``, a matrix of shape ``(n, 4)``, with ``a > 0`` and ``0 <= c < d <= 1``.
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
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
  >>> estimates = weighted_likelihood(items, numpy.array([[1, 0, numpy.nan], [1, 1, 1]]))
  >>> estimates.theta.round(4), estimates.se.round(4)
  (array([-0.5154,  1.8903]), array([1.4041, 1.9431]))
  """
  bounds = _check_bounds(bounds)
  tol = _check_tol(tol)
  matrix, items = responses_and_items(responses, items)
  rows: list[tuple[float, float, int]] = []
  for row, (row_items, row_responses) in enumerate(_answered(items, matrix)):
    theta, se, evaluations, messages = _weighted_likelihood_root(row_items, row_responses, bounds, tol)
    rows.append((theta, se, evaluations))
    for message in messages:
      warnings.warn(f"examinee in row {row}: {message}", IntegrationWarning, stacklevel=2)
  return _estimates(rows)


def _modified_score(theta: npt.ArrayLike, items: FloatArray, responses: FloatArray) -> FloatArray:
  """Return Warm's modified score at each ability: the likelihood's score plus J / (2 I)."""
  thetas = numpy.atleast_1d(numpy.asarray(theta, dtype=float))
  log_p, log_q = irt.log_response_probabilities(thetas, items)
  first, second = irt.icc_derivatives(thetas, items)
  # P and Q from their logarithms keep their digits where one of them rounds to 1
  p, q = numpy.exp(log_p), numpy.exp(log_q)
  pq = numpy.maximum(numpy.exp(log_p + log_q), numpy.finfo(float).tiny)
  # u - P is Q for a correct response and -P for an incorrect one
  residual = numpy.where(responses == 1, q, -p)
  score = numpy.sum(residual * first / pq, axis=-1)
  information = numpy.sum(first**2 / pq, axis=-1)
  j = numpy.sum(first * second / pq, axis=-1)
  correction = numpy.divide(j, 2.0 * information, out=numpy.zeros_like(j), where=information > 0.0)
  return score + correction


def _weighted_likelihood_root(
  items: FloatArray, responses: FloatArray, bounds: _Bounds, tol: float
) -> tuple[float, float, int, list[str]]:
  """Return one examinee's WLE, its standard error, the score evaluations it took and the ranking's warnings."""
  if not responses.size:
    return math.nan, math.nan, 0, []

  def score(theta: float) -> float:
    return float(_modified_score(theta, items, responses)[0])

  low, high = bounds
  scan = numpy.linspace(low, high, _SCAN_POINTS)
  values = _modified_score(scan, items, responses)
  evaluations = scan.size
  # Maxima of the weighted likelihood: + to - sign changes of its derivative (a zero at the right end of a bracket is a
  # valid Brent endpoint), and any bound the modified score points outward from
  candidates: list[float] = []
  for index in numpy.flatnonzero((values[:-1] > 0.0) & (values[1:] <= 0.0)).tolist():
    root, result = brentq(score, scan[index], scan[index + 1], xtol=tol, full_output=True)
    candidates.append(float(root))
    evaluations += result.function_calls
  if values[0] <= 0.0:
    candidates.append(low)
  if values[-1] > 0.0:
    candidates.append(high)
  messages: list[str] = []
  if len(candidates) == 1:
    theta = candidates[0]
  else:
    theta, ranking_evaluations, messages = _most_likely(score, low, candidates)
    evaluations += ranking_evaluations
  return theta, irt.see(theta, items), evaluations, messages


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
