r"""Ability estimation: each examinee's ability, with its standard error, from responses to items of known parameters.

Each estimator takes a response matrix with one row per examinee and one column per item, holding 1, 0 or NaN where the
examinee did not answer the item, and the items, one row ``[a, b, c, d]`` per column. It returns an
:py:class:`AbilityEstimates`: for each examinee an estimate, its standard error and the number of abilities at which the
estimator evaluated its objective or the objective's derivatives. An examinee's estimate depends only on the items it
answered, so its row gives the same estimate alone as in any matrix. Below, :math:`L_j(\theta)` is the likelihood of
examinee :math:`j`'s responses :math:`u_{ij}`, the product of :math:`P_i^{u_{ij}} Q_i^{1 - u_{ij}}` over the items
:math:`i` it answered, and :math:`I(\theta)` the test information of those items, both as :py:mod:`irtlib.irt` defines
them.

**Maximum likelihood.** The estimate maximizes :math:`L_j(\theta)`, and its standard error is :math:`1 /
\sqrt{I(\hat\theta)}` (:cite:t:`Magis2012`, p. 4, Equations 2 and 3). The likelihood of responses all correct rises with
the ability and that of responses all incorrect falls, so their estimates are :math:`+\infty` and :math:`-\infty`
(:cite:t:`Warm1989`, p. 442). As :math:`\theta \to -\infty` each :math:`P_i` tends to :math:`c_i`, and as :math:`\theta
\to +\infty` to :math:`d_i`, so the likelihood tends to :math:`\prod_i c_i^{u_{ij}} (1 - c_i)^{1 - u_{ij}}` and to
:math:`\prod_i d_i^{u_{ij}} (1 - d_i)^{1 - u_{ij}}`: under the 3PL and 4PL, responses that are neither all correct nor
all incorrect can also have their supremum at an infinity. The maximum likelihood estimate is the maximizer over the
extended real line: :math:`+\infty` or :math:`-\infty` wherever the likelihood comes closest to its supremum only in
that limit, with standard error :math:`\infty`, since the information tends to 0 at either infinity. An examinee who
answered no item has a constant likelihood, which every ability maximizes: its estimate is NaN, and its standard error
:math:`\infty`, the information being 0. The caller has two ways to finite numbers, never used together:

- ``bounds``: the estimate maximizes the likelihood within them, and responses all correct or all incorrect give the
  bound they point to. Bounds are always set in practice (:cite:t:`Warm1989`, p. 442), and catR 3.17 sets them by
  default: its default ``constantPatt = NULL`` estimates such responses by maximum likelihood too
  (:cite:t:`Magis2022`, p. 115), which returns the end of its ``range`` that the score points to (source
  ``R/thetaEst.R:9-10, 100-109``).
- ``fallback``: another estimator, called once on the rows whose estimate is not finite, the unanswered rows
  included, whose estimates, standard errors and evaluation counts replace theirs;
  :py:attr:`AbilityEstimates.from_fallback` marks those rows.

Where an item answered has :math:`c_i > 0` or :math:`d_i < 1`, the likelihood can have several maxima, and the search
evaluates the log-likelihood at 101 equally spaced abilities of the range, 0.12 apart on [-6, 6]. It refines the best
between its neighbours by Newton's method on the log-likelihood's first and second derivatives, from the vertex of the
parabola through the three points, keeping each step inside a bracket that shrinks to the side the first derivative
points to and bisecting the bracket where a step would leave it, and it keeps the refinement only if no worse: the
search of the MAP estimate below. When every item answered has :math:`c_i = 0` and :math:`d_i = 1`, the second
derivative is :math:`-I` (see the MAP estimate below), so the log-likelihood is concave and has at most one maximum. The
search then skips the points: the estimate is a bound where the first derivative points out of the range, and otherwise
the zero of the first derivative, found by the same bracketed Newton's method from the middle of the range. On about
38,000 random mixed patterns of 1 to 60 3PL and 4PL items, half with responses drawn from the model, 101 and 201 points
found the global maximum on [-6, 6] every time and 81 points missed it once, while Brent's method over the whole range,
with no points, missed it 203 times, by a median of 3.7 in ability (measured 2026-11-28). Without bounds the range
starts as [-6, 6]. Beyond the lower end of the range, a correct response's probability is at most its value at the end,
since it rises with the ability, and an incorrect one's at most :math:`1 - c_i`; beyond the upper end, a correct one's
is at most :math:`d_i` and an incorrect one's at most its value at the end. While these bounds leave room beyond an end
for a log-likelihood above both the best found and the two limits, that end moves out, twice as far each time, and the
new stretch is searched the same way. The estimate is then the infinity of the larger limit, :math:`-\infty` of two
equal ones, if that limit is at least the best value found, within a relative :math:`10^{-12}`, and the best ability
found otherwise. After 64 moves of an end, which only items of a vanishing discrimination need, the search stops at the
best ability found and warns.

mirt 1.48 also gives :math:`+\infty` or :math:`-\infty` to responses all correct or all incorrect
(``R/fscores.internal.R:397-405``), with no standard error, NA, where irtlib's is :math:`\infty`, and NA to both
for an examinee who answered no item (``R/fscores.internal.R:94-96, 185``). It searches the
other responses with R's ``nlm`` from the EAP estimate, up to an ability of :math:`\pm 20` (``R/fscores.R:305``;
``R/fscores.internal.R:687``), takes the standard error from ``nlm``'s numerical Hessian, the observed information
(``R/fscores.internal.R:1072-1101``), which is :math:`I` for items with :math:`c_i = 0` and :math:`d_i = 1` (see the
MAP estimate below), and turns an estimate whose standard error exceeds 30 into an infinity
(``R/fscores.internal.R:1094-1096``), which irtlib does not do.

**Maximum a posteriori.** The MAP, or Bayes modal, estimate maximizes the posterior, :math:`\log \pi(\theta) + \log
L_j(\theta)` for a prior density :math:`\pi` (:cite:t:`Magis2012`, p. 5, Equation 5; :cite:t:`Warm1989`, p. 430). For a
normal prior of variance :math:`\sigma^2` its standard error is :math:`1 / \sqrt{1 / \sigma^2 + I(\hat\theta)}`
(:cite:t:`Magis2012`, p. 5, Equation 6). For any prior, irtlib takes :math:`1 / \sqrt{I(\hat\theta) -
(\log\pi)''(\hat\theta)}`, with the prior's second derivative a central difference of step :math:`10^{-4}`: Equation 6
for a normal prior, and for a uniform one, inside its support, the maximum likelihood standard error, as catR 3.17
computes them (``R/semTheta.R:43-50, 82-89``). Where the difference is not finite, at the edge of a bounded prior's
support, the prior adds nothing; where the sum is not positive, the standard error is :math:`\infty`. The search
evaluates the log-posterior at equally spaced abilities within the bounds, 101 under a normal prior and 201 under any
other. Three or more tied at the maximum make a flat posterior, and the estimate is the middle one; otherwise the best
is refined between its neighbours, and the refinement is kept only if no worse, so the estimate stays inside the prior's
support and is the highest of the modes the points separate. With a normal prior :math:`N(\mu, \sigma^2)` the refinement
is the maximum likelihood search's Newton's method, with :math:`-(\theta - \mu) / \sigma^2` and :math:`-1 / \sigma^2`
added to the derivatives, and on items that all have :math:`c_i = 0` and :math:`d_i = 1` the log-posterior is concave
too, and the search skips the points as the maximum likelihood search does. With any other prior the refinement is
SciPy's bounded search, ``scipy.optimize.minimize_scalar``. An examinee who answered no item gets the prior's mode.

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
estimate lies within the bounds whatever the responses. An examinee who answered no item has no estimate, NaN, and
standard error :math:`\infty`, as under maximum likelihood.

mirt 1.48 maximizes :math:`\log L_j + \tfrac12 \log I` instead (``R/fscores.internal.R:705-738``), Warm's estimate when
every answered item has :math:`c_i = 0` and :math:`d_i = 1`, and takes its standard error from ``nlm``'s numerical
Hessian of that weighted log-likelihood (``R/fscores.internal.R:1103-1119``).
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
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from . import irt
from ._responses import responses_and_items
from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED, _log_terms, _LogTerms
from .quadrature import (
  LogPrior,
  QuadratureGrid,
  _NormalLogPrior,
  normal_log_prior,
  posterior,
  posterior_mean,
  posterior_variance,
)
from .typing import BoolArray, FloatArray, IntArray

# Points on which a maximum is located before Newton's method refines it, 0.12 apart on [-6, 6], and before SciPy's
# bounded search refines a posterior mode under any prior but the normal, 0.06 apart.
_NEWTON_SEARCH_POINTS = 101
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
# How a warning about one examinee begins: the examinee's row in the response matrix.
_ROW = "examinee in row {}: "
_ROW_PATTERN = re.compile(r"^examinee in row (\d+): ")
# Log-likelihoods closer than this, relative to their size, count as equal: the rounding of a sum over the items.
_RELATIVE_SLACK = 1e-12
# The most steps of a Newton refinement: bisection alone narrows [-6, 6] to 1e-6 in 24.
_NEWTON_STEPS = 100
# The smallest normal float, below which P Q is floored in Warm's modified score.
_TINY = numpy.finfo(float).tiny

type _Bounds = tuple[float, float]
type _Objective = Callable[[FloatArray], FloatArray]
type _Derivatives = Callable[[float], tuple[float, float]]


@functools.cache
def _default_grid() -> QuadratureGrid:
  """Return the grid of EAP estimates without one, built once and never handed out."""
  return QuadratureGrid.uniform()


@functools.cache
def _default_prior() -> LogPrior:
  """Return the standard normal prior of EAP and MAP estimates without one, built once."""
  return normal_log_prior()


@functools.lru_cache(maxsize=32)
def _search_points(low: float, high: float, n_points: int) -> FloatArray:
  """Return the search's equally spaced points within the bounds, built once per bounds and read-only."""
  nodes = numpy.linspace(low, high, n_points)
  nodes.flags.writeable = False
  return nodes


@dataclass(frozen=True, slots=True)
class AbilityEstimates:
  """Each examinee's ability estimate, its standard error, and the evaluations of the estimator's objective it took.

  Attributes
  ----------
  theta : numpy.ndarray
      The estimates, one per row of the response matrix.
  standard_errors : numpy.ndarray
      Their standard errors, by the estimator's formula.
  evaluations : numpy.ndarray
      For each examinee, the number of abilities at which the estimator evaluated its objective, the likelihood, the
      posterior or Warm's modified score, or the objective's derivatives, counting each ability a grid holds.
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


@dataclass(frozen=True, slots=True)
class _Examinee:
  """One examinee's responses to the items it answered: its log-likelihood, and the derivatives the searches use."""

  responses: FloatArray
  """The responses, 1 or 0."""
  items: FloatArray
  """The items answered."""
  terms: _LogTerms
  """Each response's log-probability, as a function of ability."""

  @classmethod
  def of(cls, responses: FloatArray, items: FloatArray) -> Self:
    """Return the examinee of responses and items that its caller has checked."""
    return cls(responses, items, _log_terms(items, numpy.arange(responses.size), responses == 1))

  @property
  def concave(self) -> bool:
    """Whether every item answered has c = 0 and d = 1, which makes the log-likelihood concave."""
    return not (self.items[:, 2].any() or (1.0 - self.items[:, 3]).any())

  def log_likelihood(self, theta: FloatArray) -> FloatArray:
    """Return the log-likelihood at each of an array of abilities."""
    return self.terms(theta).sum(axis=-1)

  def derivatives(self, theta: float) -> tuple[float, float]:
    """Return the log-likelihood's first and second derivatives at one ability."""
    # Each term is the log P of a signed item: with F its logistic and P = floor + (d - c) F, the term's derivatives
    # are a (P - floor)(1 - F) / P and a^2 (P - floor)(1 - F) / P (floor (1 - F) / P - F)
    slope, difficulty = self.terms.items[:, 0], self.terms.items[:, 1]
    z = slope * (theta - difficulty)
    rising, falling = expit(z), expit(-z)
    above = self.terms.dmc * rising
    probability = self.terms.floor + above
    if (probability > 0.0).all():
      share, rest = above / probability, self.terms.floor / probability
    else:
      # P underflows to 0 only above a floor of 0, where (P - floor) / P is 1 and floor / P is 0
      share = numpy.divide(above, probability, out=numpy.ones_like(above), where=probability > 0.0)
      rest = numpy.divide(self.terms.floor, probability, out=numpy.zeros_like(above), where=probability > 0.0)
    first = slope * share * falling
    return float(first.sum()), float((slope * first * (rest * falling - rising)).sum())

  def modified_score(self, theta: npt.ArrayLike) -> FloatArray:
    """Return Warm's modified score at each ability: the likelihood's score plus J / (2 I)."""
    a, b, c, d = self.items.T
    z = a * (numpy.atleast_1d(numpy.asarray(theta, dtype=float))[:, numpy.newaxis] - b)
    rising, falling = expit(z), expit(-z)
    # P and Q each from its own side of the logistic, so that neither loses its digits where the other rounds to 1
    above = (d - c) * rising
    p, q = c + above, (1.0 - d) + (d - c) * falling
    first = a * above * falling
    ratio = first / numpy.maximum(p * q, _TINY)
    information = (first * ratio).sum(axis=-1)
    j = (a * first * (falling - rising) * ratio).sum(axis=-1)
    # u - P is Q for a correct response and -P for an incorrect one
    return (numpy.where(self.responses == 1, q, -p) * ratio).sum(axis=-1) + numpy.divide(
      j, 2.0 * information, out=numpy.zeros_like(j), where=information > 0.0
    )


def _maximum(
  objective: _Objective,
  bounds: _Bounds,
  tol: float,
  derivatives: _Derivatives | None = None,
  *,
  concave: bool = False,
) -> tuple[float, float, int]:
  """Return the ability of largest objective within the bounds, the objective there and the evaluations taken.

  The best of equally spaced points, or the middle of three or more tied ones, refined between its neighbours and kept
  only if no worse: by Newton's method on ``derivatives``, the objective's first and second derivatives at one ability,
  or by SciPy's bounded search without them. A ``concave`` objective skips the points (:py:func:`_concave_maximum`).

  Returns
  -------
  tuple[float, float, int]
      The ability, the objective there, and the number of abilities the objective or its derivatives were evaluated at.

  Raises
  ------
  ValueError
      If the objective is not finite at any of the points.
  """
  if concave and derivatives is not None:
    return _concave_maximum(objective, derivatives, bounds, tol)
  nodes = _search_points(*bounds, _SEARCH_POINTS if derivatives is None else _NEWTON_SEARCH_POINTS)
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
  low, high = float(nodes[max(best - 1, 0)]), float(nodes[min(best + 1, nodes.size - 1)])
  if derivatives is None:
    found, value, used = _bounded_search(objective, low, high, tol)
  else:
    found, used = _newton(derivatives, _vertex(nodes, values, best), low, high, tol)
    value, used = float(objective(numpy.array([found]))[0]), used + 1
  # A neighbour can lie outside a narrow prior's support, so the refinement is kept only if it did at least as well as
  # the best point; the estimate then always lies inside the support
  if value >= peak:
    return found, value, nodes.size + used
  return float(nodes[best]), peak, nodes.size + used


def _bounded_search(objective: _Objective, low: float, high: float, tol: float) -> tuple[float, float, int]:
  """Return the maximum SciPy's bounded search finds between two abilities, the objective there and its evaluations."""
  # Past a narrow prior's support the objective is -inf, and SciPy's parabolic step then subtracts infinities; the
  # caller checks the result against the best point
  with numpy.errstate(invalid="ignore"):
    result = minimize_scalar(
      lambda theta: -float(objective(numpy.array([theta]))[0]),
      bounds=(low, high),
      method="bounded",
      options={"xatol": tol},
    )
  return float(result.x), -float(result.fun), int(result.nfev)


def _vertex(nodes: FloatArray, values: FloatArray, best: int) -> float:
  """Return the vertex of the parabola through the best point and its neighbours, or the best point at an end."""
  if 0 < best < nodes.size - 1:
    left, centre, right = values[best - 1 : best + 2]
    curvature = left - 2.0 * centre + right
    if curvature < 0.0:
      return float(nodes[best] + 0.5 * (nodes[1] - nodes[0]) * (left - right) / curvature)
  return float(nodes[best])


def _concave_maximum(
  objective: _Objective, derivatives: _Derivatives, bounds: _Bounds, tol: float
) -> tuple[float, float, int]:
  """Return the maximum of a concave objective within the bounds, the objective there and the evaluations taken.

  A bound is the maximum where the derivative points out of the range there; otherwise the maximum is the derivative's
  zero between the bounds, which Newton's method finds from their middle.

  Returns
  -------
  tuple[float, float, int]
      The ability, the objective there, and the number of abilities the objective or its derivatives were evaluated at.
  """
  low, high = bounds
  if derivatives(low)[0] <= 0.0:
    theta, evaluations = low, 1
  elif derivatives(high)[0] >= 0.0:
    theta, evaluations = high, 2
  else:
    theta, used = _newton(derivatives, 0.5 * (low + high), low, high, tol)
    evaluations = 2 + used
  return theta, float(objective(numpy.array([theta]))[0]), evaluations + 1


def _newton(derivatives: _Derivatives, theta: float, low: float, high: float, tol: float) -> tuple[float, int]:
  """Return a zero of the objective's derivative between two abilities, and the evaluations of the derivatives taken.

  Each step is Newton's on the derivative, within a bracket that shrinks to the side the derivative points to; a step
  that would leave the bracket, or that is longer than half the step before last, bisects it instead.

  Returns
  -------
  tuple[float, int]
      The zero, within ``tol``, and the number of abilities the derivatives were evaluated at.
  """
  last = before_last = high - low
  for evaluations in range(1, _NEWTON_STEPS + 1):
    slope, curvature = derivatives(theta)
    if slope > 0.0:
      low = theta
    else:
      high = theta
    step = -slope / curvature if curvature < 0.0 else math.inf
    # a Newton step within the tolerance ends the search before the bracket test: at the zero, theta + step can round
    # to theta, on the bracket's edge, which would bisect away from it
    if abs(step) < tol:
      return min(high, max(low, theta + step)), evaluations
    if not low < theta + step < high or abs(step) > before_last / 2.0:
      step = 0.5 * (low + high) - theta
    before_last, last = last, abs(step)
    theta += step
    if abs(step) < tol:
      return theta, evaluations
  return theta, _NEWTON_STEPS


def expected_a_posteriori(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  grid: QuadratureGrid | None = None,
  log_prior: LogPrior | None = None,
) -> AbilityEstimates:
  """Return each examinee's EAP estimate, the mean of its posterior over a grid, with the posterior standard deviation.

  Every estimate is finite, whatever the responses; an examinee who answered no item gets the prior's mean and standard
  deviation on the grid. The evaluation count is the number of grid nodes.

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
  >>> estimates = expected_a_posteriori(numpy.array([[1, 0, numpy.nan], [1, 1, 1]]), items)
  >>> estimates.theta.round(4), estimates.standard_errors.round(4)
  (array([-0.2809,  0.8555]), array([0.7837, 0.828 ]))
  """
  grid = _default_grid() if grid is None else grid
  log_prior = _default_prior() if log_prior is None else log_prior
  post = posterior(irt.log_likelihood(grid.nodes, responses, items), grid, log_prior)
  n_examinees = post.shape[0]
  return AbilityEstimates(
    theta=posterior_mean(post, grid.nodes),
    standard_errors=numpy.sqrt(posterior_variance(post, grid.nodes)),
    evaluations=numpy.full(n_examinees, grid.nodes.size),
    from_fallback=numpy.zeros(n_examinees, dtype=bool),
  )


def maximum_a_posteriori(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  log_prior: LogPrior | None = None,
  bounds: _Bounds = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's MAP estimate, the mode of its posterior within the bounds, with its standard error.

  The estimate is the highest mode the search finds, inside the prior's support, and finite whatever the responses; an
  examinee who answered no item gets the prior's mode. The evaluation count is the number of abilities at which the
  posterior or its derivatives were evaluated.

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
  return _estimates(
    _posterior_mode(row_responses, row_items, log_prior, bounds, tol)
    for row_responses, row_items in _answered(matrix, items)
  )


def _posterior_mode(
  responses: FloatArray, items: FloatArray, log_prior: LogPrior, bounds: _Bounds, tol: float
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
  examinee = _Examinee.of(responses, items)
  derivatives: _Derivatives | None = None
  if isinstance(log_prior, _NormalLogPrior):
    mean, precision = log_prior.mean, 1.0 / log_prior.sd**2

    def derivatives(theta: float) -> tuple[float, float]:
      slope, curvature = examinee.derivatives(theta)
      return slope - precision * (theta - mean), curvature - precision

  theta, _, evaluations = _maximum(
    lambda theta: examinee.log_likelihood(theta) + log_prior(theta),
    bounds,
    tol,
    derivatives,
    concave=derivatives is not None and examinee.concave,
  )
  h = _CURVATURE_STEP
  below, at, above = log_prior(numpy.array([theta - h, theta, theta + h]))
  curvature = float((above - 2.0 * at + below) / h**2)
  precision = irt.total_information(theta, items) - (curvature if math.isfinite(curvature) else 0.0)
  se = 1.0 / math.sqrt(precision) if precision > 0.0 and math.isfinite(precision) else math.inf
  return theta, se, evaluations


def maximum_likelihood(
  responses: npt.ArrayLike,
  items: FloatArray,
  *,
  bounds: _Bounds | None = None,
  fallback: Callable[[FloatArray, FloatArray], AbilityEstimates] | None = None,
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's maximum likelihood estimate, with its standard error.

  Without ``bounds`` the estimate maximizes the likelihood over the whole ability scale, infinities included: it is
  ``inf`` or ``-inf`` where the likelihood comes closest to its supremum only in that limit, as it does for responses
  all correct or all incorrect, and its standard error is then ``inf``. With ``bounds`` it maximizes the likelihood
  within them, and is finite. An examinee who answered no item has no estimate, NaN, and standard error ``inf``. A
  ``fallback`` replaces every estimate that is not finite, infinite or NaN.

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
      The estimates, their standard errors, and the number of abilities at which each examinee's likelihood or its
      derivatives were evaluated.

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
  estimates = _estimates(
    _likelihood_maximum(row_responses, row_items, bounds, tol) for row_responses, row_items in _answered(matrix, items)
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
      examinee.log_likelihood, bounds, tol, examinee.derivatives, concave=examinee.concave
    )
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
    return _maximum(examinee.log_likelihood, bounds, tol, examinee.derivatives, concave=examinee.concave)

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
  bounds: _Bounds = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  tol: float = 1e-6,
) -> AbilityEstimates:
  """Return each examinee's weighted likelihood estimate, Warm's, with its standard error.

  The estimate is the root of Warm's modified score within the bounds; of several, the one of largest weighted
  likelihood, and of none, the bound the score points to. An examinee who answered no item has no estimate, NaN, and
  standard error ``inf``. The evaluation count is the number of modified-score evaluations. A ranking integral that does
  not converge warns with :py:class:`scipy.integrate.IntegrationWarning`.

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
  rows: list[tuple[float, float, int]] = []
  for row, (row_responses, row_items) in enumerate(_answered(matrix, items)):
    theta, se, evaluations, messages = _weighted_likelihood_root(row_responses, row_items, bounds, tol)
    rows.append((theta, se, evaluations))
    for message in messages:
      warnings.warn(_ROW.format(row) + message, IntegrationWarning, skip_file_prefixes=(_PACKAGE,))
  return _estimates(rows)


def _weighted_likelihood_root(
  responses: FloatArray, items: FloatArray, bounds: _Bounds, tol: float
) -> tuple[float, float, int, list[str]]:
  """Return one examinee's WLE, its standard error, the score evaluations it took and the ranking's warnings."""
  if not responses.size:
    # every ability maximizes the weighted likelihood of no response, which gives no information
    return math.nan, math.inf, 0, []

  examinee = _Examinee.of(responses, items)

  def score(theta: float) -> float:
    return float(examinee.modified_score(theta)[0])

  low, high = bounds
  scan = numpy.linspace(low, high, _SCAN_POINTS)
  values = examinee.modified_score(scan)
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
