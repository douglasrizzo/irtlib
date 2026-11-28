"""The numerical searches behind :py:mod:`irtlib.ability`'s estimators: each examinee's maximum, alone or many at once.

The searches, their point counts and their measurements are documented with the estimators that use them, in
:py:mod:`irtlib.ability`.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .irt import _log_terms, _LogTerms
from .typing import FloatArray

# Points on which a maximum is located before Newton's method refines it: 101, 0.12 apart on [-6, 6], while no item
# answered is steeper than _STEEP_SLOPE, and above it 0.3 divided by the steepest slope apart (module docstring).
_NEWTON_SEARCH_POINTS = 101
_STEEP_SLOPE = 2.5
# The most points: 0.005 apart on [-6, 6], the spacing of a slope of 60, the steepest measured. Steeper items keep
# it, so that a pathological slope cannot exhaust memory.
_MAX_SEARCH_POINTS = 2401
# Points on which a posterior mode under any prior but the normal is located before SciPy's bounded search refines it.
_SEARCH_POINTS = 201
# Equal maxima on this many search points mean a flat objective; two can flank a symmetric peak.
_FLAT_MAXIMUM_POINTS = 3
# The most steps of a Newton refinement: bisection alone narrows [-6, 6] to 1e-6 in 24.
_NEWTON_STEPS = 100
# The smallest normal float, below which P Q is floored in Warm's modified score.
_TINY = numpy.finfo(float).tiny

type _Bounds = tuple[float, float]
type _Objective = Callable[[FloatArray], FloatArray]
type _Derivatives = Callable[[float], tuple[float, float]]


@functools.lru_cache(maxsize=32)
def _search_points(low: float, high: float, n_points: int) -> FloatArray:
  """Return the search's equally spaced points within the bounds, built once per bounds and read-only."""
  nodes = numpy.linspace(low, high, n_points)
  nodes.setflags(write=False)
  return nodes


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

  @property
  def search_points(self) -> int:
    """The number of points a search with Newton's refinement locates the maximum on, from the steepest slope."""
    steepest = float(self.items[:, 0].max()) if self.items.size else 0.0
    # a slope that is not finite gives the fewest points, on which the search then finds the objective not finite
    if not math.isfinite(steepest):
      return _NEWTON_SEARCH_POINTS
    spaced = math.ceil((_NEWTON_SEARCH_POINTS - 1) * steepest / _STEEP_SLOPE) + 1
    return min(_MAX_SEARCH_POINTS, max(_NEWTON_SEARCH_POINTS, spaced))

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
  n_points: int = _SEARCH_POINTS,
) -> tuple[float, float, int]:
  """Return the ability of largest objective within the bounds, the objective there and the evaluations taken.

  The best of ``n_points`` equally spaced points, or the middle of three or more tied ones, refined between its
  neighbours and kept only if no worse: by Newton's method on ``derivatives``, the objective's first and second
  derivatives at one ability, or by SciPy's bounded search without them. A ``concave`` objective skips the points
  (:py:func:`_concave_maximum`).

  Returns
  -------
  tuple[float, float, int]
      The ability, the objective there, and the number of evaluations of the objective or of its derivatives.

  Raises
  ------
  ValueError
      If the objective is not finite at any of the points.
  """
  if concave and derivatives is not None:
    return _concave_maximum(objective, derivatives, bounds, tol)
  # the points are shared and read-only, so a prior that wrote into them, which would move the best point, raises
  nodes = _search_points(*bounds, n_points)
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
      The ability, the objective there, and the number of evaluations of the objective or of its derivatives.
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
      The zero, within ``tol``, and the number of evaluations of the derivatives.
  """
  last = before_last = high - low
  for evaluations in range(1, _NEWTON_STEPS + 1):
    slope, curvature = derivatives(theta)
    if slope > 0.0:
      low = theta
    else:
      high = theta
    # a Newton step needs a finite negative curvature; any other bisects
    step = -slope / curvature if -math.inf < curvature < 0.0 else math.inf
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
