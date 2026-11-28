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

from . import irt
from .irt import _log_asymptotes, _log_probability, _log_terms, _LogTerms
from .typing import BoolArray, FloatArray, IntArray

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


# The most floats a batch's grid evaluates at once, rows times points times items: 8 MB.
_CHUNK_ELEMENTS = 2**20


@dataclass(frozen=True, slots=True)
class _PriorTerms:
  """A normal prior's log-density, and the mean and precision its derivatives need."""

  log_density: Callable[[FloatArray], FloatArray]
  mean: float
  precision: float


@dataclass(frozen=True, slots=True)
class _Rows:
  """Examinees' responses to the same items, each signed as :py:class:`_Examinee` signs it, for searches of every row.

  Each row's quantities sum over the items it answered, element by element as :py:class:`_Examinee`'s do, so a row
  with no missing response gets exactly its quantities alone.
  """

  answered: BoolArray
  """Which items each row answered, rows by items."""
  correct: BoolArray
  """Which it answered correctly."""
  items: FloatArray
  """The items, one row ``[a, b, c, d]`` per column of the responses."""
  slope: FloatArray
  """Each item's discrimination for each row, negated where the row answered it incorrectly."""
  floor: FloatArray
  """c where the row answered the item correctly, 1 - d elsewhere."""
  log_floor: FloatArray
  """The logarithm of ``floor``."""
  dmc: FloatArray
  """Each item's d - c."""
  log_dmc: FloatArray
  """The logarithm of ``dmc``."""

  @classmethod
  def of(cls, matrix: FloatArray, items: FloatArray) -> Self:
    """Return the rows of a response matrix and its items, which the caller has checked."""
    answered, correct = ~numpy.isnan(matrix), matrix == 1
    a, _, c, d = items.T
    log_c, log_1md, log_dmc = _log_asymptotes(items)
    return cls(
      answered,
      correct,
      items,
      numpy.where(correct, a, -a),
      numpy.where(correct, c, 1.0 - d),
      numpy.where(correct, log_c, log_1md),
      d - c,
      log_dmc,
    )

  @property
  def concave(self) -> BoolArray:
    """For each row, whether every item it answered has c = 0 and d = 1."""
    _, _, c, d = self.items.T
    return ~(self.answered & (c.astype(bool) | (1.0 - d).astype(bool))).any(axis=1)

  @property
  def search_points(self) -> IntArray:
    """For each row, :py:attr:`_Examinee.search_points`."""
    steepest = numpy.where(self.answered, self.items[:, 0], -numpy.inf).max(axis=1, initial=-numpy.inf)
    finite = numpy.isfinite(steepest)
    spaced = numpy.ceil((_NEWTON_SEARCH_POINTS - 1) * numpy.where(finite, steepest, 0.0) / _STEEP_SLOPE) + 1
    points = numpy.minimum(_MAX_SEARCH_POINTS, numpy.maximum(_NEWTON_SEARCH_POINTS, spaced)).astype(int)
    return numpy.where(finite, points, _NEWTON_SEARCH_POINTS)

  def log_likelihood(self, theta: FloatArray, rows: IntArray) -> FloatArray:
    """Return each of the rows' log-likelihood at its own ability, ``theta[i]`` for ``rows[i]``."""
    z = self.slope[rows] * (theta[:, numpy.newaxis] - self.items[:, 1])
    terms = _log_probability(z, self.log_floor[rows], self.log_dmc)
    return numpy.where(self.answered[rows], terms, 0.0).sum(axis=-1)

  def grid(self, nodes: FloatArray, rows: IntArray) -> FloatArray:
    """Return each of the rows' log-likelihood at every node, rows by nodes."""
    log_p, log_q = irt.log_response_probabilities(nodes, self.items)
    values = numpy.empty((rows.size, nodes.size))
    chunk = max(1, _CHUNK_ELEMENTS // (nodes.size * self.items.shape[0]))
    for start in range(0, rows.size, chunk):
      part = rows[start : start + chunk]
      terms = numpy.where(self.correct[part][:, numpy.newaxis, :], log_p, log_q)
      values[start : start + chunk] = numpy.where(self.answered[part][:, numpy.newaxis, :], terms, 0.0).sum(axis=-1)
    return values

  def derivatives(self, theta: FloatArray, rows: IntArray) -> tuple[FloatArray, FloatArray]:
    """Return each of the rows' log-likelihood's first and second derivatives at its own ability."""
    slope, floor = self.slope[rows], self.floor[rows]
    z = slope * (theta[:, numpy.newaxis] - self.items[:, 1])
    rising, falling = expit(z), expit(-z)
    above = self.dmc * rising
    probability = floor + above
    share = numpy.divide(above, probability, out=numpy.ones_like(above), where=probability > 0.0)
    rest = numpy.divide(floor, probability, out=numpy.zeros_like(above), where=probability > 0.0)
    first = slope * share * falling
    second = slope * first * (rest * falling - rising)
    answered = self.answered[rows]
    return numpy.where(answered, first, 0.0).sum(axis=-1), numpy.where(answered, second, 0.0).sum(axis=-1)

  def total_information(self, theta: FloatArray, rows: IntArray) -> FloatArray:
    """Return each of the rows' test information, over the items it answered, at its own ability."""
    return numpy.where(self.answered[rows], irt.information(theta, self.items), 0.0).sum(axis=-1)


@dataclass(frozen=True, slots=True)
class _BatchObjective:
  """A batched search's objective: each row's log-likelihood, plus a normal prior's log-density if there is one."""

  batch: _Rows
  prior: _PriorTerms | None = None

  def __call__(self, theta: FloatArray, rows: IntArray) -> FloatArray:
    """Return each of the rows' objective at its own ability, ``theta[i]`` for ``rows[i]``."""
    values = self.batch.log_likelihood(theta, rows)
    return values if self.prior is None else values + self.prior.log_density(theta)

  def derivatives(self, theta: FloatArray, rows: IntArray) -> tuple[FloatArray, FloatArray]:
    """Return each of the rows' objective's first and second derivatives at its own ability."""
    slope, curvature = self.batch.derivatives(theta, rows)
    if self.prior is None:
      return slope, curvature
    return slope - self.prior.precision * (theta - self.prior.mean), curvature - self.prior.precision

  def grid(self, nodes: FloatArray, rows: IntArray) -> FloatArray:
    """Return each of the rows' objective at every node, rows by nodes."""
    values = self.batch.grid(nodes, rows)
    return values if self.prior is None else values + self.prior.log_density(nodes)


def _maxima(
  objective: _BatchObjective, rows: IntArray, bounds: _Bounds, tol: float
) -> tuple[FloatArray, FloatArray, IntArray]:
  """Return what :py:func:`_maximum` returns for each row alone, with Newton's refinement.

  Rows whose items all have c = 0 and d = 1 skip the points, as :py:func:`_concave_maximum` does; the others are
  searched on their own number of points, :py:attr:`_Rows.search_points`.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
      Each row's ability, the objective there, and its evaluations of the objective or of its derivatives.

  Raises
  ------
  ValueError
      If a row's objective is not finite at any of its points.
  """
  theta, value = numpy.empty(rows.size), numpy.empty(rows.size)
  evaluations = numpy.zeros(rows.size, dtype=int)
  concave = objective.batch.concave[rows]
  if concave.any():
    theta[concave], value[concave], evaluations[concave] = _concave_maxima(objective, rows[concave], bounds, tol)
  points = objective.batch.search_points[rows]
  failed: list[tuple[int, int]] = []
  for n_points in numpy.unique(points[~concave]).tolist():
    group = numpy.flatnonzero(~concave & (points == n_points))
    *found, finite = _grid_maxima(objective, rows[group], _search_points(*bounds, n_points), tol)
    theta[group], value[group], evaluations[group] = found
    if not finite.all():
      failed.append((int(group[~finite][0]), n_points))
  if failed:
    # the error of the first row to fail, as a search of one row at a time raises it
    n_points = min(failed)[1]
    msg = (
      f"the log-posterior is not finite at any of the {n_points} search points in {bounds}; the prior's support "
      "must contain at least one of them"
    )
    raise ValueError(msg)
  return theta, value, evaluations


def _concave_maxima(
  objective: _BatchObjective, rows: IntArray, bounds: _Bounds, tol: float
) -> tuple[FloatArray, FloatArray, IntArray]:
  """Return what :py:func:`_concave_maximum` returns for each row alone.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
      Each row's ability, the objective there, and its evaluations of the objective or of its derivatives.
  """
  low, high = bounds
  theta = numpy.full(rows.size, low)
  evaluations = numpy.ones(rows.size, dtype=int)
  # not "> 0", so that a NaN derivative goes on to the upper bound as it does for one row
  above_low = ~(objective.derivatives(theta, rows)[0] <= 0.0)
  at_high = numpy.zeros(rows.size, dtype=bool)
  at_high[above_low] = objective.derivatives(numpy.full(int(above_low.sum()), high), rows[above_low])[0] >= 0.0
  theta[at_high], evaluations[above_low] = high, 2
  inner = numpy.flatnonzero(above_low & ~at_high)
  if inner.size:
    start = numpy.full(inner.size, 0.5 * (low + high))
    bracket = numpy.full(inner.size, low), numpy.full(inner.size, high)
    theta[inner], used = _newton_rows(objective.derivatives, rows[inner], start, bracket, tol=tol)
    evaluations[inner] = 2 + used
  return theta, objective(theta, rows), evaluations + 1


def _grid_maxima(
  objective: _BatchObjective, rows: IntArray, nodes: FloatArray, tol: float
) -> tuple[FloatArray, FloatArray, IntArray, BoolArray]:
  """Return what :py:func:`_maximum` returns, with Newton's refinement, for each row alone, from the given points.

  The points are evaluated a chunk of rows at a time (:py:func:`_best_points`), so that memory grows with the points
  of one chunk and not of every row.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
      Each row's ability, the objective there, its evaluations of the objective or of its derivatives, and whether the
      objective was finite at any point, without which the other three hold nothing.
  """
  peak, best, n_ties, start = _best_points(objective, rows, nodes)
  finite = numpy.isfinite(peak)
  theta, value = nodes[best], peak.copy()
  evaluations = numpy.full(rows.size, nodes.size)
  refined = numpy.flatnonzero(finite & (n_ties < _FLAT_MAXIMUM_POINTS))
  if refined.size:
    centre = best[refined]
    bracket = nodes[numpy.maximum(centre - 1, 0)], nodes[numpy.minimum(centre + 1, nodes.size - 1)]
    found, used = _newton_rows(objective.derivatives, rows[refined], start[refined], bracket, tol=tol)
    found_value = objective(found, rows[refined])
    # kept only if no worse than the best point, as _maximum keeps it
    better = found_value >= peak[refined]
    theta[refined[better]], value[refined[better]] = found[better], found_value[better]
    evaluations[refined] += used + 1
  return theta, value, evaluations, finite


def _best_points(
  objective: _BatchObjective, rows: IntArray, nodes: FloatArray
) -> tuple[FloatArray, IntArray, IntArray, FloatArray]:
  """Return each row's peak over the points, its best point, how many points tie at the peak, and where to refine.

  The points are evaluated a chunk of rows at a time. A row whose peak is not finite starts at its best point.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
      Each row's peak, the index of its best point, the number of points at the peak, and the refinement's start.
  """
  peak, start = numpy.empty(rows.size), numpy.empty(rows.size)
  best, n_ties = numpy.empty(rows.size, dtype=int), numpy.empty(rows.size, dtype=int)
  chunk = max(1, _CHUNK_ELEMENTS // nodes.size)
  for first in range(0, rows.size, chunk):
    part = slice(first, first + chunk)
    values = objective.grid(nodes, rows[part])
    peak[part] = values.max(axis=-1)
    best[part], n_ties[part] = _middle_ties(values, peak[part])
    finite = numpy.isfinite(peak[part])
    start[part] = nodes[best[part]]
    start[part][finite] = _vertices(nodes, values[finite], best[part][finite])
  return peak, best, n_ties, start


def _middle_ties(values: FloatArray, peak: FloatArray) -> tuple[IntArray, IntArray]:
  """Return each row's middle point of those at its peak, as :py:func:`_maximum` takes it, and their number."""
  ties = values == peak[:, numpy.newaxis]
  n_ties = ties.sum(axis=-1)
  return (ties & (ties.cumsum(axis=-1) == (n_ties // 2 + 1)[:, numpy.newaxis])).argmax(axis=-1), n_ties


def _vertices(nodes: FloatArray, values: FloatArray, best: IntArray) -> FloatArray:
  """Return what :py:func:`_vertex` returns for each row of values, with its best point."""
  rows = numpy.arange(best.size)
  left = values[rows, numpy.maximum(best - 1, 0)]
  right = values[rows, numpy.minimum(best + 1, nodes.size - 1)]
  curvature = left - 2.0 * values[rows, best] + right
  vertex = (best > 0) & (best < nodes.size - 1) & (curvature < 0.0)
  start = nodes[best].copy()
  shift = 0.5 * (nodes[1] - nodes[0]) * (left[vertex] - right[vertex]) / curvature[vertex]
  start[vertex] = nodes[best[vertex]] + shift
  return start


def _newton_rows(
  derivatives: Callable[[FloatArray, IntArray], tuple[FloatArray, FloatArray]],
  rows: IntArray,
  theta: FloatArray,
  bracket: tuple[FloatArray, FloatArray],
  *,
  tol: float,
) -> tuple[FloatArray, IntArray]:
  """Return what :py:func:`_newton` returns for each row alone, step by step the same arithmetic.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray]
      Each row's zero, within ``tol``, and its evaluations of the derivatives.
  """
  theta, low, high = theta.astype(float), bracket[0].astype(float), bracket[1].astype(float)
  last = high - low
  before_last = last.copy()
  evaluations = numpy.zeros(rows.size, dtype=int)
  active = numpy.arange(rows.size)
  for _ in range(_NEWTON_STEPS):
    if not active.size:
      break
    current = theta[active]
    slope, curvature = derivatives(current, rows[active])
    evaluations[active] += 1
    low[active] = numpy.where(slope > 0.0, current, low[active])
    high[active] = numpy.where(slope > 0.0, high[active], current)
    # a Newton step needs a finite negative curvature; any other bisects
    newton = (curvature > -math.inf) & (curvature < 0.0)
    step = numpy.full(active.size, math.inf)
    step[newton] = -slope[newton] / curvature[newton]
    # a Newton step within the tolerance ends the row's search before the bracket test, as in _newton
    settled = numpy.abs(step) < tol
    theta[active[settled]] = numpy.minimum(
      high[active[settled]], numpy.maximum(low[active[settled]], current[settled] + step[settled])
    )
    active, current, step = active[~settled], current[~settled], step[~settled]
    bisect = ~((low[active] < current + step) & (current + step < high[active])) | (
      numpy.abs(step) > before_last[active] / 2.0
    )
    step[bisect] = 0.5 * (low[active][bisect] + high[active][bisect]) - current[bisect]
    before_last[active], last[active] = last[active], numpy.abs(step)
    theta[active] = current + step
    active = active[numpy.abs(step) >= tol]
  return theta, evaluations
