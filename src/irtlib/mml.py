"""Marginal maximum likelihood calibration of the 1PL to 4PL with the EM algorithm, with optional priors."""

import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import numpy
from scipy.special import logit, logsumexp, ndtri

from ._m_step import (
  _GUESSING,
  _INTERCEPT,
  _SLOPE,
  _UPPER,
  _counted,
  _covariances,
  _free_gradient,
  _m_step,
  _prior_terms,
  _PriorTerm,
  _to_items,
)
from ._responses import _N_COLUMNS
from .bank import IrtModel, ItemBank
from .calibration import Calibration, ConvergenceWarning, Likelihood, _valid_items, default_grid, validate_responses
from .irt import log_likelihood
from .quadrature import QuadratureGrid, _log_masses, normal_log_prior, posterior
from .typing import FloatArray, IntArray

# mirt's starting slope, 0.5 on the normal-ogive scale times 1.702, and the scale of its starting intercept
_START_SLOPE = 0.5 * 1.702
_START_INTERCEPT_SCALE = 1.702 / math.sqrt(0.75)
# mirt's starting guessing parameter and upper asymptote (source R/04-PrepData.R:70-71), and its starting intercept
# where the proportion correct is 0 or 1 (R/05-model.elements.R:63-64)
_START_GUESSING = 0.15
_START_UPPER = 0.85
_START_EXTREME_INTERCEPT = 2.0
# mirt's central-difference step for Oakes's identity (ADR-0007)
_OAKES_STEP = 1e-5
# abilities are normal with mean 0 and variance 1 (ADR-0008)
_ABILITY_PRIOR = normal_log_prior()


@dataclass(frozen=True, slots=True)
class Priors:
  """Priors on the parameters of every item, each given as a mean and a standard deviation (ADR-0008).

  A prior applies to each item's parameter, and once to a parameter the items share, such as the 1PL's slope.

  Parameters
  ----------
  slope : tuple[float, float] or None
      The mean and standard deviation of the log of every slope: a lognormal prior on the slope.
  intercept : tuple[float, float] or None
      The mean and standard deviation of a normal prior on every intercept, ``-a * b``. It keeps the estimates of
      items answered correctly by everyone or by no one finite.
  guessing : tuple[float, float] or None
      The mean and standard deviation of a normal prior on the logit of every guessing parameter; for the 3PL and
      4PL.
  upper : tuple[float, float] or None
      The mean and standard deviation of a normal prior on the logit of every upper asymptote; for the 4PL.

  Raises
  ------
  ValueError
      If a mean is not finite, or a standard deviation is not finite and positive.

  Examples
  --------
  >>> Priors(guessing=(-1.386, 0.5)).guessing
  (-1.386, 0.5)
  """

  slope: tuple[float, float] | None = None
  intercept: tuple[float, float] | None = None
  guessing: tuple[float, float] | None = None
  upper: tuple[float, float] | None = None

  def __post_init__(self) -> None:
    """Check each prior's mean and standard deviation."""
    for name, prior in [
      ("slope", self.slope),
      ("intercept", self.intercept),
      ("guessing", self.guessing),
      ("upper", self.upper),
    ]:
      if prior is None:
        continue
      if len(prior) != 2:  # noqa: PLR2004 - a mean and a standard deviation
        msg = f"the {name} prior must be a mean and a standard deviation, got {prior}"
        raise ValueError(msg)
      mean, sd = prior
      if not (math.isfinite(mean) and math.isfinite(sd) and sd > 0):
        msg = f"the {name} prior must be a finite mean and a finite, positive standard deviation, got {prior}"
        raise ValueError(msg)


class Acceleration(StrEnum):
  """How :py:class:`MMLCalibrator` accelerates its EM cycles.

  ``"squarem"`` extrapolates from every two cycles and keeps an extrapolation only where its log-posterior is above
  the second cycle's, so no cycle lowers the log-posterior; ``"none"`` runs plain EM. Both reach the same maximum by
  the same stopping rule, ``"squarem"`` in fewer cycles. A member equals its value, so either can be passed where an
  acceleration is expected.

  Examples
  --------
  >>> Acceleration("squarem") is Acceleration.SQUAREM
  True
  """

  NONE = "none"
  SQUAREM = "squarem"


def _prior_terms_of(priors: Priors) -> tuple[_PriorTerm, ...]:
  """Return priors as terms of the M-step's objective, one per parameter column."""
  return tuple(
    _PriorTerm(column, float(prior[0]), float(prior[1]))
    for column, prior in [
      (_SLOPE, priors.slope),
      (_INTERCEPT, priors.intercept),
      (_GUESSING, priors.guessing),
      (_UPPER, priors.upper),
    ]
    if prior is not None
  )


def marginal_log_likelihood(responses: FloatArray, items: FloatArray, grid: QuadratureGrid | None = None) -> float:
  """Return the marginal log-likelihood of a response matrix, with abilities normal with mean 0 and variance 1.

  Parameters
  ----------
  responses : numpy.ndarray
      The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN for a missing response.
  items : numpy.ndarray
      A matrix of items ``[a, b, c, d]``, one row per column of ``responses``.
  grid : irtlib.quadrature.QuadratureGrid, optional
      The nodes the abilities are integrated over. Default is 61 equally spaced nodes on [-6, 6].

  Returns
  -------
  float
      The log-likelihood, summed over examinees.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses fail :py:func:`irtlib.calibration.validate_responses`, or ``items`` is not a matrix of four
      columns with one row per item.

  Examples
  --------
  >>> import numpy
  >>> round(marginal_log_likelihood(numpy.array([[1.0], [0.0]]), numpy.array([[1.0, 0.0, 0.0, 1.0]])), 6)
  -1.386294
  """
  matrix = validate_responses(responses)
  grid = grid if grid is not None else default_grid()
  return _marginal(log_likelihood(grid.nodes, matrix, items), grid)


def _marginal(log_likelihoods: FloatArray, grid: QuadratureGrid) -> float:
  """Return the marginal log-likelihood from each examinee's log-likelihood at the grid's nodes."""
  # in log space throughout, so that the masses of nodes far in the tails need not be representable
  log_masses = _log_masses(grid, _ABILITY_PRIOR)
  log_masses -= logsumexp(log_masses)
  return float(numpy.sum(logsumexp(log_likelihoods + log_masses, axis=1)))


class MMLCalibrator:
  """Calibrate the 1PL to 4PL by marginal maximum likelihood with the EM algorithm, with optional priors.

  Each call to :py:meth:`calibrate` estimates the items of one response matrix, which may have missing responses.
  Abilities are normal with mean 0 and variance 1. With priors on the item parameters, the estimates are Bayes modal:
  they maximize the log-likelihood plus the log-prior.

  The :doc:`/methods/calibration/mml` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  model : irtlib.IrtModel or str, optional
      ``"2PL"``, the default, estimates a slope per item; ``"1PL"`` one slope shared by every item; ``"3PL"`` also a
      guessing parameter per item, best with a guessing prior; ``"4PL"`` also an upper asymptote per item, best with
      priors on both asymptotes and thousands of examinees.
  fixed_slope : float, optional
      A slope every item keeps instead of estimating it, with any model.
  priors : Priors, optional
      Priors on the slope, the intercept and the logits of the guessing parameter and upper asymptote of every item.
  tolerance : float, optional
      The stopping rule: calibration stops when no estimated parameter changes by more than this in a cycle, an
      E-step and an M-step. Default is 1e-6.
  max_cycles : int, optional
      The most cycles a calibration runs before it stops unconverged. Default is 500.
  acceleration : Acceleration or str, optional
      ``"squarem"``, the default, extrapolates between cycles to reach the same estimates in fewer of them; the
      extrapolations are not cycles. ``"none"`` runs plain EM.
  grid : irtlib.quadrature.QuadratureGrid, optional
      The nodes the abilities are integrated over. Default is 61 equally spaced nodes on [-6, 6].

  Raises
  ------
  ValueError
      If ``fixed_slope`` is not finite and positive; a slope prior comes with ``fixed_slope``, a guessing prior with
      a model other than the 3PL and 4PL, or an upper prior with a model other than the 4PL; ``tolerance`` is not
      positive; ``max_cycles`` is less than 1; or ``acceleration`` is not an :py:class:`Acceleration`.

  Examples
  --------
  >>> import numpy
  >>> responses = numpy.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float)
  >>> calibration = MMLCalibrator("1PL").calibrate(responses)
  >>> calibration.converged, calibration.n_parameters
  (True, 4)
  """

  def __init__(  # noqa: PLR0913 - a calibrator takes its settings at construction (ADR-0006)
    self,
    model: IrtModel | str = IrtModel.PL2,
    *,
    fixed_slope: float | None = None,
    priors: Priors | None = None,
    tolerance: float = 1e-6,
    max_cycles: int = 500,
    acceleration: Acceleration | str = Acceleration.SQUAREM,
    grid: QuadratureGrid | None = None,
  ) -> None:
    self._model = IrtModel(model)
    self._acceleration = Acceleration(acceleration)
    match self._model:
      case IrtModel.PL1 | IrtModel.PL2 | IrtModel.PL3 | IrtModel.PL4:
        pass
      case _:
        msg = f"MMLCalibrator calibrates the 1PL to the 4PL, got {self._model}"
        raise ValueError(msg)
    if fixed_slope is not None and not (math.isfinite(fixed_slope) and fixed_slope > 0):
      msg = f"fixed_slope must be finite and positive, got {fixed_slope}"
      raise ValueError(msg)
    if not tolerance > 0:
      msg = f"tolerance must be positive, got {tolerance}"
      raise ValueError(msg)
    if max_cycles < 1:
      msg = f"max_cycles must be at least 1, got {max_cycles}"
      raise ValueError(msg)
    priors = priors if priors is not None else Priors()
    if priors.slope is not None and fixed_slope is not None:
      msg = "a slope prior needs an estimated slope, not fixed_slope"
      raise ValueError(msg)
    if priors.guessing is not None and self._model not in {IrtModel.PL3, IrtModel.PL4}:
      msg = f"a guessing prior needs the 3PL or 4PL, got {self._model}"
      raise ValueError(msg)
    if priors.upper is not None and self._model is not IrtModel.PL4:
      msg = f"an upper prior needs the 4PL, got {self._model}"
      raise ValueError(msg)
    self._priors = _prior_terms_of(priors)
    self._fixed_slope = fixed_slope
    self._tolerance = tolerance
    self._max_cycles = max_cycles
    self._grid = grid if grid is not None else default_grid()

  def calibrate(self, responses: FloatArray, start: ItemBank | None = None) -> Calibration:
    """Estimate the items of a response matrix.

    Parameters
    ----------
    responses : numpy.ndarray
        The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN for a missing response.
    start : irtlib.ItemBank, optional
        Items to start from, one per column; the 1PL starts from their mean slope. By default, the slope 0.851 and
        an intercept from each item's proportion correct.

    Returns
    -------
    Calibration
        The estimates, the covariance of each item's estimates and the log-likelihood at them, with the cycles run,
        which do not count the extrapolations of an acceleration, and the largest change of a parameter in the last
        one. If the stopping rule is not met within ``max_cycles``, the result says so and a
        :py:class:`~irtlib.calibration.ConvergenceWarning` is issued; if the information matrix cannot be inverted,
        or gives a variance that is not positive, the result has no covariances or standard errors and a
        ``RuntimeWarning`` is issued; if an item's estimates are not valid in an item bank, such as a miskeyed item's
        slope that is not positive, the result's bank leaves the item out and a ``RuntimeWarning`` names it.

    Raises
    ------
    TypeError
        If the responses are neither Booleans, integers nor floats.
    ValueError
        If the responses fail :py:func:`irtlib.calibration.validate_responses`; an item was answered correctly by
        every examinee who answered it or by none, which leaves it no finite estimate; the expected information of
        the parameters is singular; or ``start`` has a different number of items, or leaves an item's guessing
        parameter at or above its upper asymptote.
    """
    matrix = validate_responses(responses)
    proportion = _proportion_correct(matrix)
    extreme = numpy.flatnonzero((proportion == 0) | (proportion == 1))
    if extreme.size and not any(prior.column == _INTERCEPT for prior in self._priors):
      msg = (
        f"the items in columns {extreme.tolist()} were answered correctly by every examinee who answered them or by "
        "none, so they have no finite estimate; remove them, or give an intercept prior"
      )
      raise ValueError(msg)
    if start is not None and start.n_items != matrix.shape[1]:
      msg = f"start has {start.n_items} items, the responses {matrix.shape[1]}"
      raise ValueError(msg)
    index, values = self._layout(proportion, start)
    values, cycles, change = self._cycle(matrix, index, values)
    converged = change <= self._tolerance
    if not converged:
      warnings.warn(
        f"calibration stopped after {cycles} cycles with a largest change of {change:.3g}, above the tolerance "
        f"{self._tolerance:g}",
        ConvergenceWarning,
        stacklevel=2,
      )
    estimates = _to_items(values)
    covariances = _covariances(values, index, _observed_information(matrix, values, index, self._grid, self._priors))
    if covariances is None:
      warnings.warn(
        "the information matrix could not be inverted, or gave a variance that is not positive, so the calibration "
        "reports no covariances or standard errors; the model may not be identified by these responses",
        RuntimeWarning,
        stacklevel=2,
      )
    return Calibration(
      estimates=estimates,
      items=_valid_items(estimates),
      covariances=covariances,
      log_likelihood=marginal_log_likelihood(matrix, estimates, self._grid),
      likelihood=Likelihood.MARGINAL,
      n_parameters=int(index.max()) + 1,
      item_parameters=numpy.count_nonzero(index >= 0, axis=1),
      cycles=cycles,
      converged=converged,
      change=change,
      grid=self._grid,
      abilities=None,
    )

  def _cycle(self, matrix: FloatArray, index: IntArray, values: FloatArray) -> tuple[FloatArray, int, float]:
    """Run EM cycles until the stopping rule is met or the cycles run out.

    Returns
    -------
    tuple[numpy.ndarray, int, float]
        The parameters, the cycles run and the largest change of a free parameter in the last cycle.
    """
    cycles = _Cycles(matrix, index, self._grid, self._priors, self._tolerance, self._max_cycles)
    match self._acceleration:
      case Acceleration.NONE:
        while not cycles.done:
          values = cycles.run(values)
      case Acceleration.SQUAREM:
        values = _squarem(cycles, values)
      case _:
        assert_never(self._acceleration)
    return values, cycles.count, cycles.change

  def _layout(self, proportion: FloatArray, start: ItemBank | None) -> tuple[IntArray, FloatArray]:
    """Return the index of each item parameter in the free parameters, and each item's starting parameters.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The index of each item's slope, intercept and logits of c and d in the free parameters, -1 for a fixed
        parameter, the 1PL's slopes sharing one; and each item's starting slope, intercept and logits of c and d.

    Raises
    ------
    ValueError
        If the start leaves an item's guessing parameter at or above its upper asymptote.
    """
    n_items = proportion.size
    values = numpy.empty((n_items, _N_COLUMNS))
    if start is None:
      values[:, 0] = _START_SLOPE
      with numpy.errstate(divide="ignore"):
        intercept = ndtri(proportion) * _START_INTERCEPT_SCALE
      values[:, 1] = numpy.where(numpy.isfinite(intercept), intercept, _START_EXTREME_INTERCEPT)
    else:
      a, b = start.discrimination, start.difficulty
      if self._fixed_slope is not None:
        values[:, 0] = self._fixed_slope
      else:
        values[:, 0] = a.mean() if self._model is IrtModel.PL1 else a
      # the start's difficulties are kept whatever slope the items start from
      values[:, 1] = -values[:, 0] * b
    values[:, 2] = -numpy.inf  # c = 0
    if self._model in {IrtModel.PL3, IrtModel.PL4}:
      guessing = start.pseudo_guessing if start is not None else numpy.zeros(n_items)
      values[:, 2] = logit(numpy.where(guessing > 0, guessing, _START_GUESSING))
    values[:, 3] = numpy.inf  # d = 1
    if self._model is IrtModel.PL4:
      upper = start.upper_asymptote if start is not None else numpy.ones(n_items)
      values[:, 3] = logit(numpy.where(upper < 1, upper, _START_UPPER))
    index = numpy.full((n_items, _N_COLUMNS), -1)
    if self._fixed_slope is not None:
      values[:, 0] = self._fixed_slope
      index[:, 1] = numpy.arange(n_items)
    elif self._model is IrtModel.PL1:
      index[:, 0] = 0
      index[:, 1] = numpy.arange(1, n_items + 1)
    else:
      index[:, 0] = numpy.arange(n_items)
      index[:, 1] = numpy.arange(n_items, 2 * n_items)
    if not numpy.all(values[:, 2] < values[:, 3]):
      msg = "start must leave every item's guessing parameter below its upper asymptote, with c = 0 taken as 0.15"
      raise ValueError(msg)
    if self._model in {IrtModel.PL3, IrtModel.PL4}:
      index[:, 2] = numpy.arange(index.max() + 1, index.max() + 1 + n_items)
    if self._model is IrtModel.PL4:
      index[:, 3] = numpy.arange(index.max() + 1, index.max() + 1 + n_items)
    return index, values


def _proportion_correct(matrix: FloatArray) -> FloatArray:
  """Return each item's proportion correct among the examinees who answered it."""
  answered = ~numpy.isnan(matrix)
  return numpy.where(answered, matrix, 0.0).sum(axis=0) / answered.sum(axis=0)


def _expected_counts(matrix: FloatArray, values: FloatArray, grid: QuadratureGrid) -> tuple[FloatArray, FloatArray]:
  """Return the E-step's expected numbers of correct responses and of responses, shape (nodes, items)."""
  return _counts(matrix, log_likelihood(grid.nodes, matrix, _to_items(values)), grid)


def _counts(matrix: FloatArray, log_likelihoods: FloatArray, grid: QuadratureGrid) -> tuple[FloatArray, FloatArray]:
  """Return the expected counts of the E-step from each examinee's log-likelihood at the grid's nodes."""
  answered = ~numpy.isnan(matrix)
  posteriors = posterior(log_likelihoods, grid, _ABILITY_PRIOR)
  return posteriors.T @ numpy.where(answered, matrix, 0.0), posteriors.T @ answered.astype(float)


@dataclass(slots=True)
class _Cycles:
  """The EM cycles of one calibration, counted, with the largest change of a free parameter in the last one."""

  matrix: FloatArray
  index: IntArray
  grid: QuadratureGrid
  priors: tuple[_PriorTerm, ...]
  tolerance: float
  max_cycles: int
  count: int = 0
  change: float = math.inf

  @property
  def done(self) -> bool:
    """Whether the stopping rule is met, the change is NaN, or the cycles have run out."""
    # a NaN change stops the cycles too, and the calibration reports it as not converged
    return not self.change > self.tolerance or self.count >= self.max_cycles

  def run(self, values: FloatArray, counts: tuple[FloatArray, FloatArray] | None = None) -> FloatArray:
    """Return the parameters after a cycle from ``values``, whose E-step's counts are computed unless given."""
    counts = counts if counts is not None else _expected_counts(self.matrix, values, self.grid)
    new_values = _m_step(values, self.index, self.grid.nodes, *counts, priors=self.priors)
    free = self.index >= 0
    self.count += 1
    self.change = float(numpy.max(numpy.abs(new_values[free] - values[free])))
    return new_values

  def expect(self, values: FloatArray) -> tuple[tuple[FloatArray, FloatArray], float]:
    """Return the E-step's counts at ``values`` and the log-posterior there: marginal log-likelihood plus log-prior."""
    log_likelihoods = log_likelihood(self.grid.nodes, self.matrix, _to_items(values))
    log_prior = _prior_terms(values, self.priors, _counted(self.index))[0]
    return _counts(self.matrix, log_likelihoods, self.grid), _marginal(log_likelihoods, self.grid) + log_prior


def _squarem(cycles: _Cycles, values: FloatArray) -> FloatArray:
  """Run SQUAREM's iterations from ``values`` until the stopping rule is met or the cycles run out."""
  counts: tuple[FloatArray, FloatArray] | None = None
  while True:
    first = cycles.run(values, counts)
    if cycles.done:
      return first
    second = cycles.run(first)
    if cycles.done:
      return second
    extrapolated = _extrapolated(values, first, second, cycles.index)
    counts, objective = cycles.expect(second)
    if extrapolated is not None:
      extrapolated_counts, extrapolated_objective = cycles.expect(extrapolated)
      # Du & Varadhan's (2020) test with eta = 0; a log-posterior that is NaN fails it
      if extrapolated_objective > objective:
        # the stabilization step
        values, counts = cycles.run(extrapolated, extrapolated_counts), None
        if cycles.done:
          return values
        continue
    values = second


def _extrapolated(start: FloatArray, first: FloatArray, second: FloatArray, index: IntArray) -> FloatArray | None:
  """Return SQUAREM's extrapolation from the parameters of three successive cycles, or None if it is not kept."""
  point = _squarem_point(*(_free_vector(values, index) for values in (start, first, second)))
  if not numpy.all(numpy.isfinite(point)):
    return None
  free = index >= 0
  extrapolated = start.copy()
  extrapolated[free] = point[index[free]]
  if not numpy.all(extrapolated[:, _GUESSING] < extrapolated[:, _UPPER]):
    return None
  return extrapolated


def _squarem_point(start: FloatArray, first: FloatArray, second: FloatArray) -> FloatArray:
  """Return SQUAREM's extrapolation from three successive iterates of a fixed-point iteration."""
  r = first - start
  v = second - first - r
  # a v of zero makes the steplength infinite and the point not finite, which the caller refuses
  with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
    alpha = -numpy.linalg.norm(r) / numpy.linalg.norm(v)
    return start - 2 * alpha * r + alpha**2 * v


def _free_vector(values: FloatArray, index: IntArray) -> FloatArray:
  """Return the free parameters, a shared one once, from the rows of item parameters."""
  free = index >= 0
  vector = numpy.empty(int(index.max()) + 1)
  vector[index[free]] = values[free]
  return vector


def _observed_information(
  matrix: FloatArray, values: FloatArray, index: IntArray, grid: QuadratureGrid, priors: tuple[_PriorTerm, ...]
) -> FloatArray:
  """Return the observed information of the free parameters by Oakes's identity, log-prior included."""
  # The score is the M-step's gradient at the E-step's own parameters (Oakes, 1999, Equation 5); its central
  # difference, the E-step recomputed at each shifted point, sums Oakes's two terms (Equation 6).
  free = index >= 0
  n_free = int(index.max()) + 1
  hessian = numpy.empty((n_free, n_free))
  for parameter in range(n_free):
    gradients = []
    for sign in (1.0, -1.0):
      shifted = values.copy()
      shifted[free] += sign * _OAKES_STEP * (index[free] == parameter)
      counts = _expected_counts(matrix, shifted, grid)
      gradients.append(_free_gradient(shifted, index, grid.nodes, *counts, priors=priors))
    hessian[:, parameter] = (gradients[0] - gradients[1]) / (2 * _OAKES_STEP)
  return -(hessian + hessian.T) / 2
