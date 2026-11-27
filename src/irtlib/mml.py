r"""Marginal maximum likelihood calibration of the 1PL to 4PL with the EM algorithm, with optional priors.

Calibration has to estimate item parameters without the examinees' abilities. Marginal maximum likelihood
integrates the abilities out over a population distribution and maximizes the likelihood of the item parameters
alone; Bock and Aitkin's EM algorithm makes it feasible for any number of items, with one small problem per item in
each cycle :cite:p:`Bock1981` (pp. 444, 447).

Ability is normal with mean 0 and variance 1 (ADR-0008), approximated on nodes :math:`X_q` with masses :math:`A_q`
that sum to 1: the normal density at equally spaced nodes, normalized (:cite:t:`Bock1981`, p. 449). Examinee
:math:`j`'s marginal probability is :math:`\tilde P_j = \sum_q L_j(X_q) A_q`, where :math:`L_j` is the product of
:math:`P_i^{u_{ij}} (1 - P_i)^{1 - u_{ij}}` over the items :math:`i` examinee :math:`j` answered, and the marginal
log-likelihood is :math:`\sum_j \log \tilde P_j` (:cite:t:`Bock1981`, pp. 445-446, Equations 6 and 8).

Each cycle has two steps (:cite:t:`Baker1987`, p. 132, Equations 42-46; :cite:t:`Bock1981`, p. 447):

- the E-step computes, for each item and node, the expected number of examinees answering the item there,
  :math:`\bar N_{iq} = \sum_j L_j(X_q) A_q / \tilde P_j` over the examinees who answered it, and the expected number
  answering it correctly, :math:`\bar r_{iq}`, the same sum weighted by :math:`u_{ij}`;
- the M-step maximizes :math:`\sum_q [\bar r_{iq} \log P_i(X_q) + (\bar N_{iq} - \bar r_{iq}) \log Q_i(X_q)]` in the
  item parameters, the grouped log-likelihood of :py:mod:`irtlib.irt`.

The prior masses enter only through the expected counts. With missing responses the sums run over the examinees
who answered each item, so the expected sample size :math:`\bar N_{iq}` is per item; the sources define it over
complete response patterns, shared by every item.

The M-step solves in mirt's parameters, the slope, the intercept and the logit of :math:`c`, by Fisher scoring with step
halving (ADR-0007), on all free parameters at once: the system is one block per item unless a parameter is shared. The
1PL shares one slope, estimated, with abilities :math:`N(0, 1)` (:cite:t:`Thissen1982`, pp. 175-176, Equation 1); fixing
the slope at 1 for every item gives the model with slopes 1 and abilities :math:`N(0, 1)` (ADR-0008). No cycle decreases
the marginal likelihood :cite:p:`Dempster1977` (p. 7, Theorem 1), and convergence is linear, slow near the solution
(:cite:t:`Dempster1977`, p. 10; :cite:t:`Bock1981`, p. 447). A calibration stops when no estimated parameter changes by
more than its tolerance in a cycle (ADR-0007).

With priors on the item parameters, the estimates are Bayes modal: they maximize the log-likelihood plus the
log-prior, and the EM algorithm carries over with the derivatives of the log-prior added to the M-step
(:cite:t:`Mislevy1986`, p. 182, Equation 14). The priors are mirt's, on mirt's parameters (ADR-0008): lognormal on
the slope, whose log-density keeps the :math:`-\log a` term, normal on the intercept, and normal on the logit of
:math:`c`, which keeps :math:`c` inside :math:`(0, 1)` (:cite:t:`Mislevy1986`, p. 183). In the M-step's scoring
information, a log-prior's negative second derivative counts only where it is positive, so the system stays positive
definite. The 3PL's likelihood is "nearly flat along one or more dimensions" (:cite:t:`Mislevy1986`, p. 189), which a
prior on the logit of :math:`c` remedies. The 4PL adds an upper asymptote :math:`d < 1`, for examinees of high ability
who still miss an item (:cite:t:`Barton1981`, p. 2), estimated on its logit with a normal prior there; recovering it
takes thousands of examinees (:cite:t:`Waller2017`, p. 19; :cite:t:`Culpepper2016`, Section 5).

The standard errors are the square roots of the inverse observed information at the estimates, by Oakes's identity
:cite:p:`Oakes1999` (p. 480): the score of the marginal log-likelihood is the M-step's gradient at the E-step's own
parameters (Equation 5), and its derivative as both move together is the observed-data Hessian, the sum of the
complete-data Hessian and the missing information (Equation 6). That score, log-prior included, is differenced
centrally with step :math:`10^{-5}`, the E-step recomputed at each shifted point, so the posterior precision takes in
the log-prior's second derivative (:cite:t:`Mislevy1986`, p. 182, Equations 15-17); the result is symmetrized, which
:cite:t:`Chalmers2018` (p. 17) finds improves it (ADR-0007). Chalmers (pp. 5-7), as mirt, differences only the missing
information and takes the complete-data Hessian analytically; differencing the whole score keeps one path for every
model and prior. The standard errors of :math:`a`, :math:`b = -\zeta / \lambda`, :math:`c` and :math:`d` follow by
the delta method.

Starting values are the slope 0.851, the intercept :math:`\Phi^{-1}(p_i) \cdot 1.702 / \sqrt{0.75}`, or 2 where
:math:`p_i` is 0 or 1, the guessing parameter 0.15 and the upper asymptote 0.85, where :math:`p_i` is item :math:`i`'s
proportion correct among the examinees who answered it: mirt's starting values, computed over answered responses only
(ADR-0007); :cite:t:`Bock1981` (p. 458) name the inverse normal of the proportion correct as a starting intercept.
"""

import math
import warnings
from dataclasses import dataclass

import numpy
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm

from ._responses import _N_COLUMNS
from .bank import IrtModel, ItemBank
from .calibration import Calibration, ConvergenceWarning, _valid_items, default_grid, validate_responses
from .irt import log_likelihood, log_likelihood_gradient, log_response_probabilities, parameter_information
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
# The M-step's Fisher scoring stops when no parameter moves by more than this, or after the iterations below.
_M_STEP_TOLERANCE = 1e-12
_M_STEP_ITERATIONS = 50
_STEP_HALVINGS = 30
# mirt's central-difference step for Oakes's identity (ADR-0007)
_OAKES_STEP = 1e-5
# abilities are normal with mean 0 and variance 1 (ADR-0008)
_ABILITY_PRIOR = normal_log_prior()
_SLOPE, _INTERCEPT, _GUESSING, _UPPER = 0, 1, 2, 3


@dataclass(frozen=True, slots=True)
class _PriorTerm:
  """A prior on one parameter of every item: normal, or lognormal on the slope (ADR-0008)."""

  column: int
  mean: float
  sd: float

  @property
  def lognormal(self) -> bool:
    """Whether the prior is lognormal, as the slope's is; the others are normal."""
    return self.column == _SLOPE


def _prior_terms(
  values: FloatArray, priors: tuple[_PriorTerm, ...], counted: numpy.ndarray
) -> tuple[float, FloatArray, FloatArray]:
  """Return the log-prior summed over the free parameters, and its first and second derivatives per item parameter."""
  # counted marks the item parameters that carry their prior, one per free parameter: a slope the 1PL shares has one
  # prior, as in mirt, and its copies in the other items carry none
  log_prior = 0.0
  gradient = numpy.zeros_like(values)
  curvature = numpy.zeros_like(values)
  for prior in priors:
    x = values[:, prior.column]
    variance = prior.sd**2
    normalizer = -math.log(prior.sd) - 0.5 * math.log(2 * math.pi)
    if prior.lognormal:
      # a slope that is not positive has no lognormal density; the M-step's step halving moves away from it
      with numpy.errstate(divide="ignore", invalid="ignore"):
        log_x = numpy.where(x > 0, numpy.log(x), numpy.nan)
      deviation = log_x - prior.mean
      log_density = normalizer - log_x - deviation**2 / (2 * variance)
      gradient[:, prior.column] = -(1 + deviation / variance) / x
      curvature[:, prior.column] = (1 - (1 - deviation) / variance) / x**2
    else:
      deviation = x - prior.mean
      log_density = normalizer - deviation**2 / (2 * variance)
      gradient[:, prior.column] = -deviation / variance
      curvature[:, prior.column] = -1 / variance
    log_prior += float(numpy.sum(log_density[counted[:, prior.column]]))
  return log_prior, numpy.where(counted, gradient, 0.0), numpy.where(counted, curvature, 0.0)


def _counted(index: IntArray) -> numpy.ndarray:
  """Return the item parameters that carry their prior: the first item of each free parameter."""
  counted = numpy.zeros(index.shape, dtype=bool)
  free = numpy.flatnonzero(index.ravel() >= 0)
  _, first = numpy.unique(index.ravel()[free], return_index=True)
  counted.ravel()[free[first]] = True
  return counted


@dataclass(frozen=True, slots=True)
class Priors:
  """Priors on the parameters of every item, each given as a mean and a standard deviation (ADR-0008).

  A prior applies to each item's parameter, and once to a parameter the items share, such as the 1PL's slope.

  Attributes
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
  # in log space throughout, so that the masses of nodes far in the tails need not be representable
  log_masses = _log_masses(grid, _ABILITY_PRIOR)
  log_masses -= logsumexp(log_masses)
  return float(numpy.sum(logsumexp(log_likelihood(grid.nodes, matrix, items) + log_masses, axis=1)))


class MMLCalibrator:
  """Calibrate the 1PL to 4PL by marginal maximum likelihood with the EM algorithm, with optional priors.

  Each call to :py:meth:`calibrate` estimates the items of one response matrix, which may have missing responses.
  Abilities are normal with mean 0 and variance 1. With priors on the item parameters, the estimates are Bayes modal:
  they maximize the log-likelihood plus the log-prior.

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
      The stopping rule: calibration stops when no estimated parameter changes by more than this in a cycle.
      Default is 1e-6.
  max_cycles : int, optional
      The most cycles a calibration runs before it stops unconverged. Default is 500.
  grid : irtlib.quadrature.QuadratureGrid, optional
      The nodes the abilities are integrated over. Default is 61 equally spaced nodes on [-6, 6].

  Raises
  ------
  ValueError
      If ``fixed_slope`` is not finite and positive; a slope prior comes with ``fixed_slope``, a guessing prior with
      a model other than the 3PL and 4PL, or an upper prior with a model other than the 4PL; ``tolerance`` is not
      positive; or ``max_cycles`` is less than 1.

  Examples
  --------
  >>> import numpy
  >>> responses = numpy.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float)
  >>> calibration = MMLCalibrator("1PL").calibrate(responses)
  >>> calibration.converged, calibration.n_parameters
  (True, 4)
  """

  def __init__(
    self,
    model: IrtModel | str = IrtModel.PL2,
    *,
    fixed_slope: float | None = None,
    priors: Priors | None = None,
    tolerance: float = 1e-6,
    max_cycles: int = 500,
    grid: QuadratureGrid | None = None,
  ) -> None:
    self._model = IrtModel(model)
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
        The estimates, their standard errors and the log-likelihood at them. If the stopping rule is not met within
        ``max_cycles``, the result says so and a :py:class:`~irtlib.calibration.ConvergenceWarning` is issued; if
        the information matrix cannot be inverted, the result has no standard errors and a ``RuntimeWarning`` is
        issued; if an item's estimates are not valid in an item bank, such as a miskeyed item's slope that is not
        positive, the result's bank leaves the item out and a ``RuntimeWarning`` names it.

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
    information = _observed_information(matrix, values, index, self._grid, self._priors)
    standard_errors = _standard_errors(values, index, information)
    if standard_errors is None:
      warnings.warn(
        "the information matrix could not be inverted, so the calibration reports no standard errors; the model "
        "may not be identified by these responses",
        RuntimeWarning,
        stacklevel=2,
      )
    return Calibration(
      estimates=estimates,
      items=_valid_items(estimates),
      standard_errors=standard_errors,
      log_likelihood=marginal_log_likelihood(matrix, estimates, self._grid),
      n_parameters=int(index.max()) + 1,
      item_parameters=numpy.count_nonzero(index >= 0, axis=1),
      cycles=cycles,
      converged=converged,
      change=change,
      grid=self._grid,
    )

  def _cycle(self, matrix: FloatArray, index: IntArray, values: FloatArray) -> tuple[FloatArray, int, float]:
    """Run EM cycles until the stopping rule is met or the cycles run out.

    Returns
    -------
    tuple[numpy.ndarray, int, float]
        The parameters, the cycles run and the largest change of a free parameter in the last cycle.
    """
    free = index >= 0
    change = math.inf
    cycles = 0
    while cycles < self._max_cycles and change > self._tolerance:
      cycles += 1
      counts = _expected_counts(matrix, values, self._grid)
      new_values = _m_step(values, index, self._grid.nodes, *counts, priors=self._priors)
      change = float(numpy.max(numpy.abs(new_values[free] - values[free])))
      values = new_values
    return values, cycles, change

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
        intercept = norm.ppf(proportion) * _START_INTERCEPT_SCALE
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


def _to_items(values: FloatArray) -> FloatArray:
  """Return items [a, b, c, d] from rows of slope, intercept and logits of c and d."""
  slope, intercept, guessing, upper = values.T
  return numpy.column_stack([slope, -intercept / slope, expit(guessing), expit(upper)])


def _objective(
  values: FloatArray,
  index: IntArray,
  nodes: FloatArray,
  correct: FloatArray,
  total: FloatArray,
  *,
  priors: tuple[_PriorTerm, ...],
) -> float:
  """Return the M-step's objective: the grouped log-likelihood of the expected counts, plus the log-prior."""
  log_p, log_q = log_response_probabilities(nodes, _to_items(values))
  log_prior = _prior_terms(values, priors, _counted(index))[0]
  return float(numpy.sum(correct * log_p + (total - correct) * log_q)) + log_prior


def _expected_counts(matrix: FloatArray, values: FloatArray, grid: QuadratureGrid) -> tuple[FloatArray, FloatArray]:
  """Return the E-step's expected numbers of correct responses and of responses, shape (nodes, items)."""
  answered = ~numpy.isnan(matrix)
  posteriors = posterior(log_likelihood(grid.nodes, matrix, _to_items(values)), grid, _ABILITY_PRIOR)
  return posteriors.T @ numpy.where(answered, matrix, 0.0), posteriors.T @ answered.astype(float)


def _free_gradient(
  values: FloatArray,
  index: IntArray,
  nodes: FloatArray,
  correct: FloatArray,
  total: FloatArray,
  *,
  priors: tuple[_PriorTerm, ...],
) -> FloatArray:
  """Return the gradient of the M-step's objective with respect to the free parameters."""
  free = index >= 0
  prior_gradient = _prior_terms(values, priors, _counted(index))[1]
  gradient = log_likelihood_gradient(nodes, _to_items(values), correct, total) + prior_gradient
  return numpy.bincount(index[free], weights=gradient[free], minlength=int(index.max()) + 1)


def _free_information(
  values: FloatArray, index: IntArray, nodes: FloatArray, total: FloatArray, priors: tuple[_PriorTerm, ...]
) -> FloatArray:
  """Return the scoring information of the free parameters, summing each item's block into the shared ones."""
  # the expected information of the data, plus each log-prior's negative second derivative where it is positive, so
  # the system stays positive definite
  free = index >= 0
  both_free = free[:, :, numpy.newaxis] & free[:, numpy.newaxis, :]
  rows = numpy.broadcast_to(index[:, :, numpy.newaxis], both_free.shape)[both_free]
  columns = numpy.broadcast_to(index[:, numpy.newaxis, :], both_free.shape)[both_free]
  n_free = int(index.max()) + 1
  information = numpy.zeros((n_free, n_free))
  numpy.add.at(information, (rows, columns), parameter_information(nodes, _to_items(values), total)[both_free])
  prior_curvature = numpy.maximum(-_prior_terms(values, priors, _counted(index))[2], 0.0)
  numpy.add.at(information, (index[free], index[free]), prior_curvature[free])
  return information


def _m_step(
  values: FloatArray,
  index: IntArray,
  nodes: FloatArray,
  correct: FloatArray,
  total: FloatArray,
  *,
  priors: tuple[_PriorTerm, ...],
) -> FloatArray:
  """Return the parameters that maximize the M-step's objective, by Fisher scoring with step halving."""
  free = index >= 0
  objective = _objective(values, index, nodes, correct, total, priors=priors)
  for _ in range(_M_STEP_ITERATIONS):
    gradient = _free_gradient(values, index, nodes, correct, total, priors=priors)
    try:
      step = numpy.linalg.solve(_free_information(values, index, nodes, total, priors), gradient)
    except numpy.linalg.LinAlgError as error:
      msg = "the expected information of the item parameters is singular, so the items cannot be estimated"
      raise ValueError(msg) from error
    for _ in range(_STEP_HALVINGS):
      trial = values.copy()
      trial[free] += step[index[free]]
      trial_objective = _objective(trial, index, nodes, correct, total, priors=priors)
      # accept a step that does not lower the objective beyond rounding
      if trial_objective >= objective - 1e-12 * abs(objective):
        values, objective = trial, trial_objective
        break
      step /= 2
    else:
      # no step along the scoring direction raises the objective: the M-step is at its maximum
      break
    if numpy.max(numpy.abs(step)) <= _M_STEP_TOLERANCE:
      break
  return values


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


def _standard_errors(values: FloatArray, index: IntArray, information: FloatArray) -> FloatArray | None:
  """Return the standard errors of each item's a, b, c and d by the delta method.

  Returns
  -------
  numpy.ndarray or None
      The standard errors in the shape of the items, NaN for the parameters not estimated; None if the information
      cannot be inverted or gives a variance that is not positive.
  """
  try:
    covariance = numpy.linalg.inv(information)
  except numpy.linalg.LinAlgError:
    return None
  n_items = values.shape[0]
  rows = numpy.arange(n_items)
  slope, intercept = values[:, 0], values[:, 1]
  free_slope = index[:, 0] >= 0
  # the derivatives of a = slope and b = -intercept / slope with respect to the free parameters
  slope_jacobian = numpy.zeros((n_items, covariance.shape[0]))
  difficulty_jacobian = numpy.zeros_like(slope_jacobian)
  slope_jacobian[rows[free_slope], index[free_slope, 0]] = 1.0
  difficulty_jacobian[rows[free_slope], index[free_slope, 0]] = intercept[free_slope] / slope[free_slope] ** 2
  difficulty_jacobian[rows, index[:, 1]] = -1.0 / slope
  slope_variance = numpy.einsum("nk,kl,nl->n", slope_jacobian, covariance, slope_jacobian)[free_slope]
  difficulty_variance = numpy.einsum("nk,kl,nl->n", difficulty_jacobian, covariance, difficulty_jacobian)
  # c and d are the logistic function of their logits, whose derivative is c (1 - c), or d (1 - d)
  free_asymptotes = [index[:, column] >= 0 for column in (_GUESSING, _UPPER)]
  asymptote_variances = []
  for column, free in zip((_GUESSING, _UPPER), free_asymptotes, strict=True):
    asymptote = expit(values[free, column])
    asymptote_variances.append(
      (asymptote * (1 - asymptote)) ** 2 * covariance[index[free, column], index[free, column]]
    )
  variances = numpy.concatenate([slope_variance, difficulty_variance, *asymptote_variances])
  if not numpy.all(numpy.isfinite(variances) & (variances > 0)):
    return None
  standard_errors = numpy.full((n_items, _N_COLUMNS), numpy.nan)
  standard_errors[free_slope, 0] = numpy.sqrt(slope_variance)
  standard_errors[:, 1] = numpy.sqrt(difficulty_variance)
  for column, free, variance in zip((_GUESSING, _UPPER), free_asymptotes, asymptote_variances, strict=True):
    standard_errors[free, column] = numpy.sqrt(variance)
  return standard_errors
