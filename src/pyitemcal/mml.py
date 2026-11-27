r"""Marginal maximum likelihood calibration of the 1PL and 2PL with the EM algorithm.

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
  item parameters, the grouped log-likelihood of :py:mod:`pyitemcal.irt`.

The prior masses enter only through the expected counts. With missing responses the sums run over the examinees
who answered each item, so the expected sample size :math:`\bar N_{iq}` is per item; the sources define it over
complete response patterns, shared by every item.

The M-step solves in mirt's parameters, the slope and the intercept, by Fisher scoring with step halving (ADR-0007),
on all free parameters at once: the system is one block per item unless a parameter is shared. The 1PL shares one
slope, estimated, with abilities :math:`N(0, 1)` (:cite:t:`Thissen1982`, pp. 175-176, Equation 1); fixing the slope
at 1 for every item gives the model with slopes 1 and abilities :math:`N(0, 1)` (ADR-0008). No cycle decreases the
marginal likelihood :cite:p:`Dempster1977` (p. 7, Theorem 1), and convergence is linear, slow near the solution
(:cite:t:`Dempster1977`, p. 10; :cite:t:`Bock1981`, p. 447). A calibration stops when no estimated parameter changes
by more than its tolerance in a cycle (ADR-0007).

Starting values are the slope 0.851 and the intercept :math:`\Phi^{-1}(p_i) \cdot 1.702 / \sqrt{0.75}`, where
:math:`p_i` is item :math:`i`'s proportion correct among the examinees who answered it, mirt's starting values
computed over answered responses only (ADR-0007); :cite:t:`Bock1981` (p. 458) name the inverse normal of the proportion
correct as a starting intercept.
"""

import math
import warnings

import numpy
from catsim import IrtModel, ItemBank
from catsim.estimation import QuadratureGrid, normal_log_prior
from catsim.irt import log_response_probabilities
from catsim.typing import FloatArray, IntArray
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .calibration import Calibration, ConvergenceWarning, validate_responses
from .irt import log_likelihood_gradient, parameter_information

# mirt's starting slope, 0.5 on the normal-ogive scale times 1.702, and the scale of its starting intercept
_START_SLOPE = 0.5 * 1.702
_START_INTERCEPT_SCALE = 1.702 / math.sqrt(0.75)
_GRID_NODES = 61
# The M-step's Fisher scoring stops when no parameter moves by more than this, or after the iterations below.
_M_STEP_TOLERANCE = 1e-12
_M_STEP_ITERATIONS = 50
_STEP_HALVINGS = 30
_N_PARAMETERS = 4


def _log_masses(grid: QuadratureGrid) -> FloatArray:
  """Return the log of the normal masses of the grid's nodes, normalized to sum to 1."""
  log_masses = numpy.log(grid.weights) + normal_log_prior()(grid.nodes)
  return log_masses - logsumexp(log_masses)


def _log_joint(responses: FloatArray, items: FloatArray, grid: QuadratureGrid) -> FloatArray:
  """Return the log of L_j(X_q) A_q for every examinee and node, shape (examinees, nodes)."""
  answered = ~numpy.isnan(responses)
  correct = numpy.where(answered, responses, 0.0)
  log_p, log_q = log_response_probabilities(grid.nodes, items)
  return correct @ log_p.T + (answered - correct) @ log_q.T + _log_masses(grid)


def _default_grid() -> QuadratureGrid:
  """Return 61 equally spaced nodes on [-6, 6], mirt's grid (ADR-0007)."""
  return QuadratureGrid.uniform(_GRID_NODES)


def marginal_log_likelihood(responses: FloatArray, items: FloatArray, grid: QuadratureGrid | None = None) -> float:
  """Return the marginal log-likelihood of a response matrix, with abilities normal with mean 0 and variance 1.

  Parameters
  ----------
  responses : numpy.ndarray
      The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN for a missing response.
  items : numpy.ndarray
      A matrix of items ``[a, b, c, d]``, one row per column of ``responses``.
  grid : catsim.estimation.QuadratureGrid, optional
      The nodes the abilities are integrated over. Default is 61 equally spaced nodes on [-6, 6].

  Returns
  -------
  float
      The log-likelihood, summed over examinees.

  Raises
  ------
  ValueError
      If the responses fail :py:func:`pyitemcal.calibration.validate_responses`, or ``items`` does not have one row
      per item.

  Examples
  --------
  >>> import numpy
  >>> round(marginal_log_likelihood(numpy.array([[1.0], [0.0]]), numpy.array([[1.0, 0.0, 0.0, 1.0]])), 6)
  -1.386294
  """
  matrix = validate_responses(responses)
  items = numpy.asarray(items, dtype=float)
  if items.shape != (matrix.shape[1], _N_PARAMETERS):
    msg = f"items must have one row [a, b, c, d] per item, shape ({matrix.shape[1]}, 4), got {items.shape}"
    raise ValueError(msg)
  return float(numpy.sum(logsumexp(_log_joint(matrix, items, grid if grid is not None else _default_grid()), axis=1)))


class MMLCalibrator:
  """Calibrate the 1PL or the 2PL by marginal maximum likelihood with the EM algorithm.

  Each call to :py:meth:`calibrate` estimates the items of one response matrix, which may have missing responses.
  Abilities are normal with mean 0 and variance 1.

  Parameters
  ----------
  model : catsim.IrtModel or str, optional
      ``"2PL"``, the default, estimates a slope per item; ``"1PL"`` one slope shared by every item.
  fixed_slope : float, optional
      A slope every item keeps instead of estimating it, with either model.
  tolerance : float, optional
      The stopping rule: calibration stops when no estimated parameter changes by more than this in a cycle.
      Default is 1e-6.
  max_cycles : int, optional
      The most cycles a calibration runs before it stops unconverged. Default is 500.
  grid : catsim.estimation.QuadratureGrid, optional
      The nodes the abilities are integrated over. Default is 61 equally spaced nodes on [-6, 6].

  Raises
  ------
  ValueError
      If ``model`` is not the 1PL or 2PL, ``fixed_slope`` is not finite and positive, ``tolerance`` is not
      positive, or ``max_cycles`` is less than 1.

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
    tolerance: float = 1e-6,
    max_cycles: int = 500,
    grid: QuadratureGrid | None = None,
  ) -> None:
    self._model = IrtModel(model)
    match self._model:
      case IrtModel.PL1 | IrtModel.PL2:
        pass
      case _:
        msg = f"MMLCalibrator calibrates the 1PL and the 2PL, got {self._model}"
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
    self._fixed_slope = fixed_slope
    self._tolerance = tolerance
    self._max_cycles = max_cycles
    self._grid = grid if grid is not None else _default_grid()

  def calibrate(self, responses: FloatArray, start: ItemBank | None = None) -> Calibration:
    """Estimate the items of a response matrix.

    Parameters
    ----------
    responses : numpy.ndarray
        The response matrix: one row per examinee, one column per item, holding 1, 0 or NaN for a missing response.
    start : catsim.ItemBank, optional
        Items to start from, one per column; the 1PL starts from their mean slope. By default, the slope 0.851 and
        an intercept from each item's proportion correct.

    Returns
    -------
    Calibration
        The estimates and the log-likelihood at them. If the stopping rule is not met within ``max_cycles``, the
        result says so and a :py:class:`~pyitemcal.calibration.ConvergenceWarning` is issued.

    Raises
    ------
    ValueError
        If the responses fail :py:func:`pyitemcal.calibration.validate_responses`; an item was answered correctly by
        every examinee who answered it or by none, which leaves it no finite estimate; an item's slope estimate is
        not positive, which a catsim item bank cannot hold; the expected information of the parameters is singular;
        or ``start`` has a different number of items.
    """
    matrix = validate_responses(responses)
    proportion = _proportion_correct(matrix)
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
    items = _to_items(values)
    reversed_items = numpy.flatnonzero(items[:, 0] <= 0)
    if reversed_items.size:
      msg = (
        f"the items in columns {reversed_items.tolist()} have slope estimates that are not positive: the chance of "
        "a correct answer falls as ability rises, as for a miskeyed item; reverse their scoring or remove them"
      )
      raise ValueError(msg)
    return Calibration(
      items=ItemBank(items),
      log_likelihood=float(numpy.sum(logsumexp(_log_joint(matrix, items, self._grid), axis=1))),
      n_parameters=int(index.max()) + 1,
      cycles=cycles,
      converged=converged,
      change=change,
    )

  def _cycle(self, matrix: FloatArray, index: IntArray, values: FloatArray) -> tuple[FloatArray, int, float]:
    """Run EM cycles until the stopping rule is met or the cycles run out.

    Returns
    -------
    tuple[numpy.ndarray, int, float]
        The parameters, the cycles run and the largest change of a free parameter in the last cycle.
    """
    answered = (~numpy.isnan(matrix)).astype(float)
    correct = numpy.where(answered > 0, matrix, 0.0)
    free = index >= 0
    change = math.inf
    cycles = 0
    while cycles < self._max_cycles and change > self._tolerance:
      cycles += 1
      posterior = _posterior(matrix, _to_items(values), self._grid)
      new_values = _m_step(values, index, self._grid.nodes, posterior.T @ correct, posterior.T @ answered)
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
    """
    n_items = proportion.size
    values = numpy.empty((n_items, _N_PARAMETERS))
    if start is None:
      values[:, 0] = _START_SLOPE
      values[:, 1] = norm.ppf(proportion) * _START_INTERCEPT_SCALE
    else:
      a, b = start.discrimination, start.difficulty
      if self._fixed_slope is not None:
        values[:, 0] = self._fixed_slope
      else:
        values[:, 0] = a.mean() if self._model is IrtModel.PL1 else a
      # the start's difficulties are kept whatever slope the items start from
      values[:, 1] = -values[:, 0] * b
    values[:, 2] = -numpy.inf  # c = 0
    values[:, 3] = numpy.inf  # d = 1
    index = numpy.full((n_items, _N_PARAMETERS), -1)
    if self._fixed_slope is not None:
      values[:, 0] = self._fixed_slope
      index[:, 1] = numpy.arange(n_items)
    elif self._model is IrtModel.PL1:
      index[:, 0] = 0
      index[:, 1] = numpy.arange(1, n_items + 1)
    else:
      index[:, 0] = numpy.arange(n_items)
      index[:, 1] = numpy.arange(n_items, 2 * n_items)
    return index, values


def _proportion_correct(matrix: FloatArray) -> FloatArray:
  """Return each item's proportion correct among the examinees who answered it, refusing items with no estimate."""
  answered = ~numpy.isnan(matrix)
  proportion = numpy.where(answered, matrix, 0.0).sum(axis=0) / answered.sum(axis=0)
  extreme = numpy.flatnonzero((proportion == 0) | (proportion == 1))
  if extreme.size:
    msg = (
      f"the items in columns {extreme.tolist()} were answered correctly by every examinee who answered them or by "
      "none, so they have no finite estimate; remove them"
    )
    raise ValueError(msg)
  return proportion


def _to_items(values: FloatArray) -> FloatArray:
  """Return items [a, b, c, d] from rows of slope, intercept and logits of c and d."""
  slope, intercept, guessing, upper = values.T
  return numpy.column_stack([slope, -intercept / slope, expit(guessing), expit(upper)])


def _posterior(responses: FloatArray, items: FloatArray, grid: QuadratureGrid) -> FloatArray:
  """Return each examinee's posterior over the nodes, shape (examinees, nodes): L_j(X_q) A_q / P_j."""
  log_joint = _log_joint(responses, items, grid)
  return numpy.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def _grouped_log_likelihood(values: FloatArray, nodes: FloatArray, correct: FloatArray, total: FloatArray) -> float:
  """Return the sum over items of the M-step's objective."""
  log_p, log_q = log_response_probabilities(nodes, _to_items(values))
  return float(numpy.sum(correct * log_p + (total - correct) * log_q))


def _m_step(
  values: FloatArray, index: IntArray, nodes: FloatArray, correct: FloatArray, total: FloatArray
) -> FloatArray:
  """Return the parameters that maximize the grouped log-likelihood of the expected counts, by Fisher scoring."""
  free = index >= 0
  n_free = int(index.max()) + 1
  both_free = free[:, :, numpy.newaxis] & free[:, numpy.newaxis, :]
  rows = numpy.broadcast_to(index[:, :, numpy.newaxis], both_free.shape)[both_free]
  columns = numpy.broadcast_to(index[:, numpy.newaxis, :], both_free.shape)[both_free]
  objective = _grouped_log_likelihood(values, nodes, correct, total)
  for _ in range(_M_STEP_ITERATIONS):
    items = _to_items(values)
    gradient = numpy.bincount(
      index[free], weights=log_likelihood_gradient(nodes, items, correct, total)[free], minlength=n_free
    )
    information = numpy.zeros((n_free, n_free))
    numpy.add.at(information, (rows, columns), parameter_information(nodes, items, total)[both_free])
    try:
      step = numpy.linalg.solve(information, gradient)
    except numpy.linalg.LinAlgError as error:
      msg = "the expected information of the item parameters is singular, so the items cannot be estimated"
      raise ValueError(msg) from error
    for _ in range(_STEP_HALVINGS):
      trial = values.copy()
      trial[free] += step[index[free]]
      trial_objective = _grouped_log_likelihood(trial, nodes, correct, total)
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
