"""The item step that marginal and joint maximum likelihood share: Fisher scoring of the item parameters.

Given the number of responses to each item at each of a set of abilities, and the number correct among them, it
maximizes the grouped log-likelihood of :py:mod:`irtlib.irt`, plus any log-prior, in mirt's parameters (ADR-0007): the
quadrature nodes and the E-step's expected counts in the M-step of :py:mod:`irtlib.mml`, and the estimated abilities
and the observed counts in the item step of :py:mod:`irtlib.jml`. The method and its sources are documented with
:py:mod:`irtlib.mml`, and the derivatives with :py:mod:`irtlib.irt`.
"""

import math
from dataclasses import dataclass

import numpy
from scipy.special import expit

from ._responses import _N_COLUMNS
from .irt import log_likelihood_gradient, log_response_probabilities, parameter_information
from .typing import FloatArray, IntArray

# The M-step's Fisher scoring stops when no parameter moves by more than this, or after the iterations below.
_M_STEP_TOLERANCE = 1e-12
_M_STEP_ITERATIONS = 50
_STEP_HALVINGS = 30

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
