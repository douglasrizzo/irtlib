"""Item and person fit: Orlando and Thissen's S-X² of each item, and the mean squares of each item or examinee."""

import functools
import math
from dataclasses import dataclass

import numpy
import numpy.typing as npt
from scipy.stats import chi2

from ._responses import responses_and_items
from .ability import expected_a_posteriori
from .calibration import Calibration, _check_marginal, _complete_responses
from .irt import icc, log_response_probabilities
from .quadrature import normal_log_prior, prior_masses
from .typing import FloatArray, IntArray

# mirt 1.48 raises a response's variance whose square is below this to its square root (R/personfit.R:200)
_SQUARED_VARIANCE_FLOOR = 1e-5
# mirt 1.48 caps the standard deviation of a mean square at this before standardizing it (R/itemfit.R:586, 591)
_SD_CAP = 1.4142
# what item fit's error for a missing response says to do
_REMOVE_MISSING = "remove the examinees with missing responses"


@dataclass(frozen=True, slots=True)
class ItemFit:
  """S-X² of each item and the table it is computed from.

  Each item has its statistic, degrees of freedom and p-value. The table groups the examinees by their number correct,
  from 1 to M - 1 for M items, before any score group is collapsed: column ``k - 1`` of ``observed`` and ``expected``
  is the score group of number correct ``k``.

  Parameters
  ----------
  statistic : numpy.ndarray
      S-X² of each item; NaN where no degrees of freedom are left after collapsing.
  df : numpy.ndarray
      The degrees of freedom of each item's statistic.
  p_value : numpy.ndarray
      The probability of a statistic at least as large under the calibrated model; NaN where ``df`` is 0.
  group_sizes : numpy.ndarray
      The number of examinees in each score group, shape ``(M - 1,)``.
  observed : numpy.ndarray
      The proportion of each score group answering each item correctly, shape ``(M, M - 1)``, one row per item; NaN
      in a score group no examinee is in.
  expected : numpy.ndarray
      The proportion the calibrated model expects to answer each item correctly in each score group, shape
      ``(M, M - 1)``, one row per item.
  """

  statistic: FloatArray
  df: IntArray
  p_value: FloatArray
  group_sizes: IntArray
  observed: FloatArray
  expected: FloatArray


def s_x2(calibration: Calibration, responses: FloatArray, *, min_expected: float = 1.0) -> ItemFit:
  """Return the S-X² item-fit statistic of every item of a calibration, from the responses it was calibrated on.

  Every item is evaluated at its estimates, including the items the calibration's bank leaves out.

  The :doc:`/methods/fit/s-x2` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  calibration : Calibration
      The calibration whose items are evaluated, of the marginal likelihood, whose grid the expected counts are
      integrated over.
  responses : numpy.ndarray
      The complete response matrix the calibration estimated its items from: one row per examinee, one column per
      item, 1 or 0.
  min_expected : float, optional
      The smallest expected count a cell may keep after collapsing, at least 0. Default is 1. The expected counts
      are integrated over the calibration's grid.

  Returns
  -------
  ItemFit
      Each item's statistic, degrees of freedom and p-value, and the number of examinees in each score group with
      each item's observed and expected proportions correct there.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses fail :py:func:`irtlib.calibration.validate_responses`, have a missing response, or do
      not have one column per item of the calibration; ``min_expected`` is negative; or the calibration's
      likelihood is not marginal, so that it has no grid.
  """
  if not min_expected >= 0:
    msg = f"min_expected must be at least 0, got {min_expected}"
    raise ValueError(msg)
  matrix = _complete_responses(calibration, responses, "S-X²", _REMOVE_MISSING)
  grid = _check_marginal(calibration, "S-X²")
  masses = prior_masses(grid, normal_log_prior())
  expected = _expected_proportions(icc(grid.nodes, calibration.estimates).T, masses)
  group_sizes, correct = _score_groups(matrix)
  statistics = [
    _item_statistic(group_sizes, correct[item], expected[item], min_expected, int(parameters))
    for item, parameters in enumerate(calibration.item_parameters)
  ]
  values, degrees = zip(*statistics, strict=True)
  statistic, df = numpy.array(values, dtype=float), numpy.array(degrees, dtype=int)
  with numpy.errstate(invalid="ignore"):
    p_value = numpy.where(df > 0, chi2.sf(statistic, numpy.maximum(df, 1)), numpy.nan)
    observed = correct / group_sizes  # NaN in an empty score group
  return ItemFit(
    statistic=statistic, df=df, p_value=p_value, group_sizes=group_sizes, observed=observed, expected=expected
  )


def _score_groups(matrix: FloatArray) -> tuple[IntArray, FloatArray]:
  """Return the examinees at each number correct from 1 to M - 1, and each item's correct responses there."""
  scores = matrix.sum(axis=1).astype(int)
  length = matrix.shape[1] + 1  # numbers correct 0 to M
  correct = numpy.stack([numpy.bincount(scores, weights=column, minlength=length)[1:-1] for column in matrix.T])
  return numpy.bincount(scores, minlength=length)[1:-1], correct


def _item_statistic(
  group_sizes: IntArray, correct: FloatArray, expected: FloatArray, min_expected: float, n_parameters: int
) -> tuple[float, int]:
  """Return an item's S-X² and degrees of freedom after collapsing, NaN and 0 when no degree of freedom is left."""
  counts = numpy.stack([group_sizes - correct, correct], axis=1)
  expected_counts = group_sizes[:, numpy.newaxis] * numpy.stack([1 - expected, expected], axis=1)
  counts, expected_counts = _collapse(counts, expected_counts, min_expected)
  cells = expected_counts > 0
  df = max(int(cells.sum()) - counts.shape[0] - n_parameters, 0)
  if df == 0:
    return math.nan, 0
  return float(numpy.sum((counts[cells] - expected_counts[cells]) ** 2 / expected_counts[cells])), df


def _summed_score_likelihoods(probability: FloatArray) -> FloatArray:
  """Return the likelihood of each number-correct score at each node, shape (items + 1, nodes): Lord and Wingersky."""
  likelihoods = numpy.zeros((probability.shape[0] + 1, probability.shape[1]))
  likelihoods[0] = 1.0
  for item, correct in enumerate(probability):
    shifted = likelihoods[: item + 1] * correct
    likelihoods[: item + 1] *= 1 - correct
    likelihoods[1 : item + 2] += shifted
  return likelihoods


def _expected_proportions(probability: FloatArray, masses: FloatArray) -> FloatArray:
  """Return each item's expected proportion correct at each score from 1 to M - 1, shape (items, M - 1)."""
  n_items = probability.shape[0]
  score_probability = _summed_score_likelihoods(probability) @ masses  # scores 0 to M
  expected = numpy.empty((n_items, n_items - 1))
  for item in range(n_items):
    without = _summed_score_likelihoods(numpy.delete(probability, item, axis=0))  # scores 0 to M - 1
    correct = (without[:-1] * probability[item]) @ masses  # scores 1 to M - 1, the item answered correctly
    expected[item] = correct / score_probability[1:-1]
  return expected


def _collapse(observed: FloatArray, expected: FloatArray, min_expected: float) -> tuple[FloatArray, FloatArray]:
  """Collapse score rows of observed and expected counts, columns incorrect and correct, as mirt does."""
  reached = observed.sum(axis=1) > 0
  observed_rows, expected_rows = list(observed[reached]), list(expected[reached])
  # a row with one examinee goes into the next row, the last row into the one before
  row = 0
  while row < len(observed_rows) - 1:
    if observed_rows[row].sum() == 1:
      # popped first, so the next row moves into this index before it is added to
      moved_observed, moved_expected = observed_rows.pop(row), expected_rows.pop(row)
      observed_rows[row] += moved_observed
      expected_rows[row] += moved_expected
    else:
      row += 1
  if len(observed_rows) > 1 and observed_rows[-1].sum() == 1:
    moved_observed, moved_expected = observed_rows.pop(), expected_rows.pop()
    observed_rows[-1] += moved_observed
    expected_rows[-1] += moved_expected
  while len(observed_rows) > 1:
    small = [bool(numpy.any((cells < min_expected) & (cells != 0))) for cells in expected_rows]
    if not any(small):
      break
    row = small.index(True)
    if row == 0:
      target = 1
    elif row == len(observed_rows) - 1:
      target = row - 1
    else:
      target = row - 1 if observed_rows[row - 1].sum() <= observed_rows[row + 1].sum() else row + 1
    observed_rows[target] += observed_rows[row]
    expected_rows[target] += expected_rows[row]
    del observed_rows[row], expected_rows[row]
  return numpy.array(observed_rows, dtype=float), numpy.array(expected_rows, dtype=float)


@dataclass(frozen=True, slots=True)
class MeanSquares:
  """The outfit and infit mean squares of each item, or of each examinee, each with its standardized value.

  An item's mean squares sum over every examinee, and an examinee's over the items the examinee answered. Both mean
  squares have expectation 1 when the responses fit the model at the abilities they are evaluated at, and less when
  the abilities are estimated from the same responses.

  Parameters
  ----------
  outfit : numpy.ndarray
      The unweighted mean square of each item or examinee, Wright and Masters' :math:`u`: the mean of its squared
      standardized residuals.
  outfit_t : numpy.ndarray
      The outfit standardized by the same cube-root transformation as ``infit_t``, with the outfit's own standard
      deviation under the model; NaN where its variance under the model is not positive.
  infit : numpy.ndarray
      The weighted mean square of each item or examinee, Wright and Masters' :math:`v`: the sum of its squared
      residuals over the sum of their variances.
  infit_t : numpy.ndarray
      The infit standardized by a cube-root transformation, with its standard deviation under the model; NaN where
      its variance under the model is not positive.
  """

  outfit: FloatArray
  outfit_t: FloatArray
  infit: FloatArray
  infit_t: FloatArray


def mean_squares(calibration: Calibration, responses: FloatArray) -> MeanSquares:
  """Return the outfit and infit mean squares of every item of a 1PL calibration, from its responses.

  Each response is compared with its probability at the examinee's EAP ability estimate on the calibration's grid, with
  abilities normal with mean 0 and variance 1.

  The :doc:`/methods/fit/mean-squares` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  calibration : Calibration
      A calibration of the 1PL, whose items all have the same slope, ``c = 0`` and ``d = 1``, of the marginal
      likelihood, on whose grid the abilities are estimated.
  responses : numpy.ndarray
      The complete response matrix the calibration estimated its items from: one row per examinee, one column per
      item, 1 or 0.

  Returns
  -------
  MeanSquares
      Each item's outfit and infit, and their standardized values.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the calibration's items are not those of a 1PL, or its likelihood is not marginal, so that it has no grid;
      or the responses fail :py:func:`irtlib.calibration.validate_responses`, have a missing response, or do not have
      one column per item of the calibration.

  Examples
  --------
  >>> import numpy
  >>> from irtlib.drawing import draw_responses
  >>> from irtlib.mml import MMLCalibrator
  >>> items = numpy.column_stack([numpy.ones(4), [-1.0, -0.3, 0.3, 1.0], numpy.zeros(4), numpy.ones(4)])
  >>> responses = draw_responses(numpy.random.default_rng(1).normal(size=500), items, rng=2)
  >>> fit = mean_squares(MMLCalibrator("1PL").calibrate(responses), responses)
  >>> fit.outfit.round(3), fit.infit.round(3)
  (array([0.751, 0.772, 0.741, 0.723]), array([0.821, 0.807, 0.777, 0.791]))
  """
  items = calibration.estimates
  slope, _, guessing, upper = items.T
  if not (numpy.all(slope == slope[0]) and numpy.all(guessing == 0) and numpy.all(upper == 1)):
    msg = (
      "the mean squares are defined for the Rasch model: the calibration must be a 1PL, every item with the same "
      "slope, c = 0 and d = 1; S-X² (s_x2) evaluates the fit of any model"
    )
    raise ValueError(msg)
  matrix = _complete_responses(calibration, responses, "the mean squares", _REMOVE_MISSING)
  grid = _check_marginal(calibration, "the mean squares")
  theta = expected_a_posteriori(matrix, items, grid=grid).theta
  log_p, log_q = log_response_probabilities(theta, items)
  return _mean_squares(matrix, numpy.exp(log_p), numpy.exp(log_q))


def person_mean_squares(responses: npt.ArrayLike, items: npt.ArrayLike, theta: npt.ArrayLike) -> MeanSquares:
  """Return the outfit and infit mean squares of every examinee, from the responses at the examinees' abilities.

  Each response is compared with its probability at the examinee's ability, under the model of each item, 1PL to
  4PL. A missing response is left out, so that each examinee's mean squares sum over the items the examinee answered.
  An examinee who answered no item, or whose ability is infinite or NaN, gets NaN throughout.

  The :doc:`/methods/fit/mean-squares` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  responses : array_like
      The response matrix: one row per examinee, one column per item, 1 or 0, and NaN for a missing response.
  items : array_like
      The items, a matrix of shape ``(n, 4)``, one row ``[a, b, c, d]`` per column of ``responses``.
  theta : array_like
      Each examinee's ability, one per row of ``responses``, such as the ``theta`` of an
      :py:class:`~irtlib.ability.AbilityEstimates`.

  Returns
  -------
  MeanSquares
      Each examinee's outfit and infit, and their standardized values.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, or ``theta`` holds values other than numbers.
  ValueError
      If ``items`` is not an ``(n, 4)`` matrix, ``responses`` is not a matrix with one column per item, a response is
      not 0, 1 or NaN, or ``theta`` does not hold one ability per row of ``responses``.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.9]])
  >>> responses = numpy.array([[1, 1, 0], [0, 1, 1], [1, numpy.nan, 0]])
  >>> fit = person_mean_squares(responses, items, [0.5, 0.0, -0.5])
  >>> fit.outfit.round(3), fit.infit.round(3)
  (array([0.378, 1.99 , 0.435]), array([0.402, 1.904, 0.465]))
  """
  matrix, items = responses_and_items(responses, items)
  theta = numpy.asarray(theta)
  if theta.dtype.kind not in "iuf":
    msg = f"theta holds {theta.dtype} values; theta must hold numbers, one ability per row of responses"
    raise TypeError(msg)
  theta = theta.astype(float)
  if theta.shape != (matrix.shape[0],):
    msg = f"theta must hold one ability per row of responses, {matrix.shape[0]}, got shape {theta.shape}"
    raise ValueError(msg)
  # an examinee without a finite ability is left with no response; its theta is set to 0 only so that its
  # probabilities are finite, and its NaN responses keep them out of every sum
  finite = numpy.isfinite(theta)
  matrix[~finite] = numpy.nan
  log_p, log_q = log_response_probabilities(numpy.where(finite, theta, 0.0), items)
  return _mean_squares(matrix.T, numpy.exp(log_p).T, numpy.exp(log_q).T)


def _mean_squares(responses: FloatArray, p: FloatArray, q: FloatArray) -> MeanSquares:
  """Return the mean squares of each column of responses, from the probabilities P and Q of each, leaving out NaN."""
  answered = ~numpy.isnan(responses)
  n_responses = answered.sum(axis=0)
  variance = p * q
  # the floor applies to the variance W only, not to the kurtosis C or the squared residual, as mirt's does
  # (R/personfit.R:189-200)
  variance = numpy.where(variance**2 < _SQUARED_VARIANCE_FLOOR, math.sqrt(_SQUARED_VARIANCE_FLOOR), variance)
  kurtosis = p * q * (p**3 + q**3)
  # the residual u - P is Q for a correct response and -P for an incorrect one
  squared_residual = numpy.where(responses == 1, q, p) ** 2
  # each column's sum over its responses
  total = functools.partial(numpy.sum, axis=0, where=answered)
  # a column without responses divides 0 by 0, which leaves its mean squares and their variances NaN: an examinee who
  # answered no item; never an item, whose responses the item mean squares require complete and non-empty
  with numpy.errstate(divide="ignore", invalid="ignore"):
    outfit = total(squared_residual / variance) / n_responses
    outfit_variance = total(kurtosis / variance**2) / n_responses**2 - 1 / n_responses
    total_variance = total(variance)
    infit = total(squared_residual) / total_variance
    infit_variance = total(kurtosis - variance**2) / total_variance**2
  return MeanSquares(
    outfit=outfit,
    outfit_t=_standardized(outfit, outfit_variance),
    infit=infit,
    infit_t=_standardized(infit, infit_variance),
  )


def _standardized(mean_square: FloatArray, variance: FloatArray) -> FloatArray:
  """Return mean squares standardized by the cube-root transformation, NaN where their variance is not positive."""
  # the floor on W can leave the variance negative, which has no square root, as in mirt
  sd = numpy.minimum(numpy.sqrt(numpy.where(variance > 0, variance, numpy.nan)), _SD_CAP)
  return (numpy.cbrt(mean_square) - 1) * (3 / sd) + sd / 3
