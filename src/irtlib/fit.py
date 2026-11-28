"""Item fit: Orlando and Thissen's S-X² statistic for every model, and the mean squares of the Rasch model."""

import math
from dataclasses import dataclass

import numpy
from scipy.stats import chi2

from .ability import expected_a_posteriori
from .calibration import Calibration, _check_marginal, _complete_responses
from .irt import icc, log_response_probabilities, number_correct_probabilities
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
  """The S-X² statistic of each item, with its degrees of freedom and p-value.

  Parameters
  ----------
  statistic : numpy.ndarray
      S-X² of each item; NaN where no degrees of freedom are left after collapsing.
  df : numpy.ndarray
      The degrees of freedom of each item's statistic.
  p_value : numpy.ndarray
      The probability of a statistic at least as large under the calibrated model; NaN where ``df`` is 0.
  """

  statistic: FloatArray
  df: IntArray
  p_value: FloatArray


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
      Each item's statistic, degrees of freedom and p-value.

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
  items = calibration.estimates
  grid = _check_marginal(calibration, "S-X²")
  masses = prior_masses(grid, normal_log_prior())
  expected = _expected_proportions(grid.nodes, items, masses)
  scores = matrix.sum(axis=1).astype(int)
  n_items = items.shape[0]
  statistic = numpy.full(n_items, numpy.nan)
  df = numpy.zeros(n_items, dtype=int)
  for item in range(n_items):
    observed = numpy.stack(
      [numpy.bincount(scores, weights=(matrix[:, item] == value), minlength=n_items + 1)[1:-1] for value in (0, 1)],
      axis=1,
    )
    expected_counts = observed.sum(axis=1, keepdims=True) * numpy.stack([1 - expected[item], expected[item]], axis=1)
    observed, expected_counts = _collapse(observed, expected_counts, min_expected)
    cells = expected_counts > 0
    df[item] = max(int(cells.sum()) - observed.shape[0] - int(calibration.item_parameters[item]), 0)
    if df[item] > 0:
      statistic[item] = numpy.sum((observed[cells] - expected_counts[cells]) ** 2 / expected_counts[cells])
  with numpy.errstate(invalid="ignore"):
    p_value = numpy.where(df > 0, chi2.sf(statistic, numpy.maximum(df, 1)), numpy.nan)
  return ItemFit(statistic=statistic, df=df, p_value=p_value)


def _expected_proportions(nodes: FloatArray, items: FloatArray, masses: FloatArray) -> FloatArray:
  """Return each item's expected proportion correct at each score from 1 to M - 1, shape (items, M - 1)."""
  n_items = items.shape[0]
  score_probability = masses @ number_correct_probabilities(nodes, items)  # scores 0 to M
  probability = icc(nodes, items)  # nodes x items
  expected = numpy.empty((n_items, n_items - 1))
  for item in range(n_items):
    without = number_correct_probabilities(nodes, numpy.delete(items, item, axis=0))  # nodes x scores 0 to M - 1
    correct = (masses * probability[:, item]) @ without[:, :-1]  # scores 1 to M - 1, the item answered correctly
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
  """The outfit and infit mean squares of each item, each with its standardized value.

  Both mean squares have expectation 1 when the responses fit the model, and less when the abilities are estimated
  from the same responses, as they are here.

  Parameters
  ----------
  outfit : numpy.ndarray
      The unweighted mean square of each item, Wright and Masters' :math:`u_i`: the mean, over examinees, of the
      squared standardized residuals.
  outfit_t : numpy.ndarray
      The outfit standardized by the same cube-root transformation as ``infit_t``, with the outfit's own standard
      deviation under the model; NaN where its variance under the model is not positive.
  infit : numpy.ndarray
      The weighted mean square of each item, Wright and Masters' :math:`v_i`: the squared residuals summed over
      examinees, over the sum of their variances.
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


def _mean_squares(responses: FloatArray, p: FloatArray, q: FloatArray) -> MeanSquares:
  """Return the mean squares of complete responses, examinees by items, from the probabilities P and Q of each."""
  n_examinees = responses.shape[0]
  variance = p * q
  # the floor applies to the variance W only, not to the kurtosis C or the squared residual, as mirt's does
  # (R/personfit.R:189-200)
  variance = numpy.where(variance**2 < _SQUARED_VARIANCE_FLOOR, math.sqrt(_SQUARED_VARIANCE_FLOOR), variance)
  kurtosis = p * q * (p**3 + q**3)
  # the residual u - P is Q for a correct response and -P for an incorrect one
  squared_residual = numpy.where(responses == 1, q, p) ** 2
  outfit = numpy.sum(squared_residual / variance, axis=0) / n_examinees
  outfit_variance = numpy.sum(kurtosis / variance**2, axis=0) / n_examinees**2 - 1 / n_examinees
  total_variance = numpy.sum(variance, axis=0)
  infit = numpy.sum(squared_residual, axis=0) / total_variance
  infit_variance = numpy.sum(kurtosis - variance**2, axis=0) / total_variance**2
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
