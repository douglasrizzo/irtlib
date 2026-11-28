"""Tests of joint maximum likelihood calibration of the Rasch model and the 2PL.

Expected values come from the closed forms, published values and brute force ADR-0003 asks of a method with no parity
cases: the two-item closed form of Wright and Douglas (1977) and Andersen (1970), Wright and Douglas's bias and
correction, the joint likelihood written out over every examinee and item and maximized by SciPy, hand-worked
responses, and recovery of generating parameters.
"""

from __future__ import annotations

import functools
import math
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from irtlib.calibration import Calibration, ConvergenceWarning, Likelihood
from irtlib.cml import (
  CMLCalibrator,
  _log_symmetric_functions,  # noqa: PLC2701 - the conditional probabilities that give responses W&D's margins
)
from irtlib.drawing import draw_responses
from irtlib.jml import (
  JMLCalibrator,
  _identified,  # noqa: PLC2701 - the scale's move is checked on hand-worked values
)
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from irtlib.typing import BoolArray, FloatArray

_COMPARISON = load("comparison")
_LSAT6 = responses(_COMPARISON["data"]["lsat6"])
_LSAT7 = responses(_COMPARISON["data"]["lsat7"])


def _two_items(right_wrong: int, wrong_right: int) -> FloatArray:
  """Return two items' responses: the mixed patterns ``10`` and ``01`` so many times, and both right 5, both wrong 3."""
  return np.repeat(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float), [right_wrong, wrong_right, 5, 3], axis=0)


def _two_pl_items(slopes: FloatArray, difficulties: FloatArray) -> FloatArray:
  """Return 2PL items [a, b, 0, 1]."""
  return np.column_stack([slopes, difficulties, np.zeros_like(slopes), np.ones_like(slopes)])


def _kept_rows(matrix: FloatArray) -> BoolArray:
  """Return the mask of the examinees with some item but not every item correct."""
  scores = matrix.sum(axis=1)
  return (scores > 0) & (scores < matrix.shape[1])


def _kept(matrix: FloatArray) -> FloatArray:
  """Return the responses of the examinees with some item but not every item correct."""
  return matrix[_kept_rows(matrix)]


def _joint(parameters: FloatArray, matrix: FloatArray, *, rasch: bool) -> tuple[float, FloatArray]:
  """Return minus the joint log-likelihood, every examinee's responses to every item written out, and its gradient.

  The parameters are every examinee's ability and then, for the Rasch model, the difficulties of every item but the
  last, which is minus their sum; for the 2PL, every slope and then every intercept.
  """
  n_examinees, n_items = matrix.shape
  theta = parameters[:n_examinees]
  if rasch:
    difficulty = np.append(parameters[n_examinees:], -parameters[n_examinees:].sum())
    slope, intercept = np.ones(n_items), -difficulty
  else:
    slope, intercept = parameters[n_examinees : n_examinees + n_items], parameters[n_examinees + n_items :]
  logit = theta[:, np.newaxis] * slope + intercept
  log_likelihood = np.sum(matrix * log_expit(logit) + (1 - matrix) * log_expit(-logit))
  residual = matrix - expit(logit)
  theta_gradient = residual @ slope
  if rasch:
    difficulty_gradient = -residual.sum(axis=0)
    item_gradient = difficulty_gradient[:-1] - difficulty_gradient[-1]
  else:
    item_gradient = np.concatenate([theta @ residual, residual.sum(axis=0)])
  return -float(log_likelihood), -np.concatenate([theta_gradient, item_gradient])


def _brute_force_maximum(matrix: FloatArray, *, rasch: bool) -> tuple[FloatArray, FloatArray, FloatArray, float]:
  """Return the slopes, difficulties and abilities that maximize the written-out joint likelihood, and the maximum.

  SciPy's BFGS maximizes over every examinee's ability; the scale is then fixed as Baker (1987, pp. 112, 115) says
  LOGIST and BICAL fix it: the Rasch difficulties sum to zero, and the 2PL's abilities have mean 0 and standard
  deviation 1, the slopes and intercepts moved with them.
  """
  n_examinees, n_items = matrix.shape
  scores = matrix.sum(axis=1)
  theta = np.log(scores / (n_items - scores))
  if rasch:
    totals = matrix.sum(axis=0)
    difficulty = np.log((n_examinees - totals) / totals)
    start = np.concatenate([theta, (difficulty - difficulty.mean())[:-1]])
  else:
    start = np.concatenate([(theta - theta.mean()) / theta.std(), np.ones(n_items), np.zeros(n_items)])
  objective = functools.partial(_joint, matrix=matrix, rasch=rasch)
  result = minimize(objective, start, jac=True, method="BFGS", options={"gtol": 1e-10})
  theta = result.x[:n_examinees]
  if rasch:
    difficulty = np.append(result.x[n_examinees:], -result.x[n_examinees:].sum())
    return np.ones(n_items), difficulty, theta, -float(result.fun)
  slope, intercept = result.x[n_examinees : n_examinees + n_items], result.x[n_examinees + n_items :]
  location, scale = theta.mean(), theta.std()
  slope, intercept = slope * scale, intercept + slope * location
  return slope, -intercept / slope, (theta - location) / scale, -float(result.fun)


@pytest.mark.parametrize(("right_wrong", "wrong_right"), [(11, 1), (30, 45), (7, 7)])
def test_two_items_closed_form(right_wrong: int, wrong_right: int) -> None:
  """With two Rasch items, the joint b_2 - b_1 is twice the conditional log n_10 - log n_01; corrected, it equals it.

  Hand-worked: once the examinees with both items right or both wrong are left out, every examinee has one correct,
  so all share one ability, 0 by the symmetry of the two difficulties about it; each item's probability is then its
  share of the mixed patterns, p = n_10 / (n_10 + n_01), so b_1 = -log(p / (1 - p)) and b_2 = -b_1: twice the
  conditional difference (Wright & Douglas, 1977, p. 288; Andersen, 1970, p. 299), which the factor (M - 1) / M = 1/2
  brings back to it, the default for the Rasch model. Each standard error is (n p (1 - p))^(-1/2) for the n mixed
  patterns (Wright & Douglas, 1977, p. 285, Equation 16), with or without the correction, and the log-likelihood
  2 (n_10 log p + n_01 log(1 - p)). The abilities are 0 for the mixed patterns, with standard error
  (p (1 - p) + (1 - p) p)^(-1/2) at the uncorrected difficulties, inf for both right and -inf for both wrong; with
  correct_errors, the difficulties' standard errors are halved as the difficulties are. Fails if the examinees left
  out are counted, the correction is not (M - 1) / M or is not the default, the standard error is not Equation 16 or
  follows the correction unasked, the log-likelihood or the abilities' standard errors are taken at the corrected
  estimates, or the abilities are not each examinee's own.
  """
  matrix = _two_items(right_wrong, wrong_right)
  mixed = right_wrong + wrong_right
  proportion = right_wrong / mixed
  difference = math.log(right_wrong) - math.log(wrong_right)
  uncorrected = JMLCalibrator("1PL", correct=False).calibrate(matrix)
  corrected = JMLCalibrator("1PL").calibrate(matrix)
  np.testing.assert_allclose(uncorrected.estimates[:, 1], [-difference, difference], atol=1e-9)
  np.testing.assert_allclose(corrected.estimates[:, 1], [-difference / 2, difference / 2], atol=1e-9)
  expected_error = (mixed * proportion * (1 - proportion)) ** -0.5
  log_likelihood = 2 * (right_wrong * math.log(proportion) + wrong_right * math.log(1 - proportion))
  mixed_rows = np.arange(matrix.shape[0]) < mixed
  for calibration in (uncorrected, corrected):
    assert calibration.converged
    assert calibration.standard_errors is not None
    # the ability search compares likelihoods, so it finds the ability to about 1e-8, which moved the standard error
    # by a relative 3e-9 on CI's Windows runner (measured 2026-11-27)
    np.testing.assert_allclose(calibration.standard_errors[:, 1], expected_error, rtol=1e-6)
    np.testing.assert_allclose(calibration.log_likelihood, log_likelihood, rtol=1e-12)
    abilities = calibration.abilities
    assert abilities is not None
    np.testing.assert_allclose(abilities.theta[mixed_rows], 0.0, atol=1e-7)
    np.testing.assert_array_equal(abilities.theta[~mixed_rows], np.repeat([np.inf, -np.inf], [5, 3]))
    np.testing.assert_allclose(
      abilities.standard_errors[mixed_rows], (2 * proportion * (1 - proportion)) ** -0.5, rtol=1e-6
    )
    np.testing.assert_array_equal(abilities.standard_errors[~mixed_rows], np.inf)
    np.testing.assert_array_equal(abilities.evaluations[~mixed_rows], 0)
    assert np.all(abilities.evaluations[mixed_rows] == abilities.evaluations[0])
    assert abilities.evaluations[0] > 0
  halved = JMLCalibrator("1PL", correct_errors=True).calibrate(matrix)
  assert corrected.standard_errors is not None
  assert halved.standard_errors is not None
  np.testing.assert_array_equal(halved.standard_errors[:, 1], corrected.standard_errors[:, 1] / 2)
  np.testing.assert_array_equal(halved.estimates, corrected.estimates)


def _uniform_scores(difficulty: FloatArray, examinees_per_score: int) -> FloatArray:
  """Return responses with as many examinees at each number correct from 1 to M - 1, and the item totals expected there.

  Each item's total is its probability of a correct response given each number correct, epsilon_i gamma_(r-1)^(i) /
  gamma_r (Wright & Douglas, 1977, p. 285, Equation 20), summed over the numbers correct and rounded, the largest
  remainders rounded up so that the totals add up to the examinees' numbers correct. The matrix is built by Ryser's
  construction: each examinee, highest number correct first, answers correctly the items with the most correct
  responses still to place. Joint and conditional estimates of the Rasch model depend on these margins alone (p. 282).
  """
  n_items = difficulty.size
  scores = np.arange(1, n_items)
  log_gamma = _log_symmetric_functions(-difficulty)
  without = _log_symmetric_functions(np.where(np.eye(n_items, dtype=bool), -np.inf, -difficulty))
  probability = np.exp(-difficulty[:, np.newaxis] + without[:, scores - 1] - log_gamma[np.newaxis, scores])
  expected = examinees_per_score * probability.sum(axis=1)
  totals = np.floor(expected).astype(int)
  shortfall = examinees_per_score * int(scores.sum()) - int(totals.sum())
  totals[np.argsort(totals - expected)[:shortfall]] += 1
  remaining = totals.copy()
  matrix = np.zeros((examinees_per_score * scores.size, n_items))
  for row, score in enumerate(np.repeat(scores[::-1], examinees_per_score)):
    columns = np.argsort(-remaining, kind="stable")[:score]
    matrix[row, columns] = 1.0
    remaining[columns] -= 1
  assert not remaining.any(), "the margins admit no response matrix"
  return matrix


def test_rasch_bias_spreads_the_difficulties_outward_and_the_correction_removes_it() -> None:
  """With uniform scores and difficulties, the joint difficulties exceed the conditional ones by about 1 / (M - 1),
  and corrected they lie within Wright and Douglas's (1977, p. 288, Table 2) largest residual bias, 0.012.

  As Wright and Douglas calculate it, for 20 items with 500 examinees at each number correct and difficulties equally
  spaced on [-1, 1]: every item with |b| >= 0.5 moves away from 0, by a relative amount whose mean lies within their
  Table 1's values for 20 items, 0.0522 to 0.0560 (p. 287, Equations 31-34), against 1 / (M - 1) = 0.0526; multiplied
  by (M - 1) / M, every difficulty is within 0.012 of the conditional one, conditional maximum likelihood giving
  those. Their calculation approximates the bias at each number correct (p. 286, Equation 25). Computed exactly, with
  the likelihood equations solved by a separate Newton-Raphson, the mean relative bias grows with the spread of the
  difficulties, from 0.055 on [-1, 1] to 0.060 on [-2, 2] and 0.071 on [-3, 3], and the largest residual from 0.004 to
  0.029 and 0.090 (measured 2026-11-27), so the test takes the spread where their approximation holds. Fails if the
  estimates are not the joint maximum, the correction is not (M - 1) / M, or it multiplies difficulties not centred.
  """
  difficulty = np.linspace(-1, 1, 20)
  matrix = _uniform_scores(difficulty, 500)
  conditional = CMLCalibrator().calibrate(matrix).estimates[:, 1]
  joint = JMLCalibrator("1PL", correct=False).calibrate(matrix).estimates[:, 1]
  corrected = JMLCalibrator("1PL").calibrate(matrix).estimates[:, 1]
  far = np.abs(conditional) >= 0.5
  np.testing.assert_array_equal(np.sign(joint[far] - conditional[far]), np.sign(conditional[far]))
  relative_bias = np.mean((joint[far] - conditional[far]) / conditional[far])
  assert 0.0522 <= relative_bias <= 0.0560, relative_bias
  np.testing.assert_allclose(corrected, conditional, atol=0.012)


@pytest.mark.parametrize("data", [_LSAT6, _LSAT7], ids=["LSAT-6", "LSAT-7"])
def test_uncorrected_rasch_estimates_maximize_the_joint_likelihood(data: FloatArray) -> None:
  """With the correction off, the Rasch estimates, abilities and log-likelihood are the maximum of the joint
  likelihood written out over every examinee and item, each item's standard error is Wright and Douglas's (1977,
  p. 285) Equation 16 there, and each ability's the inverse square root of the information of its examinee's items.

  SciPy maximizes over every examinee's ability and difficulties that sum to zero, the examinees with no item or every
  item correct left out. The log-likelihood, flat at its maximum, moves by the square of the estimates' error: within
  1e-10 of the maximum, against 1e-8 to 2e-8 below it when the ability step's search stops at 1e-4 (measured
  2026-11-28). Fails if the item step or the ability step stops short of its maximum, the examinees of a number correct
  do not share an ability, the difficulties are not centred, or an examinee gets another examinee's ability.
  """
  matrix = _kept(data)
  _, difficulty, theta, maximum = _brute_force_maximum(matrix, rasch=True)
  calibration = JMLCalibrator("1PL", correct=False).calibrate(data)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulty, atol=1e-5)
  np.testing.assert_allclose(calibration.log_likelihood, maximum, rtol=0, atol=1e-10)
  probability = expit(theta[:, np.newaxis] - difficulty)
  assert calibration.standard_errors is not None
  np.testing.assert_allclose(
    calibration.standard_errors[:, 1], np.sum(probability * (1 - probability), axis=0) ** -0.5, rtol=1e-4
  )
  assert np.all(np.isnan(calibration.standard_errors[:, [0, 2, 3]]))
  kept = _kept_rows(data)
  assert calibration.abilities is not None
  np.testing.assert_allclose(calibration.abilities.theta[kept], theta, atol=1e-5)
  np.testing.assert_allclose(
    calibration.abilities.standard_errors[kept], np.sum(probability * (1 - probability), axis=1) ** -0.5, rtol=1e-4
  )


@functools.cache
def _two_pl_responses() -> FloatArray:
  """Return a 2PL response matrix of 400 examinees and 20 items, on which joint maximum likelihood converges."""
  rng = np.random.default_rng(2)
  slopes = np.linspace(0.8, 1.8, 20)
  rng.shuffle(slopes)
  return draw_responses(rng.normal(size=400), _two_pl_items(slopes, np.linspace(-1.5, 1.5, 20)), rng=rng)


@functools.cache
def _two_pl_calibration() -> Calibration:
  """Return the 2PL calibration of :py:func:`_two_pl_responses`, computed once for the tests that read it."""
  return JMLCalibrator().calibrate(_two_pl_responses())


@functools.cache
def _two_pl_maximum() -> tuple[FloatArray, FloatArray, FloatArray, float]:
  """Return :py:func:`_brute_force_maximum` of the 2PL on :py:func:`_two_pl_responses`, computed once."""
  return _brute_force_maximum(_kept(_two_pl_responses()), rasch=False)


def test_two_pl_estimates_maximize_the_joint_likelihood() -> None:
  """The 2PL estimates, abilities and log-likelihood are the maximum of the joint likelihood written out over every
  examinee and item, on LOGIST's scale, each ability's standard error is the inverse square root of the information of
  its examinee's items there, and its parameters are counted as Baker (1987, p. 112) counts them, 2M + N - 2.

  SciPy maximizes over every examinee's ability, slope and intercept, the examinees with no item or every item correct
  left out, and the abilities are then standardized to mean 0 and standard deviation 1, the items moved with them. The
  log-likelihood is within 1e-9 of the maximum, against 1.2e-7 below it when the ability step's search stops at 1e-4
  (measured 2026-11-28). Fails if the scale is fixed otherwise, the item step or the ability step stops short of its
  maximum, the identification changes the likelihood, or the abilities or their standard errors are not moved to the
  identified scale with the items.
  """
  slope, difficulty, theta, maximum = _two_pl_maximum()
  calibration = _two_pl_calibration()
  assert calibration.converged
  np.testing.assert_allclose(calibration.estimates[:, 0], slope, atol=1e-5)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulty, atol=1e-5)
  np.testing.assert_allclose(calibration.log_likelihood, maximum, rtol=0, atol=1e-9)
  kept = _kept_rows(_two_pl_responses())
  assert calibration.abilities is not None
  np.testing.assert_allclose(calibration.abilities.theta[kept], theta, atol=1e-5)
  weight = expit(slope * (theta[:, np.newaxis] - difficulty))
  information = np.sum(slope**2 * weight * (1 - weight), axis=1)
  np.testing.assert_allclose(calibration.abilities.standard_errors[kept], information**-0.5, rtol=1e-4)
  assert calibration.likelihood is Likelihood.JOINT
  assert calibration.grid is None
  assert calibration.n_parameters == 2 * 20 + _kept(_two_pl_responses()).shape[0] - 2
  np.testing.assert_array_equal(calibration.item_parameters, np.full(20, 2))


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_log_likelihoods_are_those_after_each_cycle(model: str) -> None:
  """The joint log-likelihood recorded for each cycle is the one written out over the responses, at the estimates and
  abilities of a calibration stopped after that cycle; the last is the calibration's own.

  Fails if the record is taken before a cycle's ability step, or at the abilities before the scale is fixed, which
  moves the 2PL's abilities and not its items.
  """
  matrix = _LSAT6 if model == "1PL" else _two_pl_responses()

  def calibrate(max_cycles: int) -> Calibration:
    with pytest.warns(ConvergenceWarning):
      return JMLCalibrator(model, correct=False, max_cycles=max_cycles).calibrate(matrix)

  calibration = calibrate(5)
  expected = []
  for cycles in range(1, 6):
    stopped = calibrate(cycles)
    assert stopped.abilities is not None
    rows = np.isfinite(stopped.abilities.theta)
    slope, difficulty = stopped.estimates[:, 0], stopped.estimates[:, 1]
    logit = slope * (stopped.abilities.theta[rows, np.newaxis] - difficulty)
    expected.append(np.sum(matrix[rows] * log_expit(logit) + (1 - matrix[rows]) * log_expit(-logit)))
  np.testing.assert_allclose(calibration.log_likelihoods, expected, rtol=1e-12)
  assert calibration.log_likelihoods[-1] == calibration.log_likelihood


def test_two_pl_covariances_invert_each_items_information() -> None:
  """Each 2PL item's covariance of a and b is the inverse of its information at the maximum, the abilities known, taken
  to b by the delta method (Baker, 1987, p. 119; p. 114, Equation 8), with standard errors the square roots of its
  diagonal.

  Written out at the brute-force maximum: the information of an item's slope and intercept sums P Q (theta^2, theta;
  theta, 1) over the examinees, and b = -intercept / slope has covariance -(b var(slope) + cov) / slope with the slope
  and variance (b^2 var(slope) + var(intercept) + 2 b cov) / slope^2. Fails if the information takes the abilities
  of the wrong examinees, or the delta method is wrong.
  """
  slope, difficulty, theta, _ = _two_pl_maximum()
  calibration = _two_pl_calibration()
  weight = expit(slope * (theta[:, np.newaxis] - difficulty))
  weight *= 1 - weight
  expected = np.empty((20, 2, 2))
  for item in range(20):
    information = np.array([
      [np.sum(weight[:, item] * theta**2), np.sum(weight[:, item] * theta)],
      [np.sum(weight[:, item] * theta), np.sum(weight[:, item])],
    ])
    (slope_variance, covariance), (_, intercept_variance) = np.linalg.inv(information)
    b = difficulty[item]
    expected[item, 0, 0] = slope_variance
    expected[item, 0, 1] = expected[item, 1, 0] = -(b * slope_variance + covariance) / slope[item]
    expected[item, 1, 1] = (b**2 * slope_variance + intercept_variance + 2 * b * covariance) / slope[item] ** 2
  assert calibration.covariances is not None
  np.testing.assert_allclose(calibration.covariances[:, :2, :2], expected, rtol=1e-4)
  assert np.all(np.isnan(calibration.covariances[:, 2:]))
  assert np.all(np.isnan(calibration.covariances[:, :, 2:]))
  assert calibration.standard_errors is not None
  np.testing.assert_allclose(
    calibration.standard_errors[:, :2], np.sqrt(np.diagonal(expected, axis1=1, axis2=2)), rtol=1e-4
  )


def test_items_answered_by_every_examinee_left_have_no_estimate() -> None:
  """Items answered correctly by every examinee get no estimate, and the leaving out repeats until nothing changes.

  Four items, with 5 examinees answering 1001, 12 answering 1101, 4 answering 1011 and 6 answering 0001. The fourth
  item, right for all, goes first; without it, the 0001 examinees have no item right and go; the first item is then
  right for every examinee left and goes; without it, the 1001 examinees have no item right and go (Wright & Douglas,
  1977, p. 294). The second and third items follow the two-item closed form of the 12 and the 4 examinees, twice the
  conditional one (p. 288). The bank leaves the first and fourth out with a warning naming them, and the 16 examinees
  left add one ability each to the one free difficulty (Baker, 1987, p. 112). The default correction multiplies them by
  (M - 1) / M for the M = 2 items kept, 1/2, where the four columns would give 3/4. A single pass keeps the first item.
  The 1001 and 0001 examinees, left out only once an item has gone, have ability -inf, by their responses to the items
  kept; the 1001 examinees answered the first column right, so reading the responses to every item would give inf.
  """
  patterns = np.array([[1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]], dtype=float)
  matrix = np.repeat(patterns, [5, 12, 4, 6], axis=0)
  with pytest.warns(RuntimeWarning, match=r"NaN or infinite parameters for 0 \('item_0'\), 3 \('item_3'\)") as caught:
    calibration = JMLCalibrator("1PL", correct=False).calibrate(matrix)
  assert caught[0].filename == __file__
  difference = math.log(12) - math.log(4)
  np.testing.assert_allclose(calibration.estimates[:, 1], [np.nan, -difference, difference, np.nan], atol=1e-9)
  np.testing.assert_array_equal(calibration.estimates[:, [0, 2, 3]], np.tile([1.0, 0.0, 1.0], (4, 1)))
  assert calibration.standard_errors is not None
  np.testing.assert_array_equal(np.isnan(calibration.standard_errors[:, 1]), [True, False, False, True])
  assert calibration.items is not None
  assert calibration.items.names == ("item_1", "item_2")
  assert calibration.n_parameters == 1 + 16
  np.testing.assert_array_equal(calibration.item_parameters, [0, 1, 1, 0])
  assert calibration.abilities is not None
  np.testing.assert_allclose(calibration.abilities.theta, np.repeat([-np.inf, 0.0, -np.inf], [5, 16, 6]), atol=1e-7)
  with pytest.warns(RuntimeWarning, match="NaN or infinite parameters"):
    corrected = JMLCalibrator("1PL").calibrate(matrix)
  np.testing.assert_allclose(corrected.estimates[:, 1], [np.nan, -difference / 2, difference / 2, np.nan], atol=1e-9)


@pytest.mark.parametrize("model", ["1PL", "2PL"])
def test_examinees_with_no_item_or_every_item_correct_change_nothing(model: str) -> None:
  """Adding examinees with no item and with every item correct leaves the estimates, standard errors, log-likelihood,
  number of parameters and the other examinees' abilities exactly as they were, and gives the added examinees the
  abilities -inf and inf, with standard error inf.

  They have no finite ability (Wright & Douglas, 1977, p. 294). Exact equality: fails if they enter the starting
  values, the groups of examinees or the scale, or if the abilities are matched to the rows of the examinees kept
  rather than to every row.
  """
  matrix = _LSAT6 if model == "1PL" else _two_pl_responses()
  extremes = np.vstack([np.zeros((40, matrix.shape[1])), matrix, np.ones((300, matrix.shape[1]))])
  before = JMLCalibrator(model).calibrate(matrix) if model == "1PL" else _two_pl_calibration()
  after = JMLCalibrator(model).calibrate(extremes)
  np.testing.assert_array_equal(after.estimates, before.estimates)
  assert before.standard_errors is not None
  assert after.standard_errors is not None
  np.testing.assert_array_equal(after.standard_errors, before.standard_errors)
  assert after.log_likelihood == before.log_likelihood
  assert after.n_parameters == before.n_parameters
  assert before.abilities is not None
  assert after.abilities is not None
  rows = slice(40, 40 + matrix.shape[0])
  for field in ("theta", "standard_errors", "evaluations"):
    np.testing.assert_array_equal(getattr(after.abilities, field)[rows], getattr(before.abilities, field))
  np.testing.assert_array_equal(after.abilities.theta[:40], -np.inf)
  np.testing.assert_array_equal(after.abilities.theta[-300:], np.inf)
  np.testing.assert_array_equal(after.abilities.standard_errors[:40], np.inf)
  np.testing.assert_array_equal(after.abilities.standard_errors[-300:], np.inf)


def test_recovery_of_rasch_difficulties() -> None:
  """Corrected Rasch difficulties from 2,000 examinees recover the generating ones, centred, within 0.2.

  Twenty difficulties equally spaced on [-2, 2] and normal abilities; the largest standard error is 0.07, so 0.2 is
  about three of it, and over 40 seeds the largest error was 0.19 (measured 2026-11-27). Fails if the scale is not
  centred on the difficulties, or the slopes are not fixed at 1.
  """
  difficulty = np.linspace(-2, 2, 20)
  rng = np.random.default_rng(5)
  matrix = draw_responses(rng.normal(size=2000), _two_pl_items(np.ones(20), difficulty), rng=rng)
  calibration = JMLCalibrator("1PL").calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulty, atol=0.2)
  np.testing.assert_array_equal(calibration.estimates[:, 0], np.ones(20))


def test_recovery_of_two_pl_parameters() -> None:
  """2PL slopes and difficulties from 1,000 examinees on 50 items recover the generating ones on LOGIST's scale.

  Slopes equally spaced on [0.8, 2], difficulties on [-2, 2] and normal abilities, which LOGIST's scale of mean 0 and
  standard deviation 1 matches. Joint estimates do not converge to the parameters on a test of fixed length (Baker,
  1987, pp. 117-118): here the slopes come out 5% to 11% above the generating ones on average, and the largest errors
  over 4 seeds were a relative 0.42 for a slope and 0.37 for a difficulty (measured 2026-11-27), so the tolerances are
  a relative 0.45 and 0.4. Fails if the abilities are not standardized, which leaves the slopes scaled by their
  spread, or the slopes are fixed.
  """
  rng = np.random.default_rng(0)
  slopes = np.linspace(0.8, 2.0, 50)
  rng.shuffle(slopes)
  difficulties = np.linspace(-2, 2, 50)
  matrix = draw_responses(rng.normal(size=1000), _two_pl_items(slopes, difficulties), rng=rng)
  calibration = JMLCalibrator().calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 0], slopes, rtol=0.45)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulties, atol=0.4)


def test_a_miskeyed_item_stops_the_cycles_at_the_last_complete_cycle() -> None:
  """An item scored in reverse gets a slope that is not positive in the first item step, which stops the cycles there,
  with the last complete cycle's state, and the warning names the items by their columns in the responses.

  On the 2PL responses with the last item's scoring reversed, the examinees who answer it correctly are those of low
  starting ability, so its slope given those abilities is negative; the ability step takes positive slopes only. A first
  column right for every examinee is left out, so the reversed item is the twentieth item kept and the 21st column. No
  cycle is complete, so the result holds the starting values, every slope positive, and the bank leaves out only the
  first column; its abilities are the starting ones, which no ability step evaluated, with standard errors the inverse
  square root of the information at the starting items. Fails if the warning numbers the items kept, or the result
  pairs the item step's slopes with the abilities of the cycle before.
  """
  matrix = np.column_stack([np.ones(400), _two_pl_responses()])
  matrix[:, -1] = 1 - matrix[:, -1]
  with pytest.warns(RuntimeWarning) as caught:
    calibration = JMLCalibrator().calibrate(matrix)
  messages = [str(warning.message) for warning in caught]
  assert any(
    warning.category is ConvergenceWarning
    and re.search(r"columns \[20\] a slope that is not positive, and the item in column \d+ the largest slope", message)
    for warning, message in zip(caught, messages, strict=True)
  ), messages
  assert any("NaN or infinite parameters for 0 ('item_00')" in message for message in messages), messages
  assert not any("item_20" in message for message in messages), messages
  assert not calibration.converged
  assert calibration.cycles == 0
  assert np.all(calibration.estimates[1:, 0] > 0)
  abilities = calibration.abilities
  assert abilities is not None
  np.testing.assert_array_equal(abilities.evaluations, 0)
  kept = _kept_rows(matrix[:, 1:])
  slope, difficulty = calibration.estimates[1:, 0], calibration.estimates[1:, 1]
  weight = expit(slope * (abilities.theta[kept, np.newaxis] - difficulty))
  information = np.sum(slope**2 * weight * (1 - weight), axis=1)
  np.testing.assert_allclose(abilities.standard_errors[kept], information**-0.5, rtol=1e-12)


# Responses of three items whose first item the starting abilities separate, each row a pattern and its count: right
# for every examinee with two correct and wrong for every one with one; and right for every examinee with two correct
# and for some with one, wrong for the others with one.
_SEPARATED = {
  "strictly": ([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]], [10, 10, 10, 10]),
  "tied at one ability": ([[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [10, 10, 5, 10, 10]),
}


@pytest.mark.parametrize(("patterns", "counts"), _SEPARATED.values(), ids=_SEPARATED)
def test_abilities_that_separate_an_items_responses_stop_the_cycles(
  patterns: list[list[int]], counts: list[int]
) -> None:
  """When the abilities put an item's correct responses at or above its incorrect ones, the cycles stop before its
  slope grows without bound.

  Hand-worked: the starting abilities rise with the number correct, so the first item discriminates perfectly, and its
  slope has no finite estimate (Baker, 1987, p. 117): strictly, or with correct and incorrect responses tied at the
  ability of one correct, where the likelihood still rises as the slope grows with the item's threshold at that
  ability. A column right for every examinee comes first and is left out, so the item is the second column. The
  calibration returns flagged before its first cycle. Fails if the cycles run on, raising the item's slope at each, if a
  tie is taken for no separation, or if the warning numbers the items kept rather than the columns.
  """
  rows = np.repeat(np.array(patterns, dtype=float), counts, axis=0)
  matrix = np.column_stack([np.ones(len(rows)), rows])
  with pytest.warns(RuntimeWarning) as caught:
    calibration = JMLCalibrator().calibrate(matrix)
  messages = [str(warning.message) for warning in caught]
  assert any(
    warning.category is ConvergenceWarning
    and "separate the correct and incorrect responses to the items in columns [1]," in message
    for warning, message in zip(caught, messages, strict=True)
  ), messages
  assert not calibration.converged
  assert calibration.cycles == 0


@pytest.mark.parametrize(
  ("rasch", "values", "expected_values", "expected_theta"),
  [
    (False, [[2.0, 1.0]], [[math.sqrt(3), 4.0]], [-math.sqrt(3), 1 / math.sqrt(3)]),
    (True, [[1.0, 0.5], [1.0, -1.5]], [[1.0, 1.0], [1.0, -1.0]], [-0.5, 1.5]),
  ],
  ids=["2PL", "Rasch"],
)
def test_moving_the_scale_keeps_every_logit(
  rasch: bool, values: list[list[float]], expected_values: list[list[float]], expected_theta: list[float]
) -> None:
  """Moving the scale to its identified position keeps every logit slope * theta + intercept, and so the likelihood.

  Hand-worked, for abilities 0 and 2 of groups of 1 and 3 examinees. The 2PL's abilities have mean 1.5 and standard
  deviation sqrt(0.75) over the examinees, so they move to -sqrt(3) and 1 / sqrt(3), and the item of slope 2 and
  intercept 1 to slope 2 sqrt(0.75) = sqrt(3) and intercept 1 + 2 * 1.5 = 4: its logits stay 1 and 5. The Rasch
  difficulties -0.5 and 1.5 have mean 0.5, which moves them to -1 and 1, their intercepts to 1 and -1, and the abilities
  to -0.5 and 1.5. Fails if the intercepts do not move with the slopes, which changes the likelihood of the starting
  values and of each cycle, or the groups are not counted by their sizes.
  """
  rows = np.array(values)
  items = np.column_stack([rows, np.full(len(rows), -np.inf), np.full(len(rows), np.inf)])
  moved, theta = _identified(items, np.array([0.0, 2.0]), np.array([1.0, 3.0]), rasch=rasch)
  np.testing.assert_allclose(moved[:, :2], expected_values, rtol=1e-12)
  np.testing.assert_allclose(theta, expected_theta, rtol=1e-12)


def test_running_out_of_cycles_is_flagged() -> None:
  """A calibration that runs out of cycles returns its estimates flagged as not converged, and warns at the caller."""
  with pytest.warns(ConvergenceWarning, match="1 cycles") as caught:
    calibration = JMLCalibrator("1PL", max_cycles=1).calibrate(_LSAT6)
  assert caught[0].filename == __file__
  assert not calibration.converged
  assert calibration.cycles == 1
  assert calibration.change > 1e-6


@pytest.mark.parametrize(
  ("model", "matrix", "message"),
  [
    ("1PL", np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 0.0]]), "complete responses; irtlib.mml.MMLCalibrator"),
    ("1PL", np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), "no item has a joint estimate"),
    ("2PL", np.array([[1.0, 2.0], [0.0, 1.0]]), r"responses\[0, 1\] is 2.0"),
    (
      "1PL",
      np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=float),
      r"no difficulty has a finite joint estimate: the examinees with at most 1 correct answered none of the items in "
      r"columns \[2, 3\] correctly, and those with more answered all of the items in columns \[0, 1\]",
    ),
    ("2PL", _two_items(4, 3), "every examinee kept has 1 correct"),
  ],
  ids=["missing", "no item left", "value 2", "no maximum", "2PL scores all equal"],
)
def test_invalid_responses(model: str, matrix: FloatArray, message: str) -> None:
  """Missing responses, responses that leave no item, values other than 0 and 1, Rasch responses with no maximum, and
  2PL responses whose examinees all have the same number correct are refused.

  The missing ones name the calibrator for incomplete responses. In the second, the first item is right for both
  examinees and the third for neither, which leaves each examinee with no item or every item correct. In the fourth,
  worked by hand, the examinees with one correct answered none of the last two items and the others both first ones,
  so the Rasch model's joint likelihood rises without bound as the pairs move apart (Haberman, 2004, p. 11, Theorem
  1). In the last, every examinee kept has one of two items correct, so the 2PL's abilities, standardized, would
  start with no spread.
  """
  with pytest.raises(ValueError, match=message):
    JMLCalibrator(model).calibrate(matrix)


@pytest.mark.parametrize(
  ("arguments", "message"),
  [
    ({"model": "3PL"}, "the 1PL and the 2PL"),
    ({"model": "2PL", "correct": True}, "derived for the Rasch model"),
    ({"model": "1PL", "correct": False, "correct_errors": True}, "the 1PL with correct=False"),
    ({"model": "2PL", "correct_errors": True}, "the 2PL with correct=None"),
    ({"tolerance": 0.0}, "tolerance"),
    ({"max_cycles": 0}, "max_cycles"),
  ],
)
def test_invalid_settings(arguments: dict[str, object], message: str) -> None:
  """A model other than the 1PL and 2PL, the correction for the 2PL, the correction of the standard errors where the
  difficulties are not corrected, a tolerance that is not positive, or fewer than one cycle is refused when the
  calibrator is built.
  """
  with pytest.raises(ValueError, match=message):
    JMLCalibrator(**arguments)  # ty: ignore[invalid-argument-type]
