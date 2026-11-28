"""Tests of the S-X² item-fit statistic and the mean squares, beyond the parity tests with mirt.

Expected values come from enumerating response patterns, hand-worked collapsing and mean squares, S-X²'s rejection
rate on data from the fitted model, and the mean squares' expectation on such data.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from irtlib.cml import CMLCalibrator
from irtlib.drawing import draw_responses
from irtlib.fit import (
  _collapse,  # noqa: PLC2701 - the collapsing rule is checked on hand-worked tables
  _mean_squares,  # noqa: PLC2701 - the mean squares are checked on hand-worked probabilities
  _summed_score_likelihoods,  # noqa: PLC2701 - the recursion is checked against enumeration
  mean_squares,
  s_x2,
)
from irtlib.mml import MMLCalibrator
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.calibration import Calibration
  from irtlib.typing import FloatArray

_LSAT7 = responses(load("mml_em")["data"]["lsat7"])


def test_summed_score_likelihoods_enumerate_the_patterns() -> None:
  """For four items, each score's likelihood is the sum of the probabilities of the patterns with that score.

  Lord and Wingersky's recursion (Orlando & Thissen, 2000, p. 53, Equations 6-11) at three abilities.
  """
  probability = np.array([[0.2, 0.5, 0.9], [0.7, 0.4, 0.3], [0.55, 0.6, 0.65], [0.1, 0.8, 0.95]])
  expected = np.zeros((5, 3))
  for pattern in itertools.product((0, 1), repeat=4):
    answers = np.array(pattern)[:, np.newaxis]
    expected[sum(pattern)] += np.prod(np.where(answers == 1, probability, 1 - probability), axis=0)
  np.testing.assert_allclose(_summed_score_likelihoods(probability), expected, rtol=1e-14)


def test_collapse_merges_empty_single_and_small_rows() -> None:
  """Rows are collapsed as mirt collapses them, on hand-worked tables of incorrect and correct counts.

  An empty row is dropped; a row with one examinee goes into the next, the last such row into the one before; a
  middle row with an expected count below 1 goes into the neighbour with fewer examinees.
  """
  observed = np.array([[0, 0], [1, 0], [3, 2], [5, 5], [0, 1]], dtype=float)
  expected = np.array([[0.1, 0.1], [0.6, 0.4], [2.0, 3.0], [4.0, 6.0], [0.3, 0.7]])
  merged_observed, merged_expected = _collapse(observed, expected, 1.0)
  np.testing.assert_array_equal(merged_observed, [[4, 2], [5, 6]])
  np.testing.assert_allclose(merged_expected, [[2.6, 3.4], [4.3, 6.7]])
  observed = np.array([[2, 2], [3, 1], [2, 3]], dtype=float)
  expected = np.array([[2.0, 2.0], [3.0, 0.5], [2.5, 2.5]])
  merged_observed, merged_expected = _collapse(observed, expected, 1.0)
  np.testing.assert_array_equal(merged_observed, [[5, 3], [2, 3]])
  np.testing.assert_allclose(merged_expected, [[5.0, 2.5], [2.5, 2.5]])
  # a middle row with one examinee goes into the next row, though the previous one has fewer examinees
  observed = np.array([[2, 1], [1, 0], [4, 4]], dtype=float)
  expected = np.array([[1.5, 1.5], [0.7, 0.3], [4.0, 4.0]])
  merged_observed, merged_expected = _collapse(observed, expected, 1.0)
  np.testing.assert_array_equal(merged_observed, [[2, 1], [5, 4]])
  np.testing.assert_allclose(merged_expected, [[1.5, 1.5], [4.7, 4.3]])


def test_type_one_error_rate() -> None:
  """On 2PL data from the calibrated model, S-X² rejects at about the nominal .05 level.

  Orlando and Thissen (2000, p. 57, Table 1) found .04 to .07; over 300 item tests the bounds are about three
  sampling errors around .05. A wrong expected proportion, or degrees of freedom without the item's parameters, moves
  the rate outside them; an error of one degree of freedom does not, and the parity tests catch it.
  """
  rng = np.random.default_rng(20261127)
  slopes = rng.lognormal(0.0, 0.3, 10)
  difficulties = rng.normal(size=10)
  items = np.column_stack([slopes, difficulties, np.zeros(10), np.ones(10)])
  p_values = []
  for _ in range(30):
    matrix = draw_responses(rng.normal(size=1000), items, rng=rng)
    p_values.append(s_x2(MMLCalibrator().calibrate(matrix), matrix).p_value)
  rejection_rate = np.mean(np.concatenate(p_values) < 0.05)
  assert 0.015 <= rejection_rate <= 0.10, rejection_rate


@pytest.mark.parametrize(("statistic", "name"), [(s_x2, "S-X²"), (mean_squares, "the mean squares")])
def test_incomplete_or_mismatched_responses_are_refused(
  statistic: Callable[[Calibration, FloatArray], object], name: str
) -> None:
  """S-X² and the mean squares refuse responses with a missing entry, naming themselves, or another number of items."""
  calibration = MMLCalibrator("1PL").calibrate(_LSAT7)
  with_missing = _LSAT7.copy()
  with_missing[0, 0] = np.nan
  with pytest.raises(ValueError, match=rf"^complete responses are needed for {name}; remove the examinees"):
    statistic(calibration, with_missing)
  with pytest.raises(ValueError, match="4 items, the calibration 5"):
    statistic(calibration, _LSAT7[:, :4])


def test_negative_min_expected_is_refused() -> None:
  """S-X² refuses a negative minimum expected count."""
  with pytest.raises(ValueError, match="min_expected"):
    s_x2(MMLCalibrator().calibrate(_LSAT7), _LSAT7, min_expected=-1.0)


@pytest.mark.parametrize(("statistic", "name"), [(s_x2, "S-X²"), (mean_squares, "the mean squares")])
def test_a_conditional_calibration_is_refused(
  statistic: Callable[[Calibration, FloatArray], object], name: str
) -> None:
  """S-X² and the mean squares refuse a calibration by conditional maximum likelihood, which has no grid of abilities.

  Its items are a 1PL's, so the mean squares' check of the model passes them; without the grid's check, S-X² would
  fail on the missing grid, and the mean squares would estimate abilities on a default grid the calibration never used.
  """
  with pytest.raises(ValueError, match=rf"^for {name}, a calibration must be of the marginal likelihood.*conditional$"):
    statistic(CMLCalibrator().calibrate(_LSAT7), _LSAT7)


def test_an_item_the_bank_leaves_out_is_evaluated() -> None:
  """S-X² evaluates every item at its estimates, the miskeyed one the calibration's bank leaves out included.

  Fails if S-X² reads the bank, which has one item fewer than the responses, rather than the estimates.
  """
  rng = np.random.default_rng(3)
  items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 0.5, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0], [1.2, -0.5, 0.0, 1.0]])
  matrix = draw_responses(rng.normal(size=2000), items, rng=rng)
  with pytest.warns(RuntimeWarning, match="miskeyed"):
    calibration = MMLCalibrator().calibrate(matrix)
  fit = s_x2(calibration, matrix)
  assert fit.statistic.shape == (4,)
  assert fit.df[2] == 1  # three score groups less the item's two parameters: nothing collapsed
  np.testing.assert_allclose(fit.statistic[2], _enumerated_s_x2(calibration, matrix, 2), rtol=1e-10)


def test_the_table_holds_every_score_group_before_collapsing() -> None:
  """S-X²'s table is Orlando and Thissen's N_k, O_ik and E_ik at every number correct from 1 to M - 1, uncollapsed.

  Six items, with the examinees of number correct 1 removed and all but one of number correct 2, so that the
  statistic collapses that group into the next. The table keeps both groups, the empty one with no observed
  proportion, and matches the sizes, proportions correct and Equation 12 (2000, p. 54) worked out by enumerating the
  64 response patterns. Fails if the table is taken after collapsing, the observed proportions count incorrect
  responses, the expected ones leave the item in the likelihood of the other items' score, or the items' rows are
  swapped.
  """
  rng = np.random.default_rng(4)
  items = np.column_stack([np.ones(6), np.linspace(-1.5, 1.5, 6), np.zeros(6), np.ones(6)])
  matrix = draw_responses(rng.normal(size=200), items, rng=rng)
  scores = matrix.sum(axis=1)
  matrix = matrix[(scores > 2) | (scores == 0) | (np.arange(scores.size) == np.flatnonzero(scores == 2)[0])]
  calibration = MMLCalibrator("1PL").calibrate(matrix)
  fit = s_x2(calibration, matrix)
  sizes, observed = _counted_table(matrix)
  np.testing.assert_array_equal(fit.group_sizes, sizes)
  assert fit.group_sizes[:2].tolist() == [0, 1]
  np.testing.assert_allclose(fit.observed, observed, rtol=1e-14)
  np.testing.assert_allclose(fit.expected, _enumerated_expected(calibration), rtol=1e-10)
  # the uncollapsed groups of number correct 2 to 5 leave 8 cells, less 4 groups and 2 parameters: 2 degrees of freedom
  assert (fit.df < 2).all()


def _counted_table(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
  """Return the examinees at each number correct from 1 to M - 1, and each item's proportion correct there.

  Orlando and Thissen's N_k and O_ik (2000, p. 54, Equation 13), counted group by group; NaN in an empty group.
  """
  n_items = matrix.shape[1]
  scores = matrix.sum(axis=1)
  groups = [matrix[scores == k] for k in range(1, n_items)]
  sizes = np.array([group.shape[0] for group in groups], dtype=float)
  observed = np.column_stack([group.mean(axis=0) if group.size else np.full(n_items, np.nan) for group in groups])
  return sizes, observed


def _enumerated_expected(calibration: Calibration) -> FloatArray:
  """Return each item's expected proportion correct at each number correct from 1 to M - 1, by enumeration.

  Orlando and Thissen's E_ik (2000, p. 54, Equation 12), from the likelihood of every response pattern, with the 4PL
  written out and the grid's normal masses.
  """
  a, b, c, d = calibration.estimates.T
  grid = calibration.grid
  assert grid is not None
  masses = grid.weights * np.exp(-(grid.nodes**2) / 2)
  probability = c + (d - c) / (1 + np.exp(-a * (grid.nodes[:, np.newaxis] - b)))  # nodes x items
  score_mass = np.zeros(a.size + 1)
  correct_mass = np.zeros((a.size + 1, a.size))  # by score, the mass of the patterns with each item correct
  for pattern in itertools.product((0, 1), repeat=a.size):
    answers = np.array(pattern)
    mass = np.prod(np.where(answers == 1, probability, 1 - probability), axis=1) @ (masses / masses.sum())
    score_mass[answers.sum()] += mass
    correct_mass[answers.sum()] += answers * mass
  return (correct_mass[1:-1] / score_mass[1:-1, np.newaxis]).T


def _enumerated_s_x2(calibration: Calibration, matrix: FloatArray, item: int) -> float:
  """Return an item's S-X² by enumerating every response pattern, for no collapsed score group.

  Equation 13 of Orlando and Thissen (2000, p. 54), on the counted table and the enumerated expected proportions.
  """
  sizes, observed = _counted_table(matrix)
  expected = _enumerated_expected(calibration)[item]
  return float(np.sum(sizes * (observed[item] - expected) ** 2 / (expected * (1 - expected))))


def test_mean_squares_of_a_hand_worked_case() -> None:
  """The mean squares and their standardized values match Wright and Masters' formulas worked by hand.

  Three examinees answer two items with probabilities .5, .8 and .2, and .8 throughout. A response's variance is PQ and
  its kurtosis PQ(P³ + Q³) (Wright & Masters, 1982, p. 98, Equations 5.4.3-5.4.4); the outfit and infit are the
  unweighted and weighted mean squares of Equations 5.4.6-5.4.7 (p. 99), their variances Table 5.4a (p. 100) and
  Equation 5.4.8 (p. 101), and t is (MS^(1/3) - 1)(3 / s) + s / 3 for the mean square's standard deviation s (p. 101,
  for the weighted one; mirt 1.48 standardizes the unweighted one the same way, R/itemfit.R:585-587). Fails if a mean
  square is weighted wrongly, averaged over the wrong axis or count, or standardized with the wrong variance.
  """
  responses = np.array([[1, 1], [0, 1], [1, 0]], dtype=float)
  p = np.array([[0.5, 0.8], [0.8, 0.8], [0.2, 0.8]])
  fit = _mean_squares(responses, p, 1 - p)
  # squared residuals .25, .64, .64 and .04, .04, .64 over variances .25, .16, .16 and .16 throughout
  outfit = np.array([(1 + 4 + 4) / 3, (0.25 + 0.25 + 4) / 3])
  infit = np.array([1.53 / 0.57, 0.72 / 0.48])
  np.testing.assert_allclose(fit.outfit, outfit, rtol=1e-14)
  np.testing.assert_allclose(fit.infit, infit, rtol=1e-14)
  # kurtoses .0625, .0832, .0832 and .0832 throughout: over the squared variances 1, 3.25, 3.25 and 3.25 throughout,
  # less the squared variances 0, .0576, .0576 and .0576 throughout
  outfit_sd = np.sqrt([7.5 / 9 - 1 / 3, 9.75 / 9 - 1 / 3])
  infit_sd = np.sqrt([0.1152 / 0.57**2, 0.1728 / 0.48**2])
  np.testing.assert_allclose(fit.outfit_t, (np.cbrt(outfit) - 1) * 3 / outfit_sd + outfit_sd / 3, rtol=1e-13)
  np.testing.assert_allclose(fit.infit_t, (np.cbrt(infit) - 1) * 3 / infit_sd + infit_sd / 3, rtol=1e-13)


def test_standardized_mean_squares_are_nan_without_a_positive_variance() -> None:
  """Where the floor on the variances leaves a mean square's variance under the model negative, its t is NaN, silently.

  Five incorrect responses at P = 1e-7: each variance PQ, about 1e-7, is raised to 10^-2.5 (mirt 1.48,
  R/personfit.R:200), so the unweighted mean square's variance is 5 * 1e-2 / 25 - 1 / 5 and the weighted one's
  numerator 5 * (1e-7 - 1e-5), both negative. Each mean square is 5 * (1e-7)^2 / (5 * 10^-2.5). Fails if NumPy's
  warning about the square root of a negative number escapes, or the mean squares are lost with the t values.
  """
  p = np.full((5, 1), 1e-7)
  fit = _mean_squares(np.zeros((5, 1)), p, 1 - p)
  np.testing.assert_allclose(fit.outfit, [1e-14 / 10**-2.5], rtol=1e-6)
  np.testing.assert_allclose(fit.infit, [1e-14 / 10**-2.5], rtol=1e-6)
  assert np.isnan(fit.outfit_t).all()
  assert np.isnan(fit.infit_t).all()


def test_mean_squares_are_near_one_on_rasch_data() -> None:
  """On responses drawn from the Rasch model, every item's mean squares are near their expectation of 1.

  Forty items of slope 1.5 with difficulties from -1.5 to 1.5, and 5,000 examinees. At the generating parameters the
  standard deviations of the mean squares (Wright & Masters, 1982, p. 100, Table 5.4a; p. 101, Equation 5.4.8) are at
  most .074 for the unweighted and .026 for the weighted, and estimating the abilities from the same responses lowers
  both expectations, by about 1 / 40 for Wright and Masters' estimates (p. 101): the bands are that lowering plus
  three standard deviations, .25 and .1. Fails if the probabilities are taken at abilities on another scale, such as
  the estimates times 0.7 or 1.3, or at no estimate.
  """
  rng = np.random.default_rng(20261127)
  items = np.column_stack([np.full(40, 1.5), np.linspace(-1.5, 1.5, 40), np.zeros(40), np.ones(40)])
  matrix = draw_responses(rng.normal(size=5000), items, rng=rng)
  fit = mean_squares(MMLCalibrator("1PL").calibrate(matrix), matrix)
  np.testing.assert_allclose(fit.outfit, 1.0, rtol=0, atol=0.25)
  np.testing.assert_allclose(fit.infit, 1.0, rtol=0, atol=0.1)


@pytest.mark.parametrize(
  ("column", "value"),
  [(0, 1.3), (2, 0.1), (3, 0.95)],
  ids=["own-slope", "guessing", "upper-asymptote"],
)
def test_mean_squares_refuse_a_calibration_other_than_the_1pl(column: int, value: float) -> None:
  """The mean squares refuse a calibration with an item whose slope differs from the others', or c > 0, or d < 1."""
  calibration = MMLCalibrator("1PL").calibrate(_LSAT7)
  estimates = calibration.estimates.copy()
  estimates[2, column] = value
  with pytest.raises(ValueError, match="Rasch model"):
    mean_squares(dataclasses.replace(calibration, estimates=estimates), _LSAT7)
