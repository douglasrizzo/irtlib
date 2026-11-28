"""Tests of the S-X² item-fit statistic, beyond the parity tests with mirt.

Expected values come from enumerating response patterns, hand-worked collapsing, and the statistic's rejection rate
on data from the fitted model.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from irtlib.cml import CMLCalibrator
from irtlib.drawing import draw_responses
from irtlib.fit import (
  _collapse,  # noqa: PLC2701 - the collapsing rule is checked on hand-worked tables
  _summed_score_likelihoods,  # noqa: PLC2701 - the recursion is checked against enumeration
  s_x2,
)
from irtlib.mml import MMLCalibrator
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
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


def test_incomplete_or_mismatched_responses_are_refused() -> None:
  """S-X² refuses responses with a missing entry or another number of items, and a negative minimum expected count."""
  calibration = MMLCalibrator().calibrate(_LSAT7)
  with_missing = _LSAT7.copy()
  with_missing[0, 0] = np.nan
  with pytest.raises(ValueError, match="complete responses"):
    s_x2(calibration, with_missing)
  with pytest.raises(ValueError, match="4 items, the calibration 5"):
    s_x2(calibration, _LSAT7[:, :4])
  with pytest.raises(ValueError, match="min_expected"):
    s_x2(calibration, _LSAT7, min_expected=-1.0)


def test_a_conditional_calibration_is_refused() -> None:
  """S-X² refuses a calibration by conditional maximum likelihood, which integrated over no grid of abilities."""
  with pytest.raises(ValueError, match=r"S-X² needs a calibration of the marginal likelihood.*conditional"):
    s_x2(CMLCalibrator().calibrate(_LSAT7), _LSAT7)


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


def _enumerated_s_x2(calibration: Calibration, matrix: FloatArray, item: int) -> float:
  """Return an item's S-X² by enumerating every response pattern, for four items and no collapsed score group.

  Equations 12 and 13 of Orlando and Thissen (2000), with the 4PL written out and the grid's normal masses.
  """
  a, b, c, d = calibration.estimates.T
  grid = calibration.grid
  assert grid is not None
  nodes = grid.nodes
  masses = grid.weights * np.exp(-(nodes**2) / 2)
  probability = c + (d - c) / (1 + np.exp(-a * (nodes[:, np.newaxis] - b)))  # nodes x items
  joint = np.zeros((5, 2))  # score 0 to 4, the item incorrect or correct
  for pattern in itertools.product((0, 1), repeat=4):
    likelihood = np.prod(np.where(np.array(pattern) == 1, probability, 1 - probability), axis=1)
    joint[sum(pattern), pattern[item]] += likelihood @ (masses / masses.sum())
  expected = joint[1:4, 1] / joint[1:4].sum(axis=1)
  scores = matrix.sum(axis=1)
  groups = [matrix[scores == k, item] for k in (1, 2, 3)]
  counts = np.array([group.size for group in groups])
  observed = np.array([group.mean() for group in groups])
  return float(np.sum(counts * (observed - expected) ** 2 / (expected * (1 - expected))))
