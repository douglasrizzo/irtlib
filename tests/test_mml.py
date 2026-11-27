"""Tests of marginal maximum likelihood calibration with the EM algorithm, beyond the parity tests with mirt.

Expected values come from Thissen's (1982) published estimates, a brute-force sum, recovery of generating parameters,
and the algorithm's guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest
from catsim import ItemBank
from catsim.irt import icc
from scipy.special import expit

from pyitemcal.calibration import ConvergenceWarning
from pyitemcal.mml import MMLCalibrator, marginal_log_likelihood
from tests.mirt_fixtures import load, responses

_MML_EM = load("mml_em")
_LSAT6 = responses(_MML_EM["data"]["lsat6"])
_LSAT7 = responses(_MML_EM["data"]["lsat7"])


def test_thissen_published_estimates() -> None:
  """The 1PL with a common slope reproduces Thissen's (1982, p. 180, Table 2) LSAT-6 estimates to the decimals printed.

  LSAT-6 is Thissen's (1982, p. 179, Table 1) data; a wrong scale identification or prior moves the slope.
  """
  calibration = MMLCalibrator("1PL", tolerance=1e-8).calibrate(_LSAT6)
  np.testing.assert_allclose(calibration.items.discrimination, 0.755, atol=5e-4)
  np.testing.assert_allclose(calibration.items.difficulty, [-3.615, -1.322, -0.318, -1.730, -2.780], atol=5e-4)


def test_recovery_of_two_pl_parameters() -> None:
  """2PL estimates from 20,000 simulated examinees with 10% of responses missing recover the generating items.

  The tolerances, 15% of each slope and 0.15 for each difficulty, are over twice the largest error measured across
  seeds, and far below what a scale factor of 1.7 in the logit, a prior applied twice, or a sign error would give.
  """
  rng = np.random.default_rng(20261127)
  slopes = rng.uniform(0.7, 2.0, 10)
  difficulties = rng.uniform(-1.5, 1.5, 10)
  theta = rng.normal(size=20_000)
  matrix = (rng.random((20_000, 10)) < expit(slopes * (theta[:, np.newaxis] - difficulties))).astype(float)
  matrix[rng.random(matrix.shape) < 0.1] = np.nan
  calibration = MMLCalibrator().calibrate(matrix)
  np.testing.assert_allclose(calibration.items.discrimination, slopes, rtol=0.15)
  np.testing.assert_allclose(calibration.items.difficulty, difficulties, atol=0.15)


def test_likelihood_never_decreases() -> None:
  """The log-likelihood after each of the first cycles never decreases (Dempster, Laird & Rubin, 1977, p. 7).

  Fails if an M-step does not maximize the expected complete-data log-likelihood, or the E-step's counts are wrong.
  """
  log_likelihoods = []
  for cycles in range(1, 9):
    with pytest.warns(ConvergenceWarning):
      calibration = MMLCalibrator(tolerance=1e-14, max_cycles=cycles).calibrate(_LSAT7)
    log_likelihoods.append(calibration.log_likelihood)
  assert np.all(np.diff(log_likelihoods) >= -1e-9)
  assert log_likelihoods[-1] > log_likelihoods[0]


def test_marginal_log_likelihood_with_a_missing_response() -> None:
  """The marginal log-likelihood equals a brute-force sum that leaves a missing response out of its examinee's product.

  Four examinees, three items, one response missing; normal masses at 61 equally spaced nodes on [-6, 6].
  """
  matrix = np.array([[1.0, 0.0, 1.0], [0.0, np.nan, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
  items = np.array([[1.2, -0.4, 0.0, 1.0], [0.7, 0.9, 0.0, 1.0], [1.6, 0.1, 0.0, 1.0]])
  nodes = np.linspace(-6.0, 6.0, 61)
  masses = np.exp(-(nodes**2) / 2)
  masses /= masses.sum()
  probability = icc(nodes, items)  # nodes x items
  expected = 0.0
  for row in matrix:
    likelihood = np.ones_like(nodes)
    for item, response in enumerate(row):
      if not np.isnan(response):
        likelihood *= probability[:, item] if response == 1 else 1 - probability[:, item]
    expected += np.log(np.sum(masses * likelihood))
  np.testing.assert_allclose(marginal_log_likelihood(matrix, items), expected, rtol=1e-12)


def test_an_examinee_who_answered_nothing_changes_nothing() -> None:
  """Adding an examinee with every response missing leaves the estimates and the log-likelihood as they were."""
  with_blank = np.vstack([_LSAT7, np.full(5, np.nan)])
  calibrator = MMLCalibrator(tolerance=1e-8)
  before, after = calibrator.calibrate(_LSAT7), calibrator.calibrate(with_blank)
  np.testing.assert_allclose(after.items.items, before.items.items, atol=1e-12)
  np.testing.assert_allclose(after.log_likelihood, before.log_likelihood, rtol=1e-13)


@pytest.mark.parametrize(
  ("calibrator", "n_parameters"),
  [(MMLCalibrator("2PL"), 10), (MMLCalibrator("1PL"), 6), (MMLCalibrator("1PL", fixed_slope=1.0), 5)],
  ids=["2PL", "1PL", "fixed slopes"],
)
def test_parameter_count(calibrator: MMLCalibrator, n_parameters: int) -> None:
  """The result counts a slope per item for the 2PL, one shared slope for the 1PL, and none when slopes are fixed."""
  assert calibrator.calibrate(_LSAT7).n_parameters == n_parameters


def test_fixed_slope_is_kept() -> None:
  """Every item keeps the fixed slope exactly; only the difficulties move."""
  calibration = MMLCalibrator("2PL", fixed_slope=1.3).calibrate(_LSAT7)
  assert np.all(calibration.items.discrimination == 1.3)


def test_running_out_of_cycles_is_flagged() -> None:
  """A calibration that runs out of cycles returns its estimates flagged as not converged, and warns."""
  with pytest.warns(ConvergenceWarning, match="2 cycles"):
    calibration = MMLCalibrator(max_cycles=2).calibrate(_LSAT7)
  assert not calibration.converged
  assert calibration.cycles == 2
  assert calibration.change > 1e-6


@pytest.mark.parametrize(
  "settings", [{"model": "2PL"}, {"model": "1PL"}, {"model": "1PL", "fixed_slope": 1.0}], ids=["2PL", "1PL", "fixed"]
)
def test_start_does_not_change_the_maximum(settings: dict[str, object]) -> None:
  """Starting from other items, whose slopes differ, reaches the same estimates as the default start."""
  start = ItemBank(
    np.array([[2.0, 1.0, 0.0, 1.0], [0.5, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]] + [[1.0, -2.0, 0.0, 1.0]] * 2)
  )
  calibrator = MMLCalibrator(**settings, tolerance=1e-10)  # ty: ignore[invalid-argument-type]
  default, started = calibrator.calibrate(_LSAT7), calibrator.calibrate(_LSAT7, start=start)
  np.testing.assert_allclose(started.items.items, default.items.items, atol=1e-7)


@pytest.mark.parametrize(
  ("matrix", "message"),
  [
    (np.array([1.0, 0.0]), "matrix of examinees by items"),
    (np.array([[1.0, 2.0], [0.0, 1.0]]), "1, 0 or NaN"),
    (np.array([[1.0, np.inf], [0.0, 1.0]]), "1, 0 or NaN"),
    (np.array([[1.0, np.nan], [0.0, np.nan]]), r"columns \[1\]"),
    (np.array([[1.0, 1.0], [0.0, 1.0]]), r"columns \[1\] were answered correctly by every examinee"),
  ],
  ids=["one-dimensional", "value 2", "infinite", "unanswered item", "all correct"],
)
def test_invalid_responses(matrix: np.ndarray, message: str) -> None:
  """A matrix not of 0, 1 and NaN, or with an item no one answered or with no finite estimate, is refused."""
  with pytest.raises(ValueError, match=message):
    MMLCalibrator().calibrate(matrix)


def test_a_reverse_keyed_item_is_refused() -> None:
  """An item whose correct answers fall as ability rises gets a clear error, not catsim's item-bank error at the end.

  Its slope estimate is negative, which a catsim item bank cannot hold.
  """
  rng = np.random.default_rng(3)
  slopes = np.array([1.0, 1.5, -1.0, 1.2])
  theta = rng.normal(size=2000)
  matrix = (rng.random((2000, 4)) < expit(slopes * theta[:, np.newaxis])).astype(float)
  with pytest.raises(ValueError, match=r"columns \[2\] have slope estimates that are not positive"):
    MMLCalibrator().calibrate(matrix)


@pytest.mark.parametrize(
  ("arguments", "message"),
  [
    ({"model": "3PL"}, "1PL and the 2PL"),
    ({"fixed_slope": 0.0}, "fixed_slope"),
    ({"tolerance": 0.0}, "tolerance"),
    ({"max_cycles": 0}, "max_cycles"),
  ],
)
def test_invalid_settings(arguments: dict[str, object], message: str) -> None:
  """A model other than the 1PL and 2PL, or an impossible setting, is refused when the calibrator is built."""
  with pytest.raises(ValueError, match=message):
    MMLCalibrator(**arguments)  # ty: ignore[invalid-argument-type]


def test_mismatched_items_are_refused() -> None:
  """Items that do not match the response matrix's columns are refused, as a start and for the log-likelihood."""
  three_items = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0]] * 3))
  with pytest.raises(ValueError, match="start has 3 items"):
    MMLCalibrator().calibrate(_LSAT7, start=three_items)
  with pytest.raises(ValueError, match="one row"):
    marginal_log_likelihood(_LSAT7, three_items.items)
