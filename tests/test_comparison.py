"""Tests of model comparison: the likelihood-ratio test, AIC, BIC and G².

Expected values come from Thissen's (1982) and Bock and Aitkin's (1981) published statistics, mirt's recorded output
and hand-worked closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irtlib import ItemBank
from irtlib.calibration import Calibration, Likelihood, default_grid
from irtlib.cml import CMLCalibrator
from irtlib.comparison import aic, bic, g2, likelihood_ratio
from irtlib.drawing import draw_responses
from irtlib.mml import MMLCalibrator
from tests.mirt_fixtures import load, responses

_COMPARISON = load("comparison")
_LSAT6 = responses(_COMPARISON["data"]["lsat6"])
_LSAT7 = responses(_COMPARISON["data"]["lsat7"])
# 2PL responses whose third item is miskeyed: its slope is negative, so the 2PL's bank leaves it out
_MISKEYED = draw_responses(
  np.random.default_rng(3).normal(size=2000),
  np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 0.5, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0], [1.2, -0.5, 0.0, 1.0]]),
  rng=4,
)


def _calibration(log_likelihood: float, n_parameters: int, n_items: int = 2) -> Calibration:
  """Return a calibration result with a given log-likelihood and number of parameters."""
  estimates = np.tile([1.0, 0.0, 0.0, 1.0], (n_items, 1))
  return Calibration(
    estimates=estimates,
    items=ItemBank(estimates),
    covariances=None,
    log_likelihood=log_likelihood,
    log_likelihoods=np.array([log_likelihood]),
    likelihood=Likelihood.MARGINAL,
    n_parameters=n_parameters,
    item_parameters=np.full(n_items, 2),
    converged=True,
    change=0.0,
    grid=default_grid(),
    abilities=None,
  )


def test_information_criteria_closed_forms() -> None:
  """AIC is -2 log L + 2k and BIC -2 log L + k log N (Akaike, 1974, p. 719; Schwarz, 1978, p. 461)."""
  calibration = _calibration(-100.0, 4)
  assert aic(calibration) == pytest.approx(208.0, rel=1e-15)
  assert bic(calibration, 50) == pytest.approx(200.0 + 4 * math.log(50), rel=1e-15)


def test_thissen_published_statistics() -> None:
  """On LSAT-6, G² is 21.8 for the 1PL and 21.2 for the 2PL, a difference of 0.6 (Thissen, 1982, p. 178).

  Thissen prints one decimal. Fails if G² or the likelihood ratio is not twice a log-likelihood difference.
  """
  restricted = MMLCalibrator("1PL", tolerance=1e-8).calibrate(_LSAT6)
  general = MMLCalibrator("2PL", tolerance=1e-8).calibrate(_LSAT6)
  assert g2(restricted, _LSAT6).statistic == pytest.approx(21.8, abs=0.05)
  assert g2(general, _LSAT6).statistic == pytest.approx(21.2, abs=0.05)
  test = likelihood_ratio(restricted, general)
  assert test.statistic == pytest.approx(0.6, abs=0.05)
  assert test.df == 4


@pytest.mark.parametrize(("matrix", "published"), [(_LSAT6, 21.80), (_LSAT7, 43.90)], ids=["LSAT-6", "LSAT-7"])
def test_bock_aitkin_published_g2(matrix: np.ndarray, published: float) -> None:
  """The 1PL's G² on 25 degrees of freedom is Bock and Aitkin's (1981, p. 455, Table 3) within 0.01.

  Their table integrates on 10 points, against 61 here, which moves the second decimal of LSAT-7's value.
  """
  fit = g2(MMLCalibrator("1PL", tolerance=1e-8).calibrate(matrix), matrix)
  assert fit.df == 25
  assert fit.statistic == pytest.approx(published, abs=0.01)


def test_likelihood_ratio_refuses_models_that_cannot_be_nested() -> None:
  """A general model with no more parameters than the restricted one, or with other items, is refused."""
  with pytest.raises(ValueError, match="more parameters"):
    likelihood_ratio(_calibration(-100.0, 4), _calibration(-99.0, 4))
  with pytest.raises(ValueError, match="2 and 3 items"):
    likelihood_ratio(_calibration(-100.0, 4), _calibration(-99.0, 6, n_items=3))


def test_a_conditional_calibration_is_refused() -> None:
  """AIC, BIC, G² and the likelihood-ratio test refuse a calibration by conditional maximum likelihood.

  Its log-likelihood leaves out the likelihood of the numbers correct, which a marginal one holds, so comparing the
  two would mislead.
  """
  conditional = CMLCalibrator().calibrate(_LSAT6)
  marginal = MMLCalibrator("1PL").calibrate(_LSAT6)
  for statistic, compute in [
    ("AIC", lambda: aic(conditional)),
    ("BIC", lambda: bic(conditional, 1000)),
    ("G2", lambda: g2(conditional, _LSAT6)),
    ("the likelihood-ratio test", lambda: likelihood_ratio(conditional, marginal)),
    ("the likelihood-ratio test", lambda: likelihood_ratio(marginal, conditional)),
  ]:
    with pytest.raises(
      ValueError, match=f"^for {statistic}, a calibration must be of the marginal likelihood.*conditional$"
    ):
      compute()


def test_bic_refuses_an_empty_sample() -> None:
  """BIC needs at least one examinee, whose logarithm it takes."""
  with pytest.raises(ValueError, match="n_examinees"):
    bic(_calibration(-100.0, 4), 0)


def test_g2_refuses_incomplete_or_mismatched_responses_and_saturated_models() -> None:
  """G² refuses missing responses, another number of items, and a model with no degrees of freedom left."""
  calibration = MMLCalibrator().calibrate(_LSAT7)
  with_missing = _LSAT7.copy()
  with_missing[0, 0] = np.nan
  with pytest.raises(
    ValueError, match=r"^complete responses are needed for G2; the multinomial over patterns is not defined"
  ):
    g2(calibration, with_missing)
  with pytest.raises(ValueError, match="4 items, the calibration 5"):
    g2(calibration, _LSAT7[:, :4])
  # the 1PL of two items has as many parameters as free pattern probabilities, and is identified
  with pytest.raises(ValueError, match="no degrees of freedom"):
    g2(MMLCalibrator("1PL").calibrate(_LSAT7[:, :2]), _LSAT7[:, :2])


def test_g2_of_a_miskeyed_item_equals_g2_with_its_scoring_reversed() -> None:
  """G² of a calibration whose bank leaves out a miskeyed item equals G² with that item's scoring reversed.

  Reversing an item's scoring maps its slope to its opposite and keeps the likelihood of every pattern, so the two
  statistics are equal. Fails if G² reads the bank, which leaves the miskeyed item out, rather than the estimates.
  """
  reversed_scoring = _MISKEYED.copy()
  reversed_scoring[:, 2] = 1 - reversed_scoring[:, 2]
  calibrator = MMLCalibrator(tolerance=1e-10)
  with pytest.warns(RuntimeWarning, match="miskeyed"):
    miskeyed = calibrator.calibrate(_MISKEYED)
  expected = g2(calibrator.calibrate(reversed_scoring), reversed_scoring).statistic
  assert g2(miskeyed, _MISKEYED).statistic == pytest.approx(expected, rel=1e-6)


def test_likelihood_ratio_counts_the_items_a_bank_leaves_out() -> None:
  """The 1PL's bank of four items and the 2PL's bank of three, without the miskeyed item, are still nested models.

  The test has 8 - 5 = 3 degrees of freedom. Fails if it counts the items of the banks, which differ, rather than
  those of the estimates.
  """
  restricted = MMLCalibrator("1PL").calibrate(_MISKEYED)
  with pytest.warns(RuntimeWarning, match="miskeyed"):
    general = MMLCalibrator("2PL").calibrate(_MISKEYED)
  assert restricted.items is not None
  assert general.items is not None
  assert (restricted.items.n_items, general.items.n_items) == (4, 3)
  test = likelihood_ratio(restricted, general)
  assert test.df == 3
  assert test.statistic == pytest.approx(2 * (general.log_likelihood - restricted.log_likelihood), rel=1e-15)
