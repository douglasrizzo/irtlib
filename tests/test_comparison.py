"""Tests of model comparison: the likelihood-ratio test, AIC, BIC and G².

Expected values come from Thissen's (1982) and Bock and Aitkin's (1981) published statistics, mirt's recorded output
and hand-worked closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from catsim import ItemBank

from pyitemcal.calibration import Calibration, default_grid
from pyitemcal.comparison import aic, bic, g2, likelihood_ratio
from pyitemcal.mml import MMLCalibrator
from tests.mirt_fixtures import load, responses

_COMPARISON = load("comparison")
_LSAT6 = responses(_COMPARISON["data"]["lsat6"])
_LSAT7 = responses(_COMPARISON["data"]["lsat7"])


def _calibration(log_likelihood: float, n_parameters: int, n_items: int = 2) -> Calibration:
  """Return a calibration result with a given log-likelihood and number of parameters."""
  return Calibration(
    items=ItemBank(np.tile([1.0, 0.0, 0.0, 1.0], (n_items, 1))),
    standard_errors=None,
    log_likelihood=log_likelihood,
    n_parameters=n_parameters,
    item_parameters=np.full(n_items, 2),
    cycles=1,
    converged=True,
    change=0.0,
    grid=default_grid(),
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


def test_bic_refuses_an_empty_sample() -> None:
  """BIC needs at least one examinee, whose logarithm it takes."""
  with pytest.raises(ValueError, match="n_examinees"):
    bic(_calibration(-100.0, 4), 0)


def test_g2_refuses_incomplete_or_mismatched_responses_and_saturated_models() -> None:
  """G² refuses missing responses, another number of items, and a model with no degrees of freedom left."""
  calibration = MMLCalibrator().calibrate(_LSAT7)
  with_missing = _LSAT7.copy()
  with_missing[0, 0] = np.nan
  with pytest.raises(ValueError, match="complete responses"):
    g2(calibration, with_missing)
  with pytest.raises(ValueError, match="4 items, the calibration 5"):
    g2(calibration, _LSAT7[:, :4])
  with pytest.raises(ValueError, match="no degrees of freedom"):
    g2(MMLCalibrator().calibrate(_LSAT7[:, :2]), _LSAT7[:, :2])
