"""Parity tests with mirt, pyitemcal's reference implementation (ADR-0003).

The expected values are mirt's outputs on fixed cases, which ``tests/parity/mirt/generate.R`` wrote, with the R and
mirt versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R.

A data set's responses are stored as patterns with counts, one character per item: ``1`` correct, ``0`` incorrect and
``.`` missing. Each case records the model mirt fitted and its estimates in mirt's parameters, the slope and the
intercept of ``slope * theta + intercept``.

The fixtures are checked against an independent reference before any method uses them: the marginal log-likelihood,
summed by brute force over mirt's grid, at the parameters a fixture records must equal the log-likelihood mirt reports
there. A fixture recorded on a grid of another range, read with the slope and intercept confused, or with missing
responses counted, fails. The number of nodes cannot be checked this way: 41 or more equally spaced nodes on the same
range give the same log-likelihood within the tolerance.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from pyitemcal.mml import MMLCalibrator, marginal_log_likelihood
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from catsim.typing import FloatArray

# Closed forms, sums on the same grid, and log-likelihoods at two estimates that agree within _ESTIMATE_ATOL, which
# the log-likelihood feels only to second order: a relative 1e-10 (ADR-0003).
_CLOSED_FORM_RTOL = 1e-10
# Estimates, against mirt's recorded at TOL = 1e-8 with an exact M-step: 1e-5 (ADR-0003).
_ESTIMATE_ATOL = 1e-5
# Standard errors, computed at estimates that agree within _ESTIMATE_ATOL, by the same identity and difference step:
# a relative 1e-4.
_STANDARD_ERROR_RTOL = 1e-4

_MML_EM = load("mml_em")

# The calibrator that fits each model of the fixture: mirt's 2PL, its 2PL with the slopes constrained equal, and its
# "1PL", whose slopes are fixed at 1.
_MML_MODELS: dict[str, dict[str, Any]] = {
  "2pl": {"model": "2PL"},
  "equal_slopes": {"model": "1PL"},
  "fixed_slopes": {"model": "1PL", "fixed_slope": 1.0},
}


def _two_pl_items(slope: FloatArray, intercept: FloatArray) -> FloatArray:
  """Return 2PL items [a, b, 0, 1] from mirt's slopes and intercepts."""
  return np.column_stack([slope, -intercept / slope, np.zeros_like(slope), np.ones_like(slope)])


def _brute_force_log_likelihood(responses: FloatArray, slope: FloatArray, intercept: FloatArray) -> float:
  """Sum, over examinees, the log of the grid sum of the normal masses times the answered items' probabilities."""
  settings = _MML_EM["settings"]
  low, high = settings["theta_limits"]
  nodes = np.linspace(low, high, settings["quadrature_points"])
  log_masses = -(nodes**2) / 2 - np.log(np.sum(np.exp(-(nodes**2) / 2)))
  probability = expit(intercept[np.newaxis, :] + slope[np.newaxis, :] * nodes[:, np.newaxis])  # nodes x items
  answered = ~np.isnan(responses)
  correct = np.where(answered, responses, 0.0)
  incorrect = np.where(answered, 1.0 - responses, 0.0)
  log_likelihood = correct @ np.log(probability).T + incorrect @ np.log1p(-probability).T  # examinees x nodes
  return float(np.sum(logsumexp(log_likelihood + log_masses[np.newaxis, :], axis=1)))


@pytest.mark.parametrize("name", sorted(_MML_EM["data"]))
def test_fixed_parameter_log_likelihood(name: str) -> None:
  """mirt's log-likelihood at fixed 2PL parameters equals the brute-force sum over its 61-node grid.

  Fails if the fixture's grid range, its normal masses or its treatment of missing responses differ from what the
  parity tests assume.
  """
  data = _MML_EM["data"][name]
  fixed = data["fixed"]
  log_likelihood = _brute_force_log_likelihood(responses(data), np.array(fixed["slope"]), np.array(fixed["intercept"]))
  np.testing.assert_allclose(log_likelihood, fixed["log_likelihood"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("case", _MML_EM["cases"], ids=operator.itemgetter("id"))
def test_estimate_log_likelihood(case: dict[str, Any]) -> None:
  """mirt's log-likelihood at its recorded estimates equals the brute-force sum there.

  Fails if a case's slopes and intercepts are recorded out of place, or the case names the wrong data set.
  """
  matrix = responses(_MML_EM["data"][case["data"]])
  log_likelihood = _brute_force_log_likelihood(matrix, np.array(case["slope"]), np.array(case["intercept"]))
  np.testing.assert_allclose(log_likelihood, case["log_likelihood"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("name", sorted(_MML_EM["data"]))
def test_mml_fixed_parameter_log_likelihood(name: str) -> None:
  """pyitemcal's marginal log-likelihood at fixed 2PL parameters equals mirt's, with and without missing responses.

  Fails if the E-step's grid masses or its treatment of missing responses differ from mirt's.
  """
  data = _MML_EM["data"][name]
  fixed = data["fixed"]
  items = _two_pl_items(np.array(fixed["slope"]), np.array(fixed["intercept"]))
  np.testing.assert_allclose(
    marginal_log_likelihood(responses(data), items), fixed["log_likelihood"], rtol=_CLOSED_FORM_RTOL
  )


@pytest.mark.parametrize("case", _MML_EM["cases"], ids=operator.itemgetter("id"))
def test_mml_estimates(case: dict[str, Any]) -> None:
  """MMLCalibrator's estimates and log-likelihood match mirt's for the 2PL, the common-slope 1PL and fixed slopes.

  Compared in mirt's parameters, the slope and the intercept, with the calibrator stopped at mirt's tolerance.
  """
  calibrator = MMLCalibrator(**_MML_MODELS[case["model"]], tolerance=_MML_EM["settings"]["tol"])
  calibration = calibrator.calibrate(responses(_MML_EM["data"][case["data"]]))
  slope = calibration.items.discrimination
  np.testing.assert_allclose(slope, case["slope"], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(-slope * calibration.items.difficulty, case["intercept"], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(calibration.log_likelihood, case["log_likelihood"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("case", _MML_EM["cases"], ids=operator.itemgetter("id"))
def test_mml_standard_errors(case: dict[str, Any]) -> None:
  """MMLCalibrator's standard errors of a and b match mirt's Oakes standard errors; a parameter not estimated has none.

  mirt reports the slope's standard error and, by the delta method, the difficulty's (``coef(..., IRTpars = TRUE)``).
  """
  calibrator = MMLCalibrator(**_MML_MODELS[case["model"]], tolerance=_MML_EM["settings"]["tol"])
  standard_errors = calibrator.calibrate(responses(_MML_EM["data"][case["data"]])).standard_errors
  assert standard_errors is not None
  np.testing.assert_allclose(standard_errors[:, 1], case["b_se"], rtol=_STANDARD_ERROR_RTOL)
  if "slope_se" in case:
    np.testing.assert_allclose(standard_errors[:, 0], case["slope_se"], rtol=_STANDARD_ERROR_RTOL)
  else:
    assert np.all(np.isnan(standard_errors[:, 0]))
  assert np.all(np.isnan(standard_errors[:, 2:]))
