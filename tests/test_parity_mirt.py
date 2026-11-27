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
from catsim import ItemBank
from scipy.special import expit, logsumexp

from pyitemcal.calibration import Calibration, default_grid
from pyitemcal.fit import (
  _collapse,  # noqa: PLC2701 - mirt's collapsing is compared on its own recorded tables
  s_x2,
)
from pyitemcal.mml import (
  MMLCalibrator,
  Priors,
  _prior_terms,  # noqa: PLC2701
  _PriorTerm,  # noqa: PLC2701 - the prior densities are compared with mirt's log-prior at its estimates
  marginal_log_likelihood,
)
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
# S-X² and its p-value at estimates that agree within _ESTIMATE_ATOL: a relative 1e-4.
_S_X2_RTOL = 1e-4

_MML_EM = load("mml_em")
_ITEM_FIT = load("item_fit")
_MML_PRIORS = load("mml_priors")

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


# The number of parameters mirt counts for each item of each model: a slope constrained equal across items counts for
# each item, a fixed one not at all.
_ITEM_PARAMETERS = {"2pl": 2, "equal_slopes": 2, "fixed_slopes": 1}


@pytest.mark.parametrize("case", _ITEM_FIT["cases"], ids=operator.itemgetter("id"))
def test_s_x2_at_mirts_estimates(case: dict[str, Any]) -> None:
  """S-X², its degrees of freedom after collapsing and its p-value match mirt's at mirt's own estimates.

  Fails if the summed-score recursion, the expected proportions, the collapsing or the degrees of freedom differ.
  """
  slope, intercept = np.array(case["slope"]), np.array(case["intercept"])
  calibration = Calibration(
    items=ItemBank(_two_pl_items(slope, intercept)),
    standard_errors=None,
    log_likelihood=0.0,
    n_parameters=0,
    item_parameters=np.full(slope.size, _ITEM_PARAMETERS[case["model"]]),
    cycles=0,
    converged=True,
    change=0.0,
    grid=default_grid(),
  )
  fit = s_x2(calibration, responses(_ITEM_FIT["data"][case["data"]]))
  np.testing.assert_array_equal(fit.df, case["df"])
  np.testing.assert_allclose(fit.statistic, case["s_x2"], rtol=_CLOSED_FORM_RTOL)
  np.testing.assert_allclose(fit.p_value, case["p_value"], rtol=_CLOSED_FORM_RTOL, atol=1e-14)


@pytest.mark.parametrize("case", _ITEM_FIT["cases"], ids=operator.itemgetter("id"))
def test_s_x2_of_a_calibration(case: dict[str, Any]) -> None:
  """S-X² of MMLCalibrator's own calibration matches mirt's, degrees of freedom included.

  The estimates agree within _ESTIMATE_ATOL, which moves the statistics by about a relative 1e-5.
  """
  matrix = responses(_ITEM_FIT["data"][case["data"]])
  calibration = MMLCalibrator(**_MML_MODELS[case["model"]], tolerance=_ITEM_FIT["settings"]["tol"]).calibrate(matrix)
  fit = s_x2(calibration, matrix)
  np.testing.assert_array_equal(fit.df, case["df"])
  np.testing.assert_allclose(fit.statistic, case["s_x2"], rtol=_S_X2_RTOL)
  np.testing.assert_allclose(fit.p_value, case["p_value"], rtol=_S_X2_RTOL)


@pytest.mark.parametrize("table", range(len(_ITEM_FIT["collapse"])))
def test_collapse_matches_mirt(table: int) -> None:
  """Collapsing a table of incorrect and correct counts by score gives the table mirt's collapseCells gives.

  Random tables with empty rows, rows of one examinee, tied neighbours and small expected counts exercise every rule.
  """
  case = _ITEM_FIT["collapse"][table]
  observed, expected = _collapse(np.array(case["observed"]), np.array(case["expected"]), 1.0)
  np.testing.assert_array_equal(observed, case["collapsed_observed"])
  np.testing.assert_allclose(expected, case["collapsed_expected"], rtol=1e-13, atol=1e-13)


# mirt's parameter names in its PRIOR syntax, and the Priors field and parameter column of each.
_PRIOR_ARGUMENTS = {"a1": ("slope", 0), "d": ("intercept", 1), "g": ("guessing", 2)}

_STOPS_SHORT = (
  "mirt 1.48 reports convergence with these priors where the gradient of the marginal log-posterior is still 6.6e-5 to "
  "7.4e-4 in size; pyitemcal's estimates bring it below 1e-6, at a higher log-posterior, and differ from mirt's by up "
  "to 1.7e-4 (measured 2026-11-27)"
)
_LOGNORMAL_HESSIAN = (
  "mirt 1.48's Hessian of the lognormal prior has the opposite sign on its last term (source R/utils.R:1430), which "
  "its Oakes standard errors take in; pyitemcal differentiates the lognormal log-density itself"
)


def _prior_settings(case: dict[str, Any]) -> dict[str, Any]:
  """Return the calibrator arguments that give a fixture case's model and priors."""
  priors = Priors(**{_PRIOR_ARGUMENTS[prior["parameter"]][0]: (prior["mean"], prior["sd"]) for prior in case["priors"]})
  return {"model": case["model"], "priors": priors, "tolerance": _MML_PRIORS["settings"]["tol"]}


def _marked(case: dict[str, Any], *reasons: tuple[str, str]) -> Any:
  """Return a case as a pytest parameter, a strict expected failure with its reason where one names its id."""
  for case_id, reason in reasons:
    if case["id"] == case_id:
      return pytest.param(case, id=case["id"], marks=pytest.mark.xfail(reason=reason, strict=True))
  return pytest.param(case, id=case["id"])


@pytest.mark.parametrize("case", _MML_PRIORS["cases"], ids=operator.itemgetter("id"))
def test_mml_prior_densities(case: dict[str, Any]) -> None:
  """At mirt's estimates, the summed log-prior equals mirt's log-prior: the priors are mirt's, on mirt's parameters."""
  n_items = len(case["slope"])
  values = np.column_stack([
    case["slope"],
    case["intercept"],
    case.get("guessing", np.full(n_items, -np.inf)),
    np.full(n_items, np.inf),
  ])
  priors = tuple(
    _PriorTerm(_PRIOR_ARGUMENTS[prior["parameter"]][1], prior["mean"], prior["sd"]) for prior in case["priors"]
  )
  # one prior per free parameter: the 1PL's shared slope carries it once
  counted = np.ones(values.shape, dtype=bool)
  if case["model"] == "1PL":
    counted[1:, 0] = False
  np.testing.assert_allclose(_prior_terms(values, priors, counted)[0], case["log_prior"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize(
  "case",
  [
    _marked(case, ("simulated_3pl/3pl_priors", _STOPS_SHORT), ("simulated/2pl_priors", _STOPS_SHORT))
    for case in _MML_PRIORS["cases"]
  ],
)
def test_mml_priors_estimates(case: dict[str, Any]) -> None:
  """Bayes modal estimates of the 3PL with a prior on the logit of c, and of the 2PL with priors, match mirt's."""
  calibration = MMLCalibrator(**_prior_settings(case)).calibrate(responses(_MML_PRIORS["data"][case["data"]]))
  slope = calibration.items.discrimination
  np.testing.assert_allclose(slope, case["slope"], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(-slope * calibration.items.difficulty, case["intercept"], atol=_ESTIMATE_ATOL)
  if "guessing" in case:
    guessing = calibration.items.pseudo_guessing
    np.testing.assert_allclose(np.log(guessing / (1 - guessing)), case["guessing"], atol=_ESTIMATE_ATOL)


@pytest.mark.parametrize(
  "case",
  [
    _marked(
      case,
      ("simulated_3pl/3pl_priors", _STOPS_SHORT),
      ("simulated/2pl_priors", _LOGNORMAL_HESSIAN),
      ("lsat7/1pl_priors", _LOGNORMAL_HESSIAN),
    )
    for case in _MML_PRIORS["cases"]
  ],
)
def test_mml_priors_standard_errors(case: dict[str, Any]) -> None:
  """The standard errors of a, b and c under priors match mirt's Oakes standard errors."""
  calibration = MMLCalibrator(**_prior_settings(case)).calibrate(responses(_MML_PRIORS["data"][case["data"]]))
  standard_errors = calibration.standard_errors
  assert standard_errors is not None
  np.testing.assert_allclose(standard_errors[:, 0], case["a_se"], rtol=_STANDARD_ERROR_RTOL)
  np.testing.assert_allclose(standard_errors[:, 1], case["b_se"], rtol=_STANDARD_ERROR_RTOL)
  if "c_se" in case:
    np.testing.assert_allclose(standard_errors[:, 2], case["c_se"], rtol=_STANDARD_ERROR_RTOL)
