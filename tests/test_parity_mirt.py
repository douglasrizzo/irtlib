"""Parity tests with mirt, irtlib's reference implementation (ADR-0003).

The expected values are mirt's outputs on fixed cases, which ``tests/parity/mirt/generate.R`` wrote, with the R and
mirt versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R.

A data set's responses are stored as patterns with counts, one character per item: ``1`` correct, ``0`` incorrect and
``.`` missing. Each case records the model mirt fitted and its estimates in mirt's parameters, the slope and the
intercept of ``slope * theta + intercept``.

The fixtures are checked against an independent reference before any method uses them: the marginal log-likelihood,
summed by brute force over mirt's grid, at the parameters a fixture records must equal the log-likelihood mirt reports
there. A fixture recorded on a grid of another range, read with the slope and intercept confused, or with missing
responses counted, fails. The number of nodes cannot be checked this way: 41 or more equally spaced nodes on the same
range give the same log-likelihood within the tolerance. The response-function fixture holds items with fixed
parameters instead of responses, and is checked against the 4PL information of Magis (2013, p. 307, Equation 6),
written out. The ability fixture holds items with fixed parameters and response patterns, and its EAP estimates are
checked against the posterior mean and standard deviation written out on mirt's grid; the abilities of the
mean-square cases are checked the same way, on irtlib's grid.
"""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, logit, logsumexp
from scipy.stats import lognorm, norm

from irtlib import ItemBank, irt
from irtlib._m_step import _prior_terms, _PriorTerm  # noqa: PLC2701 - mirt's log-prior at its estimates
from irtlib.ability import (
  AbilityEstimates,
  expected_a_posteriori,
  maximum_a_posteriori,
  maximum_likelihood,
  weighted_likelihood,
)
from irtlib.calibration import Calibration, ConvergenceWarning, Likelihood, default_grid
from irtlib.comparison import aic, bic, g2, likelihood_ratio
from irtlib.fit import (
  _collapse,  # noqa: PLC2701 - mirt's collapsing is compared on its own recorded tables
  _mean_squares,  # noqa: PLC2701 - mirt's guards are compared at the abilities it was given
  mean_squares,
  s_x2,
)
from irtlib.mml import Acceleration, MMLCalibrator, Priors, marginal_log_likelihood
from irtlib.quadrature import QuadratureGrid
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from irtlib.typing import FloatArray

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
# The mean squares at estimates that agree within _ESTIMATE_ATOL: a relative 1e-4, as S-X²; they moved by 1.4e-6 at
# most (measured 2026-11-27). Their standardized values divide that change by a standard deviation as small as 0.015
# on these data, about 70 times larger: within 1e-3.
_MEAN_SQUARE_RTOL = 1e-4
_STANDARDIZED_ATOL = 1e-3

_MML_EM = load("mml_em")
_ITEM_FIT = load("item_fit")
_MML_PRIORS = load("mml_priors")
_COMPARISON = load("comparison")
_RESPONSE_FUNCTIONS = load("response_functions")

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
  """irtlib's marginal log-likelihood at fixed 2PL parameters equals mirt's, with and without missing responses.

  Fails if the E-step's grid masses or its treatment of missing responses differ from mirt's.
  """
  data = _MML_EM["data"][name]
  fixed = data["fixed"]
  items = _two_pl_items(np.array(fixed["slope"]), np.array(fixed["intercept"]))
  np.testing.assert_allclose(
    marginal_log_likelihood(responses(data), items), fixed["log_likelihood"], rtol=_CLOSED_FORM_RTOL
  )


@pytest.mark.parametrize("acceleration", list(Acceleration))
@pytest.mark.parametrize("case", _MML_EM["cases"], ids=operator.itemgetter("id"))
def test_mml_estimates(case: dict[str, Any], acceleration: Acceleration) -> None:
  """MMLCalibrator's estimates and log-likelihood match mirt's for the 2PL, the common-slope 1PL and fixed slopes.

  Compared in mirt's parameters, the slope and the intercept, with the calibrator stopped at mirt's tolerance, with
  and without acceleration, which changes the path to the maximum and not the maximum.
  """
  calibrator = MMLCalibrator(
    **_MML_MODELS[case["model"]], tolerance=_MML_EM["settings"]["tol"], acceleration=acceleration
  )
  calibration = calibrator.calibrate(responses(_MML_EM["data"][case["data"]]))
  slope = calibration.estimates[:, 0]
  np.testing.assert_allclose(slope, case["slope"], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(-slope * calibration.estimates[:, 1], case["intercept"], atol=_ESTIMATE_ATOL)
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


@pytest.mark.parametrize(
  "case", [case for case in _MML_EM["cases"] if "slope_intercept_covariance" in case], ids=operator.itemgetter("id")
)
def test_mml_covariances(case: dict[str, Any]) -> None:
  """MMLCalibrator's covariance of each item's a and b is mirt's covariance of its slope and intercept, taken to a and b
  by the delta method.

  mirt's covariance matrix holds a slope constrained equal across items once, and each item's covariance of its slope
  and intercept is then the shared slope's, as in irtlib's matrix of each item. Fails if an item's matrix leaves out
  the shared slope's covariance with its intercept, or the covariance of a and b has the wrong sign.
  """
  calibrator = MMLCalibrator(**_MML_MODELS[case["model"]], tolerance=_MML_EM["settings"]["tol"])
  covariances = calibrator.calibrate(responses(_MML_EM["data"][case["data"]])).covariances
  slope, intercept = np.array(case["slope"]), np.array(case["intercept"])
  slope_variance, intercept_variance = np.array(case["slope_se"]) ** 2, np.array(case["intercept_se"]) ** 2
  covariance = np.array(case["slope_intercept_covariance"])
  # b = -intercept / slope, whose derivatives in the slope and the intercept are intercept / slope^2 and -1 / slope
  slope_derivative, intercept_derivative = intercept / slope**2, -1 / slope
  expected = np.empty((slope.size, 2, 2))
  expected[:, 0, 0] = slope_variance
  expected[:, 0, 1] = expected[:, 1, 0] = slope_derivative * slope_variance + intercept_derivative * covariance
  expected[:, 1, 1] = (
    slope_derivative**2 * slope_variance
    + 2 * slope_derivative * intercept_derivative * covariance
    + intercept_derivative**2 * intercept_variance
  )
  assert covariances is not None
  # a variance is a standard error squared, so it agrees within twice the standard errors' relative tolerance
  np.testing.assert_allclose(covariances[:, :2, :2], expected, rtol=2 * _STANDARD_ERROR_RTOL)


# The number of parameters mirt counts for each item of each model: a slope constrained equal across items counts for
# each item, a fixed one not at all.
_ITEM_PARAMETERS = {"2pl": 2, "equal_slopes": 2, "fixed_slopes": 1}


@pytest.mark.parametrize("case", _ITEM_FIT["cases"], ids=operator.itemgetter("id"))
def test_s_x2_at_mirts_estimates(case: dict[str, Any]) -> None:
  """S-X², its degrees of freedom after collapsing and its p-value match mirt's at mirt's own estimates.

  Fails if the summed-score recursion, the expected proportions, the collapsing or the degrees of freedom differ.
  """
  slope, intercept = np.array(case["slope"]), np.array(case["intercept"])
  estimates = _two_pl_items(slope, intercept)
  calibration = Calibration(
    estimates=estimates,
    items=ItemBank(estimates),
    covariances=None,
    log_likelihood=0.0,
    log_likelihoods=np.empty(0),
    likelihood=Likelihood.MARGINAL,
    n_parameters=0,
    item_parameters=np.full(slope.size, _ITEM_PARAMETERS[case["model"]]),
    converged=True,
    change=0.0,
    grid=default_grid(),
    abilities=None,
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


# The mean squares and their standardized values, by mirt's names for them.
_MEAN_SQUARES = {"outfit": "outfit", "outfit_t": "z_outfit", "infit": "infit", "infit_t": "z_infit"}


def _rasch_items(case: dict[str, Any]) -> FloatArray:
  """Return a Rasch fit's items as the 1PL's [a, b, 0, 1], with the common slope a the abilities' standard deviation."""
  sd, intercept = case["sd"], np.array(case["intercept"])
  return np.column_stack([
    np.full(intercept.size, sd),
    -intercept / sd,
    np.zeros_like(intercept),
    np.ones_like(intercept),
  ])


@pytest.mark.parametrize("case", _ITEM_FIT["mean_squares"], ids=operator.itemgetter("id"))
def test_mean_square_fixture_abilities_are_eap_on_irtlibs_grid(case: dict[str, Any]) -> None:
  """mirt's abilities of a Rasch fit are its standard deviation times the 1PL posterior mean on irtlib's grid.

  The posterior weighs 61 nodes on [-6, 6] by the standard normal density and the Rasch likelihood of each pattern,
  written out. Fails if the fixture's abilities were estimated on another grid or prior.
  """
  data = _ITEM_FIT["data"][case["data"]]
  patterns = np.array([[float(char) for char in pattern] for pattern in data["patterns"]])
  nodes = np.linspace(-6.0, 6.0, 61)
  p = expit(case["sd"] * nodes[:, np.newaxis] + np.array(case["intercept"]))  # nodes x items
  mass = np.exp(patterns @ np.log(p).T + (1 - patterns) @ np.log1p(-p).T - nodes**2 / 2)
  mean = (mass @ nodes) / mass.sum(axis=1)
  np.testing.assert_allclose(case["theta"], case["sd"] * mean, rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("case", _ITEM_FIT["mean_squares"], ids=operator.itemgetter("id"))
def test_mean_squares_at_mirts_estimates(case: dict[str, Any]) -> None:
  """The mean squares and their standardized values match mirt's for a Rasch fit, at its estimates as a 1PL.

  mirt's Rasch model, slopes 1 and the ability variance estimated, is the 1PL with a common slope equal to the ability
  standard deviation. Fails if the abilities, the residuals, the variances or the standardization differ.
  """
  estimates = _rasch_items(case)
  calibration = Calibration(
    estimates=estimates,
    items=ItemBank(estimates),
    covariances=None,
    log_likelihood=0.0,
    log_likelihoods=np.empty(0),
    likelihood=Likelihood.MARGINAL,
    n_parameters=0,
    item_parameters=np.full(estimates.shape[0], 1),
    converged=True,
    change=0.0,
    grid=default_grid(),
    abilities=None,
  )
  fit = mean_squares(calibration, responses(_ITEM_FIT["data"][case["data"]]))
  for name, mirt_name in _MEAN_SQUARES.items():
    np.testing.assert_allclose(getattr(fit, name), case[mirt_name], rtol=_CLOSED_FORM_RTOL, err_msg=name)


@pytest.mark.parametrize("case", _ITEM_FIT["mean_squares"], ids=operator.itemgetter("id"))
def test_mean_squares_of_a_calibration(case: dict[str, Any]) -> None:
  """The mean squares of MMLCalibrator's own 1PL calibration match mirt's for its Rasch fit of the same responses.

  The 1PL's slope and intercepts agree with the Rasch fit's standard deviation and intercepts within _ESTIMATE_ATOL.
  """
  matrix = responses(_ITEM_FIT["data"][case["data"]])
  fit = mean_squares(MMLCalibrator("1PL", tolerance=_ITEM_FIT["settings"]["tol"]).calibrate(matrix), matrix)
  np.testing.assert_allclose(fit.outfit, case["outfit"], rtol=_MEAN_SQUARE_RTOL)
  np.testing.assert_allclose(fit.infit, case["infit"], rtol=_MEAN_SQUARE_RTOL)
  np.testing.assert_allclose(fit.outfit_t, case["z_outfit"], rtol=0, atol=_STANDARDIZED_ATOL)
  np.testing.assert_allclose(fit.infit_t, case["z_infit"], rtol=0, atol=_STANDARDIZED_ATOL)


def test_mean_square_guards() -> None:
  """mirt's floor on a response's variance and its cap on a mean square's standard deviation are irtlib's.

  Five responses, 7 or more logits from their item, have a variance whose square is below 1e-5, and the unweighted
  mean square of every item, over eight examinees, a standard deviation above 1.4142. Fails if either guard is
  dropped or moved.
  """
  case = _ITEM_FIT["mean_square_guards"]
  log_p, log_q = irt.log_response_probabilities(np.array(case["theta"], dtype=float), np.array(case["items"]))
  fit = _mean_squares(np.array(case["responses"], dtype=float), np.exp(log_p), np.exp(log_q))
  for name, mirt_name in _MEAN_SQUARES.items():
    np.testing.assert_allclose(getattr(fit, name), case[mirt_name], rtol=_CLOSED_FORM_RTOL, err_msg=name)


# mirt's parameter names in its PRIOR syntax, and the Priors field and parameter column of each.
_PRIOR_ARGUMENTS = {"a1": ("slope", 0), "d": ("intercept", 1), "g": ("guessing", 2), "u": ("upper", 3)}

_STOPS_SHORT = (
  "mirt 1.48 reports convergence with these priors where the gradient of the marginal log-posterior is still 6.6e-5 to "
  "7.4e-4 in size; irtlib's estimates bring it below 1e-6, at a higher log-posterior, and differ from mirt's by up "
  "to 1.7e-4, and by 1.6e-2 on the 4PL, whose likelihood is nearly flat along a ridge (measured 2026-11-27)"
)
_LOGNORMAL_HESSIAN = (
  "mirt 1.48's Hessian of the lognormal prior has the opposite sign on its last term (source R/utils.R:1430), which "
  "its Oakes standard errors take in; irtlib differentiates the lognormal log-density itself"
)


def _prior_settings(case: dict[str, Any]) -> dict[str, Any]:
  """Return the calibrator arguments that give a fixture case's model and priors."""
  priors = Priors(**{_PRIOR_ARGUMENTS[prior["parameter"]][0]: (prior["mean"], prior["sd"]) for prior in case["priors"]})
  settings = _MML_PRIORS["settings"]
  return {"model": case["model"], "priors": priors, "tolerance": settings["tol"], "max_cycles": settings["max_cycles"]}


@functools.cache
def _prior_calibration(case_id: str, acceleration: Acceleration = Acceleration.SQUAREM) -> Calibration:
  """Return the calibration of a prior fixture case, computed once for the estimate and standard-error tests."""
  case = next(case for case in _MML_PRIORS["cases"] if case["id"] == case_id)
  calibrator = MMLCalibrator(**_prior_settings(case), acceleration=acceleration)
  return calibrator.calibrate(responses(_MML_PRIORS["data"][case["data"]]))


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
    case.get("upper", np.full(n_items, np.inf)),
  ])
  priors = tuple(
    _PriorTerm(_PRIOR_ARGUMENTS[prior["parameter"]][1], prior["mean"], prior["sd"]) for prior in case["priors"]
  )
  # one prior per free parameter: the 1PL's shared slope carries it once
  counted = np.ones(values.shape, dtype=bool)
  if case["model"] == "1PL":
    counted[1:, 0] = False
  np.testing.assert_allclose(_prior_terms(values, priors, counted)[0], case["log_prior"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("acceleration", list(Acceleration))
@pytest.mark.parametrize(
  "case",
  [
    _marked(
      case,
      ("simulated_3pl/3pl_priors", _STOPS_SHORT),
      ("simulated/2pl_priors", _STOPS_SHORT),
      ("simulated_4pl/4pl_priors", _STOPS_SHORT),
    )
    for case in _MML_PRIORS["cases"]
  ],
)
def test_mml_priors_estimates(case: dict[str, Any], acceleration: Acceleration) -> None:
  """Bayes modal estimates under priors on the slope, intercept and logits of c and d match mirt's.

  With and without acceleration, which changes the path to the maximum and not the maximum.
  """
  calibration = _prior_calibration(case["id"], acceleration)
  slope = calibration.estimates[:, 0]
  np.testing.assert_allclose(slope, case["slope"], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(-slope * calibration.estimates[:, 1], case["intercept"], atol=_ESTIMATE_ATOL)
  for key, asymptote in [("guessing", calibration.estimates[:, 2]), ("upper", calibration.estimates[:, 3])]:
    if key in case:
      np.testing.assert_allclose(np.log(asymptote / (1 - asymptote)), case[key], atol=_ESTIMATE_ATOL)


@pytest.mark.parametrize("case", _MML_PRIORS["cases"], ids=operator.itemgetter("id"))
def test_acceleration_keeps_the_prior_estimates(case: dict[str, Any]) -> None:
  """With and without acceleration, the Bayes modal estimates of every prior case agree within the parity tolerance.

  Where mirt stops short of the mode its estimates cannot show that acceleration keeps the maximum, so the plain EM
  calibration, stopped at the same tolerance, is the reference. Fails if an accelerated cycle runs from E-step counts
  computed at other parameters, as one reusing the first cycle's counts after a refused extrapolation does: it is then
  not an EM cycle, and the stopping rule can be met away from the mode.
  """
  plain = _prior_calibration(case["id"], Acceleration.NONE).estimates
  accelerated = _prior_calibration(case["id"], Acceleration.SQUAREM).estimates
  np.testing.assert_allclose(accelerated[:, 0], plain[:, 0], atol=_ESTIMATE_ATOL)
  np.testing.assert_allclose(accelerated[:, 0] * accelerated[:, 1], plain[:, 0] * plain[:, 1], atol=_ESTIMATE_ATOL)
  for column, key in [(2, "guessing"), (3, "upper")]:
    if key in case:
      np.testing.assert_allclose(logit(accelerated[:, column]), logit(plain[:, column]), atol=_ESTIMATE_ATOL)


def _scipy_log_prior(case: dict[str, Any], estimates: FloatArray) -> float:
  """Return the log-prior of a prior case's estimates from scipy's densities, the 1PL's shared slope counted once."""
  total = 0.0
  for prior in case["priors"]:
    mean, sd = prior["mean"], prior["sd"]
    match prior["parameter"]:
      case "a1":
        slopes = estimates[:1, 0] if case["model"] == "1PL" else estimates[:, 0]
        total += np.sum(lognorm.logpdf(slopes, s=sd, scale=np.exp(mean)))
      case "d":
        total += np.sum(norm.logpdf(-estimates[:, 0] * estimates[:, 1], mean, sd))
      case "g":
        total += np.sum(norm.logpdf(logit(estimates[:, 2]), mean, sd))
      case _:
        total += np.sum(norm.logpdf(logit(estimates[:, 3]), mean, sd))
  return float(total)


@pytest.mark.parametrize("case", _MML_PRIORS["cases"], ids=operator.itemgetter("id"))
def test_acceleration_moves_the_third_cycle_uphill(case: dict[str, Any]) -> None:
  """After three cycles, SQUAREM's estimates are elsewhere than plain EM's, at a higher log-posterior.

  The third cycle runs from the first extrapolation, which every prior case keeps, so a calibrator that ran plain EM
  under ``"squarem"`` would return plain EM's three-cycle estimates. The log-posterior is the public marginal
  log-likelihood plus scipy's prior densities. A converged accelerated calibration also takes fewer cycles.
  """
  matrix = responses(_MML_PRIORS["data"][case["data"]])
  settings: dict[str, Any] = {**_prior_settings(case), "max_cycles": 3}
  estimates, log_posteriors = {}, {}
  for acceleration in Acceleration:
    # a ConvergenceWarning, and on the 4PL one for an information matrix that cannot be inverted yet
    with pytest.warns(RuntimeWarning) as caught:
      calibration = MMLCalibrator(**settings, acceleration=acceleration).calibrate(matrix)
    assert any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
    estimates[acceleration] = calibration.estimates
    log_posteriors[acceleration] = marginal_log_likelihood(matrix, calibration.estimates) + _scipy_log_prior(
      case, calibration.estimates
    )
  assert np.max(np.abs(estimates[Acceleration.SQUAREM] - estimates[Acceleration.NONE])) > 1e-3
  assert log_posteriors[Acceleration.SQUAREM] > log_posteriors[Acceleration.NONE]
  converged = {acceleration: _prior_calibration(case["id"], acceleration).cycles for acceleration in Acceleration}
  assert converged[Acceleration.SQUAREM] < converged[Acceleration.NONE]


@pytest.mark.parametrize(
  "case",
  [
    _marked(
      case,
      ("simulated_3pl/3pl_priors", _STOPS_SHORT),
      ("simulated/2pl_priors", _LOGNORMAL_HESSIAN),
      ("lsat7/1pl_priors", _LOGNORMAL_HESSIAN),
      ("simulated_4pl/4pl_priors", _STOPS_SHORT),
    )
    for case in _MML_PRIORS["cases"]
  ],
)
def test_mml_priors_standard_errors(case: dict[str, Any]) -> None:
  """The standard errors of a, b, c and d under priors match mirt's Oakes standard errors."""
  calibration = _prior_calibration(case["id"])
  standard_errors = calibration.standard_errors
  assert standard_errors is not None
  np.testing.assert_allclose(standard_errors[:, 0], case["a_se"], rtol=_STANDARD_ERROR_RTOL)
  np.testing.assert_allclose(standard_errors[:, 1], case["b_se"], rtol=_STANDARD_ERROR_RTOL)
  for column, key in [(2, "c_se"), (3, "d_se")]:
    if key in case:
      np.testing.assert_allclose(standard_errors[:, column], case[key], rtol=_STANDARD_ERROR_RTOL)


@pytest.mark.parametrize("case", _COMPARISON["cases"], ids=operator.itemgetter("data"))
def test_model_comparison(case: dict[str, Any]) -> None:
  """The log-likelihoods, parameter counts, AIC, BIC, G² and likelihood-ratio test of the 1PL and 2PL match mirt's.

  G² and the likelihood ratio are differences of log-likelihoods that agree within a relative 1e-10, so they agree
  within about 1e-8 in absolute terms.
  """
  matrix = responses(_COMPARISON["data"][case["data"]])
  tolerance = _COMPARISON["settings"]["tol"]
  restricted = MMLCalibrator("1PL", tolerance=tolerance).calibrate(matrix)
  general = MMLCalibrator("2PL", tolerance=tolerance).calibrate(matrix)
  for calibration, expected in [(restricted, case["restricted"]), (general, case["general"])]:
    assert calibration.n_parameters == expected["n_parameters"]
    np.testing.assert_allclose(calibration.log_likelihood, expected["log_likelihood"], rtol=_CLOSED_FORM_RTOL)
    np.testing.assert_allclose(aic(calibration), expected["aic"], rtol=_CLOSED_FORM_RTOL)
    np.testing.assert_allclose(bic(calibration, case["n_examinees"]), expected["bic"], rtol=_CLOSED_FORM_RTOL)
    fit = g2(calibration, matrix)
    assert fit.df == expected["df"]
    np.testing.assert_allclose(fit.statistic, expected["g2"], atol=1e-6)
    np.testing.assert_allclose(fit.p_value, expected["p_value"], rtol=1e-6)
  test = likelihood_ratio(restricted, general)
  assert test.df == case["likelihood_ratio"]["df"]
  np.testing.assert_allclose(test.statistic, case["likelihood_ratio"]["statistic"], atol=1e-6)
  np.testing.assert_allclose(test.p_value, case["likelihood_ratio"]["p_value"], atol=1e-6)


def test_mml_four_pl_asymptote_standard_errors() -> None:
  """The 4PL's standard errors of c and d match mirt's within 2e-3, though its estimates stop short of the mode.

  mirt's estimates differ from the mode (the expected failures above), which moves these standard errors by up to
  6.3e-4; a wrong delta method, missing the c(1 - c) or d(1 - d) factor of about 0.1 to 0.2, misses by far more.
  """
  case = next(case for case in _MML_PRIORS["cases"] if case["id"] == "simulated_4pl/4pl_priors")
  standard_errors = _prior_calibration(case["id"]).standard_errors
  assert standard_errors is not None
  np.testing.assert_allclose(standard_errors[:, 2], case["c_se"], rtol=2e-3)
  np.testing.assert_allclose(standard_errors[:, 3], case["d_se"], rtol=2e-3)


def _four_pl_information(theta: FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL item information of Magis (2013, p. 307, Equation 6), written out, every ability by every item."""
  a, b, c, d = items.T
  p = c + (d - c) * expit(a * (theta[..., np.newaxis] - b))
  return a**2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))


@pytest.mark.parametrize("case", _RESPONSE_FUNCTIONS["reliability"], ids=operator.itemgetter("id"))
def test_reliability_fixture_is_the_information_over_the_information_plus_one(case: dict[str, Any]) -> None:
  """mirt's rxx curve is I / (I + 1) of the recorded items, by Magis's Equation 6 written out.

  Fails if the curve was recorded on another grid, for other items, or with the abilities' variance other than 1.
  """
  information = np.sum(_four_pl_information(np.array(case["thetas"]), np.array(case["items"])), axis=-1)
  np.testing.assert_allclose(case["reliability"], information / (information + 1), rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("case", _RESPONSE_FUNCTIONS["reliability"], ids=operator.itemgetter("id"))
def test_reliability(case: dict[str, Any]) -> None:
  """irtlib's reliability matches mirt's rxx curve for 2PL, 3PL and 4PL items; 1 - 1/I or a per-item ratio misses."""
  items = np.array(case["items"])
  values = [irt.reliability(theta, items) for theta in case["thetas"]]
  np.testing.assert_allclose(values, case["reliability"], rtol=_CLOSED_FORM_RTOL)


def _area(lower: float, upper: float, item: FloatArray) -> float:
  """Return the integral of an item's information by quadrature of Equation 6, up to 30 logits beyond its difficulty."""
  a, b = item[0], item[1]
  value, _ = quad(
    lambda theta: float(_four_pl_information(np.array(theta), item[np.newaxis, :])[0]),
    max(lower, b - 30 / a),
    min(upper, b + 30 / a),
    points=[b],
    limit=200,
    epsabs=1e-13,
    epsrel=1e-12,
  )
  return value


@pytest.mark.parametrize("case", _RESPONSE_FUNCTIONS["info_integral"], ids=operator.itemgetter("id"))
def test_info_integral_fixture_is_the_integral_of_the_information(case: dict[str, Any]) -> None:
  """mirt's areainfo equals SciPy's quadrature of Equation 6 written out, within 1e-10.

  Fails if the fixture's bounds or items are recorded out of place, or mirt integrated to a looser tolerance. The
  quadrature stops 30 logits beyond the difficulty, where the information left is below 2e-13.
  """
  items = np.array(_RESPONSE_FUNCTIONS["items"])
  lower, upper = float(case["lower"]), float(case["upper"])
  expected = [_area(lower, upper, item) for item in items]
  np.testing.assert_allclose(case["info"], expected, rtol=_CLOSED_FORM_RTOL, atol=1e-12)


@pytest.mark.parametrize("case", _RESPONSE_FUNCTIONS["info_integral"], ids=operator.itemgetter("id"))
def test_info_integral(case: dict[str, Any]) -> None:
  """irtlib's closed-form integral matches mirt's areainfo, a numerical integration to a relative 1e-12.

  The bounds are finite, half-infinite and the whole scale, for 2PL, 3PL and 4PL items, so a wrong weight on log P or
  log Q, or a log term not dropped at an infinite bound, misses.
  """
  items = np.array(_RESPONSE_FUNCTIONS["items"])
  values = irt.info_integral(float(case["lower"]), float(case["upper"]), items)
  np.testing.assert_allclose(values, case["info"], rtol=_CLOSED_FORM_RTOL)


_ABILITY = load("ability")
_ESTIMATORS = {
  "EAP": expected_a_posteriori,
  "MAP": maximum_a_posteriori,
  "ML": maximum_likelihood,
  "WLE": weighted_likelihood,
}
_NO_RESPONSE = (
  "for an examinee who answered no item mirt 1.48 returns NA (R/fscores.internal.R:94-96, 185); irtlib's posterior "
  "is then the prior, whose mean, mode and standard deviation it returns"
)
_NO_RESPONSE_STANDARD_ERROR = (
  "for an examinee who answered no item mirt 1.48 returns NA (R/fscores.internal.R:94-96, 185); irtlib's maximum "
  "likelihood and weighted likelihood estimates are NaN with standard error inf, the information being 0"
)
_OBSERVED_INFORMATION = (
  "mirt 1.48 takes the standard error from nlm's numerical Hessian, the observed information, plus the prior's "
  "precision for MAP (R/fscores.internal.R:1064-1068, 1083-1101); irtlib's is the expected information, which equals "
  "the observed only when every answered item has c = 0 and d = 1"
)
_INFINITE_STANDARD_ERROR = (
  "mirt 1.48 gives an infinite estimate no standard error, NA (R/fscores.internal.R:397-405); irtlib's is inf, since "
  "the information tends to 0 at either infinity"
)
_SUPREMUM = (
  "where the likelihood comes closest to its supremum only at an infinity, irtlib's estimate is that infinity; mirt "
  "1.48's nlm stops at its max_theta, -20, or short of the supremum, at 2.48, and turns an estimate whose standard "
  "error exceeds 30 into an infinity (R/fscores.internal.R:1094-1096), which gives -inf for the first pattern"
)
_WEIGHTED = (
  "mirt 1.48 maximizes log L + log I / 2 (R/fscores.internal.R:705-738), which is Warm's estimate only when every "
  "answered item has c = 0 and d = 1 (Warm, 1989, p. 431); irtlib solves Warm's modified score equation"
)
_WEIGHTED_STANDARD_ERROR = (
  "mirt 1.48's WLE standard error comes from nlm's numerical Hessian of its weighted log-likelihood; irtlib's is "
  "1 / sqrt(I), catR's default"
)


def _floats(values: list[Any]) -> FloatArray:
  """Return a list of JSON numbers, or of the strings json.R writes for infinities and NaN, as a float array."""
  return np.array([float(value) for value in values], dtype=float)


def _ability_estimates(case: dict[str, Any]) -> AbilityEstimates:
  """Return irtlib's estimates of a case's patterns, by the case's method on its bank, with irtlib's defaults."""
  return _ESTIMATORS[case["method"]](responses(case), np.array(_ABILITY["banks"][case["bank"]]))


def test_ability_fixture_uses_irtlibs_grid() -> None:
  """mirt's EAP grid in the fixture is irtlib's default, 121 nodes on [-6, 6]; another grid tests other settings."""
  grid = QuadratureGrid.uniform()
  settings = _ABILITY["settings"]
  assert (settings["quadpts"], *settings["theta_lim"]) == (grid.nodes.size, grid.nodes[0], grid.nodes[-1])


@pytest.mark.parametrize("case", [case for case in _ABILITY["cases"] if case["id"] in {"eap/2pl", "eap/4pl"}])
def test_ability_fixture_eap_is_the_posterior_mean(case: dict[str, Any]) -> None:
  """mirt's EAP estimates and standard errors are the mean and standard deviation of the posterior, written out.

  The posterior weighs 121 nodes on [-6, 6] by the standard normal density and the 4PL likelihood of the answered items.
  A fixture recorded on another grid or prior, or with its patterns or banks out of place, fails.
  """
  items = np.array(_ABILITY["banks"][case["bank"]])
  nodes = np.linspace(-6.0, 6.0, 121)
  a, b, c, d = items.T
  p = c + (d - c) / (1.0 + np.exp(-a * (nodes[:, np.newaxis] - b)))
  matrix = responses(case)
  answered = ~np.isnan(matrix)
  log_likelihood = np.where(answered, matrix, 0.0) @ np.log(p).T + np.where(answered, 1 - matrix, 0.0) @ np.log1p(-p).T
  mass = np.exp(log_likelihood - nodes**2 / 2)
  mass /= mass.sum(axis=1, keepdims=True)
  mean = mass @ nodes
  np.testing.assert_allclose(_floats(case["theta"]), mean, rtol=_CLOSED_FORM_RTOL)
  np.testing.assert_allclose(
    _floats(case["se"]), np.sqrt(np.sum(mass * (nodes - mean[:, np.newaxis]) ** 2, axis=1)), rtol=_CLOSED_FORM_RTOL
  )


def _ability_cases(*reasons: tuple[str, str]) -> list[Any]:
  """Return the ability cases as parameters, each a strict expected failure where a reason names its id."""
  return [_marked(case, *reasons) for case in _ABILITY["cases"]]


@pytest.mark.parametrize(
  "case",
  _ability_cases(
    ("eap_unanswered/2pl", _NO_RESPONSE),
    ("map_unanswered/2pl", _NO_RESPONSE),
    ("ml_supremum/4pl", _SUPREMUM),
    ("wle/4pl", _WEIGHTED),
  ),
)
def test_ability_estimates(case: dict[str, Any]) -> None:
  """irtlib's EAP, MAP, ML and WLE estimates, with its default settings, match mirt's fscores.

  EAP is a sum on the same grid, compared as a closed form; the others are searches, compared within the estimates'
  tolerance. Infinite estimates match by sign, and NaN, for no estimate, by position.
  """
  estimates = _ability_estimates(case)
  if case["method"] == "EAP":
    np.testing.assert_allclose(estimates.theta, _floats(case["theta"]), rtol=_CLOSED_FORM_RTOL)
  else:
    np.testing.assert_allclose(estimates.theta, _floats(case["theta"]), rtol=0, atol=_ESTIMATE_ATOL)


@pytest.mark.parametrize(
  "case",
  _ability_cases(
    ("eap_unanswered/2pl", _NO_RESPONSE),
    ("map_unanswered/2pl", _NO_RESPONSE),
    ("map/4pl", _OBSERVED_INFORMATION),
    ("ml/4pl", _OBSERVED_INFORMATION),
    ("ml_constant/2pl", _INFINITE_STANDARD_ERROR),
    ("ml_constant/4pl", _INFINITE_STANDARD_ERROR),
    ("ml_supremum/4pl", _SUPREMUM),
    ("ml_unanswered/2pl", _NO_RESPONSE_STANDARD_ERROR),
    ("wle/2pl", _WEIGHTED_STANDARD_ERROR),
    ("wle/4pl", _WEIGHTED_STANDARD_ERROR),
    ("wle_unanswered/2pl", _NO_RESPONSE_STANDARD_ERROR),
  ),
)
def test_ability_standard_errors(case: dict[str, Any]) -> None:
  """irtlib's standard errors match mirt's: the posterior standard deviation for EAP, 1 / sqrt(I) for 2PL items.

  On 2PL items the observed information of mirt's numerical Hessian is the expected information irtlib uses, plus the
  prior's precision for MAP, so the two agree within the Hessian's accuracy.
  """
  estimates = _ability_estimates(case)
  rtol = _CLOSED_FORM_RTOL if case["method"] == "EAP" else _STANDARD_ERROR_RTOL
  np.testing.assert_allclose(estimates.standard_errors, _floats(case["se"]), rtol=rtol)
