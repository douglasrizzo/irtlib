"""Parity tests with catR, the second reference implementation of the response functions and ability estimators.

The expected values are catR's outputs on fixed cases, which ``tests/parity/catr/generate.R`` wrote, with the R and
catR versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R. Each
estimation case records catR's settings, and the tests call the irtlib estimator that matches them. A case where irtlib
departs from catR on purpose is a strict expected failure with the reason (``_DEPARTURES``), so a departure that closes
fails the run (ADR-0003).

The fixtures are checked against an independent reference before irtlib is compared with them: catR's probabilities and
information at the recorded abilities and items must equal the 4PL of Magis (2013, pp. 306-307, Equations 1 and 6),
written out here; its EAP estimates must be the trapezoid posterior mean and standard deviation written out, and its
other estimates a bound of their range or a root of their estimating equation written out.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from irtlib import irt
from irtlib.ability import (
  AbilityEstimates,
  expected_a_posteriori,
  maximum_a_posteriori,
  maximum_likelihood,
  weighted_likelihood,
)
from irtlib.quadrature import GridPrior, LogPrior, QuadratureGrid, normal_log_prior, uniform_log_prior

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.typing import FloatArray

_DATA = Path(__file__).parent / "parity" / "catr"

# Closed forms, and quadratures on the same nodes with the same weights: a relative 1e-10 (ADR-0003).
_CLOSED_FORM_RTOL = 1e-10
# catR's thetaEst solves its estimating equations with uniroot at its default tolerance, .Machine$double.eps ** 0.25,
# which is 2 ** -13; irtlib's estimators search to their own tol, 1e-6. Each estimate may miss the root by its
# package's tolerance, so the two may differ by their sum.
_ESTIMATE_ATOL = 2.0**-13 + 1e-6

_ML_MINIMUM = (
  "the likelihood has a local minimum inside the range; catR's thetaEst solves the likelihood equation with uniroot "
  "(R/thetaEst.R:100-102), which converges to that minimum, while irtlib maximizes the likelihood"
)
_WL_ROOTS = (
  "Warm's modified score has three roots in the range, two maxima of the weighted likelihood and a minimum; catR's "
  "uniroot converges to the lower maximum (-0.59), and irtlib returns the one of larger weighted likelihood (0.66)"
)
# The estimation cases irtlib departs from on purpose, by case id, with the reason: each must fail its assertion, in
# the estimate and in the standard error taken there.
_DEPARTURES = {
  "ml/two_stationary_points": _ML_MINIMUM,
  "ml/rising_at_bound": _ML_MINIMUM,
  "wl/three_wl_roots": _WL_ROOTS,
}


def _load(name: str) -> dict[str, Any]:
  """Return the fixture ``name``, such as ``"irt"``."""
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


_IRT = _load("irt")
_ESTIMATION = _load("estimation")


def _probability(thetas: FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL probability, written out, every ability against every item."""
  a, b, c, d = items.T
  return c + (d - c) / (1 + np.exp(-a * (thetas[:, np.newaxis] - b)))


class TestData:
  """The recorded data come from the pinned catR and match the 4PL written out."""

  def test_recorded_with_the_pinned_catr(self) -> None:
    """A catR version pinned in generate.R but never run would leave the fixture recorded with another version."""
    pinned = re.search(r'^CATR_VERSION <- "([^"]+)"$', (_DATA / "generate.R").read_text(encoding="utf-8"), re.MULTILINE)

    assert pinned is not None
    assert {_IRT["catr_version"], _ESTIMATION["catr_version"]} == {pinned.group(1)}

  def test_one_r_run(self) -> None:
    """Files written by different R versions would compare irtlib with two references at once."""
    assert _IRT["r_version"] == _ESTIMATION["r_version"]

  def test_departures_name_recorded_cases(self) -> None:
    """A departure whose case was renamed or dropped would mark nothing and hide the change."""
    assert set(_DEPARTURES) <= {case["id"] for case in _ESTIMATION["cases"]}

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_probabilities_are_the_4pl(self, case: dict[str, Any]) -> None:
    """catR's Pi is Magis's Equation 1 at the recorded abilities and items.

    A fixture recorded with catR's constant D at 1.7, with the item columns in another order, or with the abilities and
    items paired differently fails.
    """
    expected = _probability(np.array(case["thetas"]), np.array(case["items"]))

    np.testing.assert_allclose(case["probability"], expected, rtol=_CLOSED_FORM_RTOL)

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_information_is_the_4pl_information(self, case: dict[str, Any]) -> None:
    """catR's Ii is Magis's Equation 6 at the recorded abilities and items; a fixture of another quantity fails."""
    items = np.array(case["items"])
    _, _, c, d = items.T
    p = _probability(np.array(case["thetas"]), items)
    expected = items[:, 0] ** 2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))

    np.testing.assert_allclose(case["information"], expected, rtol=_CLOSED_FORM_RTOL)


class TestItemResponseFunctions:
  """catR's Pi and Ii, on 1PL to 4PL items, against irtlib.irt."""

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_probability(self, case: dict[str, Any]) -> None:
    """A wrong logistic, or a guessing or slipping parameter entering it the wrong way, would move icc off catR's Pi."""
    np.testing.assert_allclose(
      irt.icc(np.array(case["thetas"]), np.array(case["items"])), case["probability"], rtol=_CLOSED_FORM_RTOL
    )

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_information(self, case: dict[str, Any]) -> None:
    """An information formula that drops c or d, or squares the wrong term, would move off catR's Ii."""
    np.testing.assert_allclose(
      irt.information(np.array(case["thetas"]), np.array(case["items"])), case["information"], rtol=_CLOSED_FORM_RTOL
    )


def _estimation_cases() -> list[Any]:
  """Return the estimation cases as parameters, each a strict expected failure where irtlib departs from catR."""
  return [
    pytest.param(
      case,
      id=case["id"],
      marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=_DEPARTURES[case["id"]])]
      if case["id"] in _DEPARTURES
      else [],
    )
    for case in _ESTIMATION["cases"]
  ]


def _log_prior(prior: dict[str, Any]) -> LogPrior:
  """Return irtlib's log prior for catR's priorDist and priorPar: the normal's mean and sd, or the uniform's bounds."""
  first, second = prior["parameters"]
  if prior["distribution"] == "norm":
    return normal_log_prior(first, second)
  return uniform_log_prior(first, second)


def _trapezoid_grid(low: float, high: float, nodes: int) -> QuadratureGrid:
  """Return catR's EAP grid, parInt = c(low, high, nodes), with the trapezoid weights of its integrate.catR."""
  weights = np.full(nodes, (high - low) / (nodes - 1))
  weights[[0, -1]] /= 2
  return QuadratureGrid(nodes=np.linspace(low, high, nodes), weights=weights)


def _case_responses(case: dict[str, Any]) -> FloatArray:
  """Return the case's responses as one row over the whole bank, NaN for the items the pattern leaves out."""
  row = np.full((1, len(_ESTIMATION["bank"])), np.nan)
  row[0, case["items"]] = case["responses"]
  return row


def _estimate(case: dict[str, Any]) -> AbilityEstimates:
  """Return irtlib's estimate of the case, by the estimator and settings that match catR's method and settings."""
  items, responses = np.array(_ESTIMATION["bank"]), _case_responses(case)
  match case["method"]:
    case "EAP":
      grid = _trapezoid_grid(**case["grid"])
      return expected_a_posteriori(responses, items, grid=grid, log_prior=_log_prior(case["prior"]))
    case "BM":
      bounds = tuple(case["range"])
      return maximum_a_posteriori(responses, items, log_prior=_log_prior(case["prior"]), bounds=bounds)
    case "WL":
      return weighted_likelihood(responses, items, bounds=tuple(case["range"]))
    case _:
      return maximum_likelihood(responses, items, bounds=tuple(case["range"]))


def _estimating_equation(case: dict[str, Any]) -> Callable[[float], float]:
  """Return the left side of the case's estimating equation, written out: the score, plus the prior's or Warm's term.

  The score of the log-likelihood is the sum of (u - P) P' / (P Q) (Magis & Raiche, 2012, p. 4); BM adds the derivative
  of the log-prior, (mean - theta) / sd^2 for a normal prior and 0 for a uniform one, and WL adds J / (2 I) (p. 6,
  Equations 12 and 13).
  """
  items = np.array(_ESTIMATION["bank"])[case["items"]]
  u = np.array(case["responses"], dtype=float)
  a, b, c, d = items.T

  def equation(theta: float) -> float:
    logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    p = c + (d - c) * logistic
    first = a * (d - c) * logistic * (1.0 - logistic)
    second = a * first * (1.0 - 2.0 * logistic)
    pq = p * (1.0 - p)
    value = float(np.sum((u - p) * first / pq))
    if case["method"] == "WL":
      value += float(np.sum(first * second / pq) / (2.0 * np.sum(first**2 / pq)))
    if case["method"] == "BM" and case["prior"]["distribution"] == "norm":
      mean, sd = case["prior"]["parameters"]
      value += (mean - theta) / sd**2
    return value

  return equation


class TestEstimationData:
  """catR's estimates are what its estimators define, before irtlib is compared with them."""

  @pytest.mark.parametrize(
    "case", [case for case in _ESTIMATION["cases"] if case["method"] == "EAP"], ids=operator.itemgetter("id")
  )
  def test_eap_is_the_trapezoid_posterior_mean(self, case: dict[str, Any]) -> None:
    """catR's EAP and its standard error are the posterior mean and standard deviation on its grid, written out.

    The posterior weighs the trapezoid nodes by the prior's density and the 4PL likelihood of the pattern. A fixture
    recorded on another grid or prior, or with its items or responses out of place, fails.
    """
    grid = case["grid"]
    nodes = np.linspace(grid["low"], grid["high"], grid["nodes"])
    weights = np.where((nodes == nodes[0]) | (nodes == nodes[-1]), 0.5, 1.0)
    first, second = case["prior"]["parameters"]
    if case["prior"]["distribution"] == "norm":
      weights *= np.exp(-0.5 * ((nodes - first) / second) ** 2)
    else:
      weights *= (nodes >= first) & (nodes <= second)
    p = _probability(nodes, np.array(_ESTIMATION["bank"])[case["items"]])
    mass = weights * np.prod(np.where(np.array(case["responses"]), p, 1.0 - p), axis=1)
    mass /= mass.sum()
    mean = mass @ nodes

    np.testing.assert_allclose(case["theta"], mean, rtol=_CLOSED_FORM_RTOL)
    np.testing.assert_allclose(case["se"], np.sqrt(mass @ (nodes - mean) ** 2), rtol=_CLOSED_FORM_RTOL)

  @pytest.mark.parametrize(
    "case", [case for case in _ESTIMATION["cases"] if case["method"] != "EAP"], ids=operator.itemgetter("id")
  )
  def test_a_search_ends_at_a_bound_or_a_root(self, case: dict[str, Any]) -> None:
    """catR's ML, BM and WL estimates are a bound of their range or within its tolerance of a root of their equation.

    A fixture recorded with another prior, range or estimator, or with its items or responses out of place, puts its
    estimate where the equation written out does not change sign.
    """
    equation = _estimating_equation(case)
    theta = case["theta"]
    bound = case["prior"]["parameters"] if case.get("prior", {}).get("distribution") == "unif" else case["range"]

    assert theta in bound or equation(theta - _ESTIMATE_ATOL) * equation(theta + _ESTIMATE_ATOL) <= 0


class TestEstimators:
  """catR's thetaEst and semTheta, with ML, BM, EAP and WL, against the matching irtlib estimator.

  ML is maximum_likelihood and BM maximum_a_posteriori, each with catR's range as its bounds, EAP is
  expected_a_posteriori on catR's grid with its trapezoid weights, and WL is weighted_likelihood on catR's range. EAP is
  a quadrature on the same nodes, so it is compared as a closed form; the others search, and are compared within the
  estimates' tolerance.
  """

  @pytest.mark.parametrize("case", _estimation_cases())
  def test_estimate(self, case: dict[str, Any]) -> None:
    """A wrong estimating equation, prior, range or grid would move the estimate off catR's thetaEst.

    catR's default for responses all correct or all incorrect, the bound of its range, is that of irtlib's maximum
    likelihood estimate with bounds.
    """
    theta = _estimate(case).theta[0]

    if case["method"] == "EAP":
      assert theta == pytest.approx(case["theta"], rel=_CLOSED_FORM_RTOL, abs=0)
    else:
      assert theta == pytest.approx(case["theta"], rel=0, abs=_ESTIMATE_ATOL)

  @pytest.mark.parametrize("case", _estimation_cases())
  def test_standard_error(self, case: dict[str, Any]) -> None:
    """A standard error that left out the prior's curvature, or added Warm's correction, would differ from semTheta's.

    Away from EAP the two standard errors are taken at estimates that differ by up to the estimates' tolerance, so they
    may differ by that much times the standard error's slope, which generate.R records; the 1 added to the slope leaves
    room for the central-difference curvature of the prior.
    """
    se = _estimate(case).standard_errors[0]

    if case["method"] == "EAP":
      assert se == pytest.approx(float(case["se"]), rel=_CLOSED_FORM_RTOL, abs=0)
    else:
      assert se == pytest.approx(float(case["se"]), rel=0, abs=_ESTIMATE_ATOL * (abs(case["se_slope"]) + 1))

  @pytest.mark.parametrize(
    "case", [case for case in _ESTIMATION["cases"] if case["method"] == "EAP"], ids=operator.itemgetter("id")
  )
  def test_eap_from_a_grid_prior(self, case: dict[str, Any]) -> None:
    """EAP from a GridPrior, the route catsim's estimator takes, gives catR's estimate and standard error too.

    catR's uniform prior on (-3, 3) over its grid on [-4, 4] has no density at the outer nodes, whose log-prior is
    -inf; a GridPrior that gave those nodes mass, or turned them into NaN, would move both off catR's.
    """
    items, responses = np.array(_ESTIMATION["bank"]), _case_responses(case)
    grid_prior = GridPrior(_trapezoid_grid(**case["grid"]), _log_prior(case["prior"]))

    estimates = expected_a_posteriori(responses, items, grid_prior=grid_prior)

    assert estimates.theta[0] == pytest.approx(case["theta"], rel=_CLOSED_FORM_RTOL, abs=0)
    assert estimates.standard_errors[0] == pytest.approx(float(case["se"]), rel=_CLOSED_FORM_RTOL, abs=0)
