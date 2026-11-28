"""Tests of marginal maximum likelihood calibration with the EM algorithm, beyond the parity tests with mirt.

Expected values come from Thissen's (1982) published estimates, a brute-force sum, recovery of generating parameters,
and the algorithm's guarantees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import lognorm, norm

from irtlib import ItemBank
from irtlib.calibration import ConvergenceWarning
from irtlib.drawing import draw_responses
from irtlib.mml import (
  Acceleration,
  MMLCalibrator,
  Priors,
  _extrapolated,  # noqa: PLC2701
  _prior_terms,  # noqa: PLC2701
  _PriorTerm,  # noqa: PLC2701 - the prior's derivatives are checked against scipy's densities
  _squarem_point,  # noqa: PLC2701 - the extrapolation is checked on linear iterations, where it has a closed form
  marginal_log_likelihood,
)
from irtlib.quadrature import QuadratureGrid
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.typing import FloatArray

_MML_EM = load("mml_em")
_LSAT6 = responses(_MML_EM["data"]["lsat6"])
_LSAT7 = responses(_MML_EM["data"]["lsat7"])


def _two_pl_items(slopes: FloatArray, difficulties: FloatArray) -> FloatArray:
  """Return 2PL items [a, b, 0, 1]."""
  return np.column_stack([slopes, difficulties, np.zeros_like(slopes), np.ones_like(slopes)])


def test_thissen_published_estimates() -> None:
  """The 1PL with a common slope reproduces Thissen's (1982, p. 180, Table 2) LSAT-6 estimates to the decimals printed.

  LSAT-6 is Thissen's (1982, p. 179, Table 1) data; a wrong scale identification or prior moves the slope.
  """
  calibration = MMLCalibrator("1PL", tolerance=1e-8).calibrate(_LSAT6)
  np.testing.assert_allclose(calibration.estimates[:, 0], 0.755, atol=5e-4)
  np.testing.assert_allclose(calibration.estimates[:, 1], [-3.615, -1.322, -0.318, -1.730, -2.780], atol=5e-4)


def test_recovery_of_two_pl_parameters() -> None:
  """2PL estimates from 20,000 simulated examinees with 10% of responses missing recover the generating items.

  The tolerances, 15% of each slope and 0.15 for each difficulty, are over twice the largest error measured across
  seeds, and far below what a scale factor of 1.7 in the logit, a prior applied twice, or a sign error would give.
  """
  rng = np.random.default_rng(20261127)
  slopes = rng.uniform(0.7, 2.0, 10)
  difficulties = rng.uniform(-1.5, 1.5, 10)
  theta = rng.normal(size=20_000)
  matrix = draw_responses(theta, _two_pl_items(slopes, difficulties), rng=rng)
  matrix[rng.random(matrix.shape) < 0.1] = np.nan
  calibration = MMLCalibrator().calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 0], slopes, rtol=0.15)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulties, atol=0.15)


_GUESSING_PRIOR = (float(logit(0.2)), 0.5)


@pytest.mark.parametrize("acceleration", list(Acceleration))
@pytest.mark.parametrize(
  ("model", "matrix", "start"),
  [
    ("2PL", _LSAT7, None),
    ("3PL", _LSAT6, ItemBank(np.tile([0.3, 2.0, 0.3, 1.0], (5, 1)))),
  ],
  ids=["LSAT-7 2PL", "LSAT-6 3PL from far"],
)
def test_log_posterior_never_decreases(
  model: str, matrix: FloatArray, start: ItemBank | None, acceleration: Acceleration
) -> None:
  """The log-posterior after each of the first cycles never decreases, with or without acceleration.

  No EM cycle lowers it (Dempster, Laird & Rubin, 1977, pp. 6-7), and SQUAREM keeps an extrapolation only where it
  does not. The 3PL, with a prior on the logit of c, starts from slopes of 0.3, difficulties of 2 and guessing
  parameters of 0.3, where SQUAREM's second extrapolation would lower the log-posterior by about 16. Fails if that
  extrapolation is kept, if an M-step does not maximize the expected complete-data log-posterior, or if the E-step's
  counts are wrong. The log-posterior is the public marginal log-likelihood plus scipy's density of the prior.
  """
  priors = Priors(guessing=_GUESSING_PRIOR) if model == "3PL" else None
  log_posteriors = []
  for cycles in range(1, 9):
    calibrator = MMLCalibrator(model, priors=priors, tolerance=1e-14, max_cycles=cycles, acceleration=acceleration)
    # a ConvergenceWarning, and from the far start one for an information matrix that cannot be inverted yet
    with pytest.warns(RuntimeWarning):
      calibration = calibrator.calibrate(matrix, start=start)
    log_prior = np.sum(norm.logpdf(logit(calibration.estimates[:, 2]), *_GUESSING_PRIOR)) if priors else 0.0
    log_posteriors.append(marginal_log_likelihood(matrix, calibration.estimates) + log_prior)
  assert np.all(np.diff(log_posteriors) >= -1e-9), np.diff(log_posteriors)
  assert log_posteriors[-1] > log_posteriors[0]


def test_squarem_point_of_a_linear_iteration() -> None:
  """SQUAREM's extrapolation from three iterates of x -> x* + c (x - x*) is the fixed point x*, for any c below 1.

  With e the first iterate's error, r = (c - 1) e and v = (c - 1)^2 e, so the steplength -|r| / |v| is -1 / (1 - c)
  and the extrapolation's error, (1 - alpha (c - 1))^2 e (Du & Varadhan, 2020, p. 3), vanishes: for c = 0.9,
  alpha = -10 and x0 - 2 alpha r + alpha^2 v = x0 - 2 e + e. Fails if the steplength's sign or norms, the coefficients
  2 and alpha^2, or the r in v are wrong.
  """
  fixed_point = np.array([0.3, -1.2, 2.0])
  start = np.array([1.0, 0.5, -0.5])
  for contraction in (0.9, 0.5, -0.5):
    first = fixed_point + contraction * (start - fixed_point)
    second = fixed_point + contraction * (first - fixed_point)
    np.testing.assert_allclose(_squarem_point(start, first, second), fixed_point, rtol=0, atol=1e-12)


def test_extrapolation_counts_a_shared_slope_once() -> None:
  """With the 1PL's slope shared by two items, the steplength's norms count the slope once.

  The free parameters (slope, two intercepts) start 0.2, 1 and 1 from (1, 0.5, -0.5) and contract at rates 0.5, 0.9
  and 0.9, so r = (-0.1, -0.1, -0.1), v = (0.05, 0.01, 0.01) and alpha = -10/3; each error becomes
  (1 - alpha (c - 1))^2 times itself, 4/9 of it. Counting the slope once per item gives alpha = -2.77 and misses.
  """
  index = np.array([[0, 1, -1, -1], [0, 2, -1, -1]])
  start, first, second = (
    np.array([[slope, first_intercept, -np.inf, np.inf], [slope, second_intercept, -np.inf, np.inf]])
    for slope, first_intercept, second_intercept in [(1.2, 1.5, 0.5), (1.1, 1.4, 0.4), (1.05, 1.31, 0.31)]
  )
  extrapolated = _extrapolated(start, first, second, index)
  assert extrapolated is not None
  expected_slope, expected_intercepts = 1.0 + 0.2 * 4 / 9, np.array([0.5, -0.5]) + 4 / 9
  np.testing.assert_allclose(extrapolated[:, 0], expected_slope, rtol=0, atol=1e-12)
  np.testing.assert_allclose(extrapolated[:, 1], expected_intercepts, rtol=0, atol=1e-12)
  assert np.all(extrapolated[:, 2] == -np.inf)
  assert np.all(extrapolated[:, 3] == np.inf)


@pytest.mark.parametrize(
  ("index", "iterates"),
  [
    (
      np.array([[0, 1, -1, -1]]),
      [[1.0, 0.0, -np.inf, np.inf], [1.25, 0.25, -np.inf, np.inf], [1.5, 0.5, -np.inf, np.inf]],
    ),
    (np.array([[0, 1, 2, 3]]), [[1.0, 0.0, -1.0, 1.0], [1.0, 0.0, -0.7, 0.7], [1.0, 0.0, -0.46, 0.46]]),
  ],
  ids=["no second difference", "guessing above the upper asymptote"],
)
def test_extrapolation_that_leaves_the_parameter_space_is_not_kept(
  index: np.ndarray, iterates: list[list[float]]
) -> None:
  """An extrapolation that is not finite, or puts a guessing parameter at or above its upper asymptote, is refused.

  Iterates that move by the same step, exactly representable, have v = 0 and an infinite steplength. Logits of c
  and d that contract at rate 0.8 toward 0.5 and -0.5 extrapolate to exactly those values, alpha being -5, where c
  is above d and the E-step's log-likelihood is NaN.
  """
  start, first, second = (np.array([row]) for row in iterates)
  assert _extrapolated(start, first, second, index) is None


def _louis_information(matrix: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
  """Return Louis's (1982, p. 227, Equation 3.2) observed information of complete 2PL responses, slopes first.

  The expected complete-data information, which for the 2PL holds no data, less the posterior covariance of the
  complete-data score, from each examinee's posterior over 61 nodes on [-6, 6] with normal masses.
  """
  nodes = np.linspace(-6.0, 6.0, 61)
  log_masses = -(nodes**2) / 2 - np.log(np.sum(np.exp(-(nodes**2) / 2)))
  probability = expit(intercept + slope * nodes[:, np.newaxis])  # nodes x items
  log_joint = matrix @ np.log(probability).T + (1 - matrix) @ np.log1p(-probability).T + log_masses
  posterior = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
  posterior /= posterior.sum(axis=1, keepdims=True)  # examinees x nodes
  # the complete-data score at each node: (u - P) X for the slopes, (u - P) for the intercepts
  residual = matrix[:, np.newaxis, :] - probability[np.newaxis, :, :]
  score = np.concatenate([residual * nodes[np.newaxis, :, np.newaxis], residual], axis=2)
  n_items = slope.size
  blocks = np.zeros((nodes.size, 2 * n_items, 2 * n_items))
  weight = probability * (1 - probability)
  for item in range(n_items):
    for row, column, factor in [(0, 0, nodes**2), (0, 1, nodes), (1, 0, nodes), (1, 1, np.ones_like(nodes))]:
      blocks[:, row * n_items + item, column * n_items + item] = weight[:, item] * factor
  mean_score = np.einsum("jq,jqp->jp", posterior, score)
  covariance_of_score = np.einsum("jq,jqp,jqr->pr", posterior, score, score) - mean_score.T @ mean_score
  return np.einsum("jq,qpr->pr", posterior, blocks) - covariance_of_score


def test_standard_errors_match_louis_exact_form() -> None:
  """The standard errors match those of Louis's exact observed information, computed independently here.

  Oakes's identity by central differences agrees with Louis's form to about 1e-11 (Chalmers, 2018, p. 13). Fails if
  the missing information has the wrong sign or scale, or the delta method to a and b is wrong.
  """
  calibration = MMLCalibrator(tolerance=1e-10).calibrate(_LSAT7)
  slope = calibration.estimates[:, 0]
  intercept = -slope * calibration.estimates[:, 1]
  covariance = np.linalg.inv(_louis_information(_LSAT7, slope, intercept))
  n_items = slope.size
  jacobian = np.zeros((n_items, 2 * n_items))
  jacobian[np.arange(n_items), np.arange(n_items)] = intercept / slope**2
  jacobian[np.arange(n_items), n_items + np.arange(n_items)] = -1 / slope
  assert calibration.standard_errors is not None
  np.testing.assert_allclose(calibration.standard_errors[:, 0], np.sqrt(np.diag(covariance)[:n_items]), rtol=1e-7)
  np.testing.assert_allclose(
    calibration.standard_errors[:, 1], np.sqrt(np.einsum("np,pr,nr->n", jacobian, covariance, jacobian)), rtol=1e-7
  )


def test_unidentified_model_reports_no_standard_errors() -> None:
  """A 2PL of one item, whose slope and intercept one proportion correct cannot both identify, has no standard errors.

  The information matrix cannot be inverted: the result holds None and a RuntimeWarning says why. With the slope
  fixed, the same item has a standard error for its difficulty.
  """
  matrix = (np.random.default_rng(0).random((500, 1)) < 0.6).astype(float)
  with pytest.warns(RuntimeWarning, match="could not be inverted"):
    calibration = MMLCalibrator().calibrate(matrix)
  assert calibration.standard_errors is None
  fixed = MMLCalibrator(fixed_slope=1.0).calibrate(matrix)
  assert fixed.standard_errors is not None
  assert np.isnan(fixed.standard_errors[0, 0])
  assert fixed.standard_errors[0, 1] > 0


def test_standard_errors_cover_the_generating_parameters() -> None:
  """Over 60 simulated calibrations, the standard errors match the estimates' spread and their intervals cover.

  Each parameter's mean standard error is within 30% of its estimates' standard deviation, and 91% to 98% of the
  95% intervals contain the generating value: about three sampling errors of each statistic from its target.
  """
  slopes = np.array([0.8, 1.0, 1.2, 1.5, 2.0])
  difficulties = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
  truth = np.concatenate([slopes, difficulties])
  rng = np.random.default_rng(20261127)
  estimates, standard_errors = [], []
  for _ in range(60):
    theta = rng.normal(size=1000)
    matrix = draw_responses(theta, _two_pl_items(slopes, difficulties), rng=rng)
    calibration = MMLCalibrator().calibrate(matrix)
    assert calibration.standard_errors is not None
    estimates.append(np.concatenate([calibration.estimates[:, 0], calibration.estimates[:, 1]]))
    standard_errors.append(np.concatenate([calibration.standard_errors[:, 0], calibration.standard_errors[:, 1]]))
  estimates, standard_errors = np.array(estimates), np.array(standard_errors)
  ratio = standard_errors.mean(axis=0) / estimates.std(axis=0, ddof=1)
  coverage = np.mean(np.abs(estimates - truth) <= 1.96 * standard_errors)
  assert np.all((ratio > 0.7) & (ratio < 1.3)), ratio
  assert 0.91 <= coverage <= 0.98, coverage


@pytest.mark.parametrize(("n_nodes", "bound"), [(61, 6.0), (81, 40.0)], ids=["default grid", "wide grid"])
def test_marginal_log_likelihood_with_a_missing_response(n_nodes: int, bound: float) -> None:
  """The marginal log-likelihood equals a brute-force sum that leaves a missing response out of its examinee's product.

  Four examinees, three items, one response missing; normal masses at 61 equally spaced nodes on [-6, 6], the default
  grid, and at 81 on [-40, 40], whose outer masses underflow to 0, which a log of the masses would warn about.
  """
  matrix = np.array([[1.0, 0.0, 1.0], [0.0, np.nan, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
  items = np.array([[1.2, -0.4, 0.0, 1.0], [0.7, 0.9, 0.0, 1.0], [1.6, 0.1, 0.0, 1.0]])
  nodes = np.linspace(-bound, bound, n_nodes)
  masses = np.exp(-(nodes**2) / 2)
  masses /= masses.sum()
  probability = expit(items[:, 0] * (nodes[:, np.newaxis] - items[:, 1]))  # the 2PL, nodes x items
  expected = 0.0
  for row in matrix:
    likelihood = np.ones_like(nodes)
    for item, response in enumerate(row):
      if not np.isnan(response):
        likelihood *= probability[:, item] if response == 1 else 1 - probability[:, item]
    expected += np.log(np.sum(masses * likelihood))
  grid = QuadratureGrid.uniform(n_nodes, -bound, bound)
  np.testing.assert_allclose(marginal_log_likelihood(matrix, items, grid), expected, rtol=1e-12)


def test_an_examinee_who_answered_nothing_changes_nothing() -> None:
  """Adding an examinee with every response missing leaves the estimates and the log-likelihood as they were."""
  with_blank = np.vstack([_LSAT7, np.full(5, np.nan)])
  calibrator = MMLCalibrator(tolerance=1e-8)
  before, after = calibrator.calibrate(_LSAT7), calibrator.calibrate(with_blank)
  np.testing.assert_allclose(after.estimates, before.estimates, atol=1e-12)
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
  assert np.all(calibration.estimates[:, 0] == 1.3)


@pytest.mark.parametrize("acceleration", list(Acceleration))
@pytest.mark.parametrize("max_cycles", [1, 2, 3])
def test_running_out_of_cycles_is_flagged(max_cycles: int, acceleration: Acceleration) -> None:
  """A calibration that runs out of cycles returns its estimates flagged as not converged, and warns.

  SQUAREM's first iteration on LSAT-7 runs two cycles, extrapolates, and runs a third from the extrapolation, so the
  three limits stop it after each kind of cycle; a calibration that runs past its limit fails.
  """
  with pytest.warns(ConvergenceWarning, match=f"{max_cycles} cycles"):
    calibration = MMLCalibrator(max_cycles=max_cycles, acceleration=acceleration).calibrate(_LSAT7)
  assert not calibration.converged
  assert calibration.cycles == max_cycles
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
  np.testing.assert_allclose(started.estimates, default.estimates, atol=1e-7)


@pytest.mark.parametrize(
  ("matrix", "message"),
  [
    (np.array([1.0, 0.0]), "must be a matrix"),
    (np.empty((3, 0)), "at least one examinee and one item"),
    (np.array([[1.0, 2.0], [0.0, 1.0]]), r"responses\[0, 1\] is 2.0"),
    (np.array([[1.0, np.inf], [0.0, 1.0]]), r"responses\[0, 1\] is inf"),
    (np.array([[1.0, np.nan], [0.0, np.nan]]), r"columns \[1\]"),
    (np.array([[1.0, 1.0], [0.0, 1.0]]), r"columns \[1\] were answered correctly by every examinee"),
  ],
  ids=["one-dimensional", "no items", "value 2", "infinite", "unanswered item", "all correct"],
)
def test_invalid_responses(matrix: np.ndarray, message: str) -> None:
  """A matrix not of 0, 1 and NaN, with no item, or with an item no one answered or none can estimate, is refused."""
  with pytest.raises(ValueError, match=message):
    MMLCalibrator().calibrate(matrix)


def test_responses_that_are_not_numbers_are_refused() -> None:
  """A matrix of strings is refused with a TypeError, rather than read as the numbers it spells."""
  with pytest.raises(TypeError, match="responses must be Booleans"):
    MMLCalibrator().calibrate(np.array([["1", "0"], ["0", "1"]]))


def test_a_miskeyed_item_is_reported_and_left_out_of_the_bank() -> None:
  """An item whose correct answers fall as ability rises keeps its negative slope estimate, and the bank leaves it out.

  A warning names the item and points at the caller of ``calibrate``, and the bank's items keep the names of their
  columns, so a bank that renumbered them, an error that discarded the estimates, a bank holding the miskeyed item,
  or a warning pointing inside irtlib fails.
  """
  rng = np.random.default_rng(3)
  items = _two_pl_items(np.array([1.0, 1.5, -1.0, 1.2]), np.zeros(4))
  matrix = draw_responses(rng.normal(size=2000), items, rng=rng)
  with pytest.warns(RuntimeWarning, match=r"discrimination <= 0 for 2 \('item_2'\).*miskeyed item") as caught:
    calibration = MMLCalibrator().calibrate(matrix)
  assert caught[0].filename == __file__
  assert calibration.estimates.shape == (4, 4)
  assert -1.3 < calibration.estimates[2, 0] < -0.7
  assert calibration.items is not None
  assert calibration.items.names == ("item_0", "item_1", "item_3")
  np.testing.assert_array_equal(calibration.items.items, calibration.estimates[[0, 1, 3]])


@pytest.mark.parametrize(
  ("arguments", "message"),
  [
    ({"model": "5PL"}, "not a valid IrtModel"),
    ({"priors": Priors(guessing=(-1.4, 0.5))}, "guessing prior needs the 3PL or 4PL"),
    ({"model": "3PL", "priors": Priors(upper=(2.2, 0.5))}, "upper prior needs the 4PL"),
    ({"fixed_slope": 1.0, "priors": Priors(slope=(0.0, 0.5))}, "slope prior needs an estimated slope"),
    ({"fixed_slope": 0.0}, "fixed_slope"),
    ({"tolerance": 0.0}, "tolerance"),
    ({"max_cycles": 0}, "max_cycles"),
    ({"acceleration": "Ramsay"}, "not a valid Acceleration"),
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
  with pytest.raises(ValueError, match="one item per column"):
    marginal_log_likelihood(_LSAT7, three_items.items)


@pytest.mark.parametrize(
  ("prior", "density"),
  [
    (_PriorTerm(0, 0.2, 0.5), lambda x: lognorm.logpdf(x, s=0.5, scale=np.exp(0.2))),
    (_PriorTerm(1, -0.3, 2.0), lambda x: norm.logpdf(x, -0.3, 2.0)),
    (_PriorTerm(2, float(logit(0.2)), 0.5), lambda x: norm.logpdf(x, logit(0.2), 0.5)),
  ],
  ids=["lognormal slope", "normal intercept", "normal logit c"],
)
def test_prior_terms_match_the_densities(prior: _PriorTerm, density: Callable[[np.ndarray], np.ndarray]) -> None:
  """Each log-prior equals scipy's log-density, and its derivatives the central differences of that density.

  The lognormal slope prior keeps the -log a term of the lognormal density; mirt 1.48's Hessian of it has the
  opposite sign on its last term (source R/utils.R:1430), which this test would catch here.
  """
  values = np.array([[0.7, -1.1, -1.8, np.inf], [1.4, 0.4, -1.2, np.inf], [2.3, 1.9, -0.6, np.inf]])
  column = values[:, prior.column]
  log_prior, gradient, curvature = _prior_terms(values, (prior,), np.ones(values.shape, dtype=bool))
  np.testing.assert_allclose(log_prior, np.sum(density(column)), rtol=1e-12)
  step = 1e-5
  first = (density(column + step) - density(column - step)) / (2 * step)
  second = (density(column + step) - 2 * density(column) + density(column - step)) / step**2
  np.testing.assert_allclose(gradient[:, prior.column], first, rtol=1e-7)
  np.testing.assert_allclose(curvature[:, prior.column], second, rtol=1e-4)


def test_estimates_are_the_posterior_mode() -> None:
  """With priors on the slope and intercept, the estimates make the log-posterior's gradient vanish.

  The gradient is a central difference of the public marginal log-likelihood plus scipy's log-densities of the
  priors, independent of the calibrator's own score.
  """
  matrix = responses(_MML_EM["data"]["simulated"])
  priors = Priors(slope=(0.0, 0.5), intercept=(0.0, 2.0))
  calibration = MMLCalibrator(priors=priors, tolerance=1e-10).calibrate(matrix)
  slope = calibration.estimates[:, 0]
  intercept = -slope * calibration.estimates[:, 1]

  def log_posterior(slope: np.ndarray, intercept: np.ndarray) -> float:
    items = np.column_stack([slope, -intercept / slope, np.zeros_like(slope), np.ones_like(slope)])
    prior = np.sum(lognorm.logpdf(slope, s=0.5)) + np.sum(norm.logpdf(intercept, 0.0, 2.0))
    return marginal_log_likelihood(matrix, items) + prior

  step = 1e-5
  gradient = []
  for item in range(slope.size):
    shift = np.zeros_like(slope)
    shift[item] = step
    gradient.extend([
      (log_posterior(slope + shift, intercept) - log_posterior(slope - shift, intercept)) / (2 * step),
      (log_posterior(slope, intercept + shift) - log_posterior(slope, intercept - shift)) / (2 * step),
    ])
  np.testing.assert_allclose(gradient, 0.0, atol=1e-4)


def test_intercept_prior_keeps_an_extreme_item_finite() -> None:
  """An item every examinee answered correctly is refused without an intercept prior and estimated with one."""
  matrix = _LSAT7.copy()
  matrix[:, 0] = 1.0
  with pytest.raises(ValueError, match="no finite estimate"):
    MMLCalibrator().calibrate(matrix)
  calibration = MMLCalibrator(priors=Priors(intercept=(0.0, 2.0))).calibrate(matrix)
  assert np.all(np.isfinite(calibration.estimates))
  assert calibration.estimates[0, 1] < calibration.estimates[1:, 1].min()


def test_recovery_of_three_pl_parameters() -> None:
  """3PL estimates with a prior on the logit of c, from 10,000 simulated examinees, recover the generating items.

  Slopes within 25%, difficulties within 0.25 and guessing parameters within 0.08, which a guessing parameter
  estimated on the wrong scale, or a prior pulling the wrong way, would exceed.
  """
  rng = np.random.default_rng(20261130)
  slopes = rng.uniform(1.0, 2.0, 8)
  difficulties = rng.uniform(-1.0, 1.5, 8)
  guessing = rng.uniform(0.1, 0.3, 8)
  theta = rng.normal(size=10_000)
  matrix = draw_responses(theta, np.column_stack([slopes, difficulties, guessing, np.ones(8)]), rng=rng)
  calibration = MMLCalibrator("3PL", priors=Priors(guessing=(float(logit(0.2)), 0.5))).calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 0], slopes, rtol=0.25)
  np.testing.assert_allclose(calibration.estimates[:, 1], difficulties, atol=0.25)
  np.testing.assert_allclose(calibration.estimates[:, 2], guessing, atol=0.08)


@pytest.mark.parametrize(
  "arguments", [{"intercept": (0.0, 0.0)}, {"slope": (float("nan"), 0.5)}, {"guessing": (0.0, float("inf"))}]
)
def test_invalid_priors(arguments: dict[str, tuple[float, float]]) -> None:
  """A prior whose mean is not finite, or whose standard deviation is not finite and positive, is refused."""
  with pytest.raises(ValueError, match="prior must be a finite mean"):
    Priors(**arguments)


def test_a_shared_slope_carries_its_prior_once() -> None:
  """With the 1PL's shared slope, the slope prior counts once, not once per item.

  Five copies of the prior would pull the slope as five priors do; the estimate equals the maximum of the marginal
  log-likelihood plus one lognormal log-density, found here by a one-dimensional search.
  """
  calibration = MMLCalibrator("1PL", priors=Priors(slope=(0.5, 0.3)), tolerance=1e-10).calibrate(_LSAT7)
  slope = calibration.estimates[0, 0]
  intercept = -slope * calibration.estimates[:, 1]

  def log_posterior(candidate: float) -> float:
    items = np.column_stack([np.full(5, candidate), -intercept / candidate, np.zeros(5), np.ones(5)])
    return marginal_log_likelihood(_LSAT7, items) + lognorm.logpdf(candidate, s=0.3, scale=np.exp(0.5))

  step = 1e-5
  derivative = (log_posterior(slope + step) - log_posterior(slope - step)) / (2 * step)
  assert abs(derivative) < 1e-3


def test_four_pl_estimates_are_the_posterior_mode() -> None:
  """4PL estimates with priors on the logits of c and d make the log-posterior's gradient in those logits vanish.

  mirt 1.48 stops short of this mode (the parity tests' expected failures), so the check is independent: central
  differences of the public marginal log-likelihood plus scipy's normal log-densities of the priors.
  """
  data = load("mml_priors")["data"]["simulated_4pl"]
  matrix = responses(data)
  center_c, center_d = float(logit(0.2)), float(logit(0.9))
  priors = Priors(guessing=(center_c, 0.2), upper=(center_d, 0.2))
  calibration = MMLCalibrator("4PL", priors=priors, tolerance=1e-8, max_cycles=5000).calibrate(matrix)
  items = calibration.estimates.copy()

  def log_posterior(candidate: FloatArray) -> float:
    prior = np.sum(norm.logpdf(logit(candidate[:, 2]), center_c, 0.2)) + np.sum(
      norm.logpdf(logit(candidate[:, 3]), center_d, 0.2)
    )
    return marginal_log_likelihood(matrix, candidate) + prior

  step = 1e-5
  gradient = []
  for item in range(items.shape[0]):
    for column in (2, 3):
      shifted = []
      for sign in (1.0, -1.0):
        candidate = items.copy()
        candidate[item, column] = expit(logit(items[item, column]) + sign * step)
        shifted.append(log_posterior(candidate))
      gradient.append((shifted[0] - shifted[1]) / (2 * step))
  np.testing.assert_allclose(gradient, 0.0, atol=1e-3)


def test_a_start_with_guessing_above_the_upper_asymptote_is_refused() -> None:
  """A 4PL start whose items keep a guessing parameter at or above the upper asymptote is refused, naming the start.

  A start's c of 0 becomes mirt's 0.15, which exceeds an upper asymptote of 0.1.
  """
  start = ItemBank(np.tile([1.0, 0.0, 0.0, 0.1], (5, 1)))
  with pytest.raises(ValueError, match="start must leave every item's guessing parameter below"):
    MMLCalibrator("4PL").calibrate(_LSAT7, start=start)
