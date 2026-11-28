"""Tests of the ability estimators of irtlib.ability.

Expected values come from hand-worked cases, closed forms and brute-force references written out here: the 4PL
likelihood and information, Warm's modified score, the posterior on a grid, dense grids searched for their maximum, and
the exact bias of an estimator by enumerating every response pattern.
"""

from __future__ import annotations

import functools
import math
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from irtlib import ability
from irtlib.ability import (
  AbilityEstimates,
  expected_a_posteriori,
  maximum_a_posteriori,
  maximum_likelihood,
  weighted_likelihood,
)
from irtlib.quadrature import GridPrior, QuadratureGrid, normal_log_prior, uniform_log_prior

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.quadrature import LogPrior
  from irtlib.typing import FloatArray

_NAN = np.nan
_ESTIMATORS = {
  "eap": expected_a_posteriori,
  "map": maximum_a_posteriori,
  "ml": maximum_likelihood,
  "wle": weighted_likelihood,
}
_SEARCHES = {name: _ESTIMATORS[name] for name in ("map", "ml", "wle")}

# Four 4PL items, guessing and slipping, and a response matrix with missing responses, a constant pattern and an
# examinee who answered nothing.
_ITEMS = np.array([[1.2, -1.0, 0.2, 1.0], [0.8, 0.0, 0.1, 0.95], [1.6, 0.7, 0.25, 1.0], [1.0, 1.5, 0.0, 0.9]])
_RESPONSES = np.array([
  [1.0, 1.0, 0.0, 1.0],
  [1.0, _NAN, 0.0, _NAN],
  [_NAN, 0.0, 1.0, 1.0],
  [1.0, 1.0, _NAN, 1.0],
  [_NAN, _NAN, _NAN, _NAN],
])


def _probability(theta: FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL probability, written out, every ability (rows) against every item (columns)."""
  a, b, c, d = items.T
  return c + (d - c) / (1.0 + np.exp(-a * (np.asarray(theta, dtype=float)[..., np.newaxis] - b)))


def _log_likelihood(theta: FloatArray, items: FloatArray, responses: FloatArray) -> FloatArray:
  """Return one examinee's log-likelihood at each ability, written out, over the items it answered.

  P and Q each come from the logistic of their own sign of the logit, so that neither is 1 - (a number close to 1).
  """
  answered = ~np.isnan(responses)
  a, b, c, d = items[answered].T
  logit = a * (np.asarray(theta, dtype=float)[..., np.newaxis] - b)
  p, q = c + (d - c) * expit(logit), (1 - d) + (d - c) * expit(-logit)
  return np.sum(np.where(responses[answered] == 1, np.log(p), np.log(q)), axis=-1)


def _information(theta: float, items: FloatArray) -> float:
  """Return the test information of 4PL items, the sum of P'^2 / (P Q), written out."""
  a, b, c, d = items.T
  logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
  p = c + (d - c) * logistic
  derivative = a * (d - c) * logistic * (1.0 - logistic)
  return float(np.sum(derivative**2 / (p * (1.0 - p))))


def _warm_modified_score(theta: float, items: FloatArray, responses: FloatArray) -> float:
  """Return Warm's (1989, p. 431, Equation 10) modified score, written out."""
  a, b, c, d = items.T
  logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
  p = c + (d - c) * logistic
  first = a * (d - c) * logistic * (1.0 - logistic)
  second = a * first * (1.0 - 2.0 * logistic)
  pq = p * (1.0 - p)
  information = np.sum(first**2 / pq)
  return float(np.sum((responses - p) * first / pq) + np.sum(first * second / pq) / (2.0 * information))


def _rows(responses: npt.ArrayLike) -> FloatArray:
  """Return a one-row response matrix."""
  return np.asarray(responses, dtype=float)[np.newaxis]


class TestEveryEstimator:
  """What the four estimators share: rows estimated apart, missing responses, the checks of their arguments."""

  @pytest.mark.parametrize("estimator", _ESTIMATORS.values(), ids=_ESTIMATORS.keys())
  def test_a_row_is_estimated_as_the_examinee_alone_on_the_items_it_answered(
    self, estimator: Callable[..., AbilityEstimates]
  ) -> None:
    """Each row gives the estimate of its examinee alone, its missing items left out of the call.

    An estimator that let one row's responses reach another's estimate, counted a missing response as incorrect or
    correct, or answered an unanswered row from its neighbour, fails.
    """
    together = estimator(_RESPONSES, _ITEMS)

    for row, responses in enumerate(_RESPONSES):
      answered = ~np.isnan(responses)
      alone = estimator(_rows(responses[answered]), _ITEMS[answered])

      np.testing.assert_allclose(together.theta[row], alone.theta[0], rtol=1e-12, atol=1e-14)
      np.testing.assert_allclose(together.standard_errors[row], alone.standard_errors[0], rtol=1e-12)
      assert together.evaluations[row] == alone.evaluations[0]
    assert together.theta.shape == together.standard_errors.shape == together.evaluations.shape == (len(_RESPONSES),)

  @pytest.mark.parametrize("estimator", _ESTIMATORS.values(), ids=_ESTIMATORS.keys())
  def test_no_estimate_comes_from_a_fallback(self, estimator: Callable[..., AbilityEstimates]) -> None:
    """Only a maximum likelihood fallback marks a row; a mask left uninitialized or True would mislabel estimates."""
    assert not estimator(_RESPONSES, _ITEMS).from_fallback.any()

  @pytest.mark.parametrize("estimator", _SEARCHES.values(), ids=_SEARCHES.keys())
  @pytest.mark.parametrize("tol", [0.0, -1e-6, math.nan])
  def test_a_tolerance_that_is_not_positive_is_rejected(
    self, estimator: Callable[..., AbilityEstimates], tol: float
  ) -> None:
    """A tolerance that is not positive fails before the search, not inside SciPy."""
    with pytest.raises(ValueError, match="tol must be positive"):
      estimator(_RESPONSES, _ITEMS, tol=tol)

  @pytest.mark.parametrize("estimator", _SEARCHES.values(), ids=_SEARCHES.keys())
  @pytest.mark.parametrize(
    "bounds",
    [
      pytest.param((3.0, -3.0), id="reversed"),
      pytest.param((1.0, 1.0), id="empty"),
      pytest.param((-math.inf, 6.0), id="infinite"),
      pytest.param((math.nan, 6.0), id="nan"),
    ],
  )
  def test_bounds_that_are_not_a_finite_interval_are_rejected(
    self, estimator: Callable[..., AbilityEstimates], bounds: tuple[float, float]
  ) -> None:
    """Bounds that are not a finite interval with low < high fail before the search, not inside SciPy."""
    with pytest.raises(ValueError, match="bounds must be finite with low < high"):
      estimator(_RESPONSES, _ITEMS, bounds=bounds)

  @pytest.mark.parametrize("estimator", _ESTIMATORS.values(), ids=_ESTIMATORS.keys())
  def test_items_of_the_wrong_shape_or_count_are_rejected(self, estimator: Callable[..., AbilityEstimates]) -> None:
    """Items that are not one (n, 4) row per column of the responses, or responses that are not 0, 1 or NaN, fail.

    Unchecked, a response matrix with fewer columns than items would be scored against the first items only.
    """
    with pytest.raises(ValueError, match="items must be an"):
      estimator(_RESPONSES, _ITEMS[:, :3])
    with pytest.raises(ValueError, match="one item per column"):
      estimator(_RESPONSES, _ITEMS[:3])
    with pytest.raises(ValueError, match="a response must be 0 or 1"):
      estimator(np.full((1, 4), 0.5), _ITEMS)

  @pytest.mark.parametrize(
    "estimator",
    [
      pytest.param(functools.partial(maximum_likelihood, bounds=(0.5, 2.0)), id="ml"),
      pytest.param(functools.partial(maximum_a_posteriori, bounds=(0.5, 2.0)), id="map"),
      pytest.param(functools.partial(weighted_likelihood, bounds=(0.5, 2.0)), id="wle"),
    ],
  )
  def test_searches_stay_inside_their_bounds(self, estimator: Callable[..., AbilityEstimates]) -> None:
    """Responses whose estimate is 0 on [-6, 6] are estimated at the lower bound of [0.5, 2].

    A correct answer to a 1PL item at b = 1 and an incorrect one to an item at b = -1 make the likelihood, the test
    information and the standard normal prior symmetric about 0 and decreasing beyond it, so the likelihood, the
    posterior and the weighted likelihood all peak at 0 and fall across [0.5, 2]. A search that ignored its bounds
    returned 0.
    """
    items = np.array([[1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]])

    assert estimator(_rows([1.0, 0.0]), items).theta[0] == pytest.approx(0.5, abs=1e-5)

  @pytest.mark.parametrize(
    ("estimator", "expected"),
    [
      pytest.param(functools.partial(maximum_likelihood, bounds=(-1.0, 1.0)), 1.0, id="ml"),
      pytest.param(
        functools.partial(expected_a_posteriori, grid=QuadratureGrid.uniform(n_nodes=3, low=-1.0, high=1.0)),
        np.exp(-0.5) * np.tanh(0.5) / (np.exp(-0.5) + 0.5),
        id="eap",
      ),
      pytest.param(
        functools.partial(maximum_a_posteriori, bounds=(-1.0, 1.0)), brentq(lambda t: t - expit(-t), 0.0, 1.0), id="map"
      ),
      pytest.param(functools.partial(weighted_likelihood, bounds=(-1.0, 1.0)), 1.0, id="wle"),
    ],
  )
  @pytest.mark.parametrize("correct", [1.0, 0.0], ids=["correct", "incorrect"])
  def test_a_constant_pattern_has_a_finite_estimate_inside_the_bounds(
    self, estimator: Callable[..., AbilityEstimates], expected: float, correct: float
  ) -> None:
    """One answer to a 1PL item at b = 0 is estimated inside [-1, 1]; an incorrect one mirrors a correct one.

    With bounds the maximum likelihood estimate is the bound the answer points to. The WLE's root, b +- ln 3, is beyond
    the bounds, so its estimate is the bound. MAP with the standard normal prior solves theta = sigma(-theta). EAP
    weighs the nodes -1, 0 and 1 by e^(-1/2) sigma(-1), 1/2 and e^(-1/2) sigma(1) after a correct answer, a mean of
    e^(-1/2) tanh(1/2) / (e^(-1/2) + 1/2). An estimator that ignored its bounds returned +-1.10 (WLE) or +-inf (ML).
    """
    theta = estimator(_rows([correct]), np.array([[1.0, 0.0, 0.0, 1.0]])).theta[0]

    assert theta == pytest.approx(expected if correct else -expected, abs=1e-5)


class TestMaximumLikelihood:
  """The maximum likelihood estimate, over the extended real line or within bounds, and its fallback."""

  @pytest.mark.parametrize(
    "items",
    [
      pytest.param(
        np.array([
          [1.2, -1.5, 0.2, 1.0],
          [0.8, -0.6, 0.0, 1.0],
          [1.6, 0.0, 0.15, 1.0],
          [1.0, 0.7, 0.0, 0.92],
          [2.0, 1.3, 0.1, 1.0],
          [1.4, -0.3, 0.0, 0.95],
        ]),
        id="guessing_and_slipping",
      ),
      pytest.param(
        np.array([
          [1.2, -1.5, 0.0, 1.0],
          [0.8, -0.6, 0.0, 1.0],
          [1.6, 0.0, 0.0, 1.0],
          [1.0, 0.7, 0.0, 1.0],
          [2.0, 1.3, 0.0, 1.0],
          [1.4, -0.3, 0.0, 1.0],
        ]),
        id="2PL",
      ),
    ],
  )
  def test_estimate_is_the_maximum_of_the_likelihood(self, items: FloatArray) -> None:
    """Each examinee's estimate is the argmax of its log-likelihood, written out, on a grid 1e-4 apart.

    Some responses are missing. On items that guess and slip the search refines the best of its points; on 2PL items,
    whose log-likelihood is concave, it skips them. A search that read missing responses as incorrect, stopped at a
    local maximum, or refined toward a zero of the wrong derivative misses the grid's argmax.
    """
    responses = np.array([
      [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
      [1.0, _NAN, 1.0, 0.0, _NAN, 0.0],
      [0.0, 1.0, 1.0, _NAN, 0.0, _NAN],
    ])
    grid = np.linspace(-6.0, 6.0, 120_001)

    estimates = maximum_likelihood(responses, items)

    for row, theta in zip(responses, estimates.theta, strict=True):
      assert theta == pytest.approx(grid[np.argmax(_log_likelihood(grid, items, row))], abs=1e-4)

  @pytest.mark.parametrize(
    ("items", "responses", "infinite"),
    [
      pytest.param(
        [[1.0, 2.0, 0.25, 1.0], [1.5, -1.0, 0.0, 1.0]], [1.0, 0.0], False, id="a_maximum_above_the_limit_at_-5.6"
      ),
      pytest.param(
        [
          [1.97, -2.537, 0.0, 0.85],
          [1.613, -0.069, 0.25, 0.85],
          [2.319, -1.723, 0.25, 0.95],
          [1.817, -2.204, 0.0, 0.85],
          [1.401, 0.036, 0.25, 1.0],
        ],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        False,
        id="a_maximum_beyond_-6_above_the_limit",
      ),
      pytest.param(
        [[1.17, -0.38, 0.0, 0.85], [1.78, 2.96, 0.1, 0.85]], [0.0, 1.0], False, id="a_maximum_beyond_6_above_the_limit"
      ),
      pytest.param(
        [[2.076, 2.492, 0.1, 0.85], [2.031, 1.022, 0.25, 1.0]], [1.0, 0.0], True, id="rising_to_the_limit_at_-inf"
      ),
      pytest.param(
        [
          [0.914, 0.166, 0.1, 0.95],
          [0.475, 0.084, 0.25, 0.95],
          [2.325, -2.052, 0.25, 0.85],
          [1.826, -1.874, 0.0, 0.85],
        ],
        [0.0, 0.0, 1.0, 0.0],
        True,
        id="a_local_maximum_below_the_limit_at_-inf",
      ),
      pytest.param(
        [[1.25, 1.313, 0.1, 0.85], [0.353, -1.438, 0.1, 1.0], [0.965, -1.703, 0.1, 1.0], [2.072, 2.647, 0.0, 0.95]],
        [0.0, 1.0, 1.0, 1.0],
        True,
        id="rising_to_the_limit_at_+inf",
      ),
      pytest.param(
        [[2.362, 2.953, 0.1, 0.85], [1.491, 1.296, 0.25, 0.85]], [1.0, 0.0], False, id="a_maximum_between_flat_tails"
      ),
    ],
  )
  def test_estimate_maximizes_the_likelihood_over_the_extended_real_line(
    self, items: list[list[float]], responses: list[float], infinite: bool
  ) -> None:
    """Responses that are neither all correct nor all incorrect get the maximizer over the extended real line.

    The reference compares the largest log-likelihood on a grid 1e-3 apart over [-120, 120] with the limits at -inf
    and +inf, the products of c or 1 - c, and of d or 1 - d, written out. The cases were found by a random search over
    3PL and 4PL patterns. A search that stopped at the end of [-6, 6] when the likelihood was below the limit there
    returned -inf for the second case, whose maximum at -8.39 is above the limit by 6e-6, and +inf for the third, whose
    maximum at 7.01 is above it by 3e-4; one that kept an interior maximum below a limit returned it for the fifth;
    Brent's search alone on [-6, 6] missed the maximum at 4.61 of the last.
    """
    item_matrix, pattern = np.array(items), np.array(responses)
    grid = np.linspace(-120.0, 120.0, 240_001)
    values = _log_likelihood(grid, item_matrix, pattern)
    _, _, c, d = item_matrix.T
    with np.errstate(divide="ignore"):
      low = np.sum(np.log(np.where(pattern == 1, c, 1 - c)))
      high = np.sum(np.log(np.where(pattern == 1, d, 1 - d)))
    expected = (math.inf if high > low else -math.inf) if infinite else float(grid[np.argmax(values)])

    estimate = maximum_likelihood(_rows(pattern), item_matrix)

    assert (max(low, high) >= values.max()) == infinite
    assert estimate.theta[0] == pytest.approx(expected, abs=1e-3)
    assert estimate.standard_errors[0] == (
      math.inf if infinite else pytest.approx(_information(estimate.theta[0], item_matrix) ** -0.5, rel=1e-12)
    )

  def test_a_search_that_cannot_settle_warns_from_the_callers_line(self) -> None:
    """Items of discrimination 1e-21 put the maximum near -6.9e20, beyond the reach of 64 moves; the search warns.

    The two items act as one, so the likelihood P (1 - P) peaks where P = 1/2, at a logit of -ln 2. A search that ran
    out of moves returned a point short of it, -2.2e20, as if it were the maximum.
    """
    items = np.array([[1e-21, -1.0, 0.25, 1.0], [1e-21, 1.0, 0.25, 1.0]])

    def estimate() -> AbilityEstimates:
      return maximum_likelihood(_rows([0.0, 1.0]), items)

    with pytest.warns(RuntimeWarning, match="without settling") as caught:
      estimates = estimate()

    call_line = estimate.__code__.co_firstlineno + 1
    assert {(warning.filename, warning.lineno) for warning in caught} == {(__file__, call_line)}
    assert -6.9e20 < estimates.theta[0] < -1e20

  @pytest.mark.parametrize(("correct", "expected"), [(1.0, math.inf), (0.0, -math.inf)], ids=["correct", "incorrect"])
  def test_a_constant_pattern_is_estimated_at_its_infinity(self, correct: float, expected: float) -> None:
    """Responses all correct get +inf and all incorrect -inf, with an infinite standard error and no evaluation.

    The likelihood of such responses rises (falls) with the ability whatever the items, so its supremum is reached
    only at the infinity. A search returned the end of its range instead, a Dodd step a finite value, and a standard
    error from an information at the estimate NaN or 0.
    """
    estimates = maximum_likelihood(np.array([[correct, _NAN, correct, correct]]), _ITEMS)

    assert (estimates.theta[0], estimates.standard_errors[0], estimates.evaluations[0]) == (expected, math.inf, 0)

  @pytest.mark.parametrize(("correct", "expected"), [(1.0, 2.5), (0.0, -1.5)], ids=["correct", "incorrect"])
  def test_bounds_give_a_constant_pattern_the_bound_it_points_to(self, correct: float, expected: float) -> None:
    """With bounds, responses all correct get the upper bound and all incorrect the lower, with the Fisher SE.

    The standard error is 1 / sqrt(I) at the bound, I written out. A search would return a point within its
    tolerance of the bound, and an estimate without bounds the infinity.
    """
    estimates = maximum_likelihood(np.array([[correct, correct, _NAN, correct]]), _ITEMS, bounds=(-1.5, 2.5))

    assert estimates.theta[0] == expected
    assert estimates.standard_errors[0] == pytest.approx(_information(expected, _ITEMS[[0, 1, 3]]) ** -0.5, rel=1e-12)
    assert estimates.evaluations[0] == 0

  @pytest.mark.parametrize("estimator", [maximum_likelihood, weighted_likelihood], ids=["ml", "wle"])
  def test_an_examinee_who_answered_nothing_has_no_estimate(self, estimator: Callable[..., AbilityEstimates]) -> None:
    """A row with every response missing gets NaN for the estimate and inf for its standard error, after no evaluation.

    Its likelihood is 1 everywhere, so every ability maximizes it, and it carries no information. The search returned
    a bound of its range, and a standard error of NaN or from the information of items not answered would be wrong.
    """
    estimates = estimator(np.full((1, 4), _NAN), _ITEMS)

    assert np.isnan(estimates.theta[0])
    assert estimates.standard_errors[0] == math.inf
    assert estimates.evaluations[0] == 0

  def test_standard_error_is_the_inverse_root_of_the_information_at_the_estimate(self) -> None:
    """Identical 1PL items at b = 0, half answered correctly, give theta = 0 and SE = 2 / sqrt(n).

    The likelihood P^k Q^k peaks where P = 1/2, at 0, where each item's information is 1/4: 16 answered items give 0.5
    and 4 give 1. An SE that was the variance, or from the information of every item rather than the answered ones,
    fails.
    """
    items = np.tile([1.0, 0.0, 0.0, 1.0], (16, 1))
    responses = np.array([[1.0, 0.0] * 8, [1.0, 0.0] * 2 + [_NAN] * 12])

    estimates = maximum_likelihood(responses, items)

    np.testing.assert_allclose(estimates.theta, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(estimates.standard_errors, [0.5, 1.0], rtol=1e-9)

  def test_tolerance_is_honoured(self) -> None:
    """A fine tolerance gives the maximum SciPy finds at 1e-12 on the written-out likelihood; a coarse one is cheaper.

    Near the maximum the log-likelihood changes by less than its rounding within about 1e-8, so the two agree to 1e-7.
    A search that ignored tol stopped at SciPy's default of 1e-5 whatever was asked, and took as many evaluations.
    """
    items = np.array([[1.2, -1.0, 0.0, 1.0], [0.8, 0.3, 0.0, 1.0], [1.6, 0.9, 0.0, 1.0], [1.1, -0.4, 0.0, 1.0]])
    pattern = np.array([1.0, 1.0, 0.0, 0.0])
    reference = minimize_scalar(
      lambda theta: -float(_log_likelihood(np.array(theta), items, pattern)),
      bounds=(-2.0, 2.0),
      method="bounded",
      options={"xatol": 1e-12},
    ).x

    coarse, fine = (maximum_likelihood(_rows(pattern), items, tol=tol) for tol in (1e-2, 1e-10))

    assert fine.theta[0] == pytest.approx(reference, abs=1e-7)
    assert coarse.theta[0] == pytest.approx(reference, abs=1e-2)
    assert coarse.evaluations[0] < fine.evaluations[0]

  @pytest.mark.parametrize("theta", [-4.0, -1.3, 0.0, 0.8, 3.5])
  def test_the_refinements_derivatives_are_the_log_likelihoods(self, theta: float) -> None:
    """The search's first and second derivatives equal central differences of the written-out log-likelihood.

    The items guess and slip. A wrong first derivative moves the estimate; a wrong second one only slows the search,
    whose bracket still converges, so only this comparison catches it.
    """
    items = np.array([[1.2, -1.5, 0.2, 1.0], [0.8, -0.6, 0.0, 0.9], [1.6, 0.0, 0.15, 0.95], [2.0, 1.3, 0.1, 1.0]])
    responses = np.array([1.0, 0.0, 1.0, 0.0])
    h = 1e-4
    below, at, above = _log_likelihood(np.array([theta - h, theta, theta + h]), items, responses)

    first, second = ability._Examinee.of(responses, items).derivatives(theta)  # noqa: SLF001

    assert first == pytest.approx((above - below) / (2 * h), rel=1e-6)
    assert second == pytest.approx((above - 2 * at + below) / h**2, rel=1e-5)

  def test_an_infinite_curvature_does_not_end_the_refinement(self) -> None:
    """A derivative of 5 with a curvature of -inf is no zero: the refinement bisects up to the bracket's upper end.

    Dividing the slope by the infinite curvature gave a step of 0, which ended the search at once where the slope was 5.
    """
    theta, evaluations = ability._newton(lambda _: (5.0, -math.inf), 0.0, -1.0, 1.0, 1e-6)  # noqa: SLF001

    assert theta == pytest.approx(1.0, abs=1e-6)
    assert evaluations > 1

  @pytest.mark.parametrize(("slope", "points"), [(1.0, 101), (5.0, 201)])
  def test_evaluations_count_the_search_points_and_the_refinement(self, slope: float, points: int) -> None:
    """Under the 3PL the count is the search points, each evaluation of the derivatives, and the refined point.

    The likelihood below falls across the whole range: the guessed correct answer to the far too hard item is worth c
    everywhere, and the incorrect answer to the easy one ever less. So the best point is the lower bound, its bracket
    reaches no lower, and one evaluation of the derivatives, pointing down, ends the refinement there. The points are
    101 for slopes up to 2.5 and ceil(40 a) + 1 above, 201 for a slope of 5. A count of the refinement alone, without
    the search points, or with points that ignore the slope, fails.
    """
    items = np.array([[slope, 10.0, 0.2, 1.0], [slope, -8.0, 0.2, 1.0]])

    estimates = maximum_likelihood(_rows([1.0, 0.0]), items, bounds=(-6.0, 6.0))

    assert estimates.theta[0] == -6.0
    assert estimates.evaluations[0] == points + 1 + 1

  def test_steep_items_narrow_the_spacing_of_the_search_points(self) -> None:
    """A peak narrower than 0.12, between two steep items, is the estimate: the argmax of the written-out likelihood.

    A correct answer to the item of slope 24.8 at -1.44 and an incorrect one to the item of slope 19.4 at -1.5 meet in a
    peak about 0.06 wide near -1.48, 0.07 above the plateau below it in log-likelihood. Points 0.12 apart step over it
    and stop at -6; spaced by 0.3 / 24.8 they find it.
    """
    items = np.array([[15.8, 2.89, 0.18, 1.0], [24.8, -1.44, 0.14, 1.0], [19.4, -1.5, 0.14, 1.0]])
    pattern = np.array([0.0, 1.0, 0.0])
    fine = np.linspace(-6.0, 6.0, 120_001)

    estimates = maximum_likelihood(_rows(pattern), items, bounds=(-6.0, 6.0))

    assert estimates.theta[0] == pytest.approx(fine[np.argmax(_log_likelihood(fine, items, pattern))], abs=1e-4)

  def test_a_concave_likelihood_skips_the_search_points(self) -> None:
    """Without guessing or slipping the log-likelihood is concave, and a maximum beyond a bound costs two evaluations.

    The two 2PL items lie far below the range, so the derivative at the lower bound already points down: the estimate
    is that bound, after one evaluation of the derivatives and one of the log-likelihood there. A search that still
    evaluated the 101 points counts 103.
    """
    items = np.array([[1.0, -10.0, 0.0, 1.0], [1.0, -9.5, 0.0, 1.0]])

    estimates = maximum_likelihood(_rows([1.0, 0.0]), items, bounds=(-6.0, 6.0))

    assert estimates.theta[0] == -6.0
    assert estimates.evaluations[0] == 2


class TestFallback:
  """The estimator a maximum likelihood estimate falls back to where it is not finite."""

  _EAP = staticmethod(functools.partial(expected_a_posteriori, log_prior=normal_log_prior(0.0, 1.0)))

  def test_a_fallback_replaces_exactly_the_estimates_that_are_not_finite(self) -> None:
    """Rows whose estimate is infinite or NaN take the fallback's estimate, standard error and count; others are kept.

    The constant pattern and the unanswered row of the matrix are replaced by EAP on those rows alone, and the mask
    marks them. A fallback applied to every row, or to the infinite ones but not the unanswered one, or that shifted the
    other rows' values, fails.
    """
    plain = maximum_likelihood(_RESPONSES, _ITEMS)
    replaced = ~np.isfinite(plain.theta)
    substitute = self._EAP(_RESPONSES[replaced], _ITEMS)

    estimates = maximum_likelihood(_RESPONSES, _ITEMS, fallback=self._EAP)

    np.testing.assert_array_equal(estimates.from_fallback, [False, False, False, True, True])
    np.testing.assert_array_equal(estimates.from_fallback, replaced)
    for field in ("theta", "standard_errors", "evaluations"):
      np.testing.assert_array_equal(getattr(estimates, field)[~replaced], getattr(plain, field)[~replaced])
      np.testing.assert_array_equal(getattr(estimates, field)[replaced], getattr(substitute, field))

  def test_a_fallback_is_called_once_on_the_rows_it_replaces(self) -> None:
    """The fallback sees only the rows to replace, and the items, in one call, and is not called when none is needed.

    Calling it row by row, or with the whole matrix, fails; so does calling it for responses whose estimates are all
    finite.
    """
    calls: list[tuple[FloatArray, FloatArray]] = []

    def recording(responses: FloatArray, items: FloatArray) -> AbilityEstimates:
      calls.append((responses, items))
      return self._EAP(responses, items)

    maximum_likelihood(_RESPONSES, _ITEMS, fallback=recording)
    maximum_likelihood(_RESPONSES[:3], _ITEMS, fallback=recording)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], _RESPONSES[3:])
    np.testing.assert_array_equal(calls[0][1], _ITEMS)

  def test_a_fallbacks_warnings_name_the_callers_row_and_line(self) -> None:
    """A warning from the fallback names the examinee's row in the caller's matrix, from the caller's line.

    The fallback sees only the rows it replaces, here the third alone, after a wrong answer to a steep item that
    guesses, where the WLE's ranking integral does not converge. Passed through, its warning named row 0 of that
    sub-matrix and pointed at the line inside irtlib that called the fallback.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0], [200.0, 0.0, 0.2, 1.0]])
    responses = np.array([[1.0, 0.0], [1.0, 0.0], [_NAN, 0.0]])

    def estimate() -> AbilityEstimates:
      return maximum_likelihood(responses, items, fallback=weighted_likelihood)

    with pytest.warns(IntegrationWarning, match=r"^examinee in row 2: .*subdivisions") as caught:
      estimates = estimate()

    call_line = estimate.__code__.co_firstlineno + 1
    assert {(warning.filename, warning.lineno) for warning in caught} == {(__file__, call_line)}
    np.testing.assert_array_equal(estimates.from_fallback, [False, False, True])

  def test_bounds_and_a_fallback_together_are_rejected(self) -> None:
    """With bounds every estimate is finite, so a fallback would never apply; the two together are an error."""
    with pytest.raises(ValueError, match="bounds and fallback cannot be given together"):
      maximum_likelihood(_RESPONSES, _ITEMS, bounds=(-4.0, 4.0), fallback=self._EAP)

  def test_a_fallback_that_returns_another_number_of_estimates_is_rejected(self) -> None:
    """A fallback that answers for the whole matrix instead of the rows it was given would scatter its estimates."""
    with pytest.raises(ValueError, match="estimates for 2 rows"):
      maximum_likelihood(_RESPONSES, _ITEMS, fallback=lambda _, items: self._EAP(_RESPONSES, items))


class TestMaximumAPosteriori:
  """The posterior mode within the bounds, and its standard error."""

  @pytest.mark.parametrize(
    "items",
    [
      pytest.param(
        np.array([[1.2, -1.5, 0.2, 1.0], [0.8, -0.5, 0.0, 1.0], [1.6, 0.5, 0.25, 0.95], [1.0, 1.5, 0.1, 1.0]]),
        id="3PL_4PL",
      ),
      pytest.param(
        np.array([[1.2, -1.5, 0.0, 1.0], [0.8, -0.5, 0.0, 1.0], [1.6, 0.5, 0.0, 1.0], [1.0, 1.5, 0.0, 1.0]]), id="2PL"
      ),
    ],
  )
  def test_estimate_maximizes_the_log_posterior(self, items: FloatArray) -> None:
    """Every response pattern's estimate is the argmax of the log-likelihood plus the log-prior on a dense grid.

    The reference is written out, with the prior N(0.5, 0.8^2), for every pattern of answers to four items, all correct
    and all incorrect included, and with each item missing in turn. On 3PL and 4PL items the search refines the best of
    its points with the prior's derivatives; on 2PL items the log-posterior is concave and the search skips them. A
    prior dropped, scaled wrongly or added with the wrong sign, in the posterior or in its derivatives, fails.
    """
    mean, sd = 0.5, 0.8
    patterns = np.array(list(product([0.0, 1.0], repeat=4)))
    missing = np.where(np.eye(4, dtype=bool)[np.arange(16) % 4], _NAN, patterns)
    responses = np.vstack([patterns, missing])
    fine = np.linspace(-6.0, 6.0, 240_001)

    estimates = maximum_a_posteriori(responses, items, log_prior=normal_log_prior(mean, sd))

    for pattern, theta in zip(responses, estimates.theta, strict=True):
      log_posterior = _log_likelihood(fine, items, pattern) - 0.5 * ((fine - mean) / sd) ** 2
      assert theta == pytest.approx(fine[np.argmax(log_posterior)], abs=1e-4), pattern

  def test_evaluations_on_the_normal_priors_newton_path(self) -> None:
    """With no answered item and the prior N(0.3, 1) the log-posterior is -(theta - 0.3)^2 / 2, found in 5 evaluations.

    Its derivative points inward at both bounds, 2 evaluations; Newton's step from the middle, 0, lands on 0.3 exactly,
    and the derivative there, 0, ends the search, 2 more; the posterior at 0.3 is the fifth. Fails if the search
    evaluated its points, bisected, or counted the bounds or the posterior wrongly.
    """
    estimates = maximum_a_posteriori(
      _rows([_NAN]), np.array([[1.0, 0.0, 0.2, 1.0]]), log_prior=normal_log_prior(0.3, 1.0)
    )

    assert estimates.theta[0] == 0.3
    assert estimates.evaluations[0] == 5

  @pytest.mark.parametrize("sd", [1e160, 1e200])
  def test_a_normal_prior_wider_than_any_float_precision_is_flat(self, sd: float) -> None:
    """A normal prior so wide that 1 / sd^2 underflows leaves the likelihood's maximum within the bounds.

    The reference is the argmax of the written-out log-likelihood on a dense grid. Computing the prior's precision as
    sd**2 overflowed and raised OverflowError.
    """
    items = np.array([[1.2, -1.0, 0.2, 1.0], [0.8, 0.0, 0.1, 0.95], [1.6, 0.7, 0.25, 1.0]])
    pattern = np.array([1.0, 0.0, 1.0])
    fine = np.linspace(-6.0, 6.0, 120_001)

    estimates = maximum_a_posteriori(_rows(pattern), items, log_prior=normal_log_prior(0.3, sd))

    assert estimates.theta[0] == pytest.approx(fine[np.argmax(_log_likelihood(fine, items, pattern))], abs=1e-4)

  def test_a_normal_prior_narrower_than_any_float_precision_has_no_mass_between_points(self) -> None:
    """A normal prior so narrow that 1 / sd^2 overflows has no mass at the search points, the documented error.

    Its mean, 0.33, is none of the 201 points, where the log-density is -inf. Computing the precision as 1 / sd**2
    raised ZeroDivisionError.
    """
    with pytest.raises(ValueError, match="not finite at any of the 201 search points"):
      maximum_a_posteriori(_rows([1.0, 0.0]), _ITEMS[:2], log_prior=normal_log_prior(0.33, 1e-200))

  def test_evaluations_add_the_refinement_to_the_search_points(self) -> None:
    """Each estimate counts the 201 search points plus every posterior evaluation of the refinement.

    The prior below records how many points each call evaluates it at: the search evaluates the posterior at 201
    points in one call, the refinement at one point per call, and the standard error's curvature, which is not a
    posterior evaluation, at three. A count that left out the refinement or added the curvature's points fails.
    """
    sizes: list[int] = []
    normal = normal_log_prior()

    def recording_prior(theta: FloatArray) -> FloatArray:
      sizes.append(theta.size)
      return normal(theta)

    estimates = maximum_a_posteriori(_rows([1.0, 0.0, 1.0, 1.0]), _ITEMS, log_prior=recording_prior)

    assert sizes.count(201) == 1
    assert estimates.evaluations[0] == 201 + sizes.count(1)

  _HARD_ITEMS = np.array([[1.2, 2.0, 0.0, 1.0], [1.4, 2.5, 0.0, 1.0], [1.1, 3.0, 0.0, 1.0]])

  @pytest.mark.parametrize(
    ("low", "high", "responses", "expected"),
    [
      pytest.param(-1.0, 1.0, [1.0, 0.0, 1.0], 1.0, id="mode_at_upper_edge"),
      pytest.param(-1.0, 1.0, [0.0, 0.0, 0.0], -1.0, id="mode_at_lower_edge"),
      pytest.param(-6.0, -5.0, [1.0, 0.0, 1.0], -5.0, id="support_away_from_the_data"),
    ],
  )
  def test_estimate_stays_inside_a_uniform_prior_narrower_than_the_bounds(
    self, low: float, high: float, responses: list[float], expected: float
  ) -> None:
    """The estimate lies in the prior's support, at the edge the likelihood points to.

    Outside the support the negative log-posterior is +inf, a flat region a bounded search cannot cross, so searching
    the whole bounds at once returned 6.0 in every case.
    """
    estimates = maximum_a_posteriori(_rows(responses), self._HARD_ITEMS, log_prior=uniform_log_prior(low, high))

    assert estimates.theta[0] == pytest.approx(expected, abs=1e-5)

  def test_estimate_stays_inside_a_prior_narrower_than_a_search_step(self) -> None:
    """Both neighbours of the only search point in the support lie outside it; the estimate still lies inside."""
    estimates = maximum_a_posteriori(_rows([1.0, 0.0, 1.0]), self._HARD_ITEMS, log_prior=uniform_log_prior(-0.01, 0.01))

    assert -0.01 <= estimates.theta[0] <= 0.01

  @pytest.mark.parametrize(
    ("low", "high"),
    [pytest.param(7.0, 8.0, id="beyond_the_bounds"), pytest.param(0.01, 0.05, id="between_search_points")],
  )
  def test_a_prior_without_mass_at_any_search_point_is_an_error(self, low: float, high: float) -> None:
    """A prior with no support at any search point leaves no mode to find; a search would return an arbitrary point."""
    with pytest.raises(ValueError, match="must contain at least one of them"):
      maximum_a_posteriori(_rows([1.0, _NAN, _NAN]), self._HARD_ITEMS, log_prior=uniform_log_prior(low, high))

  @pytest.mark.parametrize("mean", [0.5, 0.03], ids=["off_the_search_points", "midway_between_two_points"])
  def test_without_responses_the_estimate_is_the_prior_mode_within_tol(self, mean: float) -> None:
    """With no response the estimate is the prior's mode to within tol, not to the 0.06 between search points.

    A mode midway between two search points makes them tie, which is not a flat posterior: it must be refined too.
    """
    estimates = maximum_a_posteriori(
      np.full((1, 3), _NAN), self._HARD_ITEMS, log_prior=normal_log_prior(mean=mean, sd=1.0), tol=1e-8
    )

    assert estimates.theta[0] == pytest.approx(mean, abs=1e-6)

  def test_estimate_is_the_higher_mode_of_a_bimodal_prior(self) -> None:
    """The search takes the taller of two separated modes, wherever a local search from the middle would stop."""

    def log_prior(theta: FloatArray) -> FloatArray:
      low_mode = np.log(0.3) - 0.5 * ((theta + 2.0) / 0.3) ** 2
      high_mode = np.log(0.7) - 0.5 * ((theta - 2.0) / 0.3) ** 2
      return np.logaddexp(low_mode, high_mode)

    estimates = maximum_a_posteriori(np.full((1, 3), _NAN), self._HARD_ITEMS, log_prior=log_prior)

    assert estimates.theta[0] == pytest.approx(2.0, abs=1e-5)

  def test_estimate_is_the_middle_of_a_flat_posterior(self) -> None:
    """A uniform prior with no responses has no unique mode, so the middle of the bounds is returned.

    Neither the prior nor the data carry information, so the SE is infinite, and no refinement runs, so only the 201
    search points count. An SE of 0 or NaN, or a count that added a refinement, fails.
    """
    estimates = maximum_a_posteriori(np.full((1, 3), _NAN), self._HARD_ITEMS, log_prior=uniform_log_prior())

    assert estimates.theta[0] == pytest.approx(0.0, abs=1e-12)
    assert (estimates.standard_errors[0], estimates.evaluations[0]) == (math.inf, 201)

  def test_standard_error_adds_the_normal_priors_precision_to_the_information(self) -> None:
    """With an N(0, 0.8^2) prior and one 4PL item, the SE is 1 / sqrt(I + 1 / 0.8^2) at the estimate.

    I is the item's information, written out. An SE that left out the prior (the Fisher SE, 1.80 here), added its
    variance instead of its precision, or subtracted the precision, fails.
    """
    items = np.array([[1.5, 0.5, 0.2, 0.95]])

    estimates = maximum_a_posteriori(_rows([1.0]), items, log_prior=normal_log_prior(mean=0.0, sd=0.8))

    expected = 1.0 / math.sqrt(_information(estimates.theta[0], items) + 1.0 / 0.8**2)
    assert estimates.standard_errors[0] == pytest.approx(expected, rel=1e-6)

  @pytest.mark.parametrize(
    ("low", "high", "responses", "edge"),
    [
      pytest.param(-6.0, 6.0, [1.0, 0.0, 0.0], None, id="inside_the_support"),
      pytest.param(-1.0, 1.0, [1.0, 1.0, 1.0], 1.0, id="at_the_edge_of_the_support"),
    ],
  )
  def test_standard_error_under_a_uniform_prior_is_the_fisher_one(
    self, low: float, high: float, responses: list[float], edge: float | None
  ) -> None:
    """A uniform prior adds nothing to the information, so the SE is 1 / sqrt(I) at the estimate, I written out.

    All correct answers to the three hard items put the mode at the upper edge of U(-1, 1), where the prior's
    curvature, a central difference, reaches outside the support and is not finite. Taking it as it is made the
    precision infinite and the SE 0; a prior term added inside the support fails too.
    """
    estimates = maximum_a_posteriori(_rows(responses), self._HARD_ITEMS, log_prior=uniform_log_prior(low, high))

    if edge is not None:
      assert estimates.theta[0] == pytest.approx(edge, abs=1e-5)
    assert estimates.standard_errors[0] == pytest.approx(
      1.0 / math.sqrt(_information(estimates.theta[0], self._HARD_ITEMS)), rel=1e-12
    )


class TestExpectedAPosteriori:
  """The posterior mean over a grid, and the posterior standard deviation."""

  @pytest.mark.parametrize(
    ("log_prior", "expected"),
    [
      pytest.param(None, 0.0, id="standard_normal"),
      pytest.param(normal_log_prior(mean=1.2, sd=0.5), 1.2, id="shifted"),
    ],
  )
  def test_without_responses_the_estimate_is_the_prior_mean(self, log_prior: LogPrior | None, expected: float) -> None:
    """With no response the estimate is the prior's mean; an estimator that ignored a given prior returns 0."""
    estimates = expected_a_posteriori(np.full((1, 4), _NAN), _ITEMS, log_prior=log_prior)

    assert estimates.theta[0] == pytest.approx(expected, abs=1e-12)

  @pytest.mark.parametrize("precomputed", [False, True], ids=["grid_and_log_prior", "grid_prior"])
  def test_estimate_is_the_posterior_mean_over_the_given_grid_prior_and_items(self, precomputed: bool) -> None:
    """The estimate weighs the given grid's nodes by the given prior and the answered items only.

    On the 3-node grid {-1, 0, 1} with a flat prior, a correct answer to item 1 (a = 1, b = 0) gives the nodes masses
    proportional to sigma(-1), 1/2 and sigma(1), which sum to 3/2, so the posterior mean is tanh(1/2) / 1.5 = 0.3081.
    Using item 0, which the examinee did not answer, gives 0.548, the default grid 2.759 and the default normal prior
    0.253, whether the grid and prior come apart or as one GridPrior.
    """
    items = np.array([[1.0, 3.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    grid = QuadratureGrid.uniform(n_nodes=3, low=-1.0, high=1.0)
    if precomputed:
      estimates = expected_a_posteriori(_rows([_NAN, 1.0]), items, grid_prior=GridPrior(grid, uniform_log_prior()))
    else:
      estimates = expected_a_posteriori(_rows([_NAN, 1.0]), items, grid=grid, log_prior=uniform_log_prior())

    assert estimates.theta[0] == pytest.approx(np.tanh(0.5) / 1.5, rel=1e-12)

  def test_a_grid_prior_with_a_grid_or_a_log_prior_is_rejected(self) -> None:
    """A GridPrior replaces the grid and the prior, so giving either with it is an error, not a silent choice."""
    grid_prior = GridPrior(QuadratureGrid.uniform(), normal_log_prior())
    responses, items = _rows([1.0]), np.array([[1.0, 0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="grid_prior replaces grid and log_prior"):
      expected_a_posteriori(responses, items, grid_prior=grid_prior, grid=QuadratureGrid.uniform(5))
    with pytest.raises(ValueError, match="grid_prior replaces grid and log_prior"):
      expected_a_posteriori(responses, items, grid_prior=grid_prior, log_prior=uniform_log_prior())

  def test_standard_error_is_the_posterior_standard_deviation(self) -> None:
    """The mean and the SE are those of the posterior over the given grid, computed here by brute force.

    The posterior weighs each node by the N(0.5, 1.2^2) prior and the 2PL and 4PL likelihood. An SE that was the
    posterior variance, the Fisher SE at the posterior mean, or the spread of the prior or of another grid fails.
    """
    items = np.array([[1.4, -0.5, 0.0, 1.0], [0.9, 0.8, 0.15, 0.95], [1.8, 0.2, 0.0, 1.0]])
    pattern = np.array([1.0, 0.0, 1.0])
    grid = QuadratureGrid.uniform(n_nodes=41, low=-4.0, high=4.0)
    mass = grid.weights * np.exp(-0.5 * ((grid.nodes - 0.5) / 1.2) ** 2 + _log_likelihood(grid.nodes, items, pattern))
    mass /= mass.sum()
    mean = mass @ grid.nodes

    estimates = expected_a_posteriori(_rows(pattern), items, grid=grid, log_prior=normal_log_prior(mean=0.5, sd=1.2))

    assert estimates.theta[0] == pytest.approx(mean, rel=1e-12)
    assert estimates.standard_errors[0] == pytest.approx(np.sqrt(mass @ (grid.nodes - mean) ** 2), rel=1e-12)

  def test_evaluations_count_one_per_grid_node(self) -> None:
    """Each estimate evaluates the likelihood once per grid node: 31, not the items answered nor the default 121."""
    estimates = expected_a_posteriori(_RESPONSES, _ITEMS, grid=QuadratureGrid.uniform(n_nodes=31))

    np.testing.assert_array_equal(estimates.evaluations, np.full(len(_RESPONSES), 31))


class TestWeightedLikelihood:
  """Warm's weighted likelihood estimate: the root of the modified score, its ranking, and its standard error."""

  _TWO_MAXIMA = np.array([[2.13, -2.78, 0.13, 0.95], [1.11, 2.02, 0.3, 0.9]])

  def test_estimate_solves_warms_modified_score_equation(self) -> None:
    """Every estimate is an interior root of Warm's modified score, written out, for every pattern of five answers.

    The items guess and slip, where the WLE is not the maximizer of L sqrt(I), so this checks Warm's defining equation
    directly; patterns with a missing response are included. A one-step correction of the MLE, MLE + J / (2 I^2),
    returned about 2670 for a pattern whose likelihood is maximized at the search bound.
    """
    items = np.array([
      [1.1, -1.2, 0.2, 1.0],
      [0.9, -0.4, 0.15, 0.97],
      [1.5, 0.1, 0.25, 1.0],
      [1.3, 0.8, 0.1, 0.93],
      [0.7, 1.6, 0.2, 1.0],
    ])
    patterns = np.array(list(product([0.0, 1.0], repeat=5)))
    responses = np.vstack([patterns, np.where(np.eye(5, dtype=bool)[np.arange(32) % 5], _NAN, patterns)])

    estimates = weighted_likelihood(responses, items)

    for pattern, theta in zip(responses, estimates.theta, strict=True):
      answered = ~np.isnan(pattern)
      assert -6.0 < theta < 6.0, pattern
      assert _warm_modified_score(theta, items[answered], pattern[answered]) == pytest.approx(0.0, abs=1e-4), pattern

  def test_without_guessing_the_estimate_maximizes_the_likelihood_times_the_root_information(self) -> None:
    """Without guessing or slipping, w = sqrt(I) (Warm, 1989, p. 431): the estimate maximizes log L + log I / 2.

    The reference is a dense grid search of log L + 0.5 log I, written out, over every response pattern.
    """
    items = np.column_stack([np.linspace(0.8, 2.0, 5), np.linspace(-1.5, 1.5, 5), np.zeros(5), np.ones(5)])
    grid = np.linspace(-6.0, 6.0, 12_001)
    p = _probability(grid, items)
    log_information = np.log(np.sum(items[:, 0] ** 2 * p * (1.0 - p), axis=1))
    patterns = np.array(list(product([0.0, 1.0], repeat=5)))

    estimates = weighted_likelihood(patterns, items)

    for pattern, theta in zip(patterns, estimates.theta, strict=True):
      reference = grid[np.argmax(_log_likelihood(grid, items, pattern) + 0.5 * log_information)]
      assert theta == pytest.approx(reference, abs=2e-3), pattern

  def test_exact_bias_is_small_and_below_the_first_order_bias_of_the_maximum_likelihood_estimate(self) -> None:
    """The WLE removes the MLE's first-order bias, -J / (2 I^2) (Warm, 1989, p. 430, Equation 6; p. 431), on 2PL items.

    The expected bias is computed exactly by enumerating all 2^8 response patterns, weighted by their probabilities.
    Wherever the MLE's first-order bias is above 0.05, the WLE's exact bias must be less than half of it. An estimator
    that returned the MLE, or that applied the correction with the wrong sign, doubling the bias, fails.
    """
    items = np.column_stack([np.linspace(0.8, 2.0, 8), np.linspace(-2.0, 2.0, 8), np.zeros(8), np.ones(8)])
    patterns = np.array(list(product([0.0, 1.0], repeat=8)))
    estimates = weighted_likelihood(patterns, items).theta

    for theta in (-2.0, -1.0, 0.0, 1.0, 2.0):
      p = _probability(np.array(theta), items)
      probabilities = np.prod(np.where(patterns == 1, p, 1.0 - p), axis=1)
      wle_bias = float(probabilities @ (estimates - theta))
      first = items[:, 0] * p * (1.0 - p)
      second = items[:, 0] * first * (1.0 - 2.0 * p)
      information = np.sum(first**2 / (p * (1.0 - p)))
      mle_bias = -np.sum(first * second / (p * (1.0 - p))) / (2.0 * information**2)

      assert abs(wle_bias) < 0.1, theta
      if abs(mle_bias) > 0.05:
        assert abs(wle_bias) < 0.5 * abs(mle_bias), theta

  def test_standard_error_is_the_inverse_root_of_the_information_at_the_estimate(self) -> None:
    """The SE is 1 / sqrt(I) at the WLE, I written out, without the correction of Magis and Raiche's Equation 14.

    The items guess and slip, so the WLE is not the MLE. An SE taken at the MLE, the variance 1 / I, or one with
    (I'J - IJ') / (2 I^2) added to the information, fails.
    """
    items = np.array([[1.2, -1.0, 0.2, 1.0], [0.8, 0.0, 0.1, 0.95], [1.6, 0.7, 0.25, 1.0], [1.0, 1.5, 0.0, 0.9]])

    estimates = weighted_likelihood(_rows([1.0, 1.0, 0.0, 1.0]), items)

    assert estimates.standard_errors[0] == pytest.approx(
      1.0 / math.sqrt(_information(estimates.theta[0], items)), rel=1e-12
    )

  @pytest.mark.parametrize(
    ("b", "correct", "expected"),
    [
      pytest.param(10.0, 1.0, 6.0, id="correct_beyond_the_upper_bound"),
      pytest.param(-10.0, 0.0, -6.0, id="incorrect_beyond_the_lower_bound"),
    ],
  )
  def test_without_a_root_the_estimate_is_the_bound_the_score_points_to(
    self, b: float, correct: float, expected: float
  ) -> None:
    """With no root in the bounds the estimate is the bound the modified score points to.

    For one 2PL item the weighted likelihood P^1.5 Q^0.5 peaks where P = 3/4 after a correct answer (1/4 after an
    incorrect one), at b +- ln 3 / a, beyond [-6, 6] for b = +-10.
    """
    estimates = weighted_likelihood(_rows([correct]), np.array([[1.0, b, 0.0, 1.0]]))

    assert estimates.theta[0] == expected

  def test_estimate_is_the_root_of_largest_weighted_likelihood(self) -> None:
    """Of several maxima the estimate is the global maximum of the weighted likelihood.

    The weighted log-likelihood is the integral of the modified score, so its maximum is found here by integrating the
    score written out on a fine grid, after checking that the scan sees at least two maxima.
    """
    pattern = np.array([1.0, 1.0])
    scan = np.linspace(-6.0, 6.0, 121)
    scan_scores = np.array([_warm_modified_score(theta, self._TWO_MAXIMA, pattern) for theta in scan])
    fine = np.linspace(-6.0, 6.0, 24_001)
    fine_scores = np.array([_warm_modified_score(theta, self._TWO_MAXIMA, pattern) for theta in fine])
    weighted = np.concatenate([[0.0], np.cumsum((fine_scores[1:] + fine_scores[:-1]) / 2 * np.diff(fine))])

    estimates = weighted_likelihood(_rows(pattern), self._TWO_MAXIMA)

    assert np.sum((scan_scores[:-1] > 0) & (scan_scores[1:] <= 0)) + (scan_scores[0] <= 0) + (scan_scores[-1] > 0) >= 2
    assert estimates.theta[0] == pytest.approx(fine[int(np.argmax(weighted))], abs=1e-3)

  @pytest.mark.parametrize(
    ("b", "theta", "correct"),
    [
      pytest.param(5.0, -2.0, 1.0, id="correct_on_a_far_too_hard_item"),
      pytest.param(-5.0, 2.0, 0.0, id="incorrect_on_a_far_too_easy_item"),
    ],
  )
  def test_modified_score_keeps_its_digits_where_the_probability_rounds_off(
    self, b: float, theta: float, correct: float
  ) -> None:
    """At a logit of +-70 one of P and Q is below 1e-30, and the score must keep it.

    For one item without guessing or slipping the modified score is a Q + a (Q - P) / 2 after a correct answer and
    -a P + a (Q - P) / 2 after an incorrect one. Computing Q as 1 - P and clipping P Q at machine epsilon lost the
    likelihood term entirely, leaving only the correction: 5 instead of 15 for the far-too-hard item.
    """
    a = 10.0
    p, q = expit(a * (theta - b)), expit(-a * (theta - b))

    examinee = ability._Examinee.of(np.array([correct]), np.array([[a, b, 0.0, 1.0]]))  # noqa: SLF001
    value = examinee.modified_score(theta)[0]

    assert value == pytest.approx((a * q if correct else -a * p) + a * (q - p) / 2.0, rel=1e-12)

  def test_evaluations_count_the_scan_the_root_search_and_the_ranking(self) -> None:
    """The count includes every modified-score evaluation: the scan, Brent's searches and the ranking integrals.

    The reference counts SciPy's own evaluations on the score written out: the 121-point scan, Brent's search in each
    interval where the score turns from + to -, and, with several candidates, the integral from the lower bound to
    each. A count that left out the Brent steps or the integrals fails.
    """
    cases = [
      (np.array([[1.0, 0.0, 0.0, 1.0], [1.2, 0.5, 0.0, 1.0]]), np.array([1.0, 0.0]), 1),
      (self._TWO_MAXIMA, np.array([1.0, 1.0]), 2),
    ]
    for items, pattern, n_candidates in cases:
      scan = np.linspace(-6.0, 6.0, 121)
      values = np.array([_warm_modified_score(theta, items, pattern) for theta in scan])
      brackets = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0)).tolist()
      score = functools.partial(_warm_modified_score, items=items, responses=pattern)
      searches = [brentq(score, scan[k], scan[k + 1], xtol=1e-6, full_output=True) for k in brackets]
      expected = 121 + sum(result.function_calls for _, result in searches)
      if len(searches) > 1:
        expected += sum(quad(score, -6.0, root, limit=200, full_output=1)[2]["neval"] for root, _ in searches)

      estimates = weighted_likelihood(_rows(pattern), items)

      assert len(searches) == n_candidates
      assert estimates.evaluations[0] == expected

  def test_a_ranking_integral_that_does_not_converge_warns_from_the_callers_line(self) -> None:
    """A ranking integral that does not converge warns, naming the row, from the line that called the estimator.

    After a wrong answer to a steep item that guesses (a = 200, c = 0.2), the modified score spikes at b and quad runs
    out of its 200 subdivisions. quad returns that message instead of warning, so dropping it loses the warning, and a
    wrong stacklevel blames a line inside irtlib.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0], [200.0, 0.0, 0.2, 1.0]])
    responses = np.array([[1.0, _NAN], [_NAN, 0.0]])

    def estimate() -> AbilityEstimates:
      return weighted_likelihood(responses, items)

    with pytest.warns(IntegrationWarning, match=r"examinee in row 1: .*subdivisions") as caught:
      estimate()

    call_line = estimate.__code__.co_firstlineno + 1
    assert {(warning.filename, warning.lineno) for warning in caught} == {(__file__, call_line)}
