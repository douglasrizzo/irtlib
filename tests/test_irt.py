"""Tests of the item response functions, the quantities derived from them and their parameter derivatives.

Expected values come from hand-worked cases, closed forms, the published example of Magis (2013, p. 311), the 4PL
written out in the tests, brute-force sums, searches, integrals and central differences over it, and Baker's (1987,
p. 116) identity for the 2PL.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from irtlib import irt
from irtlib.irt import log_likelihood_gradient, parameter_information, probability_gradient

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.typing import FloatArray

# Items [a, b, c, d] strictly inside their bounds, so every parameter, logits included, can be perturbed.
_INTERIOR = np.array([[0.8, -0.5, 0.25, 0.95], [2.2, 0.7, 0.1, 0.9], [1.3, 1.8, 0.05, 0.99]])
_THETAS = np.linspace(-4.0, 4.0, 9)
_STEP = 1e-6


def _to_items(parameters: FloatArray) -> FloatArray:
  """Return items [a, b, c, d] from rows of slope, intercept, logit of c and logit of d."""
  slope, intercept, guessing, upper = parameters.T
  return np.column_stack([slope, -intercept / slope, expit(guessing), expit(upper)])


def _to_parameters(items: FloatArray) -> FloatArray:
  """Return rows of slope, intercept, logit of c and logit of d from items [a, b, c, d] inside their bounds."""
  a, b, c, d = items.T
  return np.column_stack([a, -a * b, logit(c), logit(d)])


def _central_differences(
  function: Callable[[FloatArray], FloatArray], parameters: FloatArray, step: float
) -> FloatArray:
  """Return central differences of a function of the items' parameters, one parameter at a time, on a new last axis."""
  columns = []
  for column in range(parameters.shape[-1]):
    shift = np.zeros_like(parameters)
    shift[:, column] = step
    columns.append((function(parameters + shift) - function(parameters - shift)) / (2 * step))
  return np.stack(columns, axis=-1)


def _log_probabilities(theta: FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray]:
  """Return log P and log Q of the 4PL written out, every ability against every item.

  Q is written out as (1 - d) + (d - c) / (1 + exp(a (theta - b))), not as 1 - P, whose cancellation the second
  differences below would magnify.
  """
  a, b, c, d = items.T
  z = a * (theta[:, np.newaxis] - b)
  return np.log(c + (d - c) / (1 + np.exp(-z))), np.log((1 - d) + (d - c) / (1 + np.exp(z)))


def _grouped_log_likelihood(parameters: FloatArray, correct: FloatArray, total: FloatArray) -> FloatArray:
  """Return each item's grouped log-likelihood at the nodes ``_THETAS``, from the 4PL written out."""
  log_p, log_q = _log_probabilities(_THETAS, _to_items(parameters))
  return np.sum(correct * log_p + (total - correct) * log_q, axis=0)


@pytest.mark.parametrize(
  ("item", "theta", "expected"),
  [
    ([1.5, 1.0, 0.2, 1.0], 0.0, [0.0, 0.1193171617, 0.1308119162, 0.0]),
    ([0.8, -0.5, 0.25, 0.95], 1.2, [0.1365220090, 0.1137683408, 0.03829505668, 0.03779858564]),
  ],
)
def test_probability_gradient_hand_worked(item: list[float], theta: float, expected: list[float]) -> None:
  """The derivatives match values worked by hand from the closed forms, which fail if a factor or the order is wrong."""
  np.testing.assert_allclose(probability_gradient(theta, np.array(item)), expected, rtol=1e-9, atol=1e-12)


def test_probability_gradient_is_the_limit_of_differences() -> None:
  """Each derivative equals the central difference of the 4PL's probability, written out, in that parameter."""
  parameters = _to_parameters(_INTERIOR)
  expected = _central_differences(lambda shifted: _probability(_THETAS, _to_items(shifted)), parameters, _STEP)
  np.testing.assert_allclose(probability_gradient(_THETAS, _INTERIOR), expected, atol=1e-9)


def test_probability_gradient_shape() -> None:
  """Every ability against every item, the four parameters last: ``shape(theta) + items.shape[:-1] + (4,)``."""
  assert probability_gradient(_THETAS, _INTERIOR).shape == (9, 3, 4)
  assert probability_gradient(0.5, _INTERIOR[0]).shape == (4,)


def test_bounds_have_no_derivative() -> None:
  """A guessing parameter of 0 and an upper asymptote of 1 have derivative 0, in every output, for every item.

  The abilities include 800 logits on either side of the difficulty, where an unguarded score of c = 0 or d = 1
  would be 0 / 0.
  """
  thetas = np.concatenate([_THETAS, [-800.0, 800.0]])
  items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.2, 1.0], [0.9, -1.0, 0.0, 0.9]])
  # the columns of the logits of c and d at their bounds, per item
  bound_columns = [[2, 3], [3], [2]]
  gradient = probability_gradient(thetas, items)
  scores = log_likelihood_gradient(thetas, items, np.zeros((11, 3)), np.ones((11, 3)))
  information = parameter_information(thetas, items, np.ones((11, 3)))
  for item, columns in enumerate(bound_columns):
    assert np.all(gradient[:, item, columns] == 0.0)
    assert np.all(scores[item, columns] == 0.0)
    assert np.all(information[item][columns, :] == 0.0)
    assert np.all(information[item][:, columns] == 0.0)


def test_log_likelihood_gradient_is_the_limit_of_differences() -> None:
  """The gradient from counts equals the central difference of the grouped log-likelihood in each parameter."""
  rng = np.random.default_rng(20261127)
  total = rng.uniform(5.0, 50.0, size=(9, 3))
  correct = total * rng.uniform(0.05, 0.95, size=(9, 3))
  parameters = _to_parameters(_INTERIOR)
  expected = _central_differences(lambda shifted: _grouped_log_likelihood(shifted, correct, total), parameters, _STEP)
  np.testing.assert_allclose(
    log_likelihood_gradient(_THETAS, _INTERIOR, correct, total), expected, rtol=1e-6, atol=1e-6
  )


def test_parameter_information_is_the_expected_negative_hessian() -> None:
  """The information equals the expected negative second derivative of one response's log-likelihood, by differences.

  At each node the expectation weights the Hessian of a correct response by P and that of an incorrect one by 1 - P.
  Fails if the information omits a term of the 3PL or 4PL, where the observed and expected information differ.
  """
  total = np.linspace(1.0, 3.0, 27).reshape(9, 3)
  parameters = _to_parameters(_INTERIOR)
  step = 1e-4

  def negative_hessian(outcome: float) -> FloatArray:
    """Return the negative Hessian of one response's log-likelihood at every node, shape (nodes, items, 4, 4)."""

    def log_likelihood(shifted: FloatArray) -> FloatArray:
      log_p, log_q = _log_probabilities(_THETAS, _to_items(shifted))
      return outcome * log_p + (1 - outcome) * log_q

    return -_central_differences(lambda outer: _central_differences(log_likelihood, outer, step), parameters, step)

  p = _probability(_THETAS, _INTERIOR)[..., np.newaxis, np.newaxis]
  expected_per_response = p * negative_hessian(1.0) + (1 - p) * negative_hessian(0.0)
  expected = np.einsum("kn,knpr->npr", total, expected_per_response)
  np.testing.assert_allclose(parameter_information(_THETAS, _INTERIOR, total), expected, rtol=1e-5, atol=1e-6)


def test_two_parameter_information_is_the_observed_one() -> None:
  """For the 2PL the observed information in the slope and intercept equals the expected one (Baker, 1987, p. 116).

  The observed information is the negative Hessian of the grouped log-likelihood of arbitrary counts, by differences;
  it contains no data only in these parameters, so a wrong parameterization fails.
  """
  items = np.array([[0.7, -1.2, 0.0, 1.0], [1.9, 0.4, 0.0, 1.0]])
  rng = np.random.default_rng(1)
  total = rng.uniform(5.0, 50.0, size=(9, 2))
  correct = total * rng.uniform(0.05, 0.95, size=(9, 2))

  def log_likelihood(location: FloatArray) -> FloatArray:
    slope, intercept = location.T
    log_p, log_q = _log_probabilities(_THETAS, np.column_stack([slope, -intercept / slope, np.zeros(2), np.ones(2)]))
    return np.sum(correct * log_p + (total - correct) * log_q, axis=0)

  location = np.column_stack([items[:, 0], -items[:, 0] * items[:, 1]])
  step = 1e-4
  observed = -_central_differences(lambda shifted: _central_differences(log_likelihood, shifted, step), location, step)
  np.testing.assert_allclose(parameter_information(_THETAS, items, total)[:, :2, :2], observed, rtol=1e-5)


def test_far_from_the_difficulty() -> None:
  """Where the logistic function underflows, the ratios take their limits, which an unguarded 0 / 0 would make NaN.

  Each item is answered incorrectly once, 800 logits from its difficulty. Above it, a 2PL item has intercept score -1
  and slope score -800, and an item with d = 0.9 has logit-of-d score -d; below it, an item with c = 0.2 has
  logit-of-c score -c. The information is then 0 for the 2PL item, c(1 - c) for the logit of c and d(1 - d) for the
  logit of d, the limits of (dP/dparameter)^2 / (PQ).
  """
  theta = np.array([800.0, -800.0])
  items = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.2, 1.0], [1.0, 0.0, 0.0, 0.9], [1.0, 0.0, 0.2, 0.9]])
  total = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
  gradient = log_likelihood_gradient(theta, items, np.zeros((2, 4)), total)
  expected_gradient = [[-800.0, -1.0, 0.0, 0.0], [0.0, 0.0, -0.2, 0.0], [0.0, 0.0, 0.0, -0.9], [0.0, 0.0, -0.2, 0.0]]
  np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)
  expected_information = np.zeros((4, 4, 4))
  expected_information[1, 2, 2] = expected_information[3, 2, 2] = 0.2 * 0.8
  expected_information[2, 3, 3] = 0.9 * 0.1
  np.testing.assert_allclose(parameter_information(theta, items, total), expected_information, atol=1e-12)


def test_tiny_asymptotes() -> None:
  """With c and d - c near the smallest doubles, the scores stay exact: a product of both would underflow to 0 / 0.

  For [1, 0, 1e-200, 2e-200] at the difficulty, P = 1.5e-200, and the scores of one incorrect response are -P times
  1/6, 1/3 and 2/3 for the intercept and the logits of c and d (the closed forms of the module docstring).
  """
  items = np.array([[1.0, 0.0, 1e-200, 2e-200]])
  gradient = log_likelihood_gradient(np.array([0.0]), items, np.zeros((1, 1)), np.ones((1, 1)))
  np.testing.assert_allclose(gradient[0], [0.0, -2.5e-201, -5e-201, -1e-200], rtol=1e-12)


# 4PL items [a, b, c, d] with guessing, slipping, both and neither, for the tests of the response functions below.
_ITEMS = np.array([[1.4, -0.5, 0.2, 0.95], [0.8, 1.0, 0.0, 1.0], [2.0, 0.3, 0.1, 1.0], [1.1, -1.2, 0.0, 0.9]])


def _probability(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL probability of Magis (2013, p. 306, Equation 1), written out, every ability against every item.

  One item, of shape (4,), gives the shape of ``theta``; a matrix of items adds an axis of items.
  """
  a, b, c, d = np.moveaxis(np.asarray(items, dtype=float), -1, 0)
  thetas = np.asarray(theta, dtype=float)
  if np.ndim(items) == 2:
    thetas = thetas[..., np.newaxis]
  return c + (d - c) / (1 + np.exp(-a * (thetas - b)))


def _information(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL item information of Magis (2013, p. 307, Equation 6), written out from ``_probability``."""
  a, _, c, d = np.moveaxis(np.asarray(items, dtype=float), -1, 0)
  p = _probability(theta, items)
  return a**2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))


class TestIcc:
  """The probability of a correct response."""

  @pytest.mark.parametrize(
    ("theta", "item", "expected"),
    [
      pytest.param(0.4, [1.3, 0.4, 0.2, 0.9], 0.55, id="theta = b gives (c + d) / 2"),
      pytest.param(0.5, [2.0, 0.0, 0.0, 1.0], 1 / (1 + math.exp(-1.0)), id="a scales the logit"),
      pytest.param(1.0, [1.0, 2.0, 0.25, 1.0], 0.25 + 0.75 / (1 + math.exp(1.0)), id="guessing floor"),
    ],
  )
  def test_icc_matches_the_4pl_formula(self, theta: float, item: list[float], expected: float) -> None:
    """P = c + (d - c) / (1 + exp(-a (theta - b))) at hand-worked points.

    Scaling the logistic term by d instead of d - c, dropping a, or flipping the sign of theta - b changes one of them.
    """
    value = irt.icc(theta, np.array(item))

    assert np.shape(value) == ()
    assert value == pytest.approx(expected, rel=1e-12)

  def test_icc_reaches_its_asymptotes_without_overflowing(self) -> None:
    """Far from b the ICC returns c and d; e^x evaluated with Python floats would raise OverflowError."""
    item = np.array([1.0, 0.0, 0.2, 0.9])

    np.testing.assert_allclose(irt.icc(np.array([-1000.0, 1000.0]), item), [0.2, 0.9], rtol=1e-12)


class TestIccDerivatives:
  """The first and second derivatives of the probability with respect to the ability."""

  @pytest.mark.parametrize(
    ("item", "expected"),
    [
      pytest.param([1.0, 0.0, 0.2, 0.9], (0.13125, -0.065625), id="a = 1"),
      pytest.param([2.0, 0.5, 0.1, 1.0], (0.3375, -0.3375), id="a = 2"),
    ],
  )
  def test_derivatives_of_hand_worked_items(self, item: list[float], expected: tuple[float, float]) -> None:
    """At a logit of ln 3, F = 3/4, so P' = a (d - c) 3/16 and P'' = a P' (1/4 - 3/4).

    Dropping a from P'' (the a = 2 case), d - c from P', or the sign of 1 - 2F changes a value.
    """
    theta = item[1] + math.log(3.0) / item[0]

    first, second = irt.icc_derivatives(theta, np.array(item))

    assert float(first) == pytest.approx(expected[0], rel=1e-12)
    assert float(second) == pytest.approx(expected[1], rel=1e-12)

  def test_derivatives_are_the_limits_of_differences(self) -> None:
    """P' and P'' equal the central first and second differences of the 4PL written out, at every ability and item."""
    thetas = np.linspace(-3.0, 3.0, 13)
    step = 1e-4
    above, at, below = (
      _probability(thetas + step, _ITEMS),
      _probability(thetas, _ITEMS),
      _probability(thetas - step, _ITEMS),
    )

    first, second = irt.icc_derivatives(thetas, _ITEMS)

    np.testing.assert_allclose(first, (above - below) / (2 * step), atol=1e-8)
    np.testing.assert_allclose(second, (above - 2 * at + below) / step**2, atol=1e-6)

  def test_derivatives_vanish_far_from_the_difficulty(self) -> None:
    """Over 1500 logits from b both derivatives are 0: F (1 - F) written as e^z / (1 + e^z)^2 is NaN past z = 710."""
    first, second = irt.icc_derivatives(np.array([-2000.0, 2000.0]), _ITEMS)

    np.testing.assert_array_equal(first, np.zeros((2, 4)))
    np.testing.assert_array_equal(second, np.zeros((2, 4)))


class TestInformation:
  """The information an item gives about the ability."""

  @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
  def test_2pl_information_at_the_difficulty_is_a_squared_over_four(self, a: float) -> None:
    """A 2PL item gives a^2 P (1 - P) = a^2 / 4 at theta = b; dropping the square of a or the 1 - P changes it."""
    assert irt.information(-0.7, np.array([a, -0.7, 0.0, 1.0])) == pytest.approx(a**2 / 4, rel=1e-12)

  @pytest.mark.parametrize("theta", [-2.0, 0.3, 1.5])
  def test_1pl_information_is_the_variance_of_the_response(self, theta: float) -> None:
    """A 1PL item's information is P (1 - P); 4PL ratios that do not cancel to 1 when c = 0 and d = 1 change it."""
    p = 1 / (1 + math.exp(-(theta - 0.4)))

    assert irt.information(theta, np.array([1.0, 0.4, 0.0, 1.0])) == pytest.approx(p * (1 - p), rel=1e-12)

  @pytest.mark.parametrize("x", [40.0, -40.0, 700.0, -1000.0, 1000.0])
  def test_information_is_accurate_far_from_difficulty(self, x: float) -> None:
    """Far from b the 2PL information a^2 F (1 - F) is tiny but finite, and exactly 0 where F underflows.

    The formula of Magis (2013, Equation 6) computes 1 - P after P has rounded to 1, so it divides 0 by 0 and returns
    NaN; so does a ratio (d - c) F / P left unguarded where P has underflowed to 0.
    """
    expected = math.exp(-abs(x)) / (1 + math.exp(-abs(x))) ** 2

    assert irt.information(x, np.array([1.0, 0.0, 0.0, 1.0])) == pytest.approx(expected, rel=1e-12, abs=0.0)

  def test_information_matches_magis_equation_6(self) -> None:
    """Where Magis's (2013, p. 307) Equation 6 is well conditioned, it agrees at every ability and item.

    Dropping c or d from either ratio, or squaring the wrong term, moves a 3PL or 4PL item off it.
    """
    thetas = np.linspace(-4.0, 4.0, 17)

    np.testing.assert_allclose(irt.information(thetas, _ITEMS), _information(thetas, _ITEMS), rtol=1e-10)


class TestTotalInformation:
  """The test information."""

  def test_total_information_sums_over_the_items(self) -> None:
    """Two 2PL items at their difficulty give 1/4 + 4/4 = 1.25; keeping one item or summing another axis does not."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])

    total = irt.total_information(0.0, items)

    assert type(total) is float
    assert total == pytest.approx(1.25, rel=1e-12)

  def test_total_information_of_one_item_gives_one_value_per_ability(self) -> None:
    """One item as a row gives its information at each ability; summing the last axis of a 1-D result adds abilities.

    A 2PL item with a = 1 at its difficulty has information 1/4, and one logit above it s(1) s(-1), s the logistic.
    """
    s_1 = 1 / (1 + math.exp(-1.0))

    values = irt.total_information(np.array([0.0, 1.0]), np.array([1.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(values, [0.25, s_1 * (1 - s_1)], rtol=1e-12)

  def test_total_information_gives_one_value_per_ability(self) -> None:
    """Two abilities give two totals; summing over the abilities as well as the items gives one number."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])
    l1, l2 = 1 / (1 + math.exp(-1.0)), 1 / (1 + math.exp(-2.0))

    values = irt.total_information(np.array([0.0, 1.0]), items)

    np.testing.assert_allclose(values, [1.25, l1 * (1 - l1) + 4 * l2 * (1 - l2)], rtol=1e-12)


class TestVarSeeAndReliability:
  """The variance and standard error of the ability estimate, and the reliability."""

  def test_var_is_the_inverse_of_the_test_information(self) -> None:
    """A 2PL item with a = 1 at its difficulty has information 1/4, so the variance is 4; returning I gives 0.25."""
    assert irt.var(0.0, np.array([[1.0, 0.0, 0.0, 1.0]])) == pytest.approx(4.0, rel=1e-12)

  def test_see_is_the_square_root_of_the_variance(self) -> None:
    """A 2PL item with a = 1 at its difficulty has variance 4, so the standard error is 2; the variance itself is 4."""
    assert irt.see(0.0, np.array([[1.0, 0.0, 0.0, 1.0]])) == pytest.approx(2.0, rel=1e-12)

  def test_reliability_is_the_information_over_the_information_plus_one(self) -> None:
    """Twenty 2PL items with a = 1 at their difficulty give I = 5, so I / (I + 1) = 5/6 for abilities N(0, 1).

    1 - 1/I gives 0.8 and 1 / (I + 1) gives 1/6.
    """
    assert irt.reliability(0.0, np.array([[1.0, 0.0, 0.0, 1.0]] * 20)) == pytest.approx(5 / 6, rel=1e-12)

  @pytest.mark.parametrize(
    ("function", "expected"),
    [(irt.var, math.inf), (irt.see, math.inf), (irt.reliability, 0.0)],
    ids=["var", "see", "reliability"],
  )
  def test_no_information(self, function: Callable[..., float], expected: float) -> None:
    """Without information the variance and standard error are +inf and the reliability 0.

    Dividing 1 by the information would raise ZeroDivisionError, and 1 - 1/I would give a reliability of -inf.
    """
    assert function(0.0, np.empty((0, 4))) == expected


class TestConfidenceInterval:
  """The confidence interval of an ability estimate."""

  @pytest.mark.parametrize(
    ("confidence", "z"),
    [(0.90, 1.6448536270), (0.95, 1.9599639845), (0.99, 2.5758293035)],
  )
  def test_interval_is_theta_plus_or_minus_the_normal_quantile_times_the_standard_error(
    self, confidence: float, z: float
  ) -> None:
    """The interval is theta +- z SE, with z the standard normal's (1 + confidence) / 2 quantile from published tables.

    Two 1PL items one logit above theta each hold information s(1) s(-1), with s the logistic, so the SE is
    1 / sqrt(2 s(1) s(-1)). A one-sided quantile (1.2816 for 90%), the variance in place of the SE, or an interval
    centred on the items' difficulty instead of theta fails.
    """
    theta = 0.5
    items = np.array([[1.0, theta + 1, 0.0, 1.0], [1.0, theta + 1, 0.0, 1.0]])
    s_1 = 1 / (1 + math.exp(-1))
    se = 1 / math.sqrt(2 * s_1 * (1 - s_1))

    lower, upper = irt.confidence_interval(theta, items, confidence=confidence)

    assert lower == pytest.approx(theta - z * se, rel=1e-9)
    assert upper == pytest.approx(theta + z * se, rel=1e-9)

  @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
  def test_rejects_a_confidence_outside_zero_and_one(self, confidence: float) -> None:
    """A confidence of 0 would give an interval of zero width, 1 an infinite one and 1.5 NaN bounds; each must raise."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      irt.confidence_interval(0.0, np.array([[1.0, 0.0, 0.0, 1.0]]), confidence=confidence)


class TestMaxInfoTheta:
  """The ability at which an item is most informative."""

  def test_max_info_theta_of_a_2pl_item_is_its_difficulty(self) -> None:
    """A 2PL item is most informative at theta = b (Magis, 2013, p. 310); another value means a wrong d = 1 branch."""
    assert irt.max_info_theta(np.array([1.0, 0.5, 0.0, 1.0])) == pytest.approx(0.5)

  def test_max_info_theta_shifts_with_guessing(self) -> None:
    """A 3PL item is most informative at b + log((1 + sqrt(1 + 8c)) / 2) / a (Magis, 2013, p. 310, Equation 23)."""
    expected = 0.3 + math.log((1 + math.sqrt(1 + 8 * 0.2)) / 2) / 1.5

    assert irt.max_info_theta(np.array([1.5, 0.3, 0.2, 1.0])) == pytest.approx(expected, rel=1e-12)

  def test_max_info_theta_of_a_symmetric_4pl_item_is_its_difficulty(self) -> None:
    """With c + d = 1 the peak is b (Magis, 2013, p. 310); the d = 1 formula would give b + 0.30 for this item."""
    assert irt.max_info_theta(np.array([1.2, -0.5, 0.1, 0.9])) == pytest.approx(-0.5, abs=1e-12)

  def test_max_info_theta_of_magis_example(self) -> None:
    """Magis's (2013, p. 311, Equation 28) item [1.1, -1, 0.2, 0.95] peaks at -0.849, printed to three decimals."""
    assert irt.max_info_theta(np.array([1.1, -1.0, 0.2, 0.95])) == pytest.approx(-0.849, abs=5e-4)

  @pytest.mark.parametrize(
    ("c", "d", "theta_star", "peak"),
    [
      (0.0, 0.8, -0.267, 0.170),
      (0.0, 0.9, -0.158, 0.206),
      (0.0, 1.0, 0.000, 0.250),
      (0.1, 0.8, -0.095, 0.124),
      (0.1, 0.9, 0.000, 0.160),
      (0.1, 1.0, 0.158, 0.206),
      (0.2, 0.8, 0.000, 0.090),
      (0.2, 0.9, 0.095, 0.124),
      (0.2, 1.0, 0.267, 0.170),
    ],
  )
  def test_max_info_theta_and_its_information_match_magis_table_1(
    self, c: float, d: float, theta_star: float, peak: float
  ) -> None:
    """The peak and the information there match Magis's (2013, p. 313) Table 1 for a = 1, b = 0, to three decimals.

    The rows cross c = 0, 0.1, 0.2 with d = 0.8, 0.9, 1, so a 4PL root taken from the wrong cosine branch, the 3PL
    formula applied to d < 1, or an information formula that drops c or d misses a printed value.
    """
    item = np.array([1.0, 0.0, c, d])

    theta = irt.max_info_theta(item)

    assert theta == pytest.approx(theta_star, abs=5e-4)
    assert irt.information(theta, item) == pytest.approx(peak, abs=5e-4)

  def test_max_info_theta_of_asymmetric_4pl_items_is_the_peak_of_the_information(self) -> None:
    """With c + d != 1 the peak moves off b; it is where a bounded search maximizes Magis's Equation 6, written out.

    The search locates the peak to 1e-9; a 4PL cubic root off by 1% moves it by over 5e-4.
    """
    items = np.array([[1.0, 0.0, 0.25, 0.6], [1.5, 0.5, 0.2, 0.9], [0.7, 1.2, 0.0, 0.8], [2.2, -0.4, 0.3, 0.97]])
    expected = [
      minimize_scalar(
        lambda theta, item=item: -float(_information(theta, item)),
        bounds=(item[1] - 3, item[1] + 3),
        method="bounded",
        options={"xatol": 1e-10},
      ).x
      for item in items
    ]

    np.testing.assert_allclose(irt.max_info_theta(items), expected, rtol=0, atol=1e-7)

  def test_max_info_theta_result_is_not_a_view_of_the_items(self) -> None:
    """Writing into the result leaves the item matrix alone; returning the difficulty column for 2PL items would not."""
    items = np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]])
    thetas = irt.max_info_theta(items)
    np.testing.assert_array_equal(thetas, [0.5, -0.2])

    thetas[0] = 99.0

    np.testing.assert_array_equal(items[:, 1], [0.5, -0.2])


class TestInfoIntegral:
  """The integral of an item's information between two abilities."""

  @pytest.mark.parametrize(
    ("lower", "upper"),
    [(-1.7, 2.4), (-math.inf, 0.5), (-0.5, math.inf)],
    ids=["finite", "from -inf", "to inf"],
  )
  def test_matches_numerical_integration(self, lower: float, upper: float) -> None:
    """The closed form agrees with quadrature of Magis's Equation 6, written out, for 1PL to 4PL items.

    An infinite bound is integrated from or to 30 logits beyond the difficulty, where the information left is below
    2e-13 and Equation 6 does not yet divide 0 by 0. A wrong weight on log P or log Q, or a dropped log term at an
    infinite bound, changes the integral of an item with guessing or slipping.
    """
    expected = [
      quad(
        lambda theta, item=item: float(_information(theta, item)),
        max(lower, item[1] - 30 / item[0]),
        min(upper, item[1] + 30 / item[0]),
        points=[item[1]],
        limit=200,
      )[0]
      for item in _ITEMS
    ]

    np.testing.assert_allclose(irt.info_integral(lower, upper, _ITEMS), expected, rtol=1e-9, atol=1e-12)

  def test_whole_scale_integral_of_2pl_item_is_its_discrimination(self) -> None:
    """For c = 0 and d = 1 the information a^2 F (1 - F) is a times dF/dtheta, so it integrates to a over the scale."""
    items = np.array([[0.7, -1.0, 0.0, 1.0], [1.3, 0.0, 0.0, 1.0], [2.2, 2.5, 0.0, 1.0]])

    np.testing.assert_allclose(irt.info_integral(-np.inf, np.inf, items), items[:, 0])


class TestLogResponseProbabilities:
  """The log-probabilities of a correct and of an incorrect response."""

  def test_log_response_probabilities_of_a_hand_worked_item(self) -> None:
    """At a logit of ln 3 the logistic term is 3/4, so [1, 0, 0.2, 0.9] has P = 0.2 + 0.7 * 3/4 = 0.725 and Q = 0.275.

    Ignoring c or d, or swapping the two outputs, changes either value.
    """
    log_p, log_q = irt.log_response_probabilities(math.log(3.0), np.array([1.0, 0.0, 0.2, 0.9]))

    assert float(log_p) == pytest.approx(math.log(0.725), rel=1e-12)
    assert float(log_q) == pytest.approx(math.log(0.275), rel=1e-12)

  def test_log_response_probabilities_match_the_4pl_near_the_difficulty(self) -> None:
    """Near the difficulty, the outputs are log P and log(1 - P) of the 4PL written out.

    c = 0 and d = 1 reach the logaddexp branches with an infinite term, the other items with two finite ones, so a
    wrong branch on any path changes a value.
    """
    thetas = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    p = _probability(thetas, _ITEMS)

    log_p, log_q = irt.log_response_probabilities(thetas, _ITEMS)

    np.testing.assert_allclose(log_p, np.log(p), rtol=1e-12)
    np.testing.assert_allclose(log_q, np.log1p(-p), rtol=1e-10)

  def test_log_response_probabilities_are_exact_where_p_rounds_to_0_or_1(self) -> None:
    """A 2PL item's log P and log Q equal -log(1 + e^-z) and -log(1 + e^z) at logits where P rounds to 0 or 1.

    Taking logarithms of P or of 1 - P once P has rounded gives -inf beyond |z| = 37 and loses precision past 25.
    """
    z = np.array([-700.0, -40.0, 40.0, 700.0])

    log_p, log_q = irt.log_response_probabilities(z, np.array([[1.0, 0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(log_p[:, 0], -np.logaddexp(0.0, -z), rtol=1e-12)
    np.testing.assert_allclose(log_q[:, 0], -np.logaddexp(0.0, z), rtol=1e-12)


# log_likelihood checks its arguments through log_likelihood_function, which must do it when built
_LOG_LIKELIHOOD_BUILDERS = [
  pytest.param(lambda responses, items: irt.log_likelihood(0.0, responses, items), id="log_likelihood"),
  pytest.param(irt.log_likelihood_function, id="log_likelihood_function"),
]


def _brute_force_log_likelihood(thetas: FloatArray, responses: FloatArray, items: FloatArray) -> FloatArray:
  """Return each examinee's log-likelihood at each ability, a sum over the examinee's answered items, one at a time."""
  values = np.zeros((responses.shape[0], thetas.size))
  for examinee, row in enumerate(responses):
    for column, theta in enumerate(thetas):
      for item, response in zip(items, row, strict=True):
        if not math.isnan(response):
          p = float(_probability(theta, item))
          values[examinee, column] += math.log(p) if response == 1 else math.log(1 - p)
  return values


class TestLogLikelihood:
  """The log-likelihood of a response matrix, and its function of the ability."""

  def test_log_likelihood_of_two_hand_worked_items(self) -> None:
    """At theta = 0, a right answer to [1, -1, 0, 1] has P = 1 / (1 + e^-1), a wrong one to [1, 0, 0.2, 0.9] Q = 0.45.

    Using P for the incorrect response, or ignoring c and d (Q = 0.5), changes the sum.
    """
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.2, 0.9]])
    expected = math.log(1 / (1 + math.exp(-1.0))) + math.log(0.45)

    values = irt.log_likelihood(0.0, np.array([[1.0, 0.0]]), items)

    assert values.shape == (1,)
    assert values[0] == pytest.approx(expected, rel=1e-12)

  def test_log_likelihood_sums_each_examinees_answered_items(self) -> None:
    """Each examinee's value at each ability is the brute-force sum of log P or log Q over the items they answered.

    The matrix has missing responses, an examinee with none (log-likelihood 0) and an item nobody answered. Counting a
    missing response as incorrect, pairing an examinee with another's responses, or transposing the examinees and
    abilities changes a value.
    """
    responses = np.array([
      [1.0, 0.0, np.nan, 1.0],
      [np.nan, np.nan, np.nan, np.nan],
      [0.0, 1.0, np.nan, np.nan],
      [1.0, 1.0, np.nan, 0.0],
      [0.0, np.nan, np.nan, 0.0],
    ])
    thetas = np.array([-2.5, -0.4, 0.0, 1.3, 2.8])

    values = irt.log_likelihood(thetas, responses, _ITEMS)

    assert values.shape == (5, 5)
    np.testing.assert_allclose(values, _brute_force_log_likelihood(thetas, responses, _ITEMS), rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(values[1], np.zeros(5))

  def test_log_likelihood_gives_each_examinee_the_shape_of_theta(self) -> None:
    """A (2, 3) array of abilities gives each examinee a (2, 3) block; flattening theta or putting it first does not."""
    thetas = np.linspace(-1.0, 1.0, 6).reshape(2, 3)
    responses = np.array([[1.0, 0.0, np.nan, 1.0], [0.0, 0.0, 1.0, np.nan], [np.nan, 1.0, 1.0, 1.0]])

    values = irt.log_likelihood(thetas, responses, _ITEMS)

    assert values.shape == (3, 2, 3)
    np.testing.assert_allclose(
      values.reshape(3, 6), _brute_force_log_likelihood(thetas.ravel(), responses, _ITEMS), rtol=1e-12
    )

  @pytest.mark.parametrize(
    "row",
    [
      pytest.param([1.0, 0.0, np.nan, 1.0], id="one missing"),
      pytest.param([0.0, np.nan, np.nan, 0.0], id="two missing, both answers incorrect"),
      pytest.param([1.0, 1.0, 0.0, 1.0], id="none missing"),
      pytest.param([np.nan, np.nan, np.nan, np.nan], id="none answered"),
    ],
  )
  def test_one_examinee_sums_the_log_probabilities_of_its_answered_items(self, row: list[float]) -> None:
    """One examinee's value at an array of abilities, and at one ability, is the brute-force sum over its answers.

    Counting a missing response as incorrect, pairing an item with another item's response or asymptotes, or giving
    one ability, a float or a 0-d array, a shape other than (1,) changes a value or the shape; with no answers the
    value is 0.
    """
    responses = np.array([row])
    thetas = np.array([-2.5, -0.4, 0.0, 1.3, 2.8])
    expected = _brute_force_log_likelihood(thetas, responses, _ITEMS)

    values = irt.log_likelihood(thetas, responses, _ITEMS)

    assert values.shape == (1, 5)
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0.0)
    for one in (1.3, np.array(1.3)):
      at_one = irt.log_likelihood(one, responses, _ITEMS)
      assert at_one.shape == (1,)
      np.testing.assert_allclose(at_one, expected[:, 3], rtol=1e-12, atol=0.0)

  def test_each_examinee_alone_gets_its_row_of_the_batch(self) -> None:
    """Each examinee's log-likelihood alone, at a (2, 3) array of abilities, is its row of the batch's.

    The batch has missing responses, an examinee with none and an item nobody answered, so its terms are no one
    examinee's. An examinee alone that summed a term it did not answer, or laid its values out other than as a
    (1, 2, 3) block, changes a value or the shape.
    """
    responses = np.array([
      [1.0, 0.0, np.nan, 1.0],
      [np.nan, np.nan, np.nan, np.nan],
      [0.0, 1.0, np.nan, np.nan],
      [1.0, 1.0, np.nan, 0.0],
      [0.0, np.nan, np.nan, 0.0],
    ])
    thetas = np.linspace(-2.0, 2.0, 6).reshape(2, 3)

    batch = irt.log_likelihood(thetas, responses, _ITEMS)

    for examinee, row in enumerate(responses):
      alone = irt.log_likelihood(thetas, row[np.newaxis], _ITEMS)
      assert alone.shape == (1, 2, 3)
      np.testing.assert_allclose(alone[0], batch[examinee], rtol=1e-12, atol=0.0)

  @pytest.mark.parametrize("z", [-700.0, -100.0, -37.0, -30.0, -25.0, 0.0, 25.0, 30.0, 37.0, 100.0, 700.0])
  def test_log_likelihood_is_exact_far_from_the_difficulty(self, z: float) -> None:
    """A 2PL item's log P and log Q equal -log(1 + e^-z) and -log(1 + e^z) at any logit.

    Computing log(1 - P) after P rounds toward 1 loses precision past z = 25 and gives -inf past z = 37.
    """
    values = irt.log_likelihood(z, np.array([[1.0], [0.0]]), np.array([[1.0, 0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(values, [-np.logaddexp(0.0, -z), -np.logaddexp(0.0, z)], rtol=1e-12)

  def test_log_likelihood_reaches_the_asymptotes(self) -> None:
    """Far from the difficulty, a guessed correct answer costs log c and a slipped incorrect one log(1 - d)."""
    items = np.array([[1.5, 0.0, 0.2, 1.0], [1.5, 0.0, 0.0, 0.9]])
    responses = np.array([[1.0, np.nan], [np.nan, 0.0]])

    values = irt.log_likelihood(np.array([-500.0, 500.0]), responses, items)

    assert values[0, 0] == pytest.approx(math.log(0.2), rel=1e-12)
    assert values[1, 1] == pytest.approx(math.log(0.1), rel=1e-12)

  def test_log_likelihood_is_zero_without_items(self) -> None:
    """Examinees with no items have log-likelihood 0 at every ability, one row each."""
    values = irt.log_likelihood(np.linspace(-1.0, 1.0, 5), np.empty((3, 0)), np.empty((0, 4)))

    np.testing.assert_array_equal(values, np.zeros((3, 5)))

  def test_log_likelihood_without_examinees_is_empty(self) -> None:
    """No examinees give an empty array with one column per ability; indexing the first examinee would raise."""
    values = irt.log_likelihood(np.linspace(-1.0, 1.0, 5), np.empty((0, 4)), _ITEMS)

    assert values.shape == (0, 5)

  def test_log_likelihood_function_keeps_the_pattern_it_was_built_from(self) -> None:
    """At theta = 1 a right answer to [1, 0, 0, 1] has log P = -log(1 + e^-1), even after the caller reuses its arrays.

    A function that reads the caller's items or responses at each call gives log Q = log 0.5 for the wrong answer to
    [1, 1, 0, 1] written over them.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    responses = np.array([[1.0]])
    log_lik = irt.log_likelihood_function(responses, items)

    items[0, 1] = 1.0
    responses[0, 0] = 0.0

    assert log_lik(1.0)[0] == pytest.approx(-math.log1p(math.exp(-1.0)), rel=1e-12)

  @pytest.mark.parametrize(
    "responses",
    [
      pytest.param(np.array([[True, False], [False, True]]), id="booleans"),
      pytest.param(np.array([[1, 0], [0, 1]], dtype=np.int64), id="integers"),
      pytest.param(np.array([[1, 0], [0, 1]], dtype=np.uint8), id="unsigned"),
      pytest.param([[1, 0], [0, 1]], id="nested lists"),
    ],
  )
  def test_log_likelihood_scores_booleans_and_integers_as_floats(self, responses: object) -> None:
    """True and 1 score as a correct response and False and 0 as an incorrect one, as 1.0 and 0.0 do.

    At theta = 0, a right answer to [1, -1, 0, 1] and a wrong one to [1, 1, 0, 1] each have log-probability
    -log(1 + e^-1), and the other pattern -log(1 + e) each. Fails if a dtype is rejected or misread.
    """
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])

    values = irt.log_likelihood(0.0, responses, items)  # ty: ignore[invalid-argument-type]

    np.testing.assert_allclose(values, [-2 * math.log1p(math.exp(-1.0)), -2 * math.log1p(math.e)], rtol=1e-12)

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  def test_log_likelihood_rejects_a_column_count_other_than_the_items(
    self, build: Callable[[object, FloatArray], object]
  ) -> None:
    """Responses with more columns than items raise when the function is built, instead of broadcasting against them."""
    with pytest.raises(ValueError, match="responses has 2 columns and items 1 rows; there must be one item per column"):
      build(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0, 1.0]]))

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  @pytest.mark.parametrize(
    "items",
    [
      pytest.param(np.array([1.0, 0.0, 0.0, 1.0]), id="one item, 1-D"),
      pytest.param(np.ones((1, 1, 4)), id="3-D"),
      pytest.param(np.ones((1, 3)), id="three columns"),
    ],
  )
  def test_log_likelihood_rejects_items_that_are_not_an_n_by_4_matrix(
    self, build: Callable[[object, FloatArray], object], items: FloatArray
  ) -> None:
    """Items that are not one (a, b, c, d) row per item raise a ValueError when the function is built.

    Fails if a single 1-D item is read as four items, or if a 3-D matrix or a missing column passes construction and
    fails later with an IndexError.
    """
    with pytest.raises(ValueError, match=r"items must be an \(n, 4\) matrix"):
      build(np.array([[1.0]]), items)

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  @pytest.mark.parametrize(
    ("responses", "message"),
    [
      pytest.param(np.array([["1", "0"]]), "responses holds <U1 values", id="strings"),
      pytest.param(np.array([[1.0, None]], dtype=object), "responses holds object values", id="none"),
      pytest.param(np.array([[1 + 0j, 0j]]), "responses holds complex128 values", id="complex"),
    ],
  )
  def test_log_likelihood_rejects_responses_that_are_not_numbers(
    self, build: Callable[[object, FloatArray], object], responses: object, message: str
  ) -> None:
    """Strings, objects such as None for a missing response, and complex numbers raise a TypeError naming the dtype.

    Fails if "1" and "0" are converted silently, or None is read as a missing response, which NaN marks.
    """
    with pytest.raises(TypeError, match=re.escape(message)):
      build(responses, np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  @pytest.mark.parametrize(
    ("responses", "message"),
    [
      pytest.param(
        np.array([[1, 0], [0, 2]]), "responses[1, 1] is 2; an integer response must be 0 or 1", id="integer two"
      ),
      pytest.param(
        np.array([[1.0, 0.5]]),
        "responses[0, 1] is 0.5; a response must be 0 or 1, or NaN where it is missing",
        id="half",
      ),
      pytest.param(
        np.array([[np.nan, 1.0], [np.inf, 0.0]]),
        "responses[1, 0] is inf; a response must be 0 or 1, or NaN where it is missing",
        id="infinity",
      ),
      pytest.param(np.array([[-1, 1]]), "responses[0, 0] is -1; an integer response must be 0 or 1", id="minus one"),
      pytest.param(
        np.array([[1, 2]], dtype=np.uint8),
        "responses[0, 1] is 2; an integer response must be 0 or 1",
        id="unsigned two",
      ),
    ],
  )
  def test_log_likelihood_rejects_a_response_other_than_0_1_or_nan(
    self, build: Callable[[object, FloatArray], object], responses: object, message: str
  ) -> None:
    """A response other than 0, 1 or NaN raises when the function is built, naming its position, value and rule.

    Fails if 2 or -1 count as correct or incorrect, if 0.5 or infinity pass, or if the check runs only once the
    function is given an ability.
    """
    with pytest.raises(ValueError, match=re.escape(message)):
      build(responses, np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  @pytest.mark.parametrize(
    "responses",
    [pytest.param(np.array([1.0, 0.0]), id="one row, 1-D"), pytest.param(np.ones((1, 1, 2)), id="3-D")],
  )
  def test_log_likelihood_rejects_responses_that_are_not_a_matrix(
    self, build: Callable[[object, FloatArray], object], responses: FloatArray
  ) -> None:
    """A vector or a 3-D array of responses raises: a vector could be one examinee or one response per examinee."""
    with pytest.raises(ValueError, match=r"responses must be a matrix, one row per examinee and one column per item"):
      build(responses, np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))

  def test_log_likelihood_of_unsigned_integer_items_is_the_formula(self) -> None:
    """Items stored as unsigned integers give the log-likelihood of the same parameters as floats.

    At theta = 0.5, a correct answer to (1, 0, 0, 1) gives log sigma(0.5) and a wrong one to (2, 1, 0, 1) gives
    log sigma(1). Fails if 1 - d is computed in the items' own dtype, where -d wraps to 255.
    """
    items = np.array([[1, 0, 0, 1], [2, 1, 0, 1]], dtype=np.uint8)
    expected = -math.log1p(math.exp(-0.5)) - math.log1p(math.exp(-1.0))

    values = irt.log_likelihood(0.5, np.array([[1.0, 0.0]]), items)  # ty: ignore[invalid-argument-type]

    assert values[0] == pytest.approx(expected, rel=1e-12)


class TestScaleConversions:
  """The conversions between abilities and scores on a reporting scale."""

  def test_theta_to_scale_maps_the_typical_range_onto_0_to_100_by_default(self) -> None:
    """By default THETA_MIN_TYPICAL = -4 becomes 0, THETA_MAX_TYPICAL = 4 becomes 100 and 2 becomes 75.

    Other default bounds move the end points, and a wrong slope or intercept moves 2.
    """
    np.testing.assert_allclose(irt.theta_to_scale(np.array([-4.0, 4.0, 2.0])), [0.0, 100.0, 75.0], rtol=1e-12)

  def test_theta_to_scale_custom_scale(self) -> None:
    """On a 200 to 800 scale each unit of theta is 75 points, so theta = 1 is 575; a slope of 1 or 100 is not."""
    assert irt.theta_to_scale(1.0, scale_min=200, scale_max=800) == pytest.approx(575.0, rel=1e-12)

  def test_scale_to_theta_maps_0_to_100_onto_the_typical_range_by_default(self) -> None:
    """By default 0 becomes THETA_MIN_TYPICAL = -4, 100 becomes THETA_MAX_TYPICAL = 4 and 75 becomes 2."""
    np.testing.assert_allclose(irt.scale_to_theta(np.array([0.0, 100.0, 75.0])), [-4.0, 4.0, 2.0], rtol=1e-12)

  def test_scale_to_theta_custom_scale(self) -> None:
    """On a 200 to 800 scale 575 is 75 points above the midpoint, theta = 1; multiplying by the slope gives 5625."""
    assert irt.scale_to_theta(575.0, scale_min=200, scale_max=800) == pytest.approx(1.0, rel=1e-12)

  @pytest.mark.parametrize(
    "function", [irt.theta_to_scale, irt.scale_to_theta], ids=["theta_to_scale", "scale_to_theta"]
  )
  @pytest.mark.parametrize(
    ("bounds", "message"),
    [
      pytest.param(
        {"theta_min": 1.0, "theta_max": 1.0}, r"theta_max \(1.0\) must be greater than theta_min \(1.0\)", id="empty"
      ),
      pytest.param(
        {"scale_min": 100, "scale_max": 0}, r"scale_max \(0\) must be greater than scale_min \(100\)", id="reversed"
      ),
    ],
  )
  def test_scale_conversions_reject_an_empty_or_reversed_range(
    self, function: Callable[..., float], bounds: dict[str, float], message: str
  ) -> None:
    """Both reject an empty or reversed range; unchecked, an empty one divides by 0, a reversed one flips."""
    with pytest.raises(ValueError, match=message):
      function(50.0, **bounds)
