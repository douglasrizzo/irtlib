"""Tests of the derivatives of the item response function with respect to the item parameters.

Expected values come from hand-worked cases, from central differences of catsim's response probabilities and
log-probabilities, which share no code with the module, and from Baker's (1987, p. 116) identity for the 2PL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from catsim.irt import icc, log_response_probabilities
from scipy.special import expit, logit

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


def _grouped_log_likelihood(parameters: FloatArray, correct: FloatArray, total: FloatArray) -> FloatArray:
  """Return each item's grouped log-likelihood at the nodes ``_THETAS``, from catsim's log-probabilities."""
  log_p, log_q = log_response_probabilities(_THETAS, _to_items(parameters))
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
  """Each derivative equals the central difference of catsim's response probability in that parameter."""
  parameters = _to_parameters(_INTERIOR)
  expected = _central_differences(lambda shifted: icc(_THETAS, _to_items(shifted)), parameters, _STEP)
  np.testing.assert_allclose(probability_gradient(_THETAS, _INTERIOR), expected, atol=1e-9)


def test_probability_gradient_shape() -> None:
  """Every ability against every item, with the four parameters last, as catsim's shapes go."""
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
      log_p, log_q = log_response_probabilities(_THETAS, _to_items(shifted))
      return outcome * log_p + (1 - outcome) * log_q

    return -_central_differences(lambda outer: _central_differences(log_likelihood, outer, step), parameters, step)

  p = icc(_THETAS, _INTERIOR)[..., np.newaxis, np.newaxis]
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
    log_p, log_q = log_response_probabilities(
      _THETAS, np.column_stack([slope, -intercept / slope, np.zeros(2), np.ones(2)])
    )
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
