"""Tests of drawing a response matrix from abilities and items.

Expected values come from the 4PL probability written out, binomial bounds on the proportions correct, and the raw
uniform numbers of a seeded generator.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from irtlib.drawing import draw_responses

# Three items: a 1PL item, a 3PL item and a 4PL item, so that the probabilities differ from item to item and from the
# logistic function alone.
_ITEMS = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.2, 1.0], [0.8, -0.5, 0.1, 0.9]])


def _probabilities(theta: float) -> list[float]:
  """Return the 4PL probability of a correct response to each of ``_ITEMS`` at ``theta``, written out."""
  return [c + (d - c) / (1 + np.exp(-a * (theta - b))) for a, b, c, d in _ITEMS.tolist()]


def test_the_proportion_correct_is_the_probability_of_a_correct_response() -> None:
  """Over 50,000 examinees at each of two abilities, each item's proportion correct is within 5 binomial SEs of P.

  The bound is 5 standard errors of a proportion, sqrt(P (1 - P) / 50,000), about 0.01. Drawing correct responses
  above P rather than below gives 1 - P, the logistic function without c and d misses the 3PL and 4PL items, and
  pairing the abilities with the wrong rows mixes the two groups.
  """
  n_examinees = 50_000
  abilities = [-1.0, 1.2]

  responses = draw_responses(np.repeat(abilities, n_examinees), _ITEMS, rng=20261127)

  for group, theta in enumerate(abilities):
    proportions = responses[group * n_examinees : (group + 1) * n_examinees].mean(axis=0)
    expected = np.array(_probabilities(theta))
    bound = 5 * np.sqrt(expected * (1 - expected) / n_examinees)
    np.testing.assert_array_less(np.abs(proportions - expected), bound)


def test_the_rows_are_drawn_one_examinee_after_another() -> None:
  """Row j is correct where the generator's next n_items uniform numbers, after rows 0 to j - 1, fall below P.

  The expected rows draw the numbers one examinee at a time, as a caller drawing one examinee per call would. Drawing
  the matrix column by column, or comparing with P at the wrong ability, gives other rows; returning Booleans fails
  the dtype.
  """
  thetas = [-0.5, 0.3, 2.0, -2.0]
  generator = np.random.default_rng(7)
  expected = [(generator.random(len(_ITEMS)) < _probabilities(theta)).astype(float) for theta in thetas]

  responses = draw_responses(np.array(thetas), _ITEMS, rng=7)

  assert responses.dtype == np.float64
  np.testing.assert_array_equal(responses, expected)


def test_the_same_seed_gives_the_same_matrix() -> None:
  """Two calls with seed 5 give the same matrix, and seed 6 another.

  Drawing from NumPy's global random state, or from an unseeded generator, gives another matrix on every call; ignoring
  the seed gives the same matrix for every seed.
  """
  thetas = np.linspace(-2.0, 2.0, 40)

  first = draw_responses(thetas, _ITEMS, rng=5)

  np.testing.assert_array_equal(draw_responses(thetas, _ITEMS, rng=5), first)
  assert not np.array_equal(draw_responses(thetas, _ITEMS, rng=6), first)


def test_a_call_without_rng_gives_another_matrix_each_time() -> None:
  """Without rng each call draws fresh numbers, so two 40 x 3 matrices differ.

  Fails if rng defaults to a fixed seed, which makes every unseeded matrix the same.
  """
  thetas = np.zeros(40)

  assert not np.array_equal(draw_responses(thetas, _ITEMS), draw_responses(thetas, _ITEMS))


@pytest.mark.parametrize(
  ("theta", "error", "message"),
  [
    pytest.param(
      0.5,
      ValueError,
      "theta must hold one ability per examinee, a one-dimensional array, got shape ()",
      id="one number",
    ),
    pytest.param([[0.5, 1.0]], ValueError, "one-dimensional array, got shape (1, 2)", id="a matrix"),
    pytest.param([True, False], TypeError, "theta holds bool values; abilities must be real numbers", id="Booleans"),
    pytest.param(["0.5"], TypeError, "theta holds <U3 values", id="strings"),
    pytest.param([0.5, np.nan], ValueError, "theta[1] is nan; abilities must be finite numbers", id="NaN"),
    pytest.param([-np.inf, 0.5], ValueError, "theta[0] is -inf; abilities must be finite numbers", id="infinite"),
  ],
)
def test_rejects_abilities_that_are_not_one_finite_number_per_examinee(
  theta: object, error: type[Exception], message: str
) -> None:
  """Each broken rule raises, naming the value or its position.

  Without the checks a single number gives a vector instead of a matrix, Booleans are read as abilities 0 and 1, and
  at a NaN ability every response is drawn incorrect.
  """
  with pytest.raises(error, match=re.escape(message)):
    draw_responses(theta, _ITEMS)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
  ("items", "message"),
  [
    pytest.param(np.array([1.0, 0.0, 0.0, 1.0]), "shape (n_items, 4), got shape (4,)", id="one row"),
    pytest.param(np.ones((2, 3)), "shape (n_items, 4), got shape (2, 3)", id="three columns"),
    pytest.param(
      np.array([[1.0, 0.0, 0.0, 1.0], [1.0, np.nan, 0.0, 1.0]]),
      "items[1] is [1.0, nan, 0.0, 1.0]; item parameters must be finite numbers",
      id="NaN",
    ),
    pytest.param(
      np.array([[np.inf, 0.0, 0.0, 1.0]]), "items[0] is [inf, 0.0, 0.0, 1.0]; item parameters must be finite", id="inf"
    ),
  ],
)
def test_rejects_items_that_are_not_a_matrix_of_four_finite_columns(items: np.ndarray, message: str) -> None:
  """A row, a matrix without four columns or a NaN or infinite parameter raises ValueError naming the shape or the item.

  Without the checks one row gives a vector of responses to one item, a NaN parameter makes every response to its item
  incorrect, and an infinite discrimination turns the item into a step at its difficulty.
  """
  with pytest.raises(ValueError, match=re.escape(message)):
    draw_responses(np.zeros(3), items)


@pytest.mark.parametrize(
  "items",
  [
    pytest.param(np.ones((1, 4), dtype=bool), id="Booleans"),
    pytest.param(np.ones((1, 4), dtype=complex), id="complex"),
  ],
)
def test_rejects_items_that_are_not_real_numbers(items: np.ndarray) -> None:
  """Boolean or complex parameters raise TypeError naming their dtype.

  Converting them to floats reads True as a = 1 and drops the imaginary part with a ComplexWarning.
  """
  with pytest.raises(TypeError, match=re.escape(f"items holds {items.dtype} values; item parameters must be real")):
    draw_responses(np.zeros(3), items)


def test_no_examinees_or_no_items_give_an_empty_matrix_of_the_right_shape() -> None:
  """No examinees give a 0 x 3 matrix for three items, and no items a 2 x 0 matrix for two examinees.

  A check that rejected an empty input, or a result shaped items by examinees, fails here.
  """
  assert draw_responses(np.array([]), _ITEMS, rng=1).shape == (0, 3)
  assert draw_responses(np.zeros(2), np.empty((0, 4)), rng=1).shape == (2, 0)
