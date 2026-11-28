"""Tests of the classical item statistics, beyond the parity tests with mirt.

Expected values come from Rizopoulos's published statistics of the LSAT data, a hand-worked matrix, and the
population values of a 2PL test, by quadrature and by enumerating its response patterns.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from irtlib.classical import item_rest_correlation, proportion_correct
from irtlib.drawing import draw_responses
from tests.mirt_fixtures import load, responses


def test_lsat_statistics_match_rizopoulos() -> None:
  """On LSAT, each item's statistics match those Rizopoulos (2006) prints, to their three decimals.

  The proportions correct are .924, .709, .553, .763 and .870 (p. 5). The item-rest correlations follow in closed
  form from the correlations with the total score that ltm prints as "Biserial correlation with Total Score", .362,
  .566, .618, .534 and .435, and from the frequencies of the total scores, 3, 20, 85, 237, 357 and 298 (p. 6): with
  the item's and the total score's standard deviations s_i and s_t, the rest score's covariance with the item is
  r s_i s_t - s_i², and its variance s_t² + s_i² - 2 r s_i s_t. The rounding of the printed values moves the derived
  correlations by at most 1.4e-3. Fails if the rest score keeps the item, which gives the printed values themselves.
  """
  matrix = responses(load("mml_em")["data"]["lsat6"])
  proportions = np.array([0.924, 0.709, 0.553, 0.763, 0.870])
  with_total = np.array([0.362, 0.566, 0.618, 0.534, 0.435])
  frequencies = np.array([3, 20, 85, 237, 357, 298])
  scores = np.arange(frequencies.size)
  mean = np.sum(frequencies * scores) / frequencies.sum()
  total_sd = np.sqrt(np.sum(frequencies * (scores - mean) ** 2) / frequencies.sum())
  item_sd = np.sqrt(proportions * (1 - proportions))
  with_rest = (with_total * total_sd - item_sd) / np.sqrt(
    total_sd**2 + item_sd**2 - 2 * with_total * item_sd * total_sd
  )
  np.testing.assert_allclose(proportion_correct(matrix), proportions, rtol=0, atol=5e-4)
  np.testing.assert_allclose(item_rest_correlation(matrix), with_rest, rtol=0, atol=1.5e-3)


def test_statistics_of_a_hand_worked_matrix_with_a_missing_response() -> None:
  """Each item's statistics run over the examinees who answered it, with rest scores over the other answered items.

  Five examinees answer three items, rows 111, 110, 100, 000 and 01., the last examinee leaving the third item out.
  The proportions correct are 3/5, 3/5 and 1/4. The first item's rest scores are 2, 1, 0, 0 and 1, so its
  correlation is .6 over the square root of 1.2 times 2.8; the second's are 2, 1, 1, 0 and 0, with the same sums;
  the third's, over the first four examinees, are 2, 2, 1 and 0, so .75 over the square root of .75 times 2.75.
  Fails if a missing response counts as incorrect, or the third item's correlation runs over all five examinees.
  """
  matrix = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, np.nan]])
  np.testing.assert_allclose(proportion_correct(matrix), [0.6, 0.6, 0.25], rtol=1e-15)
  expected = [0.6 / np.sqrt(1.2 * 2.8), 0.6 / np.sqrt(1.2 * 2.8), 0.75 / np.sqrt(0.75 * 2.75)]
  np.testing.assert_allclose(item_rest_correlation(matrix), expected, rtol=1e-14)


def test_a_perfect_correlation_is_exactly_minus_one() -> None:
  """Two items whose responses are each other's complement correlate at exactly -1, not one rounding step below.

  Each item's rest score is the other item's response, so each correlation is that of a response with its own
  complement. Fails if the rounding of the sums carries the correlation to -1.0000000000000002.
  """
  matrix = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]])
  np.testing.assert_array_equal(item_rest_correlation(matrix), [-1.0, -1.0])


@pytest.mark.parametrize(
  ("matrix", "proportions", "correlations"),
  [
    ([[1, np.nan], [0, np.nan], [1, np.nan]], [2 / 3, np.nan], [np.nan, np.nan]),
    ([[1, 1, 0], [1, 0, 1], [1, 1, 1]], [1.0, 2 / 3, 2 / 3], [np.nan, -0.5, -0.5]),
    ([[1], [0], [1]], [2 / 3], [np.nan]),
    ([[1, 0, 1]], [1.0, 0.0, 1.0], [np.nan, np.nan, np.nan]),
  ],
  ids=["unanswered-item", "constant-item", "one-item", "one-examinee"],
)
def test_undefined_statistics_are_nan(
  matrix: list[list[float]], proportions: list[float], correlations: list[float]
) -> None:
  """A statistic with no value is NaN, silently, and the other items keep theirs.

  An item nobody answered has neither statistic; an item every examinee got right, a test of one item, whose rest
  score is always 0, and a single examinee have no correlation. In the constant-item case, the second item's
  responses 1, 0, 1 meet rest scores 1, 2, 2, and the third's 0, 1, 1 meet 2, 1, 2: each a covariance of -1/3 over
  sums of squares of 2/3, a correlation of -.5. Fails if a warning about dividing by 0 escapes, or a constant gives a
  correlation.
  """
  np.testing.assert_allclose(proportion_correct(np.array(matrix)), proportions, rtol=1e-15)
  np.testing.assert_allclose(item_rest_correlation(np.array(matrix)), correlations, rtol=1e-14)


def _pearson(moments: np.ndarray) -> np.ndarray:
  """Return Pearson's correlation of a 0-1 response x and a score R from the means of x, R, xR and R², the last axis."""
  x, score, product, square = np.moveaxis(moments, -1, 0)
  return (product - x * score) / np.sqrt(x * (1 - x) * (square - score**2))


def test_statistics_recover_their_population_values() -> None:
  """On 2PL responses of 20,000 examinees, both statistics fall near their values in the population.

  Each item's population proportion correct integrates its probability of a correct response over the standard
  normal density of abilities; its standard error is [p(1 - p)/N]^(1/2) (Wright & Stone, 1979, p. 25, Equation
  1.7.2). The population item-rest correlation follows from the means of x, R, xR and R² over the 32 response
  patterns, whose probabilities are integrated the same way. The sample correlation is the same function of the
  sample means, so its standard error follows by the delta method, from that function's gradient and the four
  quantities' covariance over the patterns. The bands are four standard errors. Fails if either statistic is computed
  on the wrong scores, such as the total score with the item.
  """
  items = np.array([[0.8, -1.0, 0, 1], [1.2, -0.3, 0, 1], [1.5, 0.2, 0, 1], [1.0, 0.8, 0, 1], [2.0, 1.4, 0, 1]])
  n_examinees = 20_000
  matrix = draw_responses(np.random.default_rng(20261201).normal(size=n_examinees), items, rng=20261202)

  def probability(theta: float) -> np.ndarray:
    """Return each item's probability of a correct response at an ability."""
    return expit(items[:, 0] * (theta - items[:, 1]))

  population = np.array([quad(lambda t, i=i: probability(t)[i] * norm.pdf(t), -10, 10)[0] for i in range(5)])
  patterns = np.array(list(itertools.product((0, 1), repeat=5)), dtype=float)
  masses = np.array([
    quad(lambda t, x=x: np.prod(np.where(x == 1, probability(t), 1 - probability(t))) * norm.pdf(t), -10, 10)[0]
    for x in patterns
  ])
  rest = patterns.sum(axis=1, keepdims=True) - patterns
  quantities = np.stack([patterns, rest, patterns * rest, rest**2], axis=-1)  # patterns x items x 4
  means = np.einsum("p,pik->ik", masses, quantities)
  correlation = _pearson(means)
  step = 1e-6
  gradient = np.stack(
    [(_pearson(means + step * unit) - _pearson(means - step * unit)) / (2 * step) for unit in np.eye(4)], axis=-1
  )
  influence = np.einsum("pik,ik->pi", quantities - means, gradient)
  correlation_se = np.sqrt(masses @ influence**2 / n_examinees)
  np.testing.assert_array_less(
    np.abs(proportion_correct(matrix) - population), 4 * np.sqrt(population * (1 - population) / n_examinees)
  )
  np.testing.assert_array_less(np.abs(item_rest_correlation(matrix) - correlation), 4 * correlation_se)
