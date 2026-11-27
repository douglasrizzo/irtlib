"""Property tests of the item response functions: shapes, monotonicity, round trips and brute-force references."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from irtlib import irt
from tests.strategies import item_matrices, item_rows, response_matrices, scale_mappings

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.typing import FloatArray

# Properties stress invariants rather than sample size.
_SETTINGS = settings(max_examples=30)

_ABILITY = st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False)

_ITEM_FUNCTIONS = [
  pytest.param(irt.icc, id="icc"),
  pytest.param(lambda theta, items: irt.icc_derivatives(theta, items)[0], id="first_derivative"),
  pytest.param(lambda theta, items: irt.icc_derivatives(theta, items)[1], id="second_derivative"),
  pytest.param(irt.information, id="information"),
  pytest.param(lambda _theta, items: irt.max_info_theta(items), id="max_info_theta"),
  pytest.param(lambda theta, items: irt.info_integral(theta - 1.0, theta + 2.0, items), id="info_integral"),
  pytest.param(lambda theta, items: irt.log_response_probabilities(theta, items)[0], id="log_p"),
  pytest.param(lambda theta, items: irt.log_response_probabilities(theta, items)[1], id="log_q"),
]


def _probability(theta: float, items: FloatArray) -> FloatArray:
  """Return the 4PL probability of Magis (2013, p. 306, Equation 1), written out, at one ability for every item."""
  a, b, c, d = items.T
  return c + (d - c) / (1 + np.exp(-a * (theta - b)))


@pytest.mark.parametrize("function", _ITEM_FUNCTIONS)
@given(theta=_ABILITY, items=item_matrices())
@_SETTINGS
def test_a_matrix_gives_each_item_its_value_as_a_row(
  function: Callable[[float, FloatArray], FloatArray], theta: float, items: FloatArray
) -> None:
  """Each item's value from a matrix equals its value as a row.

  Splitting the parameters with items[:, k] fails on a row, and items[k] reads a matrix's rows as parameters.
  """
  np.testing.assert_allclose(function(theta, items), [function(theta, row) for row in items], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("function", _ITEM_FUNCTIONS[:4])
@given(thetas=st.lists(_ABILITY, min_size=1, max_size=6), items=item_matrices())
@_SETTINGS
def test_k_abilities_give_one_row_of_item_values_per_ability(
  function: Callable[[float | FloatArray, FloatArray], FloatArray], thetas: list[float], items: FloatArray
) -> None:
  """k abilities and n items give a (k, n) array whose row j holds the values at thetas[j].

  Pairing ability j with item j element-wise gives another shape, and raises when k != n.
  """
  values = function(np.array(thetas), items)

  assert values.shape == (len(thetas), len(items))
  np.testing.assert_allclose(values, [function(theta, items) for theta in thetas], rtol=1e-12, atol=1e-15)


@given(
  theta=st.floats(-4.0, 3.5, allow_nan=False, allow_infinity=False),
  delta=st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False),
  row=item_rows(),
)
@_SETTINGS
def test_icc_lies_between_its_asymptotes_and_never_falls(
  theta: float, delta: float, row: tuple[float, float, float, float]
) -> None:
  """P lies between c and d, and a higher ability never lowers it; a sign error in theta - b reverses the order."""
  item = np.array(row)
  low, high = irt.icc(theta, item), irt.icc(theta + delta, item)

  assert row[2] <= low <= high + 1e-12
  assert high <= row[3]


@given(data=st.data(), items=item_matrices(max_items=8), thetas=st.lists(_ABILITY, min_size=1, max_size=5))
@_SETTINGS
def test_log_likelihood_sums_the_log_probabilities_of_each_examinees_answered_items(
  data: st.DataObject, items: FloatArray, thetas: list[float]
) -> None:
  """Each examinee's value is the sum of log P or log(1 - P), from the 4PL written out, over the items they answered.

  The response matrices have missing responses anywhere, examinees with none and items nobody answered. The tolerance
  bounds the reference's own error: rounding P by a few eps moves its log P or log(1 - P) by about eps / P or
  eps / (1 - P).
  """
  responses = data.draw(response_matrices(len(items)))

  values = irt.log_likelihood(np.array(thetas), responses, items)

  assert values.shape == (responses.shape[0], len(thetas))
  for examinee, row in enumerate(responses):
    answered = ~np.isnan(row)
    correct = row[answered] == 1
    for value, theta in zip(values[examinee], thetas, strict=True):
      p = _probability(theta, items[answered])
      selected = np.where(correct, p, 1 - p)
      tolerance = 1e-12 + 8 * np.finfo(float).eps * float(np.sum(1 / selected))
      assert value == pytest.approx(float(np.sum(np.log(selected))), abs=tolerance)


@given(mapping=scale_mappings())
@_SETTINGS
def test_theta_scale_round_trip_identity(mapping: tuple[float, float, float, float]) -> None:
  """scale_to_theta undoes theta_to_scale for any ranges; a slope or intercept that differs between the two does not."""
  theta_min, theta_max, scale_min, scale_max = mapping
  theta = np.linspace(theta_min, theta_max, num=5)

  scores = irt.theta_to_scale(theta, scale_min, scale_max, theta_min, theta_max)

  np.testing.assert_allclose(irt.scale_to_theta(scores, scale_min, scale_max, theta_min, theta_max), theta, atol=1e-9)


@given(items=item_matrices(max_items=8))
@_SETTINGS
def test_max_info_theta_is_the_peak_of_the_information(items: FloatArray) -> None:
  """No ability within 1.2 of an item's max_info_theta gives more information, by Magis's Equation 6 written out.

  A wrong closed form, or the 3PL formula applied where d < 1, moves the peak.
  """
  for item, peak in zip(items, irt.max_info_theta(items), strict=True):
    a, _, c, d = item
    grid = np.linspace(peak - 1.2, peak + 1.2, num=25)
    p = np.array([_probability(theta, item[np.newaxis, :])[0] for theta in [peak, *grid]])
    information = a**2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))
    assert information[0] >= np.max(information[1:]) - 1e-6
