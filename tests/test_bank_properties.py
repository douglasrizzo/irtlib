"""Property tests of the item bank: how it stores and completes items, its cached values, and its CSV round trip.

The cached values and the test information are compared with the functions of irtlib.irt computed item by item.
"""

from __future__ import annotations

import tempfile
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from irtlib import ItemBank, irt
from tests.strategies import item_matrices

if TYPE_CHECKING:
  from irtlib.typing import FloatArray

# Properties stress invariants rather than sample size.
_SETTINGS = settings(max_examples=25)


@given(items=item_matrices())
@_SETTINGS
def test_the_cached_values_are_each_items_peak_and_its_information_there(items: FloatArray) -> None:
  """Item i's cached peak is irt.max_info_theta of the item, and its cached value irt.information at that peak.

  Each item is computed on its own. Passing every peak to irt.information evaluates each peak against every item, an
  n x n matrix; shifting the difficulty the wrong way evaluates each item away from its peak.
  """
  bank = ItemBank(items)
  peaks = [float(irt.max_info_theta(item)) for item in items]

  np.testing.assert_allclose(bank.max_info_thetas, peaks, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(
    bank.max_info_values, list(starmap(irt.information, zip(peaks, items, strict=True))), rtol=1e-12
  )


@given(items=item_matrices())
@_SETTINGS
def test_a_bank_stores_the_four_parameter_columns(items: FloatArray) -> None:
  """A four-column matrix is stored unchanged; completing or reordering columns it already has would change it."""
  np.testing.assert_array_equal(ItemBank(items).items, items)


@given(
  st.lists(st.floats(-2.5, 2.5, allow_nan=False, allow_infinity=False), min_size=1, max_size=14).map(
    lambda xs: np.asarray(xs, dtype=np.float64).reshape(-1, 1)
  )
)
@_SETTINGS
def test_one_column_is_read_as_difficulties(difficulties: FloatArray) -> None:
  """One column holds b, completed with a = 1, c = 0 and d = 1; reading it as the discrimination would not."""
  items = ItemBank(difficulties).items

  np.testing.assert_array_equal(items[:, 1], difficulties[:, 0])
  np.testing.assert_array_equal(items[:, [0, 2, 3]], np.tile([1.0, 0.0, 1.0], (len(difficulties), 1)))


@given(
  st.lists(
    st.tuples(
      st.floats(0.3, 2.5, allow_nan=False, allow_infinity=False),
      st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=12,
  ).map(lambda rows: np.asarray(rows, dtype=np.float64))
)
@_SETTINGS
def test_two_columns_are_read_as_discrimination_and_difficulty(two_pl: FloatArray) -> None:
  """Two columns hold a and b, completed with c = 0 and d = 1; filling the wrong columns would move them."""
  items = ItemBank(two_pl).items

  np.testing.assert_array_equal(items[:, :2], two_pl)
  np.testing.assert_array_equal(items[:, 2:], np.tile([0.0, 1.0], (len(two_pl), 1)))


@given(
  st.lists(
    st.tuples(
      st.floats(0.3, 2.5, allow_nan=False, allow_infinity=False),
      st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
      st.floats(0.0, 0.25, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=12,
  ).map(lambda rows: np.asarray(rows, dtype=np.float64))
)
@_SETTINGS
def test_three_columns_are_read_as_a_b_and_c(three_pl: FloatArray) -> None:
  """Three columns hold a, b and c, completed with d = 1; completing them with d = 0 would make every item invalid."""
  items = ItemBank(three_pl).items

  np.testing.assert_array_equal(items[:, :3], three_pl)
  np.testing.assert_array_equal(items[:, 3], 1.0)


@given(
  n=st.integers(min_value=12, max_value=80),
  seed=st.integers(min_value=0, max_value=999_999),
  model=st.sampled_from(["1PL", "2PL", "3PL", "4PL"]),
)
@_SETTINGS
def test_generate_gives_n_valid_items(n: int, seed: int, model: str) -> None:
  """Every model and seed gives n items with a > 0 and 0 <= c < d <= 1.

  An invalid draw, such as a <= 0 or c >= d, raises in the bank's checks or fails the asserted ranges.
  """
  bank = ItemBank.generate(n, model=model, rng=seed)

  assert bank.n_items == n
  assert (bank.discrimination > 0).all()
  assert (bank.pseudo_guessing >= 0).all()
  assert (bank.pseudo_guessing < bank.upper_asymptote).all()
  assert (bank.upper_asymptote <= 1).all()


@given(
  items=item_matrices(min_items=2, max_items=12),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
@_SETTINGS
def test_total_information_sums_every_item_by_default(items: FloatArray, theta: float) -> None:
  """Without item indices the test information sums the information of every item; a default subset leaves some out.

  Each item's information is irt.information of that item alone.
  """
  expected = sum(float(irt.information(theta, item)) for item in items)

  assert ItemBank(items).total_information(theta) == pytest.approx(expected, rel=1e-12)


@given(items=item_matrices(min_items=3, max_items=12), data=st.data())
@_SETTINGS
def test_get_items_gives_one_row_per_index_in_order(items: FloatArray, data: st.DataObject) -> None:
  """A tuple of indices, in any order, repeated or empty, gives row k = the parameters of item indices[k].

  Indexing the matrix with the tuple itself reads (1, 2) as one parameter and () as every item, and sorting or
  deduplicating the indices reorders or drops rows.
  """
  indices = tuple(data.draw(st.lists(st.integers(0, items.shape[0] - 1), max_size=8)))

  subset = ItemBank(items).get_items(indices)

  assert subset.shape == (len(indices), 4)
  for row, index in zip(subset, indices, strict=True):
    np.testing.assert_array_equal(row, items[index])


@given(items=item_matrices(), data=st.data())
@_SETTINGS
def test_csv_round_trip_gives_back_the_same_bank(items: FloatArray, data: st.DataObject) -> None:
  """A bank written to CSV and read back has the same parameters, bit for bit, and the same names.

  Names of any text, with delimiters, quotes, line breaks, leading spaces or other scripts, survive only if they are
  quoted where needed and decoded as written; parameters survive only if every digit their floats need is written.
  """
  names = data.draw(st.lists(st.text(min_size=1), min_size=len(items), max_size=len(items), unique=True))
  bank = ItemBank(items, names=names)

  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "bank.csv"
    bank.to_csv(path)
    loaded = ItemBank.from_csv(path)

  assert loaded.names == tuple(names)
  assert loaded.items.tobytes() == items.tobytes()
