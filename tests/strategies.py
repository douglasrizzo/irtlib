"""Hypothesis strategies shared by the property tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

if TYPE_CHECKING:
  from irtlib.typing import FloatArray


def _floats(low: float, high: float) -> st.SearchStrategy[float]:
  """Finite floats between ``low`` and ``high``."""
  return st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)


@composite
def item_rows(draw: DrawFn) -> tuple[float, float, float, float]:
  """One valid item ``(a, b, c, d)``, with ``c = 0`` and ``d = 1``, the 2PL and 3PL bounds, drawn often."""
  a = draw(_floats(0.25, 2.5))
  b = draw(_floats(-3.5, 3.5))
  c = draw(st.one_of(st.just(0.0), _floats(0.0, 0.25)))
  d = draw(st.one_of(st.just(1.0), _floats(0.55, 1.0)))
  return (a, b, c, d)


@composite
def item_matrices(draw: DrawFn, *, min_items: int = 1, max_items: int = 10) -> FloatArray:
  """An ``(n, 4)`` matrix of valid items."""
  n_items = draw(st.integers(min_value=min_items, max_value=max_items))
  return np.array([draw(item_rows()) for _ in range(n_items)], dtype=float)


@composite
def response_matrices(draw: DrawFn, n_items: int, *, max_examinees: int = 6) -> FloatArray:
  """A response matrix of 1, 0 and NaN, with one column per item, in which any row or column may be all NaN."""
  n_examinees = draw(st.integers(min_value=1, max_value=max_examinees))
  cells = draw(
    st.lists(st.sampled_from([0.0, 1.0, np.nan]), min_size=n_examinees * n_items, max_size=n_examinees * n_items)
  )
  return np.array(cells, dtype=float).reshape(n_examinees, n_items)


@composite
def scale_mappings(draw: DrawFn) -> tuple[float, float, float, float]:
  """``theta_min, theta_max, scale_min, scale_max``, each maximum above its minimum."""
  theta_min = draw(_floats(-5.0, 2.0))
  theta_max = draw(_floats(theta_min + 0.5, 6.0))
  scale_min = draw(_floats(-200.0, 400.0))
  scale_max = draw(_floats(scale_min + 1.0, 900.0))
  return (theta_min, theta_max, scale_min, scale_max)
