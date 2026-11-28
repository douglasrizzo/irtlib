"""Settings shared by the test suite."""

from collections.abc import Iterator

import matplotlib as mpl

# The tests draw figures without a display, on every platform's runner.
mpl.use("Agg")

import matplotlib.pyplot as plt
import pytest
from hypothesis import settings

# Property examples build arrays whose size, and so whose time, varies with the drawn values, so Hypothesis's 200 ms
# deadline per example would fail them on a slow machine. Each module's settings(...) inherit the profile.
settings.register_profile("irtlib", deadline=None)
settings.load_profile("irtlib")


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
  """Close every figure a test opened, so that none leaks into the next test's figure count."""
  yield
  plt.close("all")
