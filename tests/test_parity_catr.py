"""Parity tests with catR, the second reference implementation of the response functions (ADR-0003).

The expected values are catR's outputs on fixed cases, which ``tests/parity/catr/generate.R`` wrote, with the R and
catR versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R.

The fixtures are checked against an independent reference before irtlib is compared with them: catR's probabilities and
information at the recorded abilities and items must equal the 4PL of Magis (2013, pp. 306-307, Equations 1 and 6),
written out here.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from irtlib import irt

if TYPE_CHECKING:
  from irtlib.typing import FloatArray

_DATA = Path(__file__).parent / "parity" / "catr"

# Closed forms: a relative 1e-10 (ADR-0003).
_CLOSED_FORM_RTOL = 1e-10


def _load(name: str) -> dict[str, Any]:
  """Return the fixture ``name``, such as ``"irt"``."""
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


_IRT = _load("irt")


def _probability(thetas: FloatArray, items: FloatArray) -> FloatArray:
  """Return the 4PL probability, written out, every ability against every item."""
  a, b, c, d = items.T
  return c + (d - c) / (1 + np.exp(-a * (thetas[:, np.newaxis] - b)))


class TestData:
  """The recorded data come from the pinned catR and match the 4PL written out."""

  def test_recorded_with_the_pinned_catr(self) -> None:
    """A catR version pinned in generate.R but never run would leave the fixture recorded with another version."""
    pinned = re.search(r'^CATR_VERSION <- "([^"]+)"$', (_DATA / "generate.R").read_text(encoding="utf-8"), re.MULTILINE)

    assert pinned is not None
    assert _IRT["catr_version"] == pinned.group(1)

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_probabilities_are_the_4pl(self, case: dict[str, Any]) -> None:
    """catR's Pi is Magis's Equation 1 at the recorded abilities and items.

    A fixture recorded with catR's constant D at 1.7, with the item columns in another order, or with the abilities and
    items paired differently fails.
    """
    expected = _probability(np.array(case["thetas"]), np.array(case["items"]))

    np.testing.assert_allclose(case["probability"], expected, rtol=_CLOSED_FORM_RTOL)

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_information_is_the_4pl_information(self, case: dict[str, Any]) -> None:
    """catR's Ii is Magis's Equation 6 at the recorded abilities and items; a fixture of another quantity fails."""
    items = np.array(case["items"])
    _, _, c, d = items.T
    p = _probability(np.array(case["thetas"]), items)
    expected = items[:, 0] ** 2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))

    np.testing.assert_allclose(case["information"], expected, rtol=_CLOSED_FORM_RTOL)


class TestItemResponseFunctions:
  """catR's Pi and Ii, on 1PL to 4PL items, against irtlib.irt."""

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_probability(self, case: dict[str, Any]) -> None:
    """A wrong logistic, or a guessing or slipping parameter entering it the wrong way, would move icc off catR's Pi."""
    np.testing.assert_allclose(
      irt.icc(np.array(case["thetas"]), np.array(case["items"])), case["probability"], rtol=_CLOSED_FORM_RTOL
    )

  @pytest.mark.parametrize("case", _IRT["cases"], ids=operator.itemgetter("id"))
  def test_information(self, case: dict[str, Any]) -> None:
    """An information formula that drops c or d, or squares the wrong term, would move off catR's Ii."""
    np.testing.assert_allclose(
      irt.information(np.array(case["thetas"]), np.array(case["items"])), case["information"], rtol=_CLOSED_FORM_RTOL
    )
