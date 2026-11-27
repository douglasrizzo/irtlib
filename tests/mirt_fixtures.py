"""Reading the mirt fixtures of ``tests/parity/mirt/``, which ``tests/parity/mirt/generate.R`` writes (ADR-0003).

A data set's responses are stored as patterns with counts, one character per item: ``1`` correct, ``0`` incorrect and
``.`` missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from irtlib.typing import FloatArray

_DATA = Path(__file__).parent / "parity" / "mirt"


def load(name: str) -> dict[str, Any]:
  """Return the fixture ``name``, such as ``"mml_em"``."""
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


def responses(data: dict[str, Any]) -> FloatArray:
  """Expand a data set's patterns into its response matrix, NaN where a response is missing."""
  rows = [[np.nan if char == "." else float(char) for char in pattern] for pattern in data["patterns"]]
  return np.repeat(np.array(rows), data["counts"], axis=0)
