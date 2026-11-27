"""Parity tests with mirt, pyitemcal's reference implementation (ADR-0003).

The expected values are mirt's outputs on fixed cases, which ``tests/parity/mirt/generate.R`` wrote, with the R and
mirt versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R.

A data set's responses are stored as patterns with counts, one character per item: ``1`` correct, ``0`` incorrect and
``.`` missing. Each case records the model mirt fitted and its estimates in mirt's parameters, the slope and the
intercept of ``slope * theta + intercept``.

The fixtures are checked against an independent reference before any method uses them: the marginal log-likelihood,
summed by brute force over mirt's grid, at the parameters a fixture records must equal the log-likelihood mirt reports
there. A fixture recorded on a grid of another range, read with the slope and intercept confused, or with missing
responses counted, fails. The number of nodes cannot be checked this way: 41 or more equally spaced nodes on the same
range give the same log-likelihood within the tolerance.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.special import expit, logsumexp

if TYPE_CHECKING:
  from catsim.typing import FloatArray

_DATA = Path(__file__).parent / "parity" / "mirt"

# Closed forms and sums on the same grid: a relative 1e-10 (ADR-0003).
_CLOSED_FORM_RTOL = 1e-10


def _load(name: str) -> dict[str, Any]:
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


_MML_EM = _load("mml_em")


def _responses(data: dict[str, Any]) -> FloatArray:
  """Expand a data set's patterns into its response matrix, NaN where a response is missing."""
  rows = [[np.nan if char == "." else float(char) for char in pattern] for pattern in data["patterns"]]
  return np.repeat(np.array(rows), data["counts"], axis=0)


def _brute_force_log_likelihood(responses: FloatArray, slope: FloatArray, intercept: FloatArray) -> float:
  """Sum, over examinees, the log of the grid sum of the normal masses times the answered items' probabilities."""
  settings = _MML_EM["settings"]
  low, high = settings["theta_limits"]
  nodes = np.linspace(low, high, settings["quadrature_points"])
  log_masses = -(nodes**2) / 2 - np.log(np.sum(np.exp(-(nodes**2) / 2)))
  probability = expit(intercept[np.newaxis, :] + slope[np.newaxis, :] * nodes[:, np.newaxis])  # nodes x items
  answered = ~np.isnan(responses)
  correct = np.where(answered, responses, 0.0)
  incorrect = np.where(answered, 1.0 - responses, 0.0)
  log_likelihood = correct @ np.log(probability).T + incorrect @ np.log1p(-probability).T  # examinees x nodes
  return float(np.sum(logsumexp(log_likelihood + log_masses[np.newaxis, :], axis=1)))


@pytest.mark.parametrize("name", sorted(_MML_EM["data"]))
def test_fixed_parameter_log_likelihood(name: str) -> None:
  """mirt's log-likelihood at fixed 2PL parameters equals the brute-force sum over its 61-node grid.

  Fails if the fixture's grid range, its normal masses or its treatment of missing responses differ from what the
  parity tests assume.
  """
  data = _MML_EM["data"][name]
  fixed = data["fixed"]
  log_likelihood = _brute_force_log_likelihood(_responses(data), np.array(fixed["slope"]), np.array(fixed["intercept"]))
  np.testing.assert_allclose(log_likelihood, fixed["log_likelihood"], rtol=_CLOSED_FORM_RTOL)


@pytest.mark.parametrize("case", _MML_EM["cases"], ids=operator.itemgetter("id"))
def test_estimate_log_likelihood(case: dict[str, Any]) -> None:
  """mirt's log-likelihood at its recorded estimates equals the brute-force sum there.

  Fails if a case's slopes and intercepts are recorded out of place, or the case names the wrong data set.
  """
  responses = _responses(_MML_EM["data"][case["data"]])
  log_likelihood = _brute_force_log_likelihood(responses, np.array(case["slope"]), np.array(case["intercept"]))
  np.testing.assert_allclose(log_likelihood, case["log_likelihood"], rtol=_CLOSED_FORM_RTOL)
