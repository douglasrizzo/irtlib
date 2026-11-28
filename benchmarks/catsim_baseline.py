"""Time the code irtlib took from catsim against catsim's original, and check that the two agree (ADR-0004).

Each function is timed with ``timeit`` on the shapes catsim calls it with: one ability against a bank of 500 items for
the response functions and information; one examinee's responses to 20 items on catsim's default grid of 121 nodes for
the log-likelihood and the posterior; and the same responses, mixed, for the ability estimators, against catsim's
estimators on one step of an adaptive test, with bounds (-6, 6) for maximum likelihood, on that 4PL bank and on a 3PL
bank of 500, the model of most of catsim's banks. The script prints one row per
function, with the ratio of irtlib's time to catsim's, flags a ratio above 1.2, and exits with status 1 if a function's
values differ from catsim's by more than a relative 1e-12. Two cases take another tolerance:

- the estimators that search for a maximum, MAP and ML, agree within their tolerance of 1e-6, in absolute terms, since
  the two packages reach the maximum through different evaluations;
- irtlib's reliability is I / (I + 1) and catsim's 1 - 1 / I, so the check reads catsim's r as 1 / (2 - r).

Run it with the benchmarks' dependency group, which holds catsim at a fixed commit:

    uv run --group benchmarks python benchmarks/catsim_baseline.py
"""

import sys
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import catsim
import numpy
from catsim import estimation as catsim_estimation
from catsim import irt as catsim_irt

import irtlib
from irtlib import ability, irt, quadrature
from irtlib.typing import BoolArray, FloatArray

_N_BANK = 500
_N_ANSWERED = 20
_SEED = 20261127
_RTOL = 1e-12
_SEARCH_TOL = 1e-6
_BOUNDS = (-6.0, 6.0)
_SLOWER = 1.2
_REPEATS = 5


@dataclass(frozen=True, slots=True)
class Case:
  """A moved function and catsim's original, as calls with no arguments, on the shape they are timed on."""

  name: str
  shape: str
  catsim: Callable[[], object]
  irtlib: Callable[[], object]
  from_catsim: Callable[[FloatArray], FloatArray] | None = None
  """irtlib's values from catsim's, where the two define the quantity differently; None where they agree."""
  atol: float = 0.0
  """The absolute difference allowed besides the relative 1e-12: an estimator's search tolerance."""


def _values(result: object) -> FloatArray:
  """Return a result as a float array: a bank's parameters, a grid's nodes and weights, an estimate and its error."""
  if isinstance(result, irtlib.ItemBank | catsim.ItemBank):
    return result.items
  if isinstance(result, quadrature.QuadratureGrid | catsim_estimation.QuadratureGrid):
    return numpy.concatenate([result.nodes, result.weights])
  if isinstance(result, catsim.Estimate):
    return numpy.array([result.theta, result.se])
  if isinstance(result, ability.AbilityEstimates):
    return numpy.concatenate([result.theta, result.standard_errors])
  return numpy.asarray(result, dtype=float)


def _bank_cases(bank: irtlib.ItemBank, theta: float) -> list[Case]:
  """Return the cases of the response functions, information and the bank, at one ability against every item."""
  items = bank.items
  catsim_bank = catsim.ItemBank(items)
  one_bank = f"1 ability, {_N_BANK} items"
  return [
    Case("irt.icc", one_bank, lambda: catsim_irt.icc(theta, items), lambda: irt.icc(theta, items)),
    Case(
      "irt.information", one_bank, lambda: catsim_irt.information(theta, items), lambda: irt.information(theta, items)
    ),
    Case(
      "irt.total_information",
      one_bank,
      lambda: catsim_irt.total_information(theta, items),
      lambda: irt.total_information(theta, items),
    ),
    Case("irt.var", one_bank, lambda: catsim_irt.var(theta, items), lambda: irt.var(theta, items)),
    Case("irt.see", one_bank, lambda: catsim_irt.see(theta, items), lambda: irt.see(theta, items)),
    Case(
      "irt.confidence_interval",
      one_bank,
      lambda: catsim_irt.confidence_interval(theta, items),
      lambda: irt.confidence_interval(theta, items),
    ),
    Case(
      "irt.reliability",
      one_bank,
      lambda: catsim_irt.reliability(theta, items),
      lambda: irt.reliability(theta, items),
      lambda catsim_reliability: 1 / (2 - catsim_reliability),
    ),
    Case(
      "irt.log_response_probabilities",
      one_bank,
      lambda: catsim_irt.log_response_probabilities(theta, items),
      lambda: irt.log_response_probabilities(theta, items),
    ),
    Case(
      "irt.max_info_theta",
      f"{_N_BANK} items",
      lambda: catsim_irt.max_info_theta(items),
      lambda: irt.max_info_theta(items),
    ),
    Case(
      "irt.info_integral",
      f"{_N_BANK} items, [-4, 4]",
      lambda: catsim_irt.info_integral(-4.0, 4.0, items),
      lambda: irt.info_integral(-4.0, 4.0, items),
    ),
    Case(
      "irt.theta_to_scale", "1 ability", lambda: catsim_irt.theta_to_scale(theta), lambda: irt.theta_to_scale(theta)
    ),
    Case("irt.scale_to_theta", "1 score", lambda: catsim_irt.scale_to_theta(58.75), lambda: irt.scale_to_theta(58.75)),
    Case(
      "ItemBank, with its peaks",
      f"{_N_BANK} items",
      lambda: catsim.ItemBank(items).max_info_values,
      lambda: irtlib.ItemBank(items).max_info_values,
    ),
    Case(
      "ItemBank.generate",
      f"{_N_BANK} items",
      lambda: catsim.ItemBank.generate(_N_BANK, model="4PL", rng=_SEED),
      lambda: irtlib.ItemBank.generate(_N_BANK, model="4PL", rng=_SEED),
    ),
    Case("ItemBank.information", one_bank, lambda: catsim_bank.information(theta), lambda: bank.information(theta)),
  ]


def _grid_cases(answered: FloatArray, correct: BoolArray, theta: float) -> list[Case]:
  """Return the cases of the log-likelihood and the posterior, for one examinee's responses on the default grid."""
  matrix = correct.astype(float)[numpy.newaxis, :]
  grid, catsim_grid = quadrature.QuadratureGrid.uniform(), catsim_estimation.QuadratureGrid.uniform()
  prior, catsim_prior = quadrature.normal_log_prior(), catsim_estimation.normal_log_prior()
  likelihood = irt.log_likelihood_function(matrix, answered)
  catsim_likelihood = catsim_irt.log_likelihood_function(correct, answered)
  posterior = quadrature.posterior(irt.log_likelihood(grid.nodes, matrix, answered), grid, prior)
  catsim_posterior = catsim_estimation.posterior(correct, answered, catsim_grid, catsim_prior)
  on_grid = f"{_N_ANSWERED} responses, 121 nodes"
  return [
    Case(
      "irt.log_likelihood",
      on_grid,
      lambda: catsim_irt.log_likelihood(catsim_grid.nodes, correct, answered),
      lambda: irt.log_likelihood(grid.nodes, matrix, answered),
    ),
    Case(
      "irt.log_likelihood_function, a call",
      f"{_N_ANSWERED} responses, 1 ability",
      lambda: catsim_likelihood(theta),
      lambda: likelihood(theta),
    ),
    Case(
      "quadrature.QuadratureGrid.uniform",
      "121 nodes",
      catsim_estimation.QuadratureGrid.uniform,
      quadrature.QuadratureGrid.uniform,
    ),
    Case(
      "quadrature.normal_log_prior, a call", "121 nodes", lambda: catsim_prior(grid.nodes), lambda: prior(grid.nodes)
    ),
    Case(
      "quadrature.posterior, with irt.log_likelihood",
      on_grid,
      lambda: catsim_estimation.posterior(correct, answered, catsim_grid, catsim_prior),
      lambda: quadrature.posterior(irt.log_likelihood(grid.nodes, matrix, answered), grid, prior),
    ),
    Case(
      "quadrature.posterior_mean",
      "121 nodes",
      lambda: catsim_estimation.posterior_mean(catsim_posterior, catsim_grid.nodes),
      lambda: quadrature.posterior_mean(posterior, grid.nodes),
    ),
    Case(
      "quadrature.posterior_variance",
      "121 nodes",
      lambda: catsim_estimation.posterior_variance(catsim_posterior, catsim_grid.nodes),
      lambda: quadrature.posterior_variance(posterior, grid.nodes),
    ),
  ]


def _estimator_cases(bank: irtlib.ItemBank, correct: BoolArray, model: str) -> list[Case]:
  """Return the estimators' cases for responses to the bank's first items; catsim's read a step's prebuilt context."""
  answered = bank.items[: correct.size]
  matrix = correct.astype(float)[numpy.newaxis, :]
  context = catsim.StepContext(
    item_bank=catsim.ItemBank(bank.items),
    rng=numpy.random.default_rng(_SEED),
    administered_items=tuple(range(correct.size)),
    responses=tuple(bool(response) for response in correct),
  )
  eap, map_estimator = catsim_estimation.EAPEstimator(), catsim_estimation.MAPEstimator()
  search = catsim_estimation.NumericalSearchEstimator(bounds=_BOUNDS)
  warm = catsim_estimation.WarmLikelihoodEstimator()
  shape = f"{model}, {correct.size} responses, {int(correct.sum())} correct"
  return [
    Case(
      "ability.expected_a_posteriori",
      shape,
      lambda: eap.estimate(context),
      lambda: ability.expected_a_posteriori(matrix, answered),
    ),
    Case(
      "ability.maximum_a_posteriori",
      shape,
      lambda: map_estimator.estimate(context),
      lambda: ability.maximum_a_posteriori(matrix, answered),
      atol=_SEARCH_TOL,
    ),
    Case(
      "ability.maximum_likelihood, bounds (-6, 6)",
      shape,
      lambda: search.estimate(context),
      lambda: ability.maximum_likelihood(matrix, answered, bounds=_BOUNDS),
      atol=_SEARCH_TOL,
    ),
    Case(
      "ability.weighted_likelihood",
      shape,
      lambda: warm.estimate(context),
      lambda: ability.weighted_likelihood(matrix, answered),
    ),
  ]


def _cases() -> list[Case]:
  """Return every case: banks of 500 4PL and 3PL items, an ability of 0.7, and responses to each bank's first 20."""
  rng = numpy.random.default_rng(_SEED)
  bank = irtlib.ItemBank.generate(_N_BANK, model="4PL", rng=rng)
  theta = 0.7
  answered = bank.items[:_N_ANSWERED]
  correct = rng.random(_N_ANSWERED) < irt.icc(theta, answered)
  bank_3pl = irtlib.ItemBank.generate(_N_BANK, model="3PL", rng=rng)
  correct_3pl = rng.random(_N_ANSWERED) < irt.icc(theta, bank_3pl.items[:_N_ANSWERED])
  return (
    _bank_cases(bank, theta)
    + _grid_cases(answered, correct, theta)
    + _estimator_cases(bank, correct, "4PL")
    + _estimator_cases(bank_3pl, correct_3pl, "3PL")
  )


def _seconds(function: Callable[[], object]) -> float:
  """Return the fastest time of one call, from repeated timeit loops of at least 0.2 seconds each."""
  timer = timeit.Timer(function)
  number, _ = timer.autorange()
  return min(timer.repeat(repeat=_REPEATS, number=number)) / number


def _agree(case: Case) -> bool:
  """Return whether irtlib's values equal catsim's within a relative 1e-12 and the case's absolute tolerance."""
  expected = _values(case.catsim())
  if case.from_catsim is not None:
    expected = case.from_catsim(expected)
  actual = _values(case.irtlib())
  if actual.size != expected.size:
    return False
  return bool(numpy.allclose(actual.reshape(expected.shape), expected, rtol=_RTOL, atol=case.atol, equal_nan=True))


def main() -> int:
  """Time and check every case, write the table to standard output, and return 1 if any case disagrees."""
  header = f"{'function':48} {'shape':28} {'catsim':>10} {'irtlib':>10} {'ratio':>6}  agrees"
  lines = [header, "-" * len(header)]
  failed = False
  for case in _cases():
    agrees = _agree(case)
    failed |= not agrees
    catsim_time, irtlib_time = _seconds(case.catsim), _seconds(case.irtlib)
    ratio = irtlib_time / catsim_time
    flag = f" > {_SLOWER}" if ratio > _SLOWER else ""
    lines.append(
      f"{case.name:48} {case.shape:28} {catsim_time * 1e6:8.2f}us {irtlib_time * 1e6:8.2f}us {ratio:6.2f}  "
      f"{'yes' if agrees else 'NO'}{flag}"
    )
  sys.stdout.write("\n".join(lines) + "\n")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
