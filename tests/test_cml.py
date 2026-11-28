"""Tests of conditional maximum likelihood calibration of the Rasch model.

Expected values come from the published values and the brute force ADR-0003 asks of a method with no parity cases:
Baker and Harwell's (1996) symmetric functions, Bock and Aitkin's (1981) and Baker and Harwell's LSAT-6 estimates,
Andersen's (1970) two-item closed form, sums over subsets of items, and the conditional likelihood written out over
subsets of items and maximized by SciPy, with recovery of generating difficulties.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import minimize

from irtlib.calibration import ConvergenceWarning, Likelihood
from irtlib.cml import (
  CMLCalibrator,
  _log_symmetric_functions,  # noqa: PLC2701 - the summation algorithm is checked against published values
  _newton_raphson,  # noqa: PLC2701 - a stalled iteration needs starting values the calibrator never uses
)
from irtlib.drawing import draw_responses
from tests.mirt_fixtures import load, responses

if TYPE_CHECKING:
  from irtlib.typing import FloatArray

_COMPARISON = load("comparison")
_LSAT6 = responses(_COMPARISON["data"]["lsat6"])
_LSAT7 = responses(_COMPARISON["data"]["lsat7"])
# Bock and Aitkin's (1981, p. 455, Table 3) CML difficulties, from Andersen (1980), printed to three decimals; each
# section's sums to zero
_BOCK_AITKIN = {
  "LSAT-6": (_LSAT6, [-1.256, 0.475, 1.236, 0.168, -0.623]),
  "LSAT-7": (_LSAT7, [-0.641, 0.583, -0.134, 0.758, -0.566]),
}
_LSAT7_IS_OTHER_DATA = (
  "Bock and Aitkin's LSAT-7 CML column is not the conditional maximum of mirt's LSAT7, Bock and Lieberman's data, "
  "whose MML column mirt reproduces: the brute-force conditional log-likelihood there is -1184.32, against -1182.70 at "
  "the maximum; and twice the gain in conditional log-likelihood from estimating the difficulties within each "
  "number-correct group, which gives their LSAT-6 G2 of 3.1 on 12 degrees of freedom as 3.14, gives 31.3 for LSAT-7 "
  "where they print 35.9 (measured 2026-11-27)"
)


def _brute_force_log_likelihood(difficulty: FloatArray, matrix: FloatArray) -> float:
  """Return the conditional log-likelihood, each symmetric function summed over the subsets of items written out.

  Examinees with no item or every item correct are left out, as Engelen (1987, p. 11) leaves them out.
  """
  n_items = matrix.shape[1]
  scores = matrix.sum(axis=1).astype(int)
  kept = (scores > 0) & (scores < n_items)
  easiness = np.exp(-difficulty)
  gamma = [
    sum(math.prod(easiness[list(subset)]) for subset in itertools.combinations(range(n_items), r))
    for r in range(n_items + 1)
  ]
  return float(-matrix[kept].sum(axis=0) @ difficulty - sum(math.log(gamma[r]) for r in scores[kept]))


def _free_to_centred(free: FloatArray) -> FloatArray:
  """Return the difficulties that sum to zero whose first entries are ``free``."""
  return np.append(free, -free.sum())


def _brute_force_maximum(matrix: FloatArray) -> tuple[FloatArray, float]:
  """Return the difficulties, summing to zero, that maximize the brute-force conditional likelihood, and its maximum."""
  result = minimize(
    lambda free: -_brute_force_log_likelihood(_free_to_centred(free), matrix),
    np.zeros(matrix.shape[1] - 1),
    method="BFGS",
    options={"gtol": 1e-9},
  )
  return _free_to_centred(result.x), -float(result.fun)


def test_symmetric_functions_match_baker_and_harwell() -> None:
  """The symmetric functions and their first derivatives match Baker and Harwell's (1996, p. 173, Table 3) for LSAT-6.

  Each derivative is the symmetric function without the item, the summation algorithm run with its easiness left out
  (p. 182). The table's marginal row is not compared: by r gamma_r = sum_i eps_i gamma_{r-1}^(i) (p. 171, Equation
  14), its entries for r = 4 and 5 are misprinted. The values are printed to four decimals from easiness values printed
  to four.
  """
  easiness = np.array([3.5118, 0.6219, 0.2905, 0.8450, 1.8648])
  gamma = [7.1340, 16.9493, 16.7781, 7.0529, 0.9997]
  derivatives = [
    [1, 3.6222, 4.2288, 1.9273, 0.2847],
    [1, 6.5121, 12.8994, 8.7560, 1.6076],
    [1, 6.8435, 14.9612, 12.4319, 3.4414],
    [1, 6.2890, 11.6351, 6.9465, 1.1831],
    [1, 5.2692, 7.1233, 3.4946, 0.5361],
  ]
  np.testing.assert_allclose(np.exp(_log_symmetric_functions(np.log(easiness)))[1:], gamma, atol=1e-4)
  without_each = np.where(np.eye(5, dtype=bool), -np.inf, np.log(easiness))
  np.testing.assert_allclose(np.exp(_log_symmetric_functions(without_each))[:, :5], derivatives, atol=1e-4)


def test_symmetric_functions_are_sums_over_subsets() -> None:
  """For four items, each symmetric function is the sum of the products of the easiness values of every subset of its
  order, with the items left out by a log-easiness of -inf excluded from the subsets.
  """
  easiness = np.array([0.3, 2.5, 1.1, 7.0])
  for left_out in [(), (1,), (0, 3)]:
    included = [item for item in range(4) if item not in left_out]
    expected = [
      sum(math.prod(easiness[list(subset)]) for subset in itertools.combinations(included, r)) for r in range(5)
    ]
    log_easiness = np.log(easiness)
    log_easiness[list(left_out)] = -np.inf
    np.testing.assert_allclose(np.exp(_log_symmetric_functions(log_easiness)), expected, rtol=1e-14)


@pytest.mark.parametrize(("right_wrong", "wrong_right"), [(11, 1), (30, 45), (7, 7)])
def test_two_items_closed_form(right_wrong: int, wrong_right: int) -> None:
  """With two items, b_2 - b_1 = log n_10 - log n_01, the counts of each mixed pattern (Andersen, 1970, p. 299).

  Examinees with both items right or both wrong add nothing. Fails if each Newton-Raphson step is taken in full: from
  the starting values, full steps diverge on 11 to 1.
  """
  matrix = np.repeat(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float), [right_wrong, wrong_right, 5, 3], axis=0)
  difference = math.log(right_wrong) - math.log(wrong_right)
  calibration = CMLCalibrator().calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 1], [-difference / 2, difference / 2], atol=1e-9)
  assert calibration.converged


def test_lsat6_published_estimates() -> None:
  """The LSAT-6 difficulties match Bock and Aitkin's (1981, p. 455, Table 3) and Baker and Harwell's (1996, p. 171).

  Baker and Harwell print the easiness values, with a product of 1, to four decimals, so the difficulties -log(eps)
  are good to about 2e-4; Bock and Aitkin print three decimals.
  """
  difficulty = CMLCalibrator().calibrate(_LSAT6).estimates[:, 1]
  np.testing.assert_allclose(difficulty, _BOCK_AITKIN["LSAT-6"][1], atol=5e-4)
  np.testing.assert_allclose(difficulty, -np.log([3.5118, 0.6219, 0.2905, 0.8450, 1.8648]), atol=2e-4)


@pytest.mark.xfail(reason=_LSAT7_IS_OTHER_DATA, strict=True)
def test_lsat7_published_estimates() -> None:
  """The LSAT-7 difficulties match Bock and Aitkin's (1981, p. 455, Table 3) to the three decimals printed."""
  difficulty = CMLCalibrator().calibrate(_LSAT7).estimates[:, 1]
  np.testing.assert_allclose(difficulty, _BOCK_AITKIN["LSAT-7"][1], atol=5e-4)


@pytest.mark.parametrize("name", ["LSAT-6", "LSAT-7"])
def test_estimates_maximize_the_conditional_likelihood(name: str) -> None:
  """The estimates and the log-likelihood are the maximum of the conditional likelihood written out by brute force.

  SciPy's BFGS maximizes the brute-force log-likelihood over difficulties that sum to zero.
  """
  matrix = _BOCK_AITKIN[name][0]
  expected, maximum = _brute_force_maximum(matrix)
  calibration = CMLCalibrator().calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 1], expected, atol=1e-6)
  np.testing.assert_allclose(calibration.log_likelihood, maximum, rtol=1e-12)


@pytest.mark.parametrize("name", ["LSAT-6", "LSAT-7"])
def test_variances_invert_the_brute_force_hessian(name: str) -> None:
  """Each item's variance of b is the diagonal of the inverse of the brute-force log-likelihood's Hessian, centred to
  sum to zero, and its other covariances are NaN, a, c and d not being estimated.

  The Hessian is differenced centrally in the first four difficulties, the fifth their negated sum, so the covariance
  of all five is L H^-1 L^T for the map L from the four to the five. A Hessian missing the pairs' second derivatives,
  or variances of the first difficulty set to 0, fails.
  """
  matrix = _BOCK_AITKIN[name][0]
  calibration = CMLCalibrator().calibrate(matrix)
  free = calibration.estimates[:4, 1]
  step = 1e-4
  hessian = np.empty((4, 4))
  for row, column in itertools.product(range(4), repeat=2):
    values = []
    for sign_row, sign_column in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
      shifted = free.copy()
      shifted[row] += sign_row * step
      shifted[column] += sign_column * step
      values.append(sign_row * sign_column * _brute_force_log_likelihood(_free_to_centred(shifted), matrix))
    hessian[row, column] = sum(values) / (4 * step**2)
  to_centred = np.vstack([np.eye(4), -np.ones(4)])
  covariance = to_centred @ np.linalg.inv(-hessian) @ to_centred.T
  assert calibration.covariances is not None
  # a variance is a standard error squared, so it agrees within twice a standard error's relative 1e-5
  np.testing.assert_allclose(calibration.covariances[:, 1, 1], np.diag(covariance), rtol=2e-5)
  others = np.ones((4, 4), dtype=bool)
  others[1, 1] = False
  assert np.all(np.isnan(calibration.covariances[:, others]))


def test_estimates_do_not_depend_on_the_ability_distribution() -> None:
  """Difficulties from 5,000 examinees recover the generating ones, centred, whether abilities are normal or bimodal.

  Abilities half near -2 and half near 2 leave the estimates within 0.15 of the generating difficulties, as normal
  abilities do: the largest error over 40 seeds was 0.13, with standard errors up to 0.04. Marginal maximum likelihood
  with slopes 1 and normal abilities misses by 0.61 on these bimodal responses (measured 2026-11-27), so an estimator
  that assumed normal abilities fails.
  """
  difficulty = np.linspace(-2, 2, 9)
  items = np.column_stack([np.ones(9), difficulty, np.zeros(9), np.ones(9)])
  rng = np.random.default_rng(3)
  normal = rng.normal(size=5000)
  bimodal = np.where(rng.random(5000) < 0.5, -2.0, 2.0) + 0.3 * rng.normal(size=5000)
  for abilities in (normal, bimodal):
    calibration = CMLCalibrator().calibrate(draw_responses(abilities, items, rng=rng))
    np.testing.assert_allclose(calibration.estimates[:, 1], difficulty - difficulty.mean(), atol=0.15)


def test_examinees_with_no_item_or_every_item_correct_change_nothing() -> None:
  """Adding examinees with no item and with every item correct leaves the estimates, standard errors and log-likelihood
  exactly as they were.

  Exact equality: fails if they move the estimates by any amount, through the starting values too, or are counted
  with the examinees of another number correct.
  """
  extremes = np.vstack([_LSAT6, np.zeros((40, 5)), np.ones((300, 5))])
  before, after = CMLCalibrator().calibrate(_LSAT6), CMLCalibrator().calibrate(extremes)
  np.testing.assert_array_equal(after.estimates, before.estimates)
  assert before.standard_errors is not None
  assert after.standard_errors is not None
  np.testing.assert_array_equal(after.standard_errors, before.standard_errors)
  assert after.log_likelihood == before.log_likelihood


def test_items_answered_by_every_examinee_left_have_no_estimate() -> None:
  """Items answered correctly by every examinee get no estimate, and the leaving out repeats until nothing changes.

  Four items, with 5 examinees answering 1001, 12 answering 1101, 4 answering 1011 and 6 answering 0001. The fourth
  item, right for all, goes first; without it, the 0001 examinees have no item right and go; the first item is then
  right for every examinee left and goes; without it, the 1001 examinees have no item right and go. The second and
  third items follow the two-item closed form of the 12 and the 4 examinees (Andersen, 1970, p. 299). A single pass
  keeps the first item, whose estimate then diverges. The bank leaves the first and fourth out with a warning naming
  them.
  """
  patterns = np.array([[1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]], dtype=float)
  matrix = np.repeat(patterns, [5, 12, 4, 6], axis=0)
  with pytest.warns(RuntimeWarning, match=r"NaN or infinite parameters for 0 \('item_0'\), 3 \('item_3'\)") as caught:
    calibration = CMLCalibrator().calibrate(matrix)
  assert caught[0].filename == __file__
  difference = math.log(12) - math.log(4)
  np.testing.assert_allclose(calibration.estimates[:, 1], [np.nan, -difference / 2, difference / 2, np.nan], atol=1e-9)
  assert calibration.standard_errors is not None
  np.testing.assert_array_equal(np.isnan(calibration.standard_errors[:, 1]), [True, False, False, True])
  assert calibration.items is not None
  assert calibration.items.names == ("item_1", "item_2")
  assert calibration.n_parameters == 1
  np.testing.assert_array_equal(calibration.item_parameters, [0, 1, 1, 0])


# Responses with no item answered by all or none whose conditional likelihood still has no maximum, each row one
# examinee, with the number correct and the groups of columns Haberman's (2004, pp. 37-38) Theorem 10 splits them into:
# the examinees with at most that many correct answered none of the harder group, and the others all of the easier.
_NO_MAXIMUM = {
  "two pairs": ([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], 1, [0, 1], [2, 3]),
  "two pairs weighted": (
    [[1, 0, 0, 0]] * 5 + [[0, 1, 0, 0]] * 5 + [[1, 1, 0, 0]] * 10 + [[1, 1, 1, 0]] * 5 + [[1, 1, 0, 1]] * 5,
    1,
    [0, 1],
    [2, 3],
  ),
  "interleaved": (
    [[1, 0, 1, 1], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]],
    1,
    [0, 2],
    [1, 3],
  ),
  "interleaved, no 2 correct": (
    [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]],
    1,
    [0, 2],
    [1, 3],
  ),
  "after an item right for all": (
    [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 1]],
    1,
    [1, 2],
    [3, 4],
  ),
}


@pytest.mark.parametrize(("rows", "score", "easier", "harder"), _NO_MAXIMUM.values(), ids=_NO_MAXIMUM)
def test_responses_with_no_maximum_are_refused(
  rows: list[list[int]], score: int, easier: list[int], harder: list[int]
) -> None:
  """Responses whose conditional likelihood has no maximum are refused, naming the two groups of items they split into.

  In each, worked by hand, the examinees with at most ``score`` correct answered none of the harder items and the
  others all of the easier ones, so the likelihood rises without bound as the groups move apart (Haberman, 2004,
  pp. 37-38, Theorem 10), and under the sum to zero no difficulty has a finite estimate. Fails if such responses return
  estimates: Newton-Raphson runs towards the supremum, where its steps underflow and pass the stopping rule, as they
  did with differences between the groups of 35 to 124. The last case names the columns of the responses, not of the
  items kept after the first, right for all, is left out.
  """
  message = (
    rf"at most {score} correct answered none of the items in columns {re.escape(str(harder))} correctly, and those "
    rf"with more answered all of the items in columns {re.escape(str(easier))}"
  )
  with pytest.raises(ValueError, match=message):
    CMLCalibrator().calibrate(np.array(rows, dtype=float))


def test_responses_with_a_maximum_are_calibrated_however_close_to_a_split() -> None:
  """Responses whose examinees at the cut's own number correct break every split of Theorem 10 are calibrated.

  Three examinees answer the first item, the second, and the first and third. The two with one correct answered the
  first item, of the largest total, and the second, of the smallest, so every cut of the totals leaves an item they
  answered below it; the estimates are the brute-force maximum of the conditional likelihood. Fails if the examinees
  with exactly the cut's number correct are left out of the low group, which finds a split here and refuses the
  responses.
  """
  matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=float)
  expected, _ = _brute_force_maximum(matrix)
  np.testing.assert_allclose(CMLCalibrator().calibrate(matrix).estimates[:, 1], expected, atol=1e-6)


def test_a_stalled_iteration_records_the_log_likelihood_where_it_stopped() -> None:
  """An iteration whose step no halving keeps from lowering the conditional log-likelihood counts as one iteration and
  records the log-likelihood at the difficulties it leaves as they were.

  Two items, the first answered right alone by 10 examinees and the second by 5, start from difficulties -300 and 300,
  where the information is about 1e-260: the Newton-Raphson step, about 2.5e260, overshoots the maximum even halved 30
  times. The log-likelihood there is -(10 (-300) + 5 (300)) - 15 log(e^300 + e^-300) = -3000 to double precision.
  Andersen's starting values never stall like this, so the iterations are run from these directly. Fails if the stalled
  iteration records nothing, which would leave a calibration no last value, or records a refused trial.
  """
  start = np.array([-300.0, 300.0])
  difficulty, log_likelihoods, _, failure = _newton_raphson(
    start, np.array([10.0, 5.0]), np.array([0.0, 15.0, 0.0]), tolerance=1e-5, max_cycles=100
  )
  assert failure is not None
  assert failure.startswith("halving the Newton-Raphson step 30 times")
  np.testing.assert_array_equal(difficulty, start)
  assert log_likelihoods == [pytest.approx(-3000.0, rel=1e-15)]


def test_a_halved_step_under_the_tolerance_is_not_convergence() -> None:
  """A step that halving shrank below the tolerance does not stop the iterations; only a full step can.

  On two items answered 1,000 to 1, the first full step moves a difficulty by 499 and is halved 7 times, to 3.9. With a
  tolerance of 4, stopping there leaves the difficulties 0.44 from the closed form of Andersen (1970, p. 299); the
  next full step, of 0.29, meets the rule and leaves them within 0.2.
  """
  matrix = np.repeat(np.array([[1, 0], [0, 1]], dtype=float), [1000, 1], axis=0)
  difference = math.log(1000)
  calibration = CMLCalibrator(tolerance=4.0).calibrate(matrix)
  np.testing.assert_allclose(calibration.estimates[:, 1], [-difference / 2, difference / 2], atol=0.2)


@pytest.mark.parametrize(
  "matrix",
  [_LSAT6, np.repeat(np.array([[1, 0], [0, 1]], dtype=float), [1000, 1], axis=0)],
  ids=["LSAT-6", "two items 1000 to 1"],
)
def test_log_likelihoods_are_those_after_each_iteration(matrix: FloatArray) -> None:
  """The conditional log-likelihood recorded for each iteration is the brute-force one at the difficulties of a
  calibration stopped after that iteration; the last is the calibration's own.

  The two items' first step is halved 7 times, so a record of a refused trial, or of the log-likelihood before an
  iteration's step rather than after it, fails.
  """
  calibration = CMLCalibrator(tolerance=1e-10).calibrate(matrix)
  expected = []
  for iterations in range(1, calibration.cycles):
    with pytest.warns(ConvergenceWarning):
      stopped = CMLCalibrator(tolerance=1e-10, max_cycles=iterations).calibrate(matrix)
    expected.append(_brute_force_log_likelihood(stopped.estimates[:, 1], matrix))
  expected.append(_brute_force_log_likelihood(calibration.estimates[:, 1], matrix))
  np.testing.assert_allclose(calibration.log_likelihoods, expected, rtol=1e-12)
  assert calibration.log_likelihoods[-1] == calibration.log_likelihood


def test_the_result_is_a_conditional_calibration_without_a_grid() -> None:
  """The result names the conditional likelihood, has no grid, and counts one difficulty fewer than the items."""
  calibration = CMLCalibrator().calibrate(_LSAT7)
  assert calibration.likelihood is Likelihood.CONDITIONAL
  assert calibration.grid is None
  assert calibration.n_parameters == 4
  np.testing.assert_array_equal(calibration.estimates[:, [0, 2, 3]], np.tile([1.0, 0.0, 1.0], (5, 1)))


def test_running_out_of_iterations_is_flagged() -> None:
  """A calibration that runs out of iterations returns its estimates flagged as not converged, and warns."""
  with pytest.warns(ConvergenceWarning, match="1 iterations"):
    calibration = CMLCalibrator(max_cycles=1).calibrate(_LSAT6)
  assert not calibration.converged
  assert calibration.cycles == 1
  assert calibration.change > 1e-5


@pytest.mark.parametrize(
  ("matrix", "message"),
  [
    (np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 0.0]]), "complete responses; irtlib.mml.MMLCalibrator"),
    (np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), "no item has a conditional estimate"),
    (np.array([[1.0, 2.0], [0.0, 1.0]]), r"responses\[0, 1\] is 2.0"),
  ],
  ids=["missing", "no item left", "value 2"],
)
def test_invalid_responses(matrix: np.ndarray, message: str) -> None:
  """Missing responses, responses that leave no item to estimate, and values other than 0 and 1 are refused, the first
  with the calibrator for incomplete responses named.

  In the second, the first item is answered correctly by both examinees and the third by neither; without them, each
  examinee has no item or every item correct, which leaves the second item out too.
  """
  with pytest.raises(ValueError, match=message):
    CMLCalibrator().calibrate(matrix)


@pytest.mark.parametrize(
  ("arguments", "message"), [({"tolerance": 0.0}, "tolerance"), ({"max_cycles": 0}, "max_cycles")]
)
def test_invalid_settings(arguments: dict[str, float], message: str) -> None:
  """A tolerance that is not positive, or fewer than one iteration, is refused when the calibrator is built."""
  with pytest.raises(ValueError, match=message):
    CMLCalibrator(**arguments)  # ty: ignore[invalid-argument-type]
