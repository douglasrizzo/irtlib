r"""Item fit by Orlando and Thissen's S-X² statistic.

Item fit compares, item by item, the responses a calibration predicts with those observed. Orlando and Thissen group
examinees by their number-correct score, which is observed, rather than by an ability estimate, whose groups depend
on the model :cite:p:`Orlando2000` (p. 52). For item :math:`i` of :math:`M` and score :math:`k = 1, \dots, M - 1`,
with :math:`N_k` examinees at score :math:`k` and :math:`O_{ik}` the observed proportion of them answering item
:math:`i` correctly,

.. math::

   S\text{-}X^2_i = \sum_{k=1}^{M-1} N_k \frac{(O_{ik} - E_{ik})^2}{E_{ik}(1 - E_{ik})}

(:cite:t:`Orlando2000`, p. 54, Equation 13). The expected proportion is

.. math::

   E_{ik} = \frac{\sum_q P_i(X_q)\, S^{*i}_{k-1}(X_q)\, A_q}{\sum_q S_k(X_q)\, A_q}

(p. 54, Equation 12), where :math:`S_k(X_q)` is the likelihood of a score of :math:`k` at ability :math:`X_q` and
:math:`S^{*i}_k` the same without item :math:`i`, both from Lord and Wingersky's recursion over items (p. 53,
Equations 6-11): adding an item with probability :math:`T` turns :math:`S^*_k` into :math:`T S^*_{k-1} + (1 - T)
S^*_k`. The nodes and their normal masses :math:`A_q` are the calibration's grid.

The two cells of each score row, incorrect and correct, are collapsed as mirt 1.48 collapses them (source
``R/utils.R:2410-2546``): rows no examinee reached are dropped, a row with one examinee is merged into the next, the
last into the one before, and while an expected count is below ``min_expected``, the first such row is merged into
its neighbour, the first row into the second, the last into the one before and any other into the neighbour with
fewer examinees. Orlando and Thissen (p. 56) collapse from both ends toward the middle until every expected count is
at least 1. The statistic sums :math:`(O - E)^2 / E` over the cells left, and its degrees of freedom are the rows
left less the number of parameters estimated for the item (p. 56), floored at 0. An item whose collapsed table leaves
no degrees of freedom, including one collapsed to a single row, which mirt skips, has none and no statistic.

S-X² "cannot handle incomplete data directly" :cite:p:`Zhang2022` (p. 705), so it needs complete responses
(ADR-0006).
"""

from dataclasses import dataclass

import numpy
from catsim.estimation import normal_log_prior
from catsim.irt import icc
from catsim.typing import FloatArray, IntArray
from scipy.stats import chi2

from .calibration import Calibration, validate_responses


@dataclass(frozen=True, slots=True)
class ItemFit:
  """The S-X² statistic of each item, with its degrees of freedom and p-value.

  Attributes
  ----------
  statistic : numpy.ndarray
      S-X² of each item; NaN where no degrees of freedom are left after collapsing.
  df : numpy.ndarray
      The degrees of freedom of each item's statistic.
  p_value : numpy.ndarray
      The probability of a statistic at least as large under the calibrated model; NaN where ``df`` is 0.
  """

  statistic: FloatArray
  df: IntArray
  p_value: FloatArray


def s_x2(calibration: Calibration, responses: FloatArray, *, min_expected: float = 1.0) -> ItemFit:
  """Return the S-X² item-fit statistic of every item of a calibration, from the responses it was calibrated on.

  Parameters
  ----------
  calibration : Calibration
      The calibration whose items are evaluated.
  responses : numpy.ndarray
      The complete response matrix the calibration estimated its items from: one row per examinee, one column per
      item, 1 or 0.
  min_expected : float, optional
      The smallest expected count a cell may keep after collapsing, at least 0. Default is 1. The expected counts
      are integrated over the calibration's grid.

  Returns
  -------
  ItemFit
      Each item's statistic, degrees of freedom and p-value.

  Raises
  ------
  ValueError
      If the responses fail :py:func:`irtlib.calibration.validate_responses`, have a missing response, or do
      not have one column per item of the calibration, or ``min_expected`` is negative.
  """
  if not min_expected >= 0:
    msg = f"min_expected must be at least 0, got {min_expected}"
    raise ValueError(msg)
  matrix = validate_responses(responses)
  if numpy.isnan(matrix).any():
    msg = "S-X² needs complete responses; remove the examinees with missing responses"
    raise ValueError(msg)
  items = calibration.items.items
  if matrix.shape[1] != items.shape[0]:
    msg = f"the responses have {matrix.shape[1]} items, the calibration {items.shape[0]}"
    raise ValueError(msg)
  grid = calibration.grid
  masses = grid.weights * numpy.exp(normal_log_prior()(grid.nodes))
  masses /= masses.sum()
  probability = icc(grid.nodes, items).T  # items x nodes
  expected = _expected_proportions(probability, masses)
  scores = matrix.sum(axis=1).astype(int)
  n_items = items.shape[0]
  statistic = numpy.full(n_items, numpy.nan)
  df = numpy.zeros(n_items, dtype=int)
  for item in range(n_items):
    observed = numpy.stack(
      [numpy.bincount(scores, weights=(matrix[:, item] == value), minlength=n_items + 1)[1:-1] for value in (0, 1)],
      axis=1,
    )
    expected_counts = observed.sum(axis=1, keepdims=True) * numpy.stack([1 - expected[item], expected[item]], axis=1)
    observed, expected_counts = _collapse(observed, expected_counts, min_expected)
    cells = expected_counts > 0
    df[item] = max(int(cells.sum()) - observed.shape[0] - int(calibration.item_parameters[item]), 0)
    if df[item] > 0:
      statistic[item] = numpy.sum((observed[cells] - expected_counts[cells]) ** 2 / expected_counts[cells])
  with numpy.errstate(invalid="ignore"):
    p_value = numpy.where(df > 0, chi2.sf(statistic, numpy.maximum(df, 1)), numpy.nan)
  return ItemFit(statistic=statistic, df=df, p_value=p_value)


def _summed_score_likelihoods(probability: FloatArray) -> FloatArray:
  """Return the likelihood of each number-correct score at each node, shape (items + 1, nodes): Lord and Wingersky."""
  likelihoods = numpy.zeros((probability.shape[0] + 1, probability.shape[1]))
  likelihoods[0] = 1.0
  for item, correct in enumerate(probability):
    shifted = likelihoods[: item + 1] * correct
    likelihoods[: item + 1] *= 1 - correct
    likelihoods[1 : item + 2] += shifted
  return likelihoods


def _expected_proportions(probability: FloatArray, masses: FloatArray) -> FloatArray:
  """Return each item's expected proportion correct at each score from 1 to M - 1, shape (items, M - 1)."""
  n_items = probability.shape[0]
  score_probability = _summed_score_likelihoods(probability) @ masses  # scores 0 to M
  expected = numpy.empty((n_items, n_items - 1))
  for item in range(n_items):
    without = _summed_score_likelihoods(numpy.delete(probability, item, axis=0))  # scores 0 to M - 1
    correct = (without[:-1] * probability[item]) @ masses  # scores 1 to M - 1, the item answered correctly
    expected[item] = correct / score_probability[1:-1]
  return expected


def _collapse(observed: FloatArray, expected: FloatArray, min_expected: float) -> tuple[FloatArray, FloatArray]:
  """Collapse score rows of observed and expected counts, columns incorrect and correct, as mirt does."""
  reached = observed.sum(axis=1) > 0
  observed_rows, expected_rows = list(observed[reached]), list(expected[reached])
  # a row with one examinee goes into the next row, the last row into the one before
  row = 0
  while row < len(observed_rows) - 1:
    if observed_rows[row].sum() == 1:
      # popped first, so the next row moves into this index before it is added to
      moved_observed, moved_expected = observed_rows.pop(row), expected_rows.pop(row)
      observed_rows[row] += moved_observed
      expected_rows[row] += moved_expected
    else:
      row += 1
  if len(observed_rows) > 1 and observed_rows[-1].sum() == 1:
    moved_observed, moved_expected = observed_rows.pop(), expected_rows.pop()
    observed_rows[-1] += moved_observed
    expected_rows[-1] += moved_expected
  while len(observed_rows) > 1:
    small = [bool(numpy.any((cells < min_expected) & (cells != 0))) for cells in expected_rows]
    if not any(small):
      break
    row = small.index(True)
    if row == 0:
      target = 1
    elif row == len(observed_rows) - 1:
      target = row - 1
    else:
      target = row - 1 if observed_rows[row - 1].sum() <= observed_rows[row + 1].sum() else row + 1
    observed_rows[target] += observed_rows[row]
    expected_rows[target] += expected_rows[row]
    del observed_rows[row], expected_rows[row]
  return numpy.array(observed_rows, dtype=float), numpy.array(expected_rows, dtype=float)
