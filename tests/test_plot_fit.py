"""Tests for the plots of fit in irtlib.plot."""

import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import AxLine, Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

from irtlib import ItemBank
from irtlib.calibration import Calibration, Likelihood
from irtlib.fit import ItemFit, MeanSquares
from irtlib.plot import empirical_icc, invariance, pathway_map, person_fit_map, person_profile, sx2_item
from irtlib.typing import FloatArray

# Two items; empirical_icc draws the second, a 3PL item whose name holds two dollar signs.
ITEMS = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.5, -0.5, 0.2, 1.0]]), names=["first", "$5 or $6"])
A, B, C = 1.5, -0.5, 0.2
# Each examinee's ability and response to the second item, the rows shuffled. The eight who answered it rank as
# -1.5, -1.2, -0.3 | 0.1, 0.2, 0.9 | 1.4, 1.7 into three groups of sizes 3, 3 and 2, with 0, 1 and 2 correct.
# The two who did not answer it sit at the extremes, where they would move every group if they were counted.
THETA = np.array([0.9, -4.0, -1.2, 1.7, 0.2, 5.0, -1.5, 0.1, -0.3, 1.4])
RESPONSES = np.column_stack([
  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
  [0, np.nan, 0, 1, 0, np.nan, 0, 1, 0, 1],
])
# Each group's median ability and proportion correct; the means of the first two groups, -1.0 and 0.4, differ.
MEDIANS = [-1.2, 0.2, 1.55]
PROPORTIONS = [0.0, 1 / 3, 1.0]


def _icc(theta: FloatArray) -> FloatArray:
  """The second item's characteristic curve in closed form, c + (1 - c) / (1 + exp(-a (theta - b)))."""
  return C + (1 - C) / (1 + np.exp(-A * (theta - B)))


def _x(line: Line2D) -> FloatArray:
  """Return a line's x coordinates as a float array."""
  return np.asarray(line.get_xdata(), dtype=float)


def _y(line: Line2D) -> FloatArray:
  """Return a line's y coordinates as a float array."""
  return np.asarray(line.get_ydata(), dtype=float)


def _xy(line: Line2D) -> FloatArray:
  """Return a line's points as a float array of shape (n, 2)."""
  return np.asarray(line.get_xydata(), dtype=float)


def _named(ax: Axes) -> list[tuple[str, tuple[float, float]]]:
  """Return the text and the point of each annotation on the axes, after checking that every text is one."""
  annotations = [text for text in ax.texts if isinstance(text, Annotation)]
  assert len(annotations) == len(ax.texts)
  return [(annotation.get_text(), (float(annotation.xy[0]), float(annotation.xy[1]))) for annotation in annotations]


def _errorbars(ax: Axes) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return the dots' x and y and the lower and upper ends of their bars, from the axes' one errorbar container."""
  (container,) = ax.containers
  assert isinstance(container, ErrorbarContainer)
  dots, _, (bars,) = container.lines
  ends = np.array(bars.get_segments())
  return _x(dots), _y(dots), ends[:, 0, 1], ends[:, 1, 1]


def _horizontal_lines(ax: Axes) -> list[Line2D]:
  """Return the unlabelled lines of constant height, such as those ``Axes.axhline`` draws, in the order drawn."""
  return [line for line in ax.get_lines() if str(line.get_label()).startswith("_") and np.ptp(_y(line)) == 0]


@pytest.mark.parametrize("level", [0.95, 0.5])
def test_empirical_icc_groups_the_examinees_who_answered_by_ability(level: float) -> None:
  """Each dot is a group's proportion correct at its median ability, with the exact binomial interval at level.

  The examinees who answered the item fall, ranked by ability, into groups of 3, 3 and 2 with 0, 1 and 2 correct. With
  t = (1 - level) / 2, the interval of 0 of 3 is [0, 1 - t^(1/3)], of 2 of 2 [t^(1/2), 1], and of 1 of 3 starts at
  1 - (1 - t)^(1/3) and ends where the binomial probability of at most 1 correct of 3 is t. Fails if the groups follow
  the rows instead of the abilities, if an examinee who did not answer is counted, if a dot sits at the group's mean,
  or if the interval is the normal approximation, which gives the first and last groups no width at all.
  """
  ax = empirical_icc(ITEMS, 1, RESPONSES, THETA, groups=3, level=level)
  x, y, lower, upper = _errorbars(ax)
  tail = (1 - level) / 2

  np.testing.assert_allclose(x, MEDIANS)
  np.testing.assert_allclose(y, PROPORTIONS)
  np.testing.assert_allclose(lower, [0.0, 1 - (1 - tail) ** (1 / 3), math.sqrt(tail)])
  np.testing.assert_allclose(upper[[0, 2]], [1 - tail ** (1 / 3), 1.0])
  np.testing.assert_allclose(binom.cdf(1, 3, upper[1]), tail)


@pytest.mark.parametrize(
  ("n_examinees", "groups", "medians"),
  [(100, None, 4.5 + 10 * np.arange(10)), (7, 3, [1.0, 3.5, 5.5])],
  ids=["ten-by-default", "sizes-differ-by-one"],
)
def test_empirical_icc_splits_the_examinees_into_groups_of_equal_size(
  n_examinees: int, groups: int | None, medians: FloatArray
) -> None:
  """With abilities 0 to n - 1, the groups' medians show their sizes: ten groups of 10 by default, 7 into 3, 2 and 2.

  Fails if the default is not ten groups, if the sizes differ by more than one, or if the curve stops short of the
  groups: it spans b - 4 to b + 4, here -4.5 to 3.5, widened to reach the highest median.
  """
  theta = np.random.default_rng(0).permutation(n_examinees).astype(float)
  responses = np.ones((n_examinees, 2))

  if groups is None:
    ax = empirical_icc(ITEMS, 1, responses, theta)
  else:
    ax = empirical_icc(ITEMS, 1, responses, theta, groups=groups)
  curve = ax.get_lines()[0]

  np.testing.assert_allclose(_errorbars(ax)[0], medians)
  np.testing.assert_allclose(_x(curve)[[0, -1]], [B - 4, max(B + 4, medians[-1])])


def test_empirical_icc_draws_the_items_curve_with_the_dots_in_its_colour() -> None:
  """The curve is the item's characteristic curve over b - 4 to b + 4, and the dots take its colour and its label.

  Fails if the curve is another item's or another function, if a second item drawn on the same axes reuses the first's
  colour, or if the dots get a colour of their own, so that two items on one axes cannot be told apart.
  """
  ax = empirical_icc(ITEMS, 1, RESPONSES, THETA, groups=3)
  empirical_icc(ITEMS, 0, RESPONSES[:, ::-1], THETA, groups=3, ax=ax, label="other")
  first_curve, first_dots, second_curve, second_dots = ax.get_lines()
  x = _x(first_curve)

  np.testing.assert_allclose(x[[0, -1]], [B - 4, B + 4])
  np.testing.assert_allclose(_y(first_curve), _icc(x))
  assert first_dots.get_color() == first_curve.get_color() != second_curve.get_color() == second_dots.get_color()
  # a legend lists the curves before the error bars
  assert ax.get_legend_handles_labels()[1] == ["$5 or $6", "other", "$5 or $6, observed", "other, observed"]


def test_empirical_icc_needs_no_finite_ability_from_examinees_who_skipped_the_item() -> None:
  """An infinite ability, as maximum likelihood gives a perfect score, is accepted where the item was not answered.

  Fails if every ability must be finite, which would reject maximum likelihood estimates the plot never uses.
  """
  theta = THETA.copy()
  theta[[1, 5]] = [-np.inf, np.inf]

  ax = empirical_icc(ITEMS, 1, RESPONSES, theta, groups=3)

  np.testing.assert_allclose(_errorbars(ax)[0], MEDIANS)


@pytest.mark.parametrize(
  ("change", "error", "message"),
  [
    ({"item": 2}, ValueError, r"item is 2; item positions must be between 0 and 1"),
    ({"groups": 0}, ValueError, r"groups is 0; it must be between 1 and the 8 examinees who answered the item"),
    ({"groups": 9}, ValueError, r"groups is 9; it must be between 1 and the 8 examinees who answered the item"),
    ({"groups": 3.0}, TypeError, r"groups is 3.0; the number of groups must be an integer"),
    ({"groups": True}, TypeError, r"groups is True; the number of groups must be an integer, not a bool"),
    ({"level": 1.0}, ValueError, r"level is 1.0; a confidence level must lie strictly between 0 and 1"),
    ({"level": np.nan}, ValueError, r"level is nan; a confidence level must lie strictly between 0 and 1"),
    ({"theta": THETA[:-1]}, ValueError, r"theta must hold one ability per examinee, 10, got shape \(9,\)"),
    ({"theta": THETA.astype(str)}, TypeError, r"theta holds <U\d+ values; abilities must be real numbers"),
    (
      {"theta": np.where(np.arange(10) == 3, np.inf, THETA)},
      ValueError,
      r"theta\[3\] is inf; the ability of every examinee who answered the item must be finite",
    ),
    ({"responses": RESPONSES[:, :1]}, ValueError, r"responses has 1 columns and items 2 rows"),
  ],
  ids=[
    "item-past-the-bank",
    "no-groups",
    "more-groups-than-examinees",
    "float-groups",
    "bool-groups",
    "level-of-one",
    "nan-level",
    "short-theta",
    "string-theta",
    "infinite-ability-of-an-answer",
    "responses-of-another-test",
  ],
)
def test_empirical_icc_rejects_bad_input_before_drawing(
  change: dict[str, object], error: type[Exception], message: str
) -> None:
  """A bad item, number of groups, level, ability or response matrix raises before any figure exists.

  Fails if more groups than examinees leave empty groups, whose proportions are 0 / 0, if a float or a bool is taken
  for a number of groups, if a level of 1 draws infinite intervals, if an infinite ability puts a dot at infinity, or if
  a bad call leaves an empty figure open behind the error.
  """
  arguments: dict[str, object] = {"item": 1, "responses": RESPONSES, "theta": THETA, "groups": 3}
  arguments.update(change)

  with pytest.raises(error, match=message):
    empirical_icc(ITEMS, **arguments)  # ty: ignore[invalid-argument-type]
  assert plt.get_fignums() == []


# S-X² of three items, by number correct 1 and 2. Nobody answered the second item correctly at number correct 1, so its
# observed proportion there is NaN; the third item has no degrees of freedom left.
SX2_ITEMS = ItemBank(np.tile([1.0, 0.0, 0.0, 1.0], (3, 1)), names=["one", "fee $5", "three"])
SX2 = ItemFit(
  statistic=np.array([1.0, 2.345, np.nan]),
  df=np.array([1, 2, 0]),
  p_value=np.array([0.3, 0.30957, np.nan]),
  group_sizes=np.array([4, 6]),
  observed=np.array([[0.25, 0.5], [np.nan, 0.6], [0.5, 0.5]]),
  expected=np.array([[0.2, 0.6], [0.3, 0.7], [0.4, 0.6]]),
)


def test_sx2_item_draws_the_items_expected_and_observed_proportions_by_number_correct() -> None:
  """The line is the item's row of expected proportions and the dots its row of observed ones, at numbers 1 and 2.

  Fails if the table is read by column instead of by item, if the numbers correct start at 0, if a score group without
  examinees is drawn at 0, or if the dots take a colour of their own.
  """
  ax = sx2_item(SX2_ITEMS, 1, SX2)
  line, dots = ax.get_lines()

  np.testing.assert_allclose(_xy(line), [[1.0, 0.3], [2.0, 0.7]])
  np.testing.assert_allclose(_xy(dots), [[1.0, np.nan], [2.0, 0.6]])
  assert dots.get_color() == line.get_color()
  assert [line.get_label(), dots.get_label()] == ["fee $5", "fee $5, observed"]
  assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)


@pytest.mark.parametrize(
  ("item", "text"),
  [(1, "fee \\$5\n$S$-$X^2(2) = 2.35$\n$p = 0.31$"), (2, "three\n$S$-$X^2$ has no degrees of freedom")],
  ids=["statistic", "no-degrees-of-freedom"],
)
def test_sx2_item_box_gives_the_statistic_in_the_lower_right_corner(item: int, text: str) -> None:
  """The box names the item and gives S-X², its degrees of freedom and p-value to three digits, low on the right.

  Fails if the statistic of another item is shown, if NaN is printed for an item without degrees of freedom, if a
  name's dollar sign pairs up with the statistic's mathtext, or if the box sits in the upper left corner, where the
  proportions of an easy item are.
  """
  ax = sx2_item(SX2_ITEMS, item, SX2)
  ax.figure.canvas.draw()
  (box,) = ax.texts
  patch = box.get_bbox_patch()
  assert patch is not None
  extent, axes_extent = patch.get_window_extent(), ax.get_window_extent()

  assert box.get_text() == text
  assert axes_extent.x0 + axes_extent.width / 2 < extent.x0 < extent.x1 <= axes_extent.x1
  assert axes_extent.y0 <= extent.y0 < extent.y1 < axes_extent.y0 + axes_extent.height / 2


def test_sx2_item_without_the_statistic_writes_no_box() -> None:
  """statistic=False leaves the box out, so that a second item on the same axes does not cover the first's.

  Fails if the box is drawn whatever the flag.
  """
  ax = sx2_item(SX2_ITEMS, 0, SX2, statistic=False)

  assert list(ax.texts) == []


@pytest.mark.parametrize(
  ("items", "item", "message"),
  [
    (SX2_ITEMS.items[:2], 0, r"fit holds the S-X² of 3 items; it must hold one per item, 2"),
    (SX2_ITEMS, 3, r"item is 3; item positions must be between 0 and 2"),
  ],
  ids=["other-items", "item-past-the-bank"],
)
def test_sx2_item_rejects_bad_input_before_drawing(items: FloatArray | ItemBank, item: int, message: str) -> None:
  """The fit of another number of items, or an item outside the bank, raises before any figure exists.

  Fails if the plot draws another item's row of the table, or leaves an empty figure behind the error.
  """
  with pytest.raises(ValueError, match=message):
    sx2_item(items, item, SX2)
  assert plt.get_fignums() == []


# Five items, the second named with two dollar signs, and their mean squares: the outfit's standardized values are all
# 9, so a plot of them shows; the fourth item's standardized infit is NaN, and the fifth's lies on the band's edge.
FIT_ITEMS = ItemBank(
  np.column_stack([np.ones(5), [-1.0, 0.5, 1.5, 0.0, 2.0], np.zeros(5), np.ones(5)]),
  names=["a", "$b$", "c", "d", "e"],
)
FIT = MeanSquares(
  outfit=np.ones(5), outfit_t=np.full(5, 9.0), infit=np.ones(5), infit_t=np.array([0.5, 2.5, -3.0, np.nan, 2.0])
)


def test_pathway_map_draws_each_items_infit_t_against_its_difficulty() -> None:
  """Each item is a dot at its difficulty and standardized infit, an item whose infit t is NaN left out.

  Fails if the outfit's standardized values are drawn, if the difficulties are read from another column, or if a NaN
  item is drawn, which leaves a gap that matplotlib does not show but a legend counts.
  """
  ax = pathway_map(FIT_ITEMS, FIT, label="items")
  (dots,) = [line for line in ax.get_lines() if line.get_label() == "items"]

  np.testing.assert_allclose(_xy(dots), [[-1.0, 0.5], [0.5, 2.5], [1.5, -3.0], [2.0, 2.0]])
  assert ax.get_ylabel() == "Standardized infit $t$"


def test_pathway_map_names_the_items_outside_the_band_as_plain_text() -> None:
  """The items whose infit t lies beyond -2 or 2 are named beside their dots, their names not read as mathtext.

  Fails if an item inside the band, or on its edge at t = 2, is named, if a misfitting item is left unnamed, or if a
  name with two dollar signs is printed as a formula.
  """
  ax = pathway_map(FIT_ITEMS, FIT)

  assert _named(ax) == [("$b$", (0.5, 2.5)), ("c", (1.5, -3.0))]
  assert all(text.get_parse_math() is False for text in ax.texts)


def test_pathway_map_rejects_the_fit_of_other_items_before_drawing() -> None:
  """Mean squares of another number of items raise before any figure exists.

  Fails if the plot pairs the difficulties with the fit of other items, or leaves an empty figure behind the error.
  """
  fit = MeanSquares(outfit=np.ones(4), outfit_t=np.zeros(4), infit=np.ones(4), infit_t=np.zeros(4))

  with pytest.raises(ValueError, match=r"fit holds standardized infits of shape \(4,\); it must hold one per item, 5"):
    pathway_map(FIT_ITEMS, fit)
  assert plt.get_fignums() == []


def _calibration(difficulty: list[float], standard_errors: list[float] | None) -> Calibration:
  """Return a conditional calibration of the Rasch model with the given difficulties and their standard errors."""
  n_items = len(difficulty)
  covariances = np.full((n_items, 4, 4), np.nan)
  if standard_errors is not None:
    covariances[:, 1, 1] = np.square(standard_errors)
  return Calibration(
    estimates=np.column_stack([np.ones(n_items), difficulty, np.zeros(n_items), np.ones(n_items)]),
    items=None,
    covariances=None if standard_errors is None else covariances,
    log_likelihood=0.0,
    likelihood=Likelihood.CONDITIONAL,
    n_parameters=n_items - 1,
    item_parameters=np.ones(n_items, dtype=int),
    cycles=1,
    converged=True,
    change=0.0,
    grid=None,
    abilities=None,
  )


# Item 3 has no difficulty in the first calibration and item 5 none in the second, which would move either mean.
# Over the other four, the first calibration's difficulties have mean 1 and the second's 3, so the centred pairs are
# (3, 0.5), (-2, -2), (-1, -0.5) and (0, 2), with means 1.75, -2, -0.75 and 1, and differences 2.5, 0, -0.5 and -2.
# Each difference has standard error 0.5.
FIRST = _calibration([4.0, -1.0, 0.0, np.nan, 1.0, 7.0], [0.4, 0.3, 0.3, 0.3, 0.3, 0.3])
SECOND = _calibration([3.5, 1.0, 2.5, 2.0, 5.0, np.nan], [0.3, 0.4, 0.4, 0.4, 0.4, 0.4])


@pytest.mark.parametrize(("level", "k"), [(0.95, 1.959963984540054), (math.erf(math.sqrt(2)), 2.0)])
def test_invariance_draws_the_centred_pairs_inside_wright_and_masters_band(level: float, k: float) -> None:
  """The pairs are centred over the items both estimate, and the band joins (m -+ k s / 2, m +- k s / 2), by m.

  m is an item's mean difficulty, s its difference's standard error and k the normal quantile of the level: 1.96 at
  0.95, and 2 at 0.9545, Wright and Masters' k = 2. Fails if a calibration is centred over every item, NaN included, or
  over its own items only, if the band is not k standard errors of the difference wide, or if its points are joined in
  the items' order, which folds the band back on itself.
  """
  ax = invariance(FIRST, SECOND, level=level, label="items")
  identity, dots, upper, lower = ax.get_lines()
  means = np.array([-2.0, -0.75, 1.0, 1.75])

  assert isinstance(identity, AxLine)
  assert (identity.get_xy1(), identity.get_slope()) == ((0, 0), 1)
  np.testing.assert_allclose(_xy(dots), [[3.0, 0.5], [-2.0, -2.0], [-1.0, -0.5], [0.0, 2.0]])
  np.testing.assert_allclose(_xy(upper), np.column_stack([means - k / 4, means + k / 4]))
  np.testing.assert_allclose(_xy(lower), np.column_stack([means + k / 4, means - k / 4]))
  assert upper.get_color() == lower.get_color() == dots.get_color()
  assert upper.get_label() == f"{100 * level:g}% confidence band"
  assert ax.get_aspect() == 1


def test_invariance_names_the_items_outside_the_band() -> None:
  """The items whose difference exceeds k standard errors are named, by their position as in a bank of every item.

  With k s = 0.98, items 0 and 4 differ by 2.5 and -2 and fall outside, item 2 differs by -0.5 and does not. Fails if
  an item inside is named, or the names count only the items drawn, which would call item 4 item 3.
  """
  ax = invariance(FIRST, SECOND)

  assert _named(ax) == [("item_0", (3.0, 0.5)), ("item_4", (0.0, 2.0))]


@pytest.mark.parametrize(
  ("first", "second", "level", "message"),
  [
    (_calibration([0.0, 1.0], None), SECOND, 0.95, "the first calibration has no standard errors"),
    (FIRST, _calibration([0.0, 1.0], None), 0.95, "the second calibration has no standard errors"),
    (FIRST, _calibration([0.0, 1.0], [0.1, 0.1]), 0.95, "the first calibration has 6 items and the second 2"),
    (
      _calibration([np.nan, 1.0], [0.1, 0.1]),
      _calibration([0.0, 1.0], [0.1, np.nan]),
      0.95,
      "no item has a difficulty and a standard error in both calibrations",
    ),
    (FIRST, SECOND, 0.0, "level is 0.0; a confidence level must lie strictly between 0 and 1"),
  ],
  ids=["first-without-errors", "second-without-errors", "other-items", "no-common-item", "level-of-zero"],
)
def test_invariance_rejects_bad_calibrations_before_drawing(
  first: Calibration, second: Calibration, level: float, message: str
) -> None:
  """Calibrations without standard errors, of other items or with no item in common raise before any figure exists.

  Fails if the band is drawn without standard errors, if two calibrations of different items are paired by position,
  or if a bad call leaves an empty figure open behind the error.
  """
  with pytest.raises(ValueError, match=message):
    invariance(first, second, level=level)
  assert plt.get_fignums() == []


# Five items in bank order hard, easy, skipped, guessing and far, of difficulties log 4, -log 4, 0.5, 0 and -40; the
# guessing item has c = 0.2. At ability 0 they give P = 0.2, 0.8, -, 0.6 and 1 - e^-40.
PROFILE_ITEMS = ItemBank(
  np.array([
    [1.0, math.log(4), 0.0, 1.0],
    [1.0, -math.log(4), 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.0, 0.2, 1.0],
    [1.0, -40.0, 0.0, 1.0],
  ]),
  names=["hard", "easy", "skipped", "guessing", "far"],
)
# The second examinee, at ability 0, answers the hard and guessing items correctly, the easy and far ones incorrectly,
# and skips one. The first answers otherwise and has an infinite ability.
PROFILE_RESPONSES = np.array([[0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, np.nan, 1.0, 0.0]])
PROFILE_THETA = np.array([np.inf, 0.0])


def test_person_profile_draws_the_standardized_residuals_from_the_easiest_item() -> None:
  """The residuals (u - P) / sqrt(PQ), the items ordered by difficulty, are -e^20, -2, sqrt(2/3), a gap and 2.

  An incorrect response at P = 0.8 gives -sqrt(0.8 / 0.2) = -2, a correct one at P = 0.2 gives sqrt(0.8 / 0.2) = 2, and
  at P = 0.6 sqrt(0.4 / 0.6). Fails if the items keep their bank order, if another examinee's row or ability is used,
  if a skipped item is drawn as 0, if the guessing parameter is left out of P, or if P rounds to 1 forty units below
  the ability, which makes the far item's residual infinite.
  """
  ax = person_profile(PROFILE_ITEMS, PROFILE_RESPONSES, PROFILE_THETA, 1, label="examinee 1")
  (residuals,) = [line for line in ax.get_lines() if line.get_label() == "examinee 1"]

  np.testing.assert_allclose(_x(residuals), np.arange(5))
  np.testing.assert_allclose(_y(residuals), [-math.exp(20), -2.0, math.sqrt(2 / 3), np.nan, 2.0])
  assert [tick.get_text() for tick in ax.get_xticklabels()] == ["far", "easy", "guessing", "skipped", "hard"]


def test_person_profile_ticks_whole_numbers_past_thirty_items() -> None:
  """With 31 items, the ticks are whole positions chosen by matplotlib, not the names.

  Fails if every name of a long test is written on the axis, where they overlap.
  """
  items = np.tile([1.0, 0.0, 0.0, 1.0], (31, 1))

  ax = person_profile(items, np.ones((1, 31)), np.zeros(1), 0)

  assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)
  assert "item_00" not in [tick.get_text() for tick in ax.get_xticklabels()]


@pytest.mark.parametrize(
  ("change", "error", "message"),
  [
    ({"examinee": 2}, ValueError, r"examinee is 2; examinee positions must be between 0 and 1"),
    ({"examinee": -1}, ValueError, r"examinee is -1; examinee positions must be between 0 and 1"),
    ({"examinee": 1.0}, TypeError, r"examinee is 1.0; examinee positions must be integers"),
    ({"examinee": True}, TypeError, r"examinee is True; examinee positions must be integers, not bools"),
    ({"examinee": 0}, ValueError, r"theta\[0\] is inf; the ability of the examinee must be finite"),
    ({"theta": np.zeros(3)}, ValueError, r"theta must hold one ability per examinee, 2, got shape \(3,\)"),
    (
      {"responses": np.array([[np.nan] * 5, [1.0] * 5]), "examinee": 0, "theta": np.zeros(2)},
      ValueError,
      "examinee 0 answered no item, so has no residual to plot",
    ),
    ({"responses": PROFILE_RESPONSES[:, :4]}, ValueError, r"responses has 4 columns and items 5 rows"),
  ],
  ids=[
    "examinee-past-the-matrix",
    "negative-examinee",
    "float-examinee",
    "bool-examinee",
    "infinite-ability",
    "long-theta",
    "no-answers",
    "responses-of-another-test",
  ],
)
def test_person_profile_rejects_bad_input_before_drawing(
  change: dict[str, object], error: type[Exception], message: str
) -> None:
  """A bad examinee, an infinite ability or an examinee with no answers raises before any figure exists.

  Fails if a negative row draws an examinee counted from the end, if a float or a bool is read as a row, if an infinite
  ability draws residuals of 0 and infinity, if an examinee with no answers gets an empty plot, or if a bad call leaves
  an empty figure open behind the error.
  """
  arguments: dict[str, object] = {"responses": PROFILE_RESPONSES, "theta": PROFILE_THETA, "examinee": 1}
  arguments.update(change)

  with pytest.raises(error, match=message):
    person_profile(PROFILE_ITEMS, **arguments)  # ty: ignore[invalid-argument-type]
  assert plt.get_fignums() == []


def test_person_fit_map_draws_each_examinees_infit_t_against_their_ability() -> None:
  """Each examinee with a finite ability and standardized infit is a dot at the two, the others left out.

  Fails if the outfit's standardized values are drawn, if the abilities and the fit are paired out of order, or if an
  examinee with an infinite or NaN value is drawn.
  """
  theta = np.array([0.5, -1.0, np.inf, 2.0, np.nan])
  fit = MeanSquares(
    outfit=np.ones(5), outfit_t=np.full(5, 9.0), infit=np.ones(5), infit_t=np.array([1.0, -2.5, 0.3, np.nan, 0.1])
  )

  ax = person_fit_map(theta, fit, label="examinees")
  (dots,) = [line for line in ax.get_lines() if line.get_label() == "examinees"]

  np.testing.assert_allclose(_xy(dots), [[0.5, 1.0], [-1.0, -2.5]])
  assert ax.get_ylabel() == "Standardized infit $t$"


@pytest.mark.parametrize(
  ("theta", "infit_t", "error", "message"),
  [
    (np.zeros(4), np.zeros(5), ValueError, r"theta must hold one ability per examinee, 5, got shape \(4,\)"),
    (np.array(["0"] * 5), np.zeros(5), TypeError, r"theta holds <U1 values; abilities must be real numbers"),
    (
      np.zeros(6),
      np.zeros((2, 3)),
      ValueError,
      r"fit holds standardized infits of shape \(2, 3\); it must hold one per examinee",
    ),
  ],
  ids=["other-examinees", "string-theta", "fit-of-a-matrix"],
)
def test_person_fit_map_rejects_abilities_of_other_examinees_before_drawing(
  theta: FloatArray, infit_t: FloatArray, error: type[Exception], message: str
) -> None:
  """Abilities that are not one number per examinee of the fit, or a fit that is not one per examinee, raise first.

  Fails if the plot pairs the fit with the abilities of other examinees, pairs six abilities with a matrix of six
  values, or leaves an empty figure behind the error.
  """
  fit = MeanSquares(outfit=np.ones_like(infit_t), outfit_t=infit_t, infit=np.ones_like(infit_t), infit_t=infit_t)

  with pytest.raises(error, match=message):
    person_fit_map(theta, fit)
  assert plt.get_fignums() == []


_DRAW: dict[str, Callable[..., Axes]] = {
  "empirical_icc": lambda **options: empirical_icc(ITEMS, 1, RESPONSES, THETA, groups=3, **options),
  "sx2_item": lambda **options: sx2_item(SX2_ITEMS, 1, SX2, **options),
  "pathway_map": lambda **options: pathway_map(FIT_ITEMS, FIT, **options),
  "invariance": lambda **options: invariance(FIRST, SECOND, **options),
  "person_profile": lambda **options: person_profile(PROFILE_ITEMS, PROFILE_RESPONSES, PROFILE_THETA, 1, **options),
  "person_fit_map": lambda **options: person_fit_map(np.zeros(5), FIT, **options),
}


@pytest.mark.parametrize("plot", ["pathway_map", "person_profile", "person_fit_map"])
def test_fit_band_runs_from_minus_two_to_two_in_the_axes_colour(plot: str) -> None:
  """Dashed lines at -2 and 2 and a solid line at 0 mark the band, in the colour and width of the axes' spines.

  Fails if the band is drawn elsewhere or without its centre, if the lines take a colour of the style's cycle, which
  shifts the data's colours, or a colour fixed by irtlib rather than the caller's style.
  """
  with plt.style.context("dark_background"):
    ax = _DRAW[plot]()
  lines = _horizontal_lines(ax)
  spine = ax.spines["left"]

  assert sorted((float(_y(line)[0]), line.get_linestyle()) for line in lines) == [
    (-2.0, "--"),
    (0.0, "-"),
    (2.0, "--"),
  ]
  for line in lines:
    assert (line.get_color(), line.get_linewidth()) == (spine.get_edgecolor(), spine.get_linewidth())


@pytest.mark.parametrize("plot", list(_DRAW))
def test_fit_plots_draw_on_the_given_axes_with_the_title(plot: str) -> None:
  """With ax, the plot goes on that axes and no new figure opens; with title, the axes carry it.

  Fails if a plot opens a figure of its own when given axes, or drops the title.
  """
  figure, ax = plt.subplots()

  drawn = _DRAW[plot](ax=ax, title="Fit")

  assert drawn is ax
  assert plt.get_fignums() == [figure.number]
  assert ax.get_title() == "Fit"
