"""Tests for the plots of a test's functions of ability in irtlib.plot."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.markers import TICKUP

from irtlib import ItemBank
from irtlib.plot import bank_information, expected_score, relative_efficiency, reliability, standard_error
from irtlib.typing import FloatArray

# Two-parameter items: a = 1 at b = 0 and at b = 1, and a = 2 at b = 0. A 2PL item's information is a^2 P (1 - P).
ITEMS = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
# An item whose information underflows to 0 at abilities far below its difficulty, b = 4.
STEEP = np.array([[200.0, 4.0, 0.0, 1.0]])


def _logistic(x: float) -> float:
  """The logistic function, 1 / (1 + exp(-x)): a two-parameter item's characteristic curve at a (theta - b) = x."""
  return 1 / (1 + math.exp(-x))


def _information(theta: FloatArray, items: FloatArray) -> FloatArray:
  """The test information of two-parameter items in closed form, the sum of a^2 P (1 - P), one value per ability."""
  a, b = items[:, 0], items[:, 1]
  p = 1 / (1 + np.exp(-a * (theta[:, np.newaxis] - b)))
  return (a**2 * p * (1 - p)).sum(axis=1)


def _curve(ax: Axes) -> tuple[FloatArray, FloatArray]:
  """Return the abilities and the values of the first line drawn on ``ax``."""
  line = ax.get_lines()[0]
  return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


@pytest.mark.parametrize(
  ("items", "at_0"),
  [(ITEMS, 0.25 + _logistic(-1) * _logistic(1) + 1.0), (ITEMS.items[:2], 0.25 + _logistic(-1) * _logistic(1))],
  ids=["whole-bank", "two-items"],
)
def test_bank_information_sums_the_information_of_its_items_from_minus_4_to_4(
  items: ItemBank | FloatArray, at_0: float
) -> None:
  """The curve spans -4 to 4 and at each ability is the sum of the items' a^2 P (1 - P): at 0, 0.25 + P(1)P(-1) + 1.

  Fails if the information is averaged rather than summed, if the rows of a matrix are read as other items, or if the
  curve spans another range than catsim's.
  """
  x, y = _curve(bank_information(items))
  rows = items.items if isinstance(items, ItemBank) else items

  np.testing.assert_allclose([x[0], x[-1]], [-4, 4])
  np.testing.assert_allclose(np.interp(0.0, x, y), at_0)
  np.testing.assert_allclose(y, _information(x, rows))


def test_bank_information_ticks_each_items_location_in_the_curves_colour() -> None:
  """With locations, each item within the curve's abilities gets a tick on the bottom of the axes where it informs most.

  A three-parameter item with a = 0.3, b = -1 and c = 0.5 is most informative at b + log((1 + sqrt(1 + 8c)) / 2) / a,
  about 0.604, not at its difficulty; a two-parameter item at its difficulty, 0.5, and one at b = 6 beyond the curve's
  end at 4. Fails if the ticks sit at the items' difficulties, stand at y = 0 in data units, where a curve far above 0
  leaves them off the axes, take another colour than the curve's, or stretch the x or y limits past the curve's.
  """
  items = np.array([[0.3, -1.0, 0.5, 1.0], [1.0, 0.5, 0.0, 1.0], [1.0, 6.0, 0.0, 1.0]])
  plain = bank_information(items)
  ax = bank_information(items, locations=True)
  curve, ticks = ax.get_lines()

  np.testing.assert_allclose(np.asarray(ticks.get_xdata()), [-1 + math.log((1 + math.sqrt(5)) / 2) / 0.3, 0.5])
  np.testing.assert_array_equal(np.asarray(ticks.get_ydata()), [0.0, 0.0])
  assert ticks.get_transform() == ax.get_xaxis_transform()
  assert ticks.get_marker() == TICKUP
  assert ticks.get_linestyle() == "None"
  assert ticks.get_color() == curve.get_color()
  assert (ax.get_xlim(), ax.get_ylim()) == (plain.get_xlim(), plain.get_ylim())
  assert len(plain.get_lines()) == 1


@pytest.mark.parametrize(
  ("label", "ticks"), [(None, "Item locations"), ("Form A", "Form A: item locations")], ids=["unlabelled", "labelled"]
)
def test_bank_information_names_the_ticks_after_the_curves_label(label: str | None, ticks: str) -> None:
  """The ticks of a labelled curve carry its label, so that the ticks of two forms on one axes tell apart in a legend.

  Fails if every set of ticks is labelled alike, which gives a legend of two forms two identical entries.
  """
  ax = bank_information(ITEMS, locations=True, label=label)

  assert ax.get_lines()[1].get_label() == ticks


@pytest.mark.parametrize(
  ("interval", "label", "shaded"),
  [
    ((-1, 1), "Information in [-1, 1]: 0.462 of 1 (46.2%)", (-1.0, 1.0)),
    ((0, math.inf), "Information in [0, inf]: 0.5 of 1 (50.0%)", (0.0, 4.0)),
    ((-math.inf, -5), "Information in [-inf, -5]: 0.00669 of 1 (0.7%)", None),
  ],
  ids=["inside", "to-infinity", "beyond-the-curve"],
)
def test_bank_information_shades_an_interval_with_its_share_of_the_information(
  interval: tuple[float, float], label: str, shaded: tuple[float, float] | None
) -> None:
  """The shading spans the interval within the curve, under the curve, labelled with its information and share.

  For one item with a = 1 and b = 0, the information over (l, u) is P(u) - P(l) and over the whole scale 1: 0.462
  between -1 and 1, 0.5 above 0, and P(-5) = 0.00669 below -5, which the curve, from -4, does not reach. Fails if the
  share is of the information within -4 to 4, or of mirt's documented -10 to 10, rather than of the whole scale, if the
  shading starts or ends at the abilities of the curve nearest to the bounds, or goes beyond the curve's ends.
  """
  ax = bank_information(np.array([[1.0, 0.0, 0.0, 1.0]]), interval=interval)
  (area,) = ax.collections

  assert area.get_label() == label
  paths = area.get_paths()
  if shaded is None:
    assert paths == []
  else:
    (path,) = paths
    x, y = np.asarray(path.vertices).T
    np.testing.assert_allclose([x.min(), x.max()], shaded)
    top = y > 0
    np.testing.assert_allclose(y[top], _information(x[top], np.array([[1.0, 0.0, 0.0, 1.0]])))


@pytest.mark.parametrize(
  ("interval", "error", "message"),
  [
    ((1.0, 1.0), ValueError, r"interval is \(1.0, 1.0\); its lower bound must be below its upper bound"),
    ((1.0, -1.0), ValueError, r"interval is \(1.0, -1.0\); its lower bound must be below"),
    ((math.nan, 1.0), ValueError, r"interval is \(nan, 1.0\); its lower bound must be below"),
    ((0.0, 1.0, 2.0), ValueError, r"interval has shape \(3,\); it must be a pair of bounds, \(lower, upper\)"),
    (1.0, ValueError, r"interval has shape \(\); it must be a pair of bounds"),
    ((False, True), TypeError, "interval holds bool values; its bounds must be real numbers"),
    (("-1", "1"), TypeError, "interval holds <U2 values; its bounds must be real numbers"),
  ],
  ids=["empty", "reversed", "nan", "three-bounds", "one-number", "bools", "strings"],
)
def test_bank_information_rejects_an_interval_that_is_not_one_before_drawing(
  interval: object, error: type[Exception], message: str
) -> None:
  """An interval that is not two real bounds, the lower one below the upper one, raises before any figure exists.

  Fails if an empty or reversed interval shades nothing, or a negative share, without a word; if a NaN bound passes;
  if flags are read as bounds 0 and 1; or if a figure is left open behind the error.
  """
  with pytest.raises(error, match=message):
    bank_information(ITEMS, interval=interval)  # ty: ignore[invalid-argument-type]
  assert plt.get_fignums() == []


def test_standard_error_is_one_over_the_root_of_the_information() -> None:
  """At each ability the curve is 1 / sqrt of the summed information: at 0, 1 / sqrt(0.25 + P(1)P(-1) + 1).

  Fails if the standard error is the inverse of the information rather than of its root, or of the mean information.
  """
  x, y = _curve(standard_error(ITEMS, theta=np.array([-1.0, 0.0, 2.5])))

  np.testing.assert_allclose(y[1], 1 / math.sqrt(0.25 + _logistic(1) * _logistic(-1) + 1.0))
  np.testing.assert_allclose(y, 1 / np.sqrt(_information(x, ITEMS.items)))


def test_standard_error_leaves_out_the_abilities_with_no_information() -> None:
  """An item of discrimination 200 gives no information far below its difficulty, b = 4, and a^2 / 4 at it.

  Fails if 1 / sqrt(0) is computed, whose divide-by-zero warning the test settings turn into an error, or if an infinite
  standard error is drawn as a number; or if the standard error of 1 / sqrt(10000) = 0.01 at b is lost.
  """
  _, y = _curve(standard_error(STEEP))

  assert np.isnan(y[0])
  np.testing.assert_allclose(y[-1], 0.01)


@pytest.mark.parametrize(
  ("proportion", "scale", "ylabel"),
  [(False, 1.0, "Expected number correct"), (True, 1 / 3, "Expected proportion correct")],
)
def test_expected_score_sums_the_items_characteristic_curves(proportion: bool, scale: float, ylabel: str) -> None:
  """At 0 the items answer correctly with 0.5, P(-1) and 0.5: the curve is their sum, or with proportion their mean.

  Fails if the number correct is averaged, if the proportion is summed, or if either is drawn under the other's label.
  """
  ax = expected_score(ITEMS, theta=np.array([-1.0, 0.0, 1.0]), proportion=proportion)
  x, y = _curve(ax)
  a, b = ITEMS.discrimination, ITEMS.difficulty

  np.testing.assert_allclose(y[1], (1.0 + _logistic(-1)) * scale)
  np.testing.assert_allclose(y, (1 / (1 + np.exp(-a * (x[:, np.newaxis] - b)))).sum(axis=1) * scale)
  assert ax.get_ylabel() == ylabel


@pytest.mark.parametrize(("proportion", "ylim"), [(False, None), (True, (0.0, 1.0))])
def test_expected_score_scales_a_proportion_from_0_to_1(proportion: bool, ylim: tuple[float, float] | None) -> None:
  """A proportion goes on an axis from 0 to 1; a number correct keeps the axis matplotlib fits to the curves.

  Fails if a proportion's axis fits its curve, so that item sets compare on different scales, or if the number correct
  of a set drawn last cuts the curves of longer sets drawn before it on the same axes.
  """
  ax = expected_score(ITEMS, proportion=proportion)
  expected_score(ITEMS.items[:1], proportion=proportion, ax=ax)

  if ylim is None:
    # The three items' curve ends at 4 with P(4) + P(3) + P(8).
    assert ax.get_ylim()[1] >= _logistic(4.0) + _logistic(3.0) + _logistic(8.0)
  else:
    assert ax.get_ylim() == ylim


def test_reliability_is_the_information_over_itself_plus_1_on_an_axis_from_0_to_1() -> None:
  """For abilities N(0, 1) the reliability is I / (I + 1): 0.5 at the difficulty of an item of a = 2, where I = 1.

  Fails if the reliability is drawn from a population variance other than 1, or is the information itself, or if its
  axis fits the curve rather than spanning 0 to 1.
  """
  items = np.array([[2.0, 0.0, 0.0, 1.0]])
  ax = reliability(items, theta=np.array([-1.0, 0.0, 1.5]))
  x, y = _curve(ax)
  information = _information(x, items)

  np.testing.assert_allclose(y[1], 0.5)
  np.testing.assert_allclose(y, information / (information + 1))
  assert ax.get_ylim() == (0.0, 1.0)


def test_relative_efficiency_is_the_ratio_of_the_two_sets_information_with_a_line_at_1() -> None:
  """An item of a = 2 over one of a = 1, both at b = 0, gives 1 / 0.25 = 4 at 0; a dashed line marks 1 across the axes.

  Fails if the ratio is inverted, so that the reference set reads as the more efficient one, if it is a difference, or
  if the line at 1 is missing, drawn at another height, or takes a colour of the style's cycle.
  """
  items, reference = np.array([[2.0, 0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0, 1.0]])
  ax = relative_efficiency(items, reference, theta=np.array([-2.0, 0.0, 1.0]))
  curve, line = ax.get_lines()
  x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())

  np.testing.assert_allclose(y[1], 4.0)
  np.testing.assert_allclose(y, _information(x, items) / _information(x, reference))
  np.testing.assert_array_equal(np.asarray(line.get_ydata()), [1.0, 1.0])
  np.testing.assert_array_equal(np.asarray(line.get_xdata()), [0.0, 1.0])
  assert line.get_linestyle() == "--"
  assert line.get_color() == ax.spines["left"].get_edgecolor()


def test_relative_efficiency_leaves_out_the_abilities_where_the_reference_gives_no_information() -> None:
  """A reference item of discrimination 200 at b = 4 gives no information at -4: the ratio is left out there.

  Fails if the division by zero is computed, whose warning the test settings turn into an error, or if an infinite
  ratio is drawn; or if the ratio at b, 0.25 * 4 / 10000 for an item of a = 1 at b = 4 over the steep one, is lost.
  """
  _, y = _curve(relative_efficiency(np.array([[1.0, 4.0, 0.0, 1.0]]), STEEP))

  assert np.isnan(y[0])
  np.testing.assert_allclose(y[-1], 0.25 / 10000)


@pytest.mark.parametrize(
  ("items", "reference"),
  [(np.array([[-1.0, 0.0, 0.0, 1.0]]), ITEMS.items), (ITEMS.items, np.array([[-1.0, 0.0, 0.0, 1.0]]))],
  ids=["items", "reference"],
)
def test_relative_efficiency_checks_both_item_sets_before_drawing(items: FloatArray, reference: FloatArray) -> None:
  """A set with a negative discrimination raises, whichever of the two it is, before any figure exists.

  Fails if only the first set is checked, so that the ratio of a miskeyed reference is drawn, or a figure is left open.
  """
  with pytest.raises(ValueError, match="discrimination <= 0"):
    relative_efficiency(items, reference)
  assert plt.get_fignums() == []
