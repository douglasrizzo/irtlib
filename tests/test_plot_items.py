"""Tests for the plots of items in irtlib.plot."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest

from irtlib import ItemBank
from irtlib.plot import Curve, item_curve
from irtlib.typing import FloatArray

# A three-parameter item whose discrimination has more digits than the parameter box should show, as item 1 of a bank
# whose item 0 differs in every parameter. Its name has a lone dollar sign, which mathtext would pair up.
ITEM = np.array([1.1669737841771746, -0.4321, 0.2, 1.0])
ITEMS = ItemBank(np.array([[2.0, 1.5, 0.0, 0.9], ITEM]), names=["other", "fee $5"])
A, B, C = ITEM[:3]
# Birnbaum's peak of a three-parameter item's information, b + log((1 + sqrt(1 + 8c)) / 2) / a, about -0.203.
PEAK = B + np.log((1 + np.sqrt(1 + 8 * C)) / 2) / A
# A two-parameter item, whose information peaks at its difficulty, 0.
TWO_PL = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0]]))


def _icc(theta: FloatArray) -> FloatArray:
  """The item's characteristic curve in closed form, c + (1 - c) / (1 + exp(-a (theta - b)))."""
  return C + (1 - C) / (1 + np.exp(-A * (theta - B)))


def _information(theta: FloatArray) -> FloatArray:
  """The three-parameter item's information in closed form, a^2 (P - c)^2 (1 - P) / ((1 - c)^2 P)."""
  p = _icc(theta)
  return A**2 * (p - C) ** 2 * (1 - p) / ((1 - C) ** 2 * p)


def test_item_curve_draws_the_items_characteristic_curve() -> None:
  """The curve spans four units around the item's difficulty and passes through (c + d) / 2 = 0.6 at it.

  Fails if the parameters are read in another order than [a, b, c, d] or from another item of the bank, if the curve is
  centred elsewhere, or if the information is drawn instead.
  """
  ax = item_curve(ITEMS, 1)
  (line,) = ax.get_lines()
  x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())

  np.testing.assert_allclose([x[0], x[-1]], [B - 4, B + 4])
  np.testing.assert_allclose(np.interp(B, x, y), 0.6)
  np.testing.assert_allclose(y, _icc(x))
  assert ax.get_ylabel() == r"$P(\theta)$"


def test_item_curve_marks_the_peak_of_the_information_in_the_curves_colour() -> None:
  """The curve is the item's information, spanning b - 4 to its peak + 4, with a dot at Birnbaum's peak in its colour.

  For d = 1 the peak is at b + log((1 + sqrt(1 + 8c)) / 2) / a, and the information is a^2 / 6 at b when c = 0.2. Fails
  if the curve is another function or spans another range, if the dot is off the peak or off the curve, or if it takes
  the next colour of the axes' cycle instead of the curve's.
  """
  ax = item_curve(ITEMS, 1, kind=Curve.IIC)
  curve, dot = ax.get_lines()
  x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())

  np.testing.assert_allclose([x[0], x[-1]], [B - 4, PEAK + 4])
  np.testing.assert_allclose(_information(np.array(B)), A**2 / 6)
  np.testing.assert_allclose(y, _information(x))
  np.testing.assert_allclose(np.asarray(dot.get_xydata()), [[PEAK, _information(PEAK)]])
  assert dot.get_color() == curve.get_color()


def test_item_curve_spans_the_peak_of_a_flat_items_information() -> None:
  """A flat item with guessing peaks far beyond its difficulty, and its information curve still reaches the peak.

  For a = 0.05, c = 0.3 and d = 1 the peak is b + log((1 + sqrt(1 + 8c)) / 2) / a, about b + 7.0, three units past a
  span of four around b. Fails if the curve keeps to b +- 4 and leaves the dot floating beyond its end.
  """
  a, b, c = 0.05, 0.5, 0.3
  peak = b + np.log((1 + np.sqrt(1 + 8 * c)) / 2) / a

  ax = item_curve(np.array([[a, b, c, 1.0]]), 0, kind=Curve.IIC)
  curve, dot = ax.get_lines()
  x = np.asarray(curve.get_xdata())

  assert x[0] <= b - 4
  assert x[-1] >= peak + 4
  np.testing.assert_allclose(np.asarray(dot.get_xydata())[0, 0], peak)


@pytest.mark.parametrize(("kind", "curve"), [(Curve.ICC, _icc), (Curve.IIC, _information)])
def test_item_curve_draws_at_the_given_abilities(kind: Curve, curve: Callable[[FloatArray], FloatArray]) -> None:
  """With theta, the curve is drawn at those abilities and no others, through the item's value at each.

  At b the characteristic curve is (c + d) / 2 = 0.6 and the information a^2 / 6, for c = 0.2 and d = 1. Fails if theta
  is ignored for the item's own span, so that the curves of several items on one axes end at ragged abilities, or if
  the curve is evaluated elsewhere than at theta.
  """
  theta = np.array([-3.0, B, 0.5, 2.5])

  ax = item_curve(ITEMS, 1, kind=kind, theta=theta, max_info=False)
  (line,) = ax.get_lines()

  np.testing.assert_array_equal(np.asarray(line.get_xdata()), theta)
  np.testing.assert_allclose(np.asarray(line.get_ydata())[1], {Curve.ICC: 0.6, Curve.IIC: A**2 / 6}[kind])
  np.testing.assert_allclose(np.asarray(line.get_ydata()), curve(theta))


@pytest.mark.parametrize(
  ("items", "theta", "dots"),
  [
    (ITEMS, np.linspace(-1, 1, 5), [[PEAK, float(_information(np.array(PEAK)))]]),
    (ITEMS, np.linspace(-3, -1, 5), []),
    (TWO_PL, np.array([-1.0, 0.0]), [[0.0, 0.25]]),
    (TWO_PL, np.array([0.0, 1.0]), [[0.0, 0.25]]),
  ],
  ids=["peak-inside", "peak-beyond-the-end", "peak-at-the-upper-end", "peak-at-the-lower-end"],
)
def test_item_curve_marks_the_peak_only_within_the_given_abilities(
  items: ItemBank, theta: FloatArray, dots: list[list[float]]
) -> None:
  """With theta, the dot marks the peak when the abilities reach it, an end of them included, and only then.

  The three-parameter item peaks at about -0.203, the two-parameter one at b = 0, with information a^2 / 4 = 0.25 there.
  Fails if the dot is drawn beyond the end of the curve, where it marks a maximum the curve does not show and stretches
  the axes past the abilities the caller chose, or if it is left out when the peak is in range or at an end of it.
  """
  ax = item_curve(items, items.n_items - 1, kind=Curve.IIC, theta=theta)
  drawn = [np.asarray(line.get_xydata()) for line in ax.get_lines()[1:]]

  assert len(drawn) == len(dots)
  for dot, expected in zip(drawn, dots, strict=True):
    np.testing.assert_allclose(dot, [expected])


@pytest.mark.parametrize(
  ("kind", "max_info", "ylabel", "lines"),
  [
    ("icc", True, r"$P(\theta)$", ["fee $5"]),
    ("iic", True, r"$I(\theta)$", ["fee $5", "Maximum information"]),
    ("iic", False, r"$I(\theta)$", ["fee $5"]),
  ],
)
def test_item_curve_draws_the_curve_of_its_kind(kind: str, max_info: bool, ylabel: str, lines: list[str]) -> None:
  """Each kind draws its own curve on the one axes, labelled with the item's name, and the dot only with max_info.

  Fails if a kind draws another curve or names its axis after the other, if max_info=False still marks the peak, if a
  curve goes on a second axes, or if the curve is left without the item's name for a legend.
  """
  ax = item_curve(ITEMS, 1, kind=kind, max_info=max_info)

  assert ax.figure.axes == [ax]
  assert ax.get_ylabel() == ylabel
  assert [line.get_label() for line in ax.get_lines()] == lines


def test_item_curve_labels_compare_items_drawn_on_one_axes() -> None:
  """Two items on one axes get two legend entries, by name or by the label given, each curve in its own colour.

  Fails if every curve carries the same label, so that a legend of several items cannot tell them apart, if label is
  ignored, or if the second curve reuses the first one's colour.
  """
  ax = item_curve(ITEMS, 0, parameters=False)
  item_curve(ITEMS, 1, parameters=False, ax=ax, label="the second item")
  first, second = ax.get_lines()

  assert [line.get_label() for line in ax.get_legend_handles_labels()[0]] == ["other", "the second item"]
  assert first.get_color() != second.get_color()


def test_item_curve_draws_an_item_matrix_with_its_items_named_by_position() -> None:
  """An item matrix draws the same curve as a bank of it, and its box names the item as a bank built from it does.

  Fails if a matrix is read in another layout than ItemBank's, or if its item goes unnamed in the box.
  """
  ax = item_curve(ITEMS.items, 1)
  (line,) = ax.get_lines()

  np.testing.assert_allclose(np.asarray(line.get_ydata()), _icc(np.asarray(line.get_xdata())))
  assert ax.texts[0].get_text().splitlines()[0] == "item_1"


@pytest.mark.parametrize("kind", list(Curve))
def test_item_curve_box_names_the_item_and_rounds_its_parameters(kind: Curve) -> None:
  """The box gives the item's name, its dollar sign escaped, then each parameter to three significant digits.

  Fails if the box leaves the item unnamed or names another item, if a parameter is printed at full float precision
  (a = 1.1669737841771746), which makes the box wider than the axes, or if the name's lone dollar sign is left to
  mathtext, which then finds an odd number of them and prints the parameters' as literal dollar signs.
  """
  ax = item_curve(ITEMS, 1, kind=kind)

  assert [text.get_text() for text in ax.texts] == ["fee \\$5\n$a = 1.17$\n$b = -0.432$\n$c = 0.2$\n$d = 1$"]


def test_item_curve_without_parameters_writes_no_box() -> None:
  """parameters=False leaves the box out, so that a second panel of the same item does not repeat it.

  Fails if the box is drawn whatever the flag.
  """
  ax = item_curve(ITEMS, 1, kind=Curve.IIC, parameters=False)

  assert list(ax.texts) == []


@pytest.mark.parametrize(
  ("items", "kind", "theta"),
  [
    (ITEMS, Curve.ICC, None),
    (ITEMS, Curve.IIC, None),
    # An easy item's information peaks near the left end of these abilities and falls toward the right one.
    (ItemBank(np.array([[1.5, -3.5, 0.0, 1.0]])), Curve.IIC, np.linspace(-4, 4, 161)),
    # Abilities that start at the peak, from which the information falls to the right.
    (ITEMS, Curve.IIC, np.linspace(PEAK, PEAK + 8, 161)),
  ],
  ids=["icc", "iic", "iic-of-an-easy-item", "iic-from-its-peak"],
)
def test_item_curve_parameter_box_stays_inside_the_axes_and_off_the_curve(
  items: ItemBank, kind: Curve, theta: FloatArray | None
) -> None:
  """The parameter box lies within the axes and no plotted point falls under it, whichever end the curve is low at.

  Fails if the box sticks out past the edge of the axes, or sits where the curve passes: in the lower right corner,
  where the information curve's tail runs, or in the upper left one when the abilities start at or near its peak.
  """
  ax = item_curve(items, items.n_items - 1, kind=kind, theta=theta)
  ax.figure.canvas.draw()
  (annotation,) = ax.texts
  patch = annotation.get_bbox_patch()
  assert patch is not None
  box, axes_box = patch.get_window_extent(), ax.get_window_extent()

  assert axes_box.x0 <= box.x0 < box.x1 <= axes_box.x1
  assert axes_box.y0 <= box.y0 < box.y1 <= axes_box.y1
  for line in ax.get_lines():
    assert box.count_contains(line.get_transform().transform(line.get_xydata())) == 0


@pytest.mark.parametrize(
  ("item", "kind", "error", "message"),
  [
    (-1, Curve.ICC, ValueError, r"item is -1; item positions must be between 0 and 1"),
    (2, Curve.ICC, ValueError, r"item is 2; item positions must be between 0 and 1"),
    (1.0, Curve.ICC, TypeError, r"item is 1.0; item positions must be integers"),
    (True, Curve.ICC, TypeError, r"item is True; item positions must be integers, not bools"),
    (ITEM, Curve.ICC, TypeError, r"item is array\(.*\); item positions must be integers"),
    (1, "pdf", ValueError, "'pdf' is not a valid Curve"),
  ],
  ids=["negative-item", "item-past-the-bank", "float-item", "bool-item", "parameter-row", "unknown-kind"],
)
def test_item_curve_rejects_bad_input_before_drawing(
  item: object, kind: str, error: type[Exception], message: str
) -> None:
  """A position that is not an item of the bank, a non-integer, or an unknown kind raises before any figure exists.

  Fails if a negative position draws an item counted from the end of the bank, if a float such as 1.0, a bool or a row
  of parameters is read as a position, or if a bad call leaves an empty figure open behind the error.
  """
  with pytest.raises(error, match=message):
    item_curve(ITEMS, item, kind=kind)  # ty: ignore[invalid-argument-type]
  assert plt.get_fignums() == []
