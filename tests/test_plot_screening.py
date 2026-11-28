"""Tests for the plots that screen a response matrix in irtlib.plot."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from irtlib.plot import number_correct_histogram, rest_score_curve

NAN = np.nan
# Six examinees, three items. Item 0's rest scores, the correct responses to items 1 and 2: 2, 2, 1, 0, -, 1, the fifth
# examinee not answering item 0 and the sixth not answering item 1. At rest score 0, 0 of 1 is correct; at 1, 2 of 2;
# at 2, 1 of 2. The numbers correct are 3, 2, 2, 0, 1, 2.
RESPONSES = np.array([
  [1, 1, 1],
  [0, 1, 1],
  [1, 1, 0],
  [0, 0, 0],
  [NAN, 1, 0],
  [1, NAN, 1],
])


def _bars(ax: Axes) -> list[tuple[float, float]]:
  """Return each bar's centre and height."""
  bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
  return [(bar.get_x() + bar.get_width() / 2, bar.get_height()) for bar in bars]


class TestNumberCorrectHistogram:
  """Tests for number_correct_histogram()."""

  def test_number_correct_histogram_counts_the_examinees_at_each_number_correct(self) -> None:
    """The numbers correct 3, 2, 2, 0, 1 and 2 give one bar at each of 0 to 3, of heights 1, 1, 3 and 1.

    Fails if a missing response counts as correct or makes its examinee's score NaN, or if the bars are shares.
    """
    ax = number_correct_histogram(RESPONSES)

    assert _bars(ax) == [(0.0, 1), (1.0, 1), (2.0, 3), (3.0, 1)]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Number correct", "Examinees")

  def test_number_correct_histogram_draws_every_number_correct(self) -> None:
    """With one examinee at 0 and one at 2 of 3 items, the bars at 1 and at 3 are drawn empty.

    Fails if the bars stop at the highest number correct anyone got, so that the empty top of the scale does not show,
    or skip a number nobody got.
    """
    ax = number_correct_histogram(np.array([[0, 0, 0], [1, 1, 0]]), label="form A")

    assert _bars(ax) == [(0.0, 1), (1.0, 0), (2.0, 1), (3.0, 0)]
    assert ax.get_legend_handles_labels()[1] == ["form A"]

  def test_number_correct_histogram_rejects_a_matrix_without_examinees_before_drawing(self) -> None:
    """A response matrix with no rows raises before any figure exists.

    Fails if an empty histogram is drawn, or a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match="responses has no rows"):
      number_correct_histogram(np.empty((0, 3)))
    assert plt.get_fignums() == []


class TestRestScoreCurve:
  """Tests for rest_score_curve()."""

  def test_rest_score_curve_draws_the_proportion_correct_at_each_rest_score(self) -> None:
    """Item 0 is correct for 0 of 1 examinee at rest score 0, 2 of 2 at 1 and 1 of 2 at 2, named item_0.

    Fails if the item's own response counts in its score, if an examinee who did not answer the item counts, if a
    missing response to another item drops its examinee or counts as correct, or if the item goes unnamed.
    """
    ax = rest_score_curve(RESPONSES, 0)
    (line,) = ax.get_lines()

    np.testing.assert_array_equal(np.asarray(line.get_xydata()), [[0, 0.0], [1, 1.0], [2, 0.5]])
    assert line.get_label() == "item_0"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Rest score", "Proportion correct")

  def test_rest_score_curve_leaves_out_rest_scores_nobody_has(self) -> None:
    """Item 2's rest scores are 2, 1, 2, 0, 1 and 1: 0 of 1 correct at 0, 2 of 3 at 1 and 1 of 2 at 2.

    Of the first, third and fourth examinees, at rest scores 2, 2 and 0, nobody has rest score 1, which gets no point.
    Fails if a rest score with no examinee is drawn at 0, or as NaN that breaks the line, or if the label given is
    ignored.
    """
    (line,) = rest_score_curve(RESPONSES, 2, label="third").get_lines()
    (sparse,) = rest_score_curve(RESPONSES[[0, 2, 3]], 2).get_lines()

    np.testing.assert_allclose(np.asarray(line.get_xydata()), [[0, 0.0], [1, 2 / 3], [2, 0.5]])
    assert line.get_label() == "third"
    np.testing.assert_allclose(np.asarray(sparse.get_xydata()), [[0, 0.0], [2, 0.5]])

  @pytest.mark.parametrize(
    ("responses", "item", "error", "message"),
    [
      (RESPONSES, 3, ValueError, r"item is 3; item positions must be between 0 and 2"),
      (RESPONSES, -1, ValueError, r"item is -1; item positions must be between 0 and 2"),
      (RESPONSES, 1.0, TypeError, r"item is 1.0; item positions must be integers"),
      (np.array([[NAN, 1.0], [NAN, 0.0]]), 0, ValueError, "no examinee answered item 0"),
      (np.array([1, 0]), 0, ValueError, r"responses must be a matrix"),
    ],
    ids=["past-the-items", "negative", "float", "unanswered", "vector"],
  )
  def test_rest_score_curve_rejects_bad_input_before_drawing(
    self, responses: np.ndarray, item: object, error: type[Exception], message: str
  ) -> None:
    """An item that is not a column of the matrix, or that nobody answered, raises before any figure exists.

    Fails if a negative item is read from the end of the matrix, a float is truncated, an empty curve is drawn, or a
    figure is left open behind the error.
    """
    with pytest.raises(error, match=message):
      rest_score_curve(responses, item)  # ty: ignore[invalid-argument-type]
    assert plt.get_fignums() == []


_DRAW: dict[str, Callable[[Axes, str], Axes]] = {
  "number_correct_histogram": lambda ax, title: number_correct_histogram(RESPONSES, ax=ax, title=title),
  "rest_score_curve": lambda ax, title: rest_score_curve(RESPONSES, 0, ax=ax, title=title),
}


@pytest.mark.parametrize("name", list(_DRAW))
def test_screening_plot_draws_on_the_given_axes_with_the_title(name: str) -> None:
  """Each plot draws on the axes it is given, sets the title there and returns those axes, with no other axes.

  Fails if a function opens a figure of its own despite ``ax``, ignores ``title``, or returns other axes than the
  caller's.
  """
  figure, ax = plt.subplots()

  assert _DRAW[name](ax, "A title") is ax
  assert ax.get_title() == "A title"
  assert figure.axes == [ax]
  assert plt.get_fignums() == [figure.number]
