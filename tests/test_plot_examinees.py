"""Tests for the plots of examinees on the ability scale in irtlib.plot."""

import math
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from irtlib import ItemBank
from irtlib.ability import AbilityEstimates, expected_a_posteriori
from irtlib.plot import (
  ability_histogram,
  item_person_map,
  likelihood,
  population_posterior,
  precision,
  prior_posterior,
)
from irtlib.quadrature import QuadratureGrid, normal_log_prior
from irtlib.typing import FloatArray

# Two 2PL items at b = 0: answering one right and one wrong has likelihood P (1 - P), whose maximum, 1/4, is at 0.
TWO_ITEMS = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
# One Rasch item at b = 0, and a grid of five nodes 0.5 apart, each weighing 0.5, for densities worked by hand.
RASCH_ITEM = np.array([[1.0, 0.0, 0.0, 1.0]])
GRID = QuadratureGrid.uniform(5, -1.0, 1.0)
NODES = GRID.nodes


def _logistic(x: FloatArray | float) -> FloatArray:
  """The logistic function, 1 / (1 + exp(-x))."""
  return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def _density(values: FloatArray) -> FloatArray:
  """Scale values at GRID's nodes to integrate to 1 over the grid, each node weighing 0.5."""
  return values / (values.sum() * 0.5)


def _bars(ax: Axes) -> list[Rectangle]:
  """Return the bars of a plot."""
  return [patch for patch in ax.patches if isinstance(patch, Rectangle)]


def _xy(ax: Axes, label: str) -> tuple[FloatArray, FloatArray]:
  """Return the data of the line labelled ``label``."""
  (line,) = [line for line in ax.get_lines() if line.get_label() == label]
  return np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)


class TestItemPersonMap:
  """Tests for item_person_map()."""

  # A 3PL item with a = 0.3, b = -1 and c = 0.5 is most informative at b + log((1 + sqrt(1 + 8c)) / 2) / a.
  ITEMS = ItemBank(np.array([[0.3, -1.0, 0.5, 1.0]]))
  LOCATION = -1.0 + math.log((1 + math.sqrt(5)) / 2) / 0.3
  THETA = np.linspace(-3, 3, 601)

  def test_item_person_map_places_items_where_they_are_most_informative(self) -> None:
    """The item's bar, below the axis, covers the ability where it is most informative, about 0.60, not its b of -1.

    Fails if items are placed at their difficulty, which for three-parameter items lies left of where they measure.
    """
    (item_bar,) = [bar for bar in _bars(item_person_map(self.ITEMS, self.THETA)) if bar.get_height() < 0]

    assert item_bar.get_x() <= self.LOCATION < item_bar.get_x() + item_bar.get_width()
    assert not item_bar.get_x() <= -1.0 < item_bar.get_x() + item_bar.get_width()

  def test_item_person_map_draws_shares_so_that_sizes_compare(self) -> None:
    """The examinees' bars sum to 1 above the axis and the items' bars to 1 below it, each labelled with its count.

    Fails if the bars count examinees and items, which puts a population of 601 and a bank of 1 on scales 600 times
    apart, or if the legend leaves out how many of each there are.
    """
    ax = item_person_map(self.ITEMS, self.THETA)
    heights = np.array([bar.get_height() for bar in _bars(ax)])

    assert heights[heights > 0].sum() == pytest.approx(1.0)
    assert heights[heights < 0].sum() == pytest.approx(-1.0)
    assert ax.get_legend_handles_labels()[1] == ["Examinees (601)", "Items (1)"]

  def test_item_person_map_labels_the_items_shares_as_positive(self) -> None:
    """The y axis reads the items' downward bars as the positive shares they are: -0.25 is labelled 0.25.

    Fails if the items' shares are labelled with a minus sign, as plain negative heights would be.
    """
    formatter = item_person_map(self.ITEMS, self.THETA).yaxis.get_major_formatter()

    assert [formatter(-0.25, 0), formatter(0.25, 0)] == ["0.25", "0.25"]


class TestAbilityHistogram:
  """Tests for ability_histogram()."""

  THETA = np.array([-1.3, -0.2, 0.0, 0.1, 0.4, 0.45, 2.2])

  def test_ability_histogram_counts_the_examinees_in_each_bar(self) -> None:
    """Each bar's height is the number of abilities in its span, counted by brute force, and they sum to 7.

    Fails if the bars show shares or densities in place of counts, or if an examinee is dropped or counted twice.
    """
    ax = ability_histogram(self.THETA, label="group")
    bars = _bars(ax)
    counted = [
      sum(
        bar.get_x() <= theta < bar.get_x() + bar.get_width() or (index == len(bars) - 1 and theta == self.THETA.max())
        for theta in self.THETA
      )
      for index, bar in enumerate(bars)
    ]

    assert [bar.get_height() for bar in bars] == counted
    assert sum(counted) == self.THETA.size
    assert ax.get_ylabel() == "Examinees"
    assert ax.get_legend_handles_labels()[1] == ["group"]


class TestLikelihood:
  """Tests for likelihood()."""

  def test_likelihood_is_relative_to_its_maximum_and_marks_the_ml_estimate(self) -> None:
    """The curve is 4 P (1 - P): 1 at theta = 0 and 4 logistic(1) logistic(-1) at 1, the ML estimate 0 dashed.

    Fails if the curve is the log-likelihood or is not divided by its maximum, if it spans another range than -6 to 6,
    or if the dashed line marks another ability than the maximum likelihood estimate, or in another colour.
    """
    ax = likelihood([1, 0], TWO_ITEMS)
    curve, estimate = ax.get_lines()
    x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())

    np.testing.assert_allclose(y, 4 * _logistic(x) * _logistic(-x))
    assert np.interp(1.0, x, y) == pytest.approx(4 * _logistic(1) * _logistic(-1))
    assert (x[0], x[-1]) == (-6.0, 6.0)
    np.testing.assert_allclose(np.asarray(estimate.get_xdata()), [0.0, 0.0], atol=1e-6)
    assert estimate.get_linestyle() == "--"
    assert estimate.get_color() == curve.get_color()

  def test_likelihood_marks_the_estimate_and_the_true_ability_given(self) -> None:
    """With estimate 0.1 and true_theta 0.4, the dashed line is at 0.1 and a dotted line at 0.4.

    Fails if the estimate given is ignored for the ML estimate, or if the true ability is not marked.
    """
    ax = likelihood([1, 0], TWO_ITEMS, estimate=0.1, true_theta=0.4)
    _, estimate, true_ability = ax.get_lines()

    assert np.asarray(estimate.get_xdata()).tolist() == [0.1, 0.1]
    assert np.asarray(true_ability.get_xdata()).tolist() == [0.4, 0.4]
    assert true_ability.get_linestyle() == ":"

  def test_likelihood_of_correct_responses_rises_to_the_edge_where_the_estimate_is(self) -> None:
    """Two correct responses have no maximum inside the range, so the curve rises to 1 at 6, the ML estimate within it.

    Fails if the curve is divided by its value at an estimate inside the range, which puts it above 1 past it, or if
    the infinite unbounded ML estimate is marked, which draws no line.
    """
    curve, estimate = likelihood([True, True], TWO_ITEMS).get_lines()
    y = np.asarray(curve.get_ydata())

    assert (np.diff(y) > 0).all()
    assert y[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray(estimate.get_xdata()), [6.0, 6.0], atol=1e-6)

  def test_likelihood_leaves_out_the_items_not_answered(self) -> None:
    """A missing response to a third item leaves the curve of the two answered, 4 P (1 - P).

    Fails if a missing response is read as incorrect, which multiplies the curve by 1 - P and moves its peak left.
    """
    items = np.vstack([TWO_ITEMS, [[2.0, 1.0, 0.0, 1.0]]])
    curve, _ = likelihood([1.0, np.nan, 0.0], items[[0, 2, 1]]).get_lines()
    x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())

    np.testing.assert_allclose(y, 4 * _logistic(x) * _logistic(-x))

  @pytest.mark.parametrize(
    ("responses", "options", "error", "message"),
    [
      ([1, 0, 1], {}, ValueError, r"responses has shape \(3,\); it must hold one examinee's responses, one for each"),
      ([[1, 0]], {}, ValueError, r"responses has shape \(1, 2\)"),
      ([1, 2], {}, ValueError, r"responses\[0, 1\] is 2; an integer response must be 0 or 1"),
      (["1", "0"], {}, TypeError, "responses holds <U1 values"),
      ([np.nan, np.nan], {}, ValueError, "responses holds no answer"),
      ([1, 0], {"estimate": math.inf}, ValueError, "estimate is inf; it must be a finite ability"),
      ([1, 0], {"true_theta": math.nan}, ValueError, "true_theta is nan; it must be a finite ability"),
    ],
    ids=["too-many", "matrix", "not-0-or-1", "strings", "no-answer", "infinite-estimate", "nan-true-ability"],
  )
  def test_likelihood_rejects_bad_input_before_drawing(
    self, responses: object, options: dict[str, float], error: type[Exception], message: str
  ) -> None:
    """Responses that are not one per item, not 0, 1 or NaN, or all missing, and marks that are not finite, raise.

    Fails if a wrong number of responses is matched to the items by position, if a flat curve with no estimate is drawn
    for an examinee who answered nothing, if an infinite mark is drawn as no line, or if a figure is left open behind
    the error.
    """
    with pytest.raises(error, match=message):
      likelihood(responses, TWO_ITEMS, **options)  # ty: ignore[invalid-argument-type]
    assert plt.get_fignums() == []


class TestPriorPosterior:
  """Tests for prior_posterior()."""

  def test_prior_posterior_draws_each_curve_as_a_density(self) -> None:
    """After a correct response to a Rasch item at 0, the prior, likelihood and posterior are phi, P and phi P, scaled.

    On five nodes 0.5 apart, each curve is proportional to its closed form and integrates to 1 over the grid. Fails if
    a curve is drawn as masses that sum to 1, which depend on the spacing, if the posterior leaves out the prior or the
    likelihood, or if the curves are mislabelled or drawn in other line styles than dotted, dashed and solid.
    """
    ax = prior_posterior([1], RASCH_ITEM, grid=GRID, log_prior=normal_log_prior())
    prior, likely = np.exp(-(NODES**2) / 2), _logistic(NODES)

    for name, expected, linestyle in [
      ("Prior", _density(prior), ":"),
      ("Likelihood", _density(likely), "--"),
      ("Posterior", _density(prior * likely), "-"),
    ]:
      x, y = _xy(ax, name)
      np.testing.assert_allclose(x, NODES)
      np.testing.assert_allclose(y, expected)
      (line,) = [line for line in ax.get_lines() if line.get_label() == name]
      assert line.get_linestyle() == linestyle
    assert ax.get_legend() is not None

  def test_prior_posterior_of_no_answer_has_a_flat_likelihood_and_the_prior_as_posterior(self) -> None:
    """An examinee who answered no item has a likelihood of 1 / 2, flat over a grid 2 wide, and the prior as posterior.

    Fails if a missing response is read as incorrect, or if the flat likelihood has no density to scale.
    """
    ax = prior_posterior([np.nan], RASCH_ITEM, grid=GRID, log_prior=normal_log_prior())

    np.testing.assert_allclose(_xy(ax, "Likelihood")[1], np.full(5, 0.4))
    np.testing.assert_allclose(_xy(ax, "Posterior")[1], _xy(ax, "Prior")[1])

  def test_prior_posterior_defaults_to_the_posterior_of_the_eap_estimate(self) -> None:
    """Without a grid or prior, the posterior is the one whose mean is the EAP estimate, on 121 nodes from -6 to 6.

    Fails if the default grid or prior differs from expected_a_posteriori's, so that the posterior drawn is not the one
    the estimate comes from.
    """
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.5, 0.0, 0.2, 1.0], [0.8, 1.0, 0.0, 0.95]])
    x, y = _xy(prior_posterior([1, 0, np.nan], items), "Posterior")

    assert (x.size, x[0], x[-1]) == (121, -6.0, 6.0)
    # The EAP estimate of these responses, from expected_a_posteriori's own example.
    assert float((x * y).sum() * 0.1) == pytest.approx(-0.2809, abs=5e-5)
    assert float((x * y).sum() * 0.1) == pytest.approx(expected_a_posteriori([[1, 0, np.nan]], items).theta[0])

  def test_prior_posterior_rejects_a_bad_prior_before_drawing(self) -> None:
    """A log-prior that is NaN at a node raises before any figure exists.

    Fails if a bad prior draws NaN curves, or leaves a figure open behind the error.
    """
    with pytest.raises(ValueError, match=r"the log-prior is NaN or \+inf at some node"):
      prior_posterior([1], RASCH_ITEM, grid=GRID, log_prior=lambda theta: np.full_like(theta, np.nan))
    assert plt.get_fignums() == []


class TestPopulationPosterior:
  """Tests for population_posterior()."""

  def test_population_posterior_of_opposite_responses_is_the_prior(self) -> None:
    """One correct and one incorrect response to a Rasch item at 0 average back to the symmetric prior.

    Their posteriors are 2 phi P / Z and 2 phi (1 - P) / Z, with Z the prior's sum, since P(x) + P(-x) = 1: their mean
    is phi / Z. Fails if the mean is taken over unnormalized posteriors, or the curves are not densities over the grid.
    """
    ax = population_posterior([[1], [0]], RASCH_ITEM, grid=GRID, log_prior=normal_log_prior())
    prior = _density(np.exp(-(NODES**2) / 2))

    np.testing.assert_allclose(_xy(ax, "Mean posterior")[1], prior)
    np.testing.assert_allclose(_xy(ax, "Prior")[1], prior)
    (line,) = [line for line in ax.get_lines() if line.get_label() == "Prior"]
    assert line.get_linestyle() == ":"

  def test_population_posterior_counts_each_examinee_once(self) -> None:
    """An examinee who answered two items and one who answered none weigh one half each in the mean.

    The first's posterior is proportional to phi P^2 and the second's is the prior. Fails if examinees are weighted by
    the number of items they answered, which leaves the second out, or if a missing response is read as incorrect.
    """
    ax = population_posterior([[1, 1], [np.nan, np.nan]], np.vstack([RASCH_ITEM, RASCH_ITEM]), grid=GRID)
    prior = np.exp(-(NODES**2) / 2)

    np.testing.assert_allclose(
      _xy(ax, "Mean posterior")[1], (_density(prior * _logistic(NODES) ** 2) + _density(prior)) / 2
    )

  def test_population_posterior_rejects_responses_without_examinees_before_drawing(self) -> None:
    """A response matrix with no rows raises before any figure exists.

    Fails if the mean of no posteriors is drawn as NaN, with a warning, or a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match="responses has no rows"):
      population_posterior(np.empty((0, 1)), RASCH_ITEM)
    assert plt.get_fignums() == []


class TestPrecision:
  """Tests for precision()."""

  # One 2PL item with a = 2 and b = 0 has information 4 P (1 - P) at P = logistic(2 theta): 1 at theta = 0.
  ITEM = np.array([[2.0, 0.0, 0.0, 1.0]])
  ESTIMATES = AbilityEstimates(
    theta=np.array([-1.0, 0.0, np.inf, np.nan, 0.3]),
    standard_errors=np.array([0.5, 0.4, np.inf, np.inf, np.nan]),
    evaluations=np.zeros(5, dtype=int),
    from_fallback=np.zeros(5, dtype=bool),
  )

  @staticmethod
  def _test_se(theta: FloatArray, prior_precision: float) -> FloatArray:
    """The item's standard error in closed form, 1 / sqrt(4 P (1 - P) + 1 / sigma^2)."""
    p = _logistic(2 * theta)
    return 1 / np.sqrt(4 * p * (1 - p) + prior_precision)

  @pytest.mark.parametrize(("prior_sd", "prior_precision"), [(None, 0.0), (1.0, 1.0), (0.5, 4.0)])
  def test_precision_draws_the_test_standard_error_past_the_estimates(
    self, prior_sd: float | None, prior_precision: float
  ) -> None:
    """The line is 1 / sqrt(I + 1 / sigma^2), 1 at 0 without a prior and 1 / sqrt(2) with sigma = 1, from -1.5 to 0.5.

    Fails if the prior's precision is left out of the line or added as its standard deviation, if the line spans
    another range than half a unit past the finite estimates, or if it is drawn in another colour than the circles.
    """
    ax = precision(self.ESTIMATES, self.ITEM, prior_sd=prior_sd)
    x, y = _xy(ax, "Test")
    circles, line = ax.get_lines()

    assert (x[0], x[-1]) == (-1.5, 0.5)
    np.testing.assert_allclose(y, self._test_se(x, prior_precision))
    assert np.interp(0.0, x, y) == pytest.approx(1 / math.sqrt(1 + prior_precision), rel=1e-4)
    assert line.get_color() == circles.get_color()

  def test_precision_draws_each_finite_estimate_at_its_standard_error(self) -> None:
    """Only the examinees with a finite estimate and standard error are drawn, at (-1, 0.5) and (0, 0.4).

    Fails if an infinite estimate, an examinee with no estimate or a NaN standard error is drawn, or if the label given
    is ignored.
    """
    ax = precision(self.ESTIMATES, self.ITEM, label="ML")
    x, y = _xy(ax, "ML")

    np.testing.assert_array_equal(np.column_stack([x, y]), [[-1.0, 0.5], [0.0, 0.4]])
    assert ax.get_legend_handles_labels()[1] == ["ML", "Test"]

  @pytest.mark.parametrize(
    ("estimates", "prior_sd", "message"),
    [
      (ESTIMATES, 0.0, "prior_sd is 0.0; it must be finite and positive"),
      (ESTIMATES, -1.0, "prior_sd is -1.0; it must be finite and positive"),
      (ESTIMATES, math.inf, "prior_sd is inf; it must be finite and positive"),
      (ESTIMATES, math.nan, "prior_sd is nan; it must be finite and positive"),
      (
        AbilityEstimates(np.array([np.inf]), np.array([np.inf]), np.zeros(1, dtype=int), np.zeros(1, dtype=bool)),
        None,
        "no examinee has a finite estimate and standard error to plot",
      ),
    ],
    ids=["zero-sd", "negative-sd", "infinite-sd", "nan-sd", "no-finite-estimate"],
  )
  def test_precision_rejects_bad_input_before_drawing(
    self, estimates: AbilityEstimates, prior_sd: float | None, message: str
  ) -> None:
    """A prior standard deviation that is not finite and positive, or no finite estimate, raises before drawing.

    Fails if a zero or infinite prior_sd draws a line of no meaning, if an empty plot is drawn, or if a figure is left
    open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      precision(estimates, self.ITEM, prior_sd=prior_sd)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("plot", [item_person_map, ability_histogram], ids=["item_person_map", "ability_histogram"])
@pytest.mark.parametrize(
  ("theta", "error", "message"),
  [
    ([], ValueError, "theta is empty; it must hold one ability per examinee, at least one"),
    ([0.0, math.nan], ValueError, r"theta\[1\] is nan; abilities must be finite numbers"),
    (
      [[0.0, 1.0]],
      ValueError,
      r"theta must hold one ability per examinee, a one-dimensional array, got shape \(1, 2\)",
    ),
    (["a"], TypeError, "theta holds <U1 values; abilities must be real numbers"),
  ],
  ids=["empty", "nan", "matrix", "strings"],
)
def test_plot_of_abilities_rejects_abilities_that_are_not_finite_numbers_before_drawing(
  plot: Callable[..., Axes], theta: npt.ArrayLike, error: type[Exception], message: str
) -> None:
  """Abilities that are empty, NaN, not one per examinee or not numbers raise before any figure exists.

  Fails if NaN abilities are dropped silently from the shares, an empty histogram is drawn, or a figure is left open
  behind the error.
  """
  args = (TestItemPersonMap.ITEMS, theta) if plot is item_person_map else (theta,)
  with pytest.raises(error, match=message):
    plot(*args)
  assert plt.get_fignums() == []


_DRAW: dict[str, Callable[[Axes, str], Axes]] = {
  "item_person_map": lambda ax, title: item_person_map(TestItemPersonMap.ITEMS, [0.0, 1.0], ax=ax, title=title),
  "ability_histogram": lambda ax, title: ability_histogram([0.0, 1.0], ax=ax, title=title),
  "likelihood": lambda ax, title: likelihood([1, 0], TWO_ITEMS, ax=ax, title=title),
  "prior_posterior": lambda ax, title: prior_posterior([1], RASCH_ITEM, ax=ax, title=title),
  "population_posterior": lambda ax, title: population_posterior([[1], [0]], RASCH_ITEM, ax=ax, title=title),
  "precision": lambda ax, title: precision(TestPrecision.ESTIMATES, TestPrecision.ITEM, ax=ax, title=title),
}


@pytest.mark.parametrize("name", list(_DRAW))
def test_plot_draws_on_the_given_axes_with_the_title(name: str) -> None:
  """Each plot draws on the axes it is given, sets the title there and returns those axes, with no other axes.

  Fails if a function opens a figure of its own despite ``ax``, ignores ``title``, adds a second axes for a second
  scale, or returns other axes than the caller's.
  """
  figure, ax = plt.subplots()

  assert _DRAW[name](ax, "A title") is ax
  assert ax.get_title() == "A title"
  assert ax.get_lines() or ax.patches
  assert figure.axes == [ax]
  assert plt.get_fignums() == [figure.number]
