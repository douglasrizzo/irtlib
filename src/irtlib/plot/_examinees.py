"""Plots of examinees on the ability scale: their abilities, likelihoods, posteriors and standard errors."""

import math

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from .. import irt
from .._responses import response_matrix
from ..ability import AbilityEstimates, maximum_likelihood
from ..bank import ItemBank
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..quadrature import LogPrior, QuadratureGrid, normal_log_prior, posterior, prior_masses
from ..typing import FloatArray
from ._common import abilities, as_bank, prepare_axes

_LIKELIHOOD_THETAS = numpy.linspace(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED, 481)
"""The abilities a likelihood is drawn at: -6 to 6, 0.025 apart."""
_MARGIN = 0.5
"""How far the test's standard error curve reaches past the lowest and the highest estimate, in units of ability."""
_POINTS = 161
"""The abilities the test's standard error curve is evaluated at."""


def item_person_map(
  items: ItemBank | FloatArray, theta: npt.ArrayLike, *, ax: Axes | None = None, title: str | None = None
) -> Axes:
  r"""Plot examinees' abilities above the axis and, below it, the abilities where the items inform most.

  Both histograms share one :math:`\theta` scale and its bins, and each shows shares, so a bank and a population of
  any sizes compare: the examinees' bars sum to 1 above the axis and the items' bars to 1 below it. An item sits at
  ``ItemBank.max_info_thetas``, which is its difficulty for one- and two-parameter items.

  .. plot::
      :filename-prefix: item_person_map
      :caption: A 100-item bank against a population of 500 examinees.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.plot import item_person_map

      theta = numpy.random.default_rng(1).normal(0, 1, 500)
      item_person_map(ItemBank.generate(100, "3PL", rng=0), theta)
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix.
  theta : array_like
      The examinees' abilities, true or estimated, as finite numbers.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, or ``theta`` is not a non-empty one-dimensional array of finite numbers,
      before any figure is created.
  """
  bank = as_bank(items)
  values = abilities(theta)
  locations = bank.max_info_thetas
  edges = numpy.histogram_bin_edges(numpy.concatenate([values, locations]), bins="auto")
  examinees, _ = numpy.histogram(values, edges)
  counts, _ = numpy.histogram(locations, edges)
  widths = numpy.diff(edges)
  ax = prepare_axes(ax, title)
  ax.bar(edges[:-1], examinees / values.size, widths, align="edge", label=f"Examinees ({values.size})")
  ax.bar(edges[:-1], -counts / locations.size, widths, align="edge", label=f"Items ({locations.size})")
  ax.axhline(0, color=ax.spines["bottom"].get_edgecolor(), linewidth=ax.spines["bottom"].get_linewidth())
  # The items' shares are drawn downwards, as negative heights, and labelled as the positive shares they are.
  ax.yaxis.set_major_formatter(FuncFormatter(lambda share, _: f"{abs(share):g}"))
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Share of examinees (up) and items (down)")
  ax.legend(loc="upper right")
  return ax


def ability_histogram(
  theta: npt.ArrayLike, *, ax: Axes | None = None, title: str | None = None, label: str | None = None
) -> Axes:
  r"""Plot how many examinees have each ability, as a histogram.

  The bins are NumPy's ``"auto"`` bins of the abilities.

  .. plot::
      :filename-prefix: ability_histogram
      :caption: The abilities of 500 examinees.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib.plot import ability_histogram

      ability_histogram(numpy.random.default_rng(1).normal(0, 1, 500))
      plt.show()

  Parameters
  ----------
  theta : array_like
      The examinees' abilities, true or estimated, as finite numbers.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The histogram's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the histogram is on.

  Raises
  ------
  TypeError
      If ``theta`` holds values that are not real numbers, before any figure is created.
  ValueError
      If ``theta`` is not a non-empty one-dimensional array of finite numbers, before any figure is created.
  """
  values = abilities(theta)
  ax = prepare_axes(ax, title)
  ax.hist(values, bins="auto", label=label)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Examinees")
  return ax


def likelihood(  # noqa: PLR0913 - the responses and items, the two abilities marked, and every plot's ax, title, label
  responses: npt.ArrayLike,
  items: ItemBank | FloatArray,
  *,
  estimate: float | None = None,
  true_theta: float | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the likelihood of one examinee's responses across abilities, relative to its maximum, with an estimate.

  The curve is the likelihood of the responses to the items answered, divided by its maximum, from -6 to 6. A dashed
  line in the curve's colour marks the estimate, and a dotted line the true ability. Responses whose likelihood is
  highest at an infinity, such as all correct or all incorrect, rise to an edge of the range, where the maximum
  likelihood estimate sits; the :doc:`/methods/ability/ml` page gives the estimate. A mark outside -6 to 6 widens the
  axis past the curve.

  .. plot::
      :filename-prefix: likelihood
      :caption: The likelihood of one examinee's responses to 10 items.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.plot import likelihood

      items = ItemBank.generate(10, "3PL", rng=0)
      responses = draw_responses(numpy.array([0.7]), items.items, rng=0)[0]
      likelihood(responses, items, true_theta=0.7)
      plt.show()

  Parameters
  ----------
  responses : array_like
      The examinee's responses, one per item: 1 or True for a correct response, 0 or False for an incorrect one, and
      NaN for an item not answered.
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix, one per response.
  estimate : float or None, optional
      The estimate to mark, such as an EAP estimate or an adaptive test's final one. Default is None, which marks the
      maximum likelihood estimate within -6 and 6.
  true_theta : float or None, optional
      The examinee's true ability, marked by a dotted line. Default is None, which marks none.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve is on.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``responses`` does not hold one response per item, a response is not
      0, 1 or NaN, no item is answered, or ``estimate`` or ``true_theta`` is not finite, before any figure is created.
  """
  bank = as_bank(items)
  row = _examinee(responses, bank.n_items)
  if numpy.isnan(row).all():
    msg = "responses holds no answer; the likelihood of an examinee who answered no item is flat"
    raise ValueError(msg)
  marked = (
    float(maximum_likelihood(row, bank.items, bounds=(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)).theta[0])
    if estimate is None
    else _finite("estimate", estimate)
  )
  true_theta = None if true_theta is None else _finite("true_theta", true_theta)
  log_likelihood = irt.log_likelihood(_LIKELIHOOD_THETAS, row, bank.items)[0]
  ax = prepare_axes(ax, title)
  (line,) = ax.plot(_LIKELIHOOD_THETAS, numpy.exp(log_likelihood - log_likelihood.max()), label=label)
  ax.axvline(marked, color=line.get_color(), linestyle="--", linewidth=1)
  if true_theta is not None:
    ax.axvline(true_theta, color=ax.spines["bottom"].get_edgecolor(), linestyle=":", linewidth=1)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Likelihood relative to its maximum")
  return ax


def prior_posterior(
  responses: npt.ArrayLike,
  items: ItemBank | FloatArray,
  *,
  grid: QuadratureGrid | None = None,
  log_prior: LogPrior | None = None,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  r"""Plot the prior, the likelihood and the posterior of one examinee's responses, each as a density over a grid.

  Each curve integrates to 1 over the grid's nodes, with their weights: the prior dotted, the likelihood dashed and
  the posterior solid, each labelled. With no item answered, the likelihood is flat and the posterior is the prior.
  The posterior is the one the :doc:`/methods/ability/eap` estimate is the mean of.

  .. plot::
      :filename-prefix: prior_posterior
      :caption: The posterior of one examinee after 2 and after 20 items: the likelihood comes to dominate the prior.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.plot import prior_posterior

      items = ItemBank.generate(20, "3PL", rng=0)
      responses = draw_responses(numpy.array([0.7]), items.items, rng=0)[0]
      fig, (top, bottom) = plt.subplots(2, 1, sharex=True, layout="constrained")
      prior_posterior(responses[:2], items.items[:2], ax=top, title="2 items")
      prior_posterior(responses, items, ax=bottom, title="20 items")
      top.label_outer()
      bottom.set_xlim(-4, 4)
      plt.show()

  Parameters
  ----------
  responses : array_like
      The examinee's responses, one per item: 1 or True for a correct response, 0 or False for an incorrect one, and
      NaN for an item not answered.
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix, one per response.
  grid : irtlib.quadrature.QuadratureGrid or None, optional
      The nodes the curves are computed at. Default is ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6], the grid
      of :py:func:`irtlib.ability.expected_a_posteriori`.
  log_prior : irtlib.quadrature.LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curves are on.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``responses`` does not hold one response per item, a response is not
      0, 1 or NaN, or the log-prior is NaN or +inf at a node or has no mass on the grid, before any figure is created.
  """
  bank = as_bank(items)
  row = _examinee(responses, bank.n_items)
  grid = QuadratureGrid.uniform() if grid is None else grid
  log_prior = normal_log_prior() if log_prior is None else log_prior
  log_likelihood = irt.log_likelihood(grid.nodes, row, bank.items)
  curves = {
    "Prior": (prior_masses(grid, log_prior), ":"),
    "Likelihood": (posterior(log_likelihood, grid, _flat_log_prior)[0], "--"),
    "Posterior": (posterior(log_likelihood, grid, log_prior)[0], "-"),
  }
  ax = prepare_axes(ax, title)
  for name, (masses, linestyle) in curves.items():
    ax.plot(grid.nodes, masses / grid.weights, linestyle=linestyle, label=name)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Density")
  ax.legend()
  return ax


def population_posterior(
  responses: npt.ArrayLike,
  items: ItemBank | FloatArray,
  *,
  grid: QuadratureGrid | None = None,
  log_prior: LogPrior | None = None,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  r"""Plot the mean of the examinees' posteriors over a grid against the prior, each as a density.

  Each examinee's posterior is the one :py:func:`prior_posterior` draws, and every examinee counts once in the mean,
  whatever the number of items answered. Both curves are scaled to integrate to 1 over the grid's nodes, with their
  weights: the mean posterior solid and the prior dotted, each labelled for a legend.

  .. plot::
      :filename-prefix: population_posterior
      :caption: The mean posterior of 1,000 examinees of abilities drawn from N(0.5, 1.2²), against the N(0, 1) prior.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.plot import population_posterior

      items = ItemBank.generate(30, "2PL", rng=0)
      theta = numpy.random.default_rng(1).normal(0.5, 1.2, 1000)
      population_posterior(draw_responses(theta, items.items, rng=2), items)
      plt.show()

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank or as an item matrix, one per column of ``responses``.
  grid : irtlib.quadrature.QuadratureGrid or None, optional
      The nodes the posteriors are computed at, such as a calibration's ``grid``. Default is
      ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6], the grid of :py:func:`irtlib.ability.expected_a_posteriori`.
  log_prior : irtlib.quadrature.LogPrior or None, optional
      The prior's log-density. Default is ``normal_log_prior()``, the standard normal.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curves are on.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``responses`` is not a matrix with one column per item and at least
      one row, a response is not 0, 1 or NaN, or the log-prior is NaN or +inf at a node or has no mass on the grid,
      before any figure is created.
  """
  bank = as_bank(items)
  grid = QuadratureGrid.uniform() if grid is None else grid
  log_prior = normal_log_prior() if log_prior is None else log_prior
  log_likelihood = irt.log_likelihood(grid.nodes, responses, bank.items)
  if log_likelihood.shape[0] == 0:
    msg = "responses has no rows; pass the responses of at least one examinee"
    raise ValueError(msg)
  mean = posterior(log_likelihood, grid, log_prior).mean(axis=0)
  prior = prior_masses(grid, log_prior)
  ax = prepare_axes(ax, title)
  ax.plot(grid.nodes, mean / grid.weights, label="Mean posterior")
  ax.plot(grid.nodes, prior / grid.weights, linestyle=":", label="Prior")
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Density")
  ax.legend()
  return ax


def precision(
  estimates: AbilityEstimates,
  items: ItemBank | FloatArray,
  *,
  prior_sd: float | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot each examinee's standard error against their estimate, with the standard error of the test.

  Each examinee with a finite estimate and a finite standard error is an open circle; the others are left out. The
  line is the standard error at each ability of an examinee who answered every item: the :doc:`/methods/ability/ml`
  standard error or, with ``prior_sd``, the :doc:`/methods/ability/map` standard error under a normal prior of that
  standard deviation. It spans the estimates and half a unit of ability past them on each side.

  .. plot::
      :filename-prefix: precision
      :caption: The standard errors of 500 examinees' EAP estimates on a 20-item test.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.ability import expected_a_posteriori
      from irtlib.drawing import draw_responses
      from irtlib.plot import precision

      items = ItemBank.generate(20, "2PL", rng=0)
      theta = numpy.random.default_rng(1).normal(0, 1, 500)
      estimates = expected_a_posteriori(draw_responses(theta, items.items, rng=2), items.items)
      precision(estimates, items, prior_sd=1.0, label="EAP")
      plt.show()

  Parameters
  ----------
  estimates : irtlib.ability.AbilityEstimates
      The examinees' estimates and their standard errors, from any of :py:mod:`irtlib.ability`'s estimators.
  items : irtlib.ItemBank or numpy.ndarray
      The items the estimates were computed from, as a bank or as an item matrix.
  prior_sd : float or None, optional
      The standard deviation of the normal prior the estimates were computed with, for MAP and EAP estimates only;
      the line adds its precision to the test's information. Default is None, the line of maximum likelihood and
      weighted likelihood estimates.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The circles' label, for a legend. Default is None, which labels them "Examinees".

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If ``items`` is not a valid item matrix, ``prior_sd`` is not finite and positive, or no examinee has a finite
      estimate and standard error, before any figure is created.
  """
  bank = as_bank(items)
  if prior_sd is not None and not (math.isfinite(prior_sd) and prior_sd > 0):
    msg = f"prior_sd is {prior_sd}; it must be finite and positive"
    raise ValueError(msg)
  shown = numpy.isfinite(estimates.theta) & numpy.isfinite(estimates.standard_errors)
  if not shown.any():
    msg = "no examinee has a finite estimate and standard error to plot"
    raise ValueError(msg)
  theta, standard_errors = estimates.theta[shown], estimates.standard_errors[shown]
  span = numpy.linspace(theta.min() - _MARGIN, theta.max() + _MARGIN, _POINTS)
  prior_precision = 0.0 if prior_sd is None else prior_sd**-2
  # Far from every item the information underflows to 0, where the standard error is infinite and not drawn.
  with numpy.errstate(divide="ignore"):
    test = 1 / numpy.sqrt(irt.total_information(span, bank.items) + prior_precision)
  ax = prepare_axes(ax, title)
  (circles,) = ax.plot(
    theta, standard_errors, "o", markerfacecolor="none", label="Examinees" if label is None else label
  )
  ax.plot(span, test, color=circles.get_color(), label="Test")
  ax.set_xlabel(r"$\hat\theta$")
  ax.set_ylabel("Standard error")
  ax.legend()
  return ax


def _examinee(responses: npt.ArrayLike, n_items: int) -> FloatArray:
  """Return one examinee's responses as a one-row response matrix, after checking that there is one per item."""
  row = numpy.asarray(responses)
  if row.shape != (n_items,):
    msg = f"responses has shape {row.shape}; it must hold one examinee's responses, one for each of the {n_items} items"
    raise ValueError(msg)
  return response_matrix(row[numpy.newaxis])


def _finite(label: str, value: float) -> float:
  """Return an ability as a float, after checking that it is finite; ``label`` names it in the message."""
  value = float(value)
  if not math.isfinite(value):
    msg = f"{label} is {value}; it must be a finite ability"
    raise ValueError(msg)
  return value


def _flat_log_prior(theta: FloatArray) -> FloatArray:
  """Return a log prior density of 0 at every ability, under which the posterior is the normalized likelihood."""
  return numpy.zeros_like(theta)
