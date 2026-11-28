"""Plots of fit: how the responses agree with the calibrated items, item by item and examinee by examinee."""

import operator

import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.typing import ColorType
from scipy.special import betaincinv, ndtri

from .. import irt
from .._responses import responses_and_items
from ..bank import ItemBank, _generated_names
from ..calibration import Calibration
from ..fit import ItemFit, MeanSquares
from ..typing import BoolArray, FloatArray, IntArray
from ._common import (
  HALF_WIDTH,
  POINTS,
  abilities,
  as_bank,
  check_level,
  examinee_position,
  item_position,
  label_items,
  name_points,
  prepare_axes,
  reference_style,
  text_box,
)

_BAND = 2.0
"""The bound of the band a standardized fit statistic or residual is expected within, on each side of 0."""


def empirical_icc(  # noqa: PLR0913 - the items, the item, its responses and abilities, the groups, the interval's level, and every plot's ax, title, label
  items: ItemBank | FloatArray,
  item: int,
  responses: npt.ArrayLike,
  theta: npt.ArrayLike,
  *,
  groups: int = 10,
  level: float = 0.95,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot an item's characteristic curve with the proportion correct observed in groups of examinees by ability.

  The examinees who answered the item are ranked by ability and split into groups whose sizes differ by at most one,
  ten groups by default, as :cite:t:`Yen1981` (p. 246) groups them. Each group's dot is its proportion correct at the
  median of its abilities, where :cite:t:`Yen1981` (p. 248) describes Bock evaluating the curve, with an exact
  binomial (Clopper-Pearson) interval at ``level``. Dots off the curve, beyond their intervals, show misfit. The dots
  take the curve's colour, so that several items compare on one axes.

  .. plot::
      :filename-prefix: empirical_icc
      :caption: An item calibrated as a 2PL from responses simulated with guessing, against ten groups of examinees.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.ability import expected_a_posteriori
      from irtlib.drawing import draw_responses
      from irtlib.mml import MMLCalibrator
      from irtlib.plot import empirical_icc

      bank = ItemBank.generate(20, "3PL", rng=2)
      responses = draw_responses(numpy.random.default_rng(1).normal(size=2000), bank.items, rng=3)
      calibration = MMLCalibrator("2PL").calibrate(responses)
      theta = expected_a_posteriori(responses, calibration.estimates, grid=calibration.grid).theta
      empirical_icc(calibration.estimates, 0, responses, theta)
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The calibrated items, as a bank, whose names label them, or as an item matrix, whose items are named by position.
  item : int
      The item's position in ``items``; ``ItemBank.index(name)`` gives the position of a named item.
  responses : array_like
      The response matrix: one row per examinee, one column per item of ``items``, 1 or 0, and NaN for a missing
      response. Examinees who did not answer the item are left out.
  theta : array_like
      Each examinee's ability estimate, one per row of ``responses``, such as the ``theta`` of an
      :py:class:`~irtlib.ability.AbilityEstimates`; finite for every examinee who answered the item.
  groups : int, optional
      The number of groups, at least 1 and at most the number of examinees who answered the item. Default is 10.
  level : float, optional
      The confidence level of each group's interval, strictly between 0 and 1. Default is 0.95.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend, which the dots' label follows with ", observed". Default is None, which labels
      it with the item's name.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the curve and the groups are on.

  Raises
  ------
  TypeError
      If ``item`` or ``groups`` is not an integer, or the responses or ``theta`` are not numbers, before any figure is
      created.
  ValueError
      If ``items`` is not a valid item matrix, ``item`` is not a position in it, ``responses`` is not a response
      matrix with one column per item, ``theta`` does not hold one ability per examinee or is not finite for an
      examinee who answered the item, ``groups`` is less than 1 or more than the examinees who answered the item, or
      ``level`` is not strictly between 0 and 1, before any figure is created.
  """
  bank = as_bank(items)
  item = item_position("item", item, bank.n_items)
  matrix, _ = responses_and_items(responses, bank.items)
  theta = abilities(theta, matrix.shape[0])
  level = check_level(level)
  answered = ~numpy.isnan(matrix[:, item])
  _check_finite(theta, answered, "every examinee who answered the item")
  groups = _group_count(groups, int(answered.sum()))

  medians, proportions, errors = _observed(theta[answered], matrix[answered, item], groups, level)
  label = bank.names[item] if label is None else label
  ax = prepare_axes(ax, title)

  # the groups follow the ranking, so the first median is the lowest and the last the highest
  colour = _draw_icc(ax, bank.items[[item]], medians[0], medians[-1], label)
  ax.errorbar(medians, proportions, yerr=errors, fmt="o", color=colour, label=f"{label}, observed")
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Proportion correct")
  return ax


def sx2_item(  # noqa: PLR0913 - the items, the item and its fit, the statistic's box, and every plot's ax, title, label
  items: ItemBank | FloatArray,
  item: int,
  fit: ItemFit,
  *,
  statistic: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot an item's proportion correct at each number correct, observed and as S-X² expects it.

  The examinees are grouped by their number correct, from 1 to M - 1 for M items, as S-X² groups them
  (:cite:t:`Orlando2000`, p. 54), before any group is collapsed. A line joins the expected proportions, and dots of its
  colour mark the observed ones; a number correct no examinee has leaves no dot. A box in the lower right corner gives
  the item's name and its S-X² with its degrees of freedom and p-value, to three significant digits.

  .. plot::
      :filename-prefix: sx2_item
      :caption: An item calibrated as a 2PL from responses simulated with guessing, by number correct.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.fit import s_x2
      from irtlib.mml import MMLCalibrator
      from irtlib.plot import sx2_item

      bank = ItemBank.generate(20, "3PL", rng=2)
      responses = draw_responses(numpy.random.default_rng(1).normal(size=2000), bank.items, rng=3)
      calibration = MMLCalibrator("2PL").calibrate(responses)
      sx2_item(calibration.estimates, 0, s_x2(calibration, responses))
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The calibrated items, as a bank, whose names label them, or as an item matrix, whose items are named by position.
  item : int
      The item's position in ``items``; ``ItemBank.index(name)`` gives the position of a named item.
  fit : irtlib.fit.ItemFit
      The S-X² of the same items, in the same order, as :py:func:`irtlib.fit.s_x2` returns it.
  statistic : bool, optional
      Whether to write the item's name and statistic in a box. Default is True.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The line's label, for a legend, which the dots' label follows with ", observed". Default is None, which labels
      it with the item's name.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the proportions are on.

  Raises
  ------
  TypeError
      If ``item`` is not an integer, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``item`` is not a position in it, or ``fit`` is not the S-X² of as many
      items, before any figure is created.
  """
  bank = as_bank(items)
  item = item_position("item", item, bank.n_items)
  if fit.observed.shape[0] != bank.n_items:
    msg = f"fit holds the S-X² of {fit.observed.shape[0]} items; it must hold one per item, {bank.n_items}"
    raise ValueError(msg)
  scores = numpy.arange(1, fit.observed.shape[1] + 1)
  label = bank.names[item] if label is None else label
  ax = prepare_axes(ax, title)

  (line,) = ax.plot(scores, fit.expected[item], label=label)
  ax.plot(scores, fit.observed[item], "o", color=line.get_color(), label=f"{label}, observed")
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel("Number correct")
  ax.set_ylabel("Proportion correct")
  if statistic:
    # the statistic is mathtext, so a dollar sign in the name would pair up with its
    name = bank.names[item].replace("$", r"\$")
    df = int(fit.df[item])
    value = (
      f"$S$-$X^2({df}) = {fit.statistic[item]:.3g}$\n$p = {fit.p_value[item]:.3g}$"
      if df > 0
      else "$S$-$X^2$ has no degrees of freedom"
    )
    # the proportions rise with the number correct, which leaves the lower right corner clear
    text_box(ax, f"{name}\n{value}", lower_right=True)
  return ax


def pathway_map(
  items: ItemBank | FloatArray,
  fit: MeanSquares,
  *,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot each item's standardized infit against its difficulty, with the band from -2 to 2 it is expected within.

  Dashed lines mark the band and a solid line 0, as in Figure 5.4c of :cite:t:`Wright1982` (pp. 102-103), where most
  items fall within the band. The items outside it are named. An item whose standardized infit is NaN is left out.

  .. plot::
      :filename-prefix: pathway_map
      :caption: A 1PL calibration of 20 items, of which item_03 was simulated flatter and item_12 steeper.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.drawing import draw_responses
      from irtlib.fit import mean_squares
      from irtlib.mml import MMLCalibrator
      from irtlib.plot import pathway_map

      items = ItemBank.generate(20, "1PL", rng=2).items.copy()
      items[[3, 12], 0] = [0.4, 2.5]
      responses = draw_responses(numpy.random.default_rng(1).normal(size=300), items, rng=3)
      calibration = MMLCalibrator("1PL").calibrate(responses)
      pathway_map(calibration.estimates, mean_squares(calibration, responses))
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The calibrated items, as a bank, whose names label them, or as an item matrix, whose items are named by position.
  fit : irtlib.fit.MeanSquares
      The mean squares of the same items, in the same order, as :py:func:`irtlib.fit.mean_squares` returns them.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The items' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the items are on.

  Raises
  ------
  ValueError
      If ``items`` is not a valid item matrix, or ``fit`` does not hold one standardized infit per item, before any
      figure is created.
  """
  bank = as_bank(items)
  infit_t = numpy.asarray(fit.infit_t, dtype=float)
  if infit_t.shape != (bank.n_items,):
    msg = f"fit holds standardized infits of shape {infit_t.shape}; it must hold one per item, {bank.n_items}"
    raise ValueError(msg)
  shown = numpy.flatnonzero(numpy.isfinite(infit_t))
  difficulty, infit_t = bank.difficulty[shown], infit_t[shown]
  ax = prepare_axes(ax, title)

  _draw_fit_map(ax, difficulty, infit_t, label)
  outside = numpy.abs(infit_t) > _BAND
  name_points(ax, difficulty[outside], infit_t[outside], [bank.names[item] for item in shown[outside]])
  ax.set_xlabel("Difficulty $b$")
  return ax


def person_fit_map(
  theta: npt.ArrayLike,
  fit: MeanSquares,
  *,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot each examinee's standardized infit against their ability, with the band from -2 to 2 it is expected within.

  An examinee's infit is standardized as an item's is, so the band is the one of :py:func:`pathway_map`: dashed lines
  at -2 and 2 and a solid line at 0. An examinee above the band answered less predictably than the model expects, one
  below it more (:cite:t:`Wright1982`, p. 109). An examinee whose ability or standardized infit is not finite is left
  out.

  .. plot::
      :filename-prefix: person_fit_map
      :caption: 500 examinees of a 2PL test, the first 20 of whom answered every item at random.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.ability import weighted_likelihood
      from irtlib.drawing import draw_responses
      from irtlib.fit import person_mean_squares
      from irtlib.plot import person_fit_map

      bank = ItemBank.generate(30, "2PL", rng=2)
      responses = draw_responses(numpy.random.default_rng(1).normal(size=500), bank.items, rng=3)
      responses[:20] = numpy.random.default_rng(4).integers(0, 2, size=(20, 30))
      theta = weighted_likelihood(responses, bank.items).theta
      person_fit_map(theta, person_mean_squares(responses, bank.items, theta))
      plt.show()

  Parameters
  ----------
  theta : array_like
      Each examinee's ability, such as the ``theta`` of an :py:class:`~irtlib.ability.AbilityEstimates`.
  fit : irtlib.fit.MeanSquares
      The mean squares of the same examinees, in the same order, as :py:func:`irtlib.fit.person_mean_squares` returns
      them.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The examinees' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the examinees are on.

  Raises
  ------
  TypeError
      If ``theta`` holds anything but numbers, before any figure is created.
  ValueError
      If ``theta`` and ``fit`` do not hold one value per examinee each, for the same examinees, before any figure is
      created.
  """
  infit_t = numpy.asarray(fit.infit_t, dtype=float)
  if infit_t.ndim != 1:
    msg = f"fit holds standardized infits of shape {infit_t.shape}; it must hold one per examinee"
    raise ValueError(msg)
  ability = abilities(theta, infit_t.size)
  shown = numpy.isfinite(ability) & numpy.isfinite(infit_t)
  ax = prepare_axes(ax, title)

  _draw_fit_map(ax, ability[shown], infit_t[shown], label)
  ax.set_xlabel(r"Ability $\theta$")
  return ax


def invariance(
  first: Calibration,
  second: Calibration,
  *,
  level: float = 0.95,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot each item's difficulty from one calibration against its difficulty from another, with a confidence band.

  Each calibration's difficulties are centred on 0 over the items both estimate, and drawn against the identity line,
  with the confidence band at ``level`` that :cite:t:`Wright1982` (pp. 114-116, Figure 5.9) build from the standard
  error of each item's difference: an item outside it differs between the calibrations by more than its standard
  errors allow, and is named. The band takes the items' colour. An item without a difficulty and a standard error in
  both calibrations is left out.

  .. plot::
      :filename-prefix: invariance
      :caption: CML calibrations of a 1PL test on the examinees with the lower and the higher half of the scores.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.cml import CMLCalibrator
      from irtlib.drawing import draw_responses
      from irtlib.plot import invariance

      bank = ItemBank.generate(20, "1PL", rng=2)
      responses = draw_responses(numpy.random.default_rng(1).normal(size=2000), bank.items, rng=3)
      scores = responses.sum(axis=1)
      low = scores <= numpy.median(scores)
      ax = invariance(CMLCalibrator().calibrate(responses[low]), CMLCalibrator().calibrate(responses[~low]))
      ax.legend()
      plt.show()

  Parameters
  ----------
  first, second : irtlib.calibration.Calibration
      The two calibrations of the same items, in the same order, each with standard errors, such as calibrations on
      two groups of examinees; ``first`` gives the x axis and ``second`` the y axis.
  level : float, optional
      The confidence level of the band, strictly between 0 and 1. Default is 0.95.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The items' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the items are on.

  Raises
  ------
  ValueError
      If a calibration has no standard errors, the two have different numbers of items, no item has a difficulty and a
      standard error in both, or ``level`` is not strictly between 0 and 1, before any figure is created.
  """
  level = check_level(level)
  shown, x, y, difference_se = _centred_pairs(first, second)
  # each item's band reaches k standard errors of its difference, across the identity line from the item's mean
  half = ndtri((1 + level) / 2) * difference_se / 2
  ax = prepare_axes(ax, title)

  edge, width = reference_style(ax)
  ax.axline((0, 0), slope=1, color=edge, linewidth=width)
  (points,) = ax.plot(x, y, "o", label=label)
  centre = (x + y) / 2
  order = numpy.argsort(centre, kind="stable")
  colour = points.get_color()
  ax.plot((centre - half)[order], (centre + half)[order], "--", color=colour, label=f"{100 * level:g}% confidence band")
  ax.plot((centre + half)[order], (centre - half)[order], "--", color=colour)
  outside = numpy.abs(x - y) > 2 * half
  names = _generated_names(first.estimates.shape[0])
  name_points(ax, x[outside], y[outside], [names[item] for item in shown[outside]])
  ax.set_aspect("equal", adjustable="datalim")
  ax.set_xlabel("Difficulty $b$, first calibration, centred")
  ax.set_ylabel("Difficulty $b$, second calibration, centred")
  return ax


def person_profile(  # noqa: PLR0913 - the items, the responses, the abilities and the examinee, and every plot's ax, title, label
  items: ItemBank | FloatArray,
  responses: npt.ArrayLike,
  theta: npt.ArrayLike,
  examinee: int,
  *,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot one examinee's standardized residual on each item, the items in order of difficulty.

  A residual is the response less its probability at the examinee's ability, over its standard deviation: positive
  for a correct response, negative for an incorrect one. The items run from the easiest to the hardest, as
  :cite:t:`Wright1979` (pp. 166, 170) arrange them to read a response pattern, and a line joins the residuals, as in
  their Figure 7.9.3 (p. 180). Dashed lines mark the band from -2 to 2, outside which :cite:t:`Wright1982` (p. 99)
  call a response surprising, and a solid line 0. An item the examinee did not answer leaves a gap. With at most 30
  items, the ticks name them.

  .. plot::
      :filename-prefix: person_profile
      :caption: An examinee who answered the four easiest items of a 2PL test incorrectly.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from irtlib.ability import weighted_likelihood
      from irtlib.drawing import draw_responses
      from irtlib.plot import person_profile

      bank = ItemBank.generate(20, "2PL", rng=2)
      responses = draw_responses(numpy.array([1.5]), bank.items, rng=3)
      responses[0, numpy.argsort(bank.difficulty)[:4]] = 0
      theta = weighted_likelihood(responses, bank.items).theta
      person_profile(bank, responses, theta, 0)
      plt.show()

  Parameters
  ----------
  items : irtlib.ItemBank or numpy.ndarray
      The items, as a bank, whose names label them, or as an item matrix, whose items are named by position.
  responses : array_like
      The response matrix: one row per examinee, one column per item of ``items``, 1 or 0, and NaN for a missing
      response.
  theta : array_like
      Each examinee's ability, one per row of ``responses``, such as the ``theta`` of an
      :py:class:`~irtlib.ability.AbilityEstimates`; finite for ``examinee``.
  examinee : int
      The examinee's row in ``responses``.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The residuals' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the residuals are on.

  Raises
  ------
  TypeError
      If ``examinee`` is not an integer, or the responses or ``theta`` are not numbers, before any figure is created.
  ValueError
      If ``items`` is not a valid item matrix, ``responses`` is not a response matrix with one column per item,
      ``theta`` does not hold one ability per examinee, ``examinee`` is not a row of ``responses``, or the examinee's
      ability is not finite or the examinee answered no item, before any figure is created.
  """
  bank = as_bank(items)
  matrix, _ = responses_and_items(responses, bank.items)
  theta = abilities(theta, matrix.shape[0])
  examinee = examinee_position("examinee", examinee, matrix.shape[0])
  _check_finite(theta, numpy.arange(matrix.shape[0]) == examinee, "the examinee")
  row = matrix[examinee]
  if numpy.isnan(row).all():
    msg = f"examinee {examinee} answered no item, so has no residual to plot"
    raise ValueError(msg)
  order = numpy.argsort(bank.difficulty, kind="stable")
  residuals = _standardized_residuals(row, theta[examinee], bank.items)[order]
  positions = numpy.arange(bank.n_items)
  ax = prepare_axes(ax, title)

  _draw_band(ax)
  ax.plot(positions, residuals, "o-", label=label)
  label_items(ax, positions, [bank.names[item] for item in order])
  ax.set_xlabel("Item, from the easiest to the hardest")
  ax.set_ylabel("Standardized residual $z$")
  return ax


def _check_finite(theta: FloatArray, needed: BoolArray, who: str) -> None:
  """Raise a ValueError naming the first ability that is not finite among the examinees that need one."""
  unusable = numpy.flatnonzero(needed & ~numpy.isfinite(theta))
  if unusable.size:
    position = int(unusable[0])
    msg = f"theta[{position}] is {theta[position].item()!r}; the ability of {who} must be finite"
    raise ValueError(msg)


def _group_count(groups: object, n_answered: int) -> int:
  """Return the number of groups as an int, after checking that it is an integer from 1 to the examinees answering."""
  # bool is an int to operator.index, but a flag passed by mistake is not a count.
  if isinstance(groups, (bool, numpy.bool_)):
    msg = f"groups is {groups!r}; the number of groups must be an integer, not a bool"
    raise TypeError(msg)
  try:
    count = operator.index(groups)  # ty: ignore[invalid-argument-type]
  except TypeError:
    msg = f"groups is {groups!r}; the number of groups must be an integer"
    raise TypeError(msg) from None
  if not 1 <= count <= n_answered:
    msg = f"groups is {count}; it must be between 1 and the {n_answered} examinees who answered the item"
    raise ValueError(msg)
  return count


def _draw_icc(ax: Axes, row: FloatArray, low: float, high: float, label: str) -> ColorType:
  """Draw an item's curve over its difficulty +- 4, widened to reach ``low`` and ``high``, and return its colour."""
  difficulty = row[0, 1]
  grid = numpy.linspace(min(difficulty - HALF_WIDTH, low), max(difficulty + HALF_WIDTH, high), POINTS)
  (line,) = ax.plot(grid, irt.icc(grid, row)[:, 0], label=label)
  return line.get_color()


def _observed(
  ability: FloatArray, correct: FloatArray, groups: int, level: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return each ability group's median ability, proportion correct, and interval's reach below and above it."""
  cells = numpy.array_split(numpy.argsort(ability, kind="stable"), groups)
  sizes = numpy.array([cell.size for cell in cells])
  counts = numpy.array([correct[cell].sum() for cell in cells])
  proportions = counts / sizes
  lower, upper = _exact_interval(counts, sizes, level)
  medians = numpy.array([numpy.median(ability[cell]) for cell in cells])
  return medians, proportions, numpy.stack([proportions - lower, upper - proportions])


def _exact_interval(counts: IntArray | FloatArray, sizes: IntArray, level: float) -> tuple[FloatArray, FloatArray]:
  """Return the exact binomial (Clopper-Pearson) interval of each proportion ``counts / sizes`` at ``level``."""
  tail = (1 - level) / 2
  # the beta quantiles are those of the tails at the interval's ends; a count of 0 or of the size pins its end at 0 or 1
  lower = numpy.where(counts > 0, betaincinv(numpy.maximum(counts, 1), sizes - counts + 1, tail), 0.0)
  upper = numpy.where(counts < sizes, betaincinv(counts + 1, numpy.maximum(sizes - counts, 1), 1 - tail), 1.0)
  return lower, upper


def _centred_pairs(first: Calibration, second: Calibration) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
  """Return the items both calibrations estimate, their difficulties centred over them, and each difference's SE."""
  first_b, first_se = _difficulties(first, "first")
  second_b, second_se = _difficulties(second, "second")
  if first_b.size != second_b.size:
    msg = f"the first calibration has {first_b.size} items and the second {second_b.size}; they must be the same items"
    raise ValueError(msg)
  both = numpy.isfinite(first_b) & numpy.isfinite(first_se) & numpy.isfinite(second_b) & numpy.isfinite(second_se)
  if not both.any():
    msg = "no item has a difficulty and a standard error in both calibrations"
    raise ValueError(msg)
  shown = numpy.flatnonzero(both)
  x, y = first_b[shown] - first_b[shown].mean(), second_b[shown] - second_b[shown].mean()
  return shown, x, y, numpy.hypot(first_se[shown], second_se[shown])


def _difficulties(calibration: Calibration, which: str) -> tuple[FloatArray, FloatArray]:
  """Return a calibration's difficulties and their standard errors, after checking that it has standard errors."""
  if calibration.standard_errors is None:
    msg = f"the {which} calibration has no standard errors, which the confidence band is built from"
    raise ValueError(msg)
  return calibration.estimates[:, 1], calibration.standard_errors[:, 1]


def _standardized_residuals(row: FloatArray, theta: float, items: FloatArray) -> FloatArray:
  """Return each response's residual over its standard deviation at ``theta``, NaN where the response is missing."""
  log_p, log_q = irt.log_response_probabilities(theta, items)
  # (u - P) / sqrt(P Q) is sqrt(Q / P) for a correct response and -sqrt(P / Q) for an incorrect one, which the
  # log-probabilities give without rounding P or Q to 0 far from the difficulty
  sign = 2 * row - 1
  return sign * numpy.exp(sign * (log_q - log_p) / 2)


def _draw_fit_map(ax: Axes, x: FloatArray, fit_t: FloatArray, label: str | None) -> None:
  """Draw the band of a standardized fit statistic and a dot at each ``(x, fit_t)``, and name the y axis."""
  _draw_band(ax)
  ax.plot(x, fit_t, "o", label=label)
  ax.set_ylabel("Standardized infit $t$")


def _draw_band(ax: Axes) -> None:
  """Draw a solid line at 0 and dashed lines at the band's bounds, in the axes' own colour."""
  colour, width = reference_style(ax)
  ax.axhline(0, color=colour, linewidth=width)
  for bound in (-_BAND, _BAND):
    ax.axhline(bound, linestyle="--", color=colour, linewidth=width)
