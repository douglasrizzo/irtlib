"""Joint maximum likelihood calibration of the Rasch model and the 2PL."""

import math
import warnings
from dataclasses import dataclass
from typing import Self

import numpy

from ._m_step import _covariances, _free_information, _m_step, _objective, _to_items
from ._responses import _N_COLUMNS
from .ability import AbilityEstimates, maximum_likelihood
from .bank import IrtModel
from .calibration import (
  Calibration,
  ConvergenceWarning,
  Likelihood,
  _edited,
  _reject_separation,
  _valid_items,
  validate_responses,
)
from .irt import total_information
from .typing import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class _Groups:
  """The examinees kept, grouped so that the examinees of a group share an ability estimate."""

  patterns: FloatArray
  """The responses of the first examinee of each group, groups by items."""
  correct: FloatArray
  """The number of correct responses to each item in each group, groups by items."""
  total: FloatArray
  """The number of responses to each item in each group, its size, since every examinee answers every item."""
  sizes: FloatArray
  """The number of examinees in each group."""
  members: IntArray
  """The group of each examinee kept, in the order of their rows."""

  @classmethod
  def of(cls, kept: FloatArray, *, rasch: bool) -> Self:
    """Return the examinees grouped by their number correct under the Rasch model, and by their responses otherwise."""
    keys = kept.sum(axis=1, keepdims=True) if rasch else kept
    _, first, inverse, counts = numpy.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    members = inverse.ravel()
    correct = numpy.zeros((first.size, kept.shape[1]))
    numpy.add.at(correct, members, kept)
    sizes = counts.astype(float)
    return cls(kept[first], correct, numpy.outer(sizes, numpy.ones(kept.shape[1])), sizes, members)

  def separated(self, theta: FloatArray) -> IntArray:
    """Return the items whose correct responses all come at abilities at or above, or at or below, the incorrect ones.

    Parameters
    ----------
    theta : numpy.ndarray
        The ability of each group.

    Returns
    -------
    numpy.ndarray
        The indices of the items, among those kept, that the abilities separate.
    """
    abilities = theta[:, numpy.newaxis]
    right, wrong = self.correct > 0, self.correct < self.total
    lowest_right = numpy.where(right, abilities, numpy.inf).min(axis=0)
    highest_right = numpy.where(right, abilities, -numpy.inf).max(axis=0)
    lowest_wrong = numpy.where(wrong, abilities, numpy.inf).min(axis=0)
    highest_wrong = numpy.where(wrong, abilities, -numpy.inf).max(axis=0)
    return numpy.flatnonzero((lowest_right >= highest_wrong) | (highest_right <= lowest_wrong))


@dataclass(frozen=True, slots=True)
class _Fit:
  """The state of a calibration: the item parameters and the abilities of the groups, and how its cycles went."""

  index: IntArray
  """The index of each item's slope, intercept and logits of c and d among the free parameters, -1 for a fixed one."""
  values: FloatArray
  """Each item's slope, intercept and logits of c and d."""
  theta: FloatArray
  """Each group's ability."""
  evaluations: IntArray
  """The evaluations of each group's likelihood in the ability step that gave its ability, 0 for a starting one."""
  log_likelihoods: tuple[float, ...] = ()
  """The joint log-likelihood at the end of each complete cycle."""
  change: float = math.inf
  failure: str | None = None

  @property
  def cycles(self) -> int:
    """The complete cycles run."""
    return len(self.log_likelihoods)


class JMLCalibrator:
  """Calibrate the Rasch model or the 2PL by joint maximum likelihood.

  Each call to :py:meth:`calibrate` estimates the items of one complete response matrix together with the abilities
  of its examinees, with no assumption about their distribution. The estimates are not consistent for a test of fixed
  length; they serve comparisons of calibration methods and preliminary estimates. On short tests the 2PL's joint
  maximum often does not exist, its slopes growing without bound; calibration then stops early with a
  :py:class:`~irtlib.calibration.ConvergenceWarning`.

  The :doc:`/methods/calibration/jml` page gives the method, its sources and the origin of each default.

  Parameters
  ----------
  model : irtlib.IrtModel or str, optional
      ``"2PL"``, the default, estimates a slope and a difficulty per item, with the abilities given mean 0 and standard
      deviation 1; ``"1PL"`` the Rasch model, every slope 1, with the difficulties given mean 0. The 1PL of
      :py:class:`~irtlib.mml.MMLCalibrator` estimates one slope with abilities :math:`N(0, 1)` instead, so the two
      compare only after a linear change of the ability scale (ADR-0008).
  correct : bool or None, optional
      Whether to multiply the difficulties of the Rasch model by ``(M - 1) / M``, for ``M`` items kept, once the
      likelihood is maximized. Default is None, which corrects the Rasch model and not the 2PL.
  correct_errors : bool, optional
      Whether to multiply the difficulty's row and column of each item's covariances by ``(M - 1) / M`` too, and so
      its standard error by that factor, when the difficulties are corrected. Default is False, which keeps the
      covariances of the estimates before the correction.
  tolerance : float, optional
      The stopping rule: calibration stops when no item parameter changes by more than this in a cycle, an item step
      and an ability step. Default is 1e-6.
  max_cycles : int, optional
      The most cycles a calibration runs before it stops unconverged. Default is 500.

  Raises
  ------
  ValueError
      If ``model`` is neither the 1PL nor the 2PL, ``correct`` is True for the 2PL, ``correct_errors`` is True where
      the difficulties are not corrected, ``tolerance`` is not positive, or ``max_cycles`` is less than 1.

  Examples
  --------
  >>> import numpy
  >>> responses = numpy.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
  >>> calibration = JMLCalibrator("1PL", correct=False).calibrate(responses)
  >>> calibration.estimates[:, 1].round(4)  # b_2 - b_1 = 2 log(2 / 1)
  array([-0.6931,  0.6931])
  """

  def __init__(
    self,
    model: IrtModel | str = IrtModel.PL2,
    *,
    correct: bool | None = None,
    correct_errors: bool = False,
    tolerance: float = 1e-6,
    max_cycles: int = 500,
  ) -> None:
    self._model = IrtModel(model)
    match self._model:
      case IrtModel.PL1 | IrtModel.PL2:
        pass
      case _:
        msg = f"JMLCalibrator calibrates the 1PL and the 2PL, got {self._model}"
        raise ValueError(msg)
    if correct and self._model is not IrtModel.PL1:
      msg = f"the correction (M - 1) / M is derived for the Rasch model, the 1PL, not the {self._model}"
      raise ValueError(msg)
    if not tolerance > 0:
      msg = f"tolerance must be positive, got {tolerance}"
      raise ValueError(msg)
    if max_cycles < 1:
      msg = f"max_cycles must be at least 1, got {max_cycles}"
      raise ValueError(msg)
    self._correct = self._model is IrtModel.PL1 if correct is None else correct
    if correct_errors and not self._correct:
      msg = (
        "correct_errors multiplies the difficulties' covariances by the correction (M - 1) / M, which this calibrator "
        f"does not apply: the {self._model} with correct={correct}"
      )
      raise ValueError(msg)
    self._correct_errors = correct_errors
    self._tolerance = tolerance
    self._max_cycles = max_cycles

  def calibrate(self, responses: FloatArray) -> Calibration:
    """Estimate the items of a complete response matrix, with the abilities of its examinees.

    Examinees with no item or every item correct have no finite ability estimate and are left out, as are items
    answered correctly by every examinee or by none, repeatedly until neither is left.

    Parameters
    ----------
    responses : numpy.ndarray
        The response matrix: one row per examinee, one column per item, holding 1 or 0.

    Returns
    -------
    Calibration
        The estimates ``[a, b, 0, 1]``, their covariances, and the joint log-likelihood at the estimates, before
        any correction, of the examinees and items kept; the result has no grid. An item left out has NaN estimates
        and covariances, and the result's bank leaves it out, with a ``RuntimeWarning`` naming it. The result's
        ``abilities`` are every examinee's, those of the cycle the estimates come from, on the same scale, with standard
        errors one over the square root of the test information at the estimates before any correction, and the
        evaluations of the ability step that gave them, 0 for starting values. An examinee left out has the ability
        ``inf`` or ``-inf``, for every item kept right or wrong, standard error ``inf`` and 0 evaluations. Abilities at
        the corrected difficulties are :py:func:`irtlib.ability.maximum_likelihood`'s, called as
        ``maximum_likelihood(responses[:, kept], calibration.estimates[kept])`` for the items with estimates,
        ``kept = ~numpy.isnan(calibration.estimates[:, 1])``. If the stopping rule is not met within ``max_cycles``,
        the 2PL's abilities separate an item's correct responses from its incorrect ones, or an item step gives a slope
        that is not positive, the result holds the last complete cycle, says so, and a
        :py:class:`~irtlib.calibration.ConvergenceWarning` names the items by their columns; if an item's information
        cannot be inverted, or gives a variance that is not positive, the result has no covariances or standard errors
        and a ``RuntimeWarning`` is issued.

    Raises
    ------
    TypeError
        If the responses are neither Booleans, integers nor floats.
    ValueError
        If the responses fail :py:func:`irtlib.calibration.validate_responses` or have a missing response; no item is
        left once examinees and items are left out; under the Rasch model, the joint likelihood has no maximum, which
        the error explains by the two groups of items the responses split into; under the 2PL, every examinee kept has
        the same number correct, which leaves the starting abilities no spread; or an item's information is singular
        during the cycles.
    """
    matrix = validate_responses(responses)
    if numpy.isnan(matrix).any():
      msg = (
        "joint maximum likelihood needs complete responses; irtlib.mml.MMLCalibrator calibrates responses with "
        "missing entries"
      )
      raise ValueError(msg)
    kept, examinees, items = _edited(matrix, Likelihood.JOINT)
    rasch = self._model is IrtModel.PL1
    if rasch:
      _reject_separation(kept, items, Likelihood.JOINT)
    groups = _Groups.of(kept, rasch=rasch)
    fit = self._cycle(groups, _start(groups, rasch=rasch), columns=numpy.flatnonzero(items), rasch=rasch)
    if fit.failure is not None:
      warnings.warn(f"calibration stopped after {fit.cycles} cycles: {fit.failure}", ConvergenceWarning, stacklevel=2)
    kept_covariances = _covariances(
      fit.values, fit.index, _free_information(fit.values, fit.index, fit.theta, groups.total, ())
    )
    if kept_covariances is None:
      warnings.warn(
        "the information matrix could not be inverted, or gave a variance that is not positive, so the calibration "
        "reports no covariances or standard errors; an item may have no finite estimate",
        RuntimeWarning,
        stacklevel=2,
      )
    estimates = _to_items(fit.values)
    abilities = _abilities(fit, groups, estimates, matrix[:, items], examinees)
    if self._correct:
      # the difficulties are centred at 0 by the scale's identification
      factor = (kept.shape[1] - 1) / kept.shape[1]
      estimates[:, 1] *= factor
      if self._correct_errors and kept_covariances is not None:
        kept_covariances[:, 1, :] *= factor
        kept_covariances[:, :, 1] *= factor
    estimates = _every_item(estimates, items, [1.0 if rasch else numpy.nan, numpy.nan, 0.0, 1.0])
    return Calibration(
      estimates=estimates,
      items=_valid_items(estimates),
      covariances=None if kept_covariances is None else _every_item(kept_covariances, items, numpy.nan),
      log_likelihood=_objective(fit.values, fit.index, fit.theta, groups.correct, groups.total, priors=()),
      log_likelihoods=numpy.array(fit.log_likelihoods),
      likelihood=Likelihood.JOINT,
      # the free item parameters and the abilities of the examinees kept, less the constraints of the scale: one for
      # the Rasch model's mean difficulty, two for the 2PL's mean and standard deviation of the abilities
      n_parameters=int(fit.index.max()) + 1 + kept.shape[0] - (1 if rasch else 2),
      item_parameters=numpy.where(items, 1 if rasch else 2, 0),
      converged=fit.failure is None,
      change=fit.change,
      grid=None,
      abilities=abilities,
    )

  def _cycle(self, groups: _Groups, fit: _Fit, *, columns: IntArray, rasch: bool) -> _Fit:
    """Run cycles of an item and an ability step until the rule is met, the cycles run out, or a slope has no estimate.

    Parameters
    ----------
    groups : _Groups
        The examinees kept, grouped.
    fit : _Fit
        The starting item parameters and abilities.
    columns : numpy.ndarray
        The column of the responses of each item kept, for the messages.
    rasch : bool
        Whether the model is the Rasch model.

    Returns
    -------
    _Fit
        The item parameters and the abilities of the last complete cycle, on the identified scale; the joint
        log-likelihood after each complete cycle; the largest change of an item parameter in the last of them; and
        None if the rule was met, or else why the cycles stopped short of it.
    """
    free = fit.index >= 0
    values, theta, evaluations, change = fit.values, fit.theta, fit.evaluations, math.inf
    log_likelihoods: list[float] = []
    for _ in range(self._max_cycles):
      separated = numpy.array([], dtype=int) if rasch else groups.separated(theta)
      if separated.size:
        failure = (
          f"the abilities separate the correct and incorrect responses to the items in columns "
          f"{columns[separated].tolist()}, so their slopes have no finite estimate"
        )
        return _Fit(fit.index, values, theta, evaluations, tuple(log_likelihoods), change, failure)
      stepped = _m_step(values, fit.index, theta, groups.correct, groups.total, priors=())
      negative = numpy.flatnonzero(stepped[:, 0] <= 0)
      if negative.size:
        largest = int(numpy.argmax(stepped[:, 0]))
        failure = (
          f"the item step of the next cycle gave the items in columns {columns[negative].tolist()} a slope that is not "
          f"positive, and the item in column {columns[largest]} the largest slope, {stepped[largest, 0]:.3g}"
        )
        return _Fit(fit.index, values, theta, evaluations, tuple(log_likelihoods), change, failure)
      abilities = maximum_likelihood(groups.patterns, _to_items(stepped))
      stepped, theta = _identified(stepped, abilities.theta, groups.sizes, rasch=rasch)
      evaluations = abilities.evaluations
      change = float(numpy.max(numpy.abs(stepped[free] - values[free])))
      values = stepped
      log_likelihoods.append(_objective(values, fit.index, theta, groups.correct, groups.total, priors=()))
      if change <= self._tolerance:
        return _Fit(fit.index, values, theta, evaluations, tuple(log_likelihoods), change)
    failure = f"the last cycle changed an item parameter by {change:.3g}, above the tolerance {self._tolerance:g}"
    return _Fit(fit.index, values, theta, evaluations, tuple(log_likelihoods), change, failure)


def _start(groups: _Groups, *, rasch: bool) -> _Fit:
  """Return the starting item parameters and abilities, on the identified scale.

  Returns
  -------
  _Fit
      The index of each item parameter among the free ones, and the starting item parameters and abilities.

  Raises
  ------
  ValueError
      If the 2PL's starting abilities have no spread, every examinee kept having the same number correct.
  """
  n_items = groups.patterns.shape[1]
  scores = groups.patterns.sum(axis=1)
  theta = numpy.log(scores / (n_items - scores))
  totals = groups.correct.sum(axis=0)
  difficulty = numpy.log((groups.sizes.sum() - totals) / totals)
  values = numpy.tile([1.0, 0.0, -numpy.inf, numpy.inf], (n_items, 1))
  values[:, 1] = -(difficulty - difficulty.mean())
  index = numpy.full((n_items, _N_COLUMNS), -1)
  evaluations = numpy.zeros(theta.size, dtype=int)
  if rasch:
    index[:, 1] = numpy.arange(n_items)
    return _Fit(index, values, theta, evaluations)
  index[:, 0] = numpy.arange(n_items)
  index[:, 1] = numpy.arange(n_items, 2 * n_items)
  if numpy.all(scores == scores[0]):
    msg = (
      f"every examinee kept has {int(scores[0])} correct, so the 2PL's abilities, standardized to fix its scale, start "
      "with no spread"
    )
    raise ValueError(msg)
  values, theta = _identified(values, theta, groups.sizes, rasch=False)
  return _Fit(index, values, theta, evaluations)


def _identified(
  values: FloatArray, theta: FloatArray, sizes: FloatArray, *, rasch: bool
) -> tuple[FloatArray, FloatArray]:
  """Return the item parameters and abilities moved along the ability scale to its identified position.

  For the Rasch model, the difficulties have mean 0; for the 2PL, the abilities of the examinees, each group counted
  by its size, have mean 0 and standard deviation 1. The joint likelihood does not change.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray]
      The item parameters and the abilities on the identified scale.
  """
  if rasch:
    # the slopes are 1, so each difficulty is the negated intercept
    location, scale = float(numpy.mean(-values[:, 1])), 1.0
  else:
    location = float(numpy.average(theta, weights=sizes))
    scale = math.sqrt(float(numpy.average((theta - location) ** 2, weights=sizes)))
  # theta = scale * theta' + location turns slope * theta + intercept into (slope * scale) theta' + (intercept + slope *
  # location)
  moved = values.copy()
  moved[:, 1] += values[:, 0] * location
  moved[:, 0] *= scale
  return moved, (theta - location) / scale


def _abilities(
  fit: _Fit, groups: _Groups, estimates: FloatArray, responses: FloatArray, examinees: BoolArray
) -> AbilityEstimates:
  """Return every examinee's ability: its group's if it is kept, and otherwise the infinite one of its responses.

  Parameters
  ----------
  fit : _Fit
      The calibration's state, whose abilities go with ``estimates``.
  groups : _Groups
      The examinees kept, grouped.
  estimates : numpy.ndarray
      The estimates of the items kept, before any correction.
  responses : numpy.ndarray
      Every examinee's responses to the items kept.
  examinees : numpy.ndarray
      The mask of the examinees kept among the rows.

  Returns
  -------
  irtlib.ability.AbilityEstimates
      Each examinee's ability, its standard error at ``estimates``, and the evaluations that gave it.
  """
  # an examinee left out answered every item kept right, or every one wrong
  theta = numpy.where(responses[:, 0] == 1, numpy.inf, -numpy.inf)
  standard_errors = numpy.full(theta.size, numpy.inf)
  evaluations = numpy.zeros(theta.size, dtype=int)
  theta[examinees] = fit.theta[groups.members]
  standard_errors[examinees] = (total_information(fit.theta, estimates) ** -0.5)[groups.members]
  evaluations[examinees] = fit.evaluations[groups.members]
  return AbilityEstimates(theta, standard_errors, evaluations, numpy.zeros(theta.size, dtype=bool))


def _every_item(kept: FloatArray, items: BoolArray, fill: float | list[float]) -> FloatArray:
  """Return an entry for every column of the responses: that of each item kept, and ``fill`` for the others."""
  rows = numpy.full((items.size, *kept.shape[1:]), fill)
  rows[items] = kept
  return rows
