r"""Joint maximum likelihood calibration of the Rasch model and the 2PL.

Joint maximum likelihood (JML) treats every examinee's ability as a parameter, and maximizes the likelihood of the
responses over the abilities and the item parameters together, with no assumption about the distribution of ability
(:cite:t:`Baker1987`, p. 112). With :math:`N` examinees and :math:`M` items, its logarithm is

.. math::

   \sum_{j=1}^N \sum_{i=1}^M \left[u_{ij} \log P_i(\theta_j) + (1 - u_{ij}) \log Q_i(\theta_j)\right]

(:cite:t:`Baker1987`, p. 112, Equations 1 and 3), for the 2PL (p. 114, Equation 5) or for the Rasch model, whose slopes
are 1 (p. 115, Equation 9; :cite:t:`Wright1977`, p. 282, Equations 1-3). As each examinee adds a parameter, the item
estimates do not converge to the item parameters as examinees are added to a test of fixed length
(:cite:t:`Bock1981`, p. 443; :cite:t:`Baker1987`, pp. 117-118): for the Rasch model they converge to limits that differ
from the parameters by a term of order :math:`1/M` (:cite:t:`Haberman2004`, p. 13), and Haberman (p. 35) finds it
"difficult to advocate use of JMLE except for preliminary estimation of parameters".

**Examinees and items left out.** An examinee with no item or every item correct has no finite ability estimate, and
an item answered correctly by every examinee or by none no finite estimate either (:cite:t:`Wright1977`, p. 294;
:cite:t:`Haberman2004`, p. 7). Both are left out before estimation, the examinees first and then the items, repeatedly
until neither is left, as Wright and Douglas's editing does (p. 294), and the items left out get no estimate. For the
Rasch model, the joint likelihood of the responses left can still lack a maximum: it has none exactly when the
examinees with at most some number correct answered none of a group of items and the others all of the rest
(:cite:t:`Haberman2004`, p. 11, Theorem 1), the same split that leaves the conditional likelihood of
:py:mod:`irtlib.cml` none once those examinees and items are left out, and calibration raises an error naming the
groups.

**The cycles.** Each cycle has two steps, after Birnbaum's paradigm (:cite:t:`Baker1987`, p. 112):

- the item step maximizes, for each item, its grouped log-likelihood with the abilities fixed, by the Fisher scoring of
  the M-step of :py:mod:`irtlib.mml` with the estimated abilities in place of the quadrature nodes and the observed
  counts in place of the expected ones. For the Rasch model and the 2PL the Hessian is the information, so the steps
  are Newton-Raphson's (:cite:t:`Baker1987`, p. 116), those of Wright and Douglas (p. 284, Equation 11) for the
  difficulties and of Baker (p. 114, Equation 8) for the 2PL's intercept and slope;
- the ability step estimates each examinee's ability with the items fixed, by maximum likelihood
  (:py:func:`irtlib.ability.maximum_likelihood`). Under the Rasch model every examinee with the same number correct has
  the same estimate (:cite:t:`Wright1977`, p. 283; :cite:t:`Baker1987`, p. 115), and under either model every examinee
  with the same responses, so each is estimated once.

The likelihood is the same for abilities and items moved together along a linear transformation of the ability scale
(:cite:t:`Baker1987`, p. 112), so after each ability step both are moved to fix the scale (ADR-0008): for the Rasch
model so that the difficulties have mean 0, as BICAL fixes it (p. 115) and as Wright and Douglas recentre them (p. 283;
p. 284, step iv); for the 2PL so that the abilities of the examinees kept have mean 0 and standard deviation 1, its
divisor the number of examinees, as LOGIST fixes it (p. 112). LOGIST standardizes only the abilities between -3 and 3
(:cite:t:`Mislevy1989`, p. 60), where irtlib standardizes every examinee kept.

The cycles start from Wright and Douglas's (p. 283, Equation 9; p. 284, Equation 10) starting values, the abilities
:math:`\log(r / (M - r))` for :math:`r` correct and the difficulties :math:`\log((N - s_i) / s_i)` centred, for
:math:`s_i` correct responses to item :math:`i`; for the 2PL, which they do not cover, the slopes start at 1 and the
abilities are standardized. A calibration stops when no item parameter changes by more than its tolerance in a cycle,
:math:`10^{-6}` by default, in at most 500 cycles, the rule and the defaults of :py:mod:`irtlib.mml` (ADR-0007). Wright
and Douglas stop when the mean squared change of the difficulties falls below :math:`10^{-4}` (p. 285, Equation 15),
which stopped 0.0007 and 0.0009 short of the maximum on 20 items and on LSAT-6, where irtlib's rule stops within
:math:`2 \times 10^{-7}` of it (measured 2026-11-27). The ability step compares likelihoods, which leaves changes of
about :math:`10^{-8}` from cycle to cycle, so a tolerance below that is not met.

**Bias and its correction for the Rasch model.** For the Rasch model, Wright and Douglas write each difficulty
estimate as the conditional one plus a bias, :math:`d_i^* = d_i + a_{ri}` at score :math:`r`, approximating the bias by
the change in the conditional ability for score :math:`r` when item :math:`i` is left out, and its relative bias as
:math:`c_{ri} = a_{ri} / d_i` (:cite:t:`Wright1977`, p. 286, Equations 25-27). Averaged over scores and items with
:math:`|d_i| \geq 0.5`, they find it close to :math:`1 / (M - 1)` whatever the spread of the difficulties (p. 287,
Equations 31-34, Table 1), so the estimates spread outward from the conditional ones, and multiplying the centred
difficulties by the correction factor :math:`(M - 1) / M` (p. 287, Equation 35) leaves a residual bias of at most 0.012
for 20 items (p. 288, Table 2). The factor is the one "used in most versions of UCON", and with two items, for which
the joint estimates are exactly twice the conditional ones, it is exact (p. 288; :cite:t:`Haberman2004`, pp. 17-18). It
assumes item difficulties and examinees spread "more or less uniform[ly]" over the scores (:cite:t:`Wright1977`,
p. 288), and it brings the estimates close to the conditional ones, which "is not the same as" removing their bias from
the parameters (:cite:t:`Baker1987`, p. 118). Computed exactly for 20 items equally spaced, with as many examinees at
each score, the relative bias is 0.055 for difficulties within :math:`\pm 1`, as Wright and Douglas find, but 0.060
within :math:`\pm 2` and 0.071 within :math:`\pm 3`, and the largest residual after the correction 0.004, 0.029 and
0.090 (measured 2026-11-27): the correction leaves more of the bias of the items far from the centre of a wide test.
It is applied once the cycles stop, by default for the Rasch model; Wright and Douglas derive it for the Rasch model
alone.

**Standard errors.** Each item's standard errors are the square roots of the diagonal of the inverse of its item step's
information at the last cycle (:cite:t:`Baker1987`, p. 119), for the Rasch model
:math:`(\sum_r n_r p_{ri} (1 - p_{ri}))^{-1/2}`, with :math:`n_r` examinees of score :math:`r` and :math:`p_{ri}` their
probability of a correct response to item :math:`i` (:cite:t:`Wright1977`, p. 285, Equation 16); the 2PL's difficulty
takes its standard error by the delta method, as in :py:mod:`irtlib.mml`. They treat the abilities as known, and with
the correction they are those of the estimates before it. For a fixed number of items, JML's estimates are not centred
on the parameters and their asymptotic covariance is not the customary one, so "ordinary asymptotic confidence
intervals" for the difficulties cannot be built from them (:cite:t:`Haberman2004`, pp. 26-27, Theorem 7; p. 29).

Under the 2PL a slope can grow without bound, as large slopes and the abilities of the examinees who answer those items
correctly feed each other from cycle to cycle, the Heywood case; LOGIST imposes upper limits on the slopes
(:cite:t:`Baker1987`, p. 113). irtlib imposes none. An item whose correct responses the abilities all put at or above
its incorrect ones, or at or below, discriminates perfectly, and its slope has no finite estimate (p. 117), so the
cycles stop before such an item step; they stop too after an item step that gives a slope that is not positive, as a
miskeyed item can, since the ability step needs positive slopes. Either way the result says so.

The ability estimates are not part of the result (ADR-0006): with the correction off, maximum likelihood abilities at
the estimated items are the joint ones, within the tolerance. JML needs complete responses, as its sources write it, an
examinee's number correct out of every item (:cite:t:`Wright1977`, p. 282), and :py:class:`irtlib.mml.MMLCalibrator`
calibrates incomplete ones (ADR-0006).

On two SAT tests of 78 and 60 items, the largest difference between the joint and conditional estimates of the Rasch
difficulties was 0.097 and 0.126, against conditional standard errors of at most 0.008
(:cite:t:`Haberman2004`, pp. 44-45). In :cite:t:`Robitzsch2021`'s simulations, JML with the correction and extreme
scores removed (p. 816) had a mean absolute bias below 0.025 (p. 821) in 54.2% of the conditions, against 100% for
conditional maximum likelihood (p. 823, Table 2).
"""

import math
import warnings
from dataclasses import dataclass
from typing import Self

import numpy

from ._m_step import _free_information, _m_step, _objective, _standard_errors, _to_items
from ._responses import _N_COLUMNS
from .ability import maximum_likelihood
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
from .typing import BoolArray, FloatArray, IntArray

# The ability step's search tolerance, well inside the stopping rule's, so that the abilities' rounding does not move
# the items by as much as the rule allows.
_ABILITY_TOLERANCE = 1e-9


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

  @classmethod
  def of(cls, kept: FloatArray, *, rasch: bool) -> Self:
    """Return the examinees grouped by their number correct under the Rasch model, and by their responses otherwise."""
    keys = kept.sum(axis=1, keepdims=True) if rasch else kept
    _, first, inverse, counts = numpy.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    correct = numpy.zeros((first.size, kept.shape[1]))
    numpy.add.at(correct, inverse.ravel(), kept)
    sizes = counts.astype(float)
    return cls(kept[first], correct, numpy.outer(sizes, numpy.ones(kept.shape[1])), sizes)

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
  cycles: int = 0
  change: float = math.inf
  failure: str | None = None


class JMLCalibrator:
  """Calibrate the Rasch model or the 2PL by joint maximum likelihood.

  Each call to :py:meth:`calibrate` estimates the items of one complete response matrix together with the abilities
  of its examinees, with no assumption about their distribution. The estimates are not consistent for a test of fixed
  length; they serve comparisons of calibration methods and preliminary estimates.

  Parameters
  ----------
  model : irtlib.IrtModel or str, optional
      ``"2PL"``, the default, estimates a slope and a difficulty per item, with the abilities given mean 0 and standard
      deviation 1; ``"1PL"`` the Rasch model, every slope 1, with the difficulties given mean 0.
  correct : bool or None, optional
      Whether to multiply the difficulties of the Rasch model by ``(M - 1) / M``, for ``M`` items kept, once the
      likelihood is maximized. Default is None, which corrects the Rasch model and not the 2PL.
  tolerance : float, optional
      The stopping rule: calibration stops when no item parameter changes by more than this in a cycle, an item step
      and an ability step. Default is 1e-6.
  max_cycles : int, optional
      The most cycles a calibration runs before it stops unconverged. Default is 500.

  Raises
  ------
  ValueError
      If ``model`` is neither the 1PL nor the 2PL, ``correct`` is True for the 2PL, ``tolerance`` is not positive, or
      ``max_cycles`` is less than 1.

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
        The estimates ``[a, b, 0, 1]``, their standard errors, and the joint log-likelihood at the estimates, before
        any correction, of the examinees and items kept; the result has no grid. An item left out has NaN estimates
        and standard errors, and the result's bank leaves it out, with a ``RuntimeWarning`` naming it. If the stopping
        rule is not met within ``max_cycles``, the 2PL's abilities separate an item's correct responses from its
        incorrect ones, or an item step gives a slope that is not positive, the result says so and a
        :py:class:`~irtlib.calibration.ConvergenceWarning` is issued; if an item's information cannot be inverted, the
        result has no standard errors and a ``RuntimeWarning`` is issued.

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
    kept, items = _edited(matrix, Likelihood.JOINT)
    rasch = self._model is IrtModel.PL1
    if rasch:
      _reject_separation(kept, items, Likelihood.JOINT)
    groups = _Groups.of(kept, rasch=rasch)
    fit = self._cycle(groups, _start(groups, rasch=rasch), rasch=rasch)
    if fit.failure is not None:
      warnings.warn(f"calibration stopped after {fit.cycles} cycles: {fit.failure}", ConvergenceWarning, stacklevel=2)
    kept_errors = _standard_errors(
      fit.values, fit.index, _free_information(fit.values, fit.index, fit.theta, groups.total, ())
    )
    if kept_errors is None:
      warnings.warn(
        "the information matrix could not be inverted, so the calibration reports no standard errors; an item may "
        "have no finite estimate",
        RuntimeWarning,
        stacklevel=2,
      )
    estimates = _to_items(fit.values)
    if self._correct:
      # the difficulties are centred at 0 by the scale's identification
      estimates[:, 1] *= (kept.shape[1] - 1) / kept.shape[1]
    estimates = _every_item(estimates, items, [1.0 if rasch else numpy.nan, numpy.nan, 0.0, 1.0])
    return Calibration(
      estimates=estimates,
      items=_valid_items(estimates),
      standard_errors=None if kept_errors is None else _every_item(kept_errors, items, [numpy.nan] * _N_COLUMNS),
      log_likelihood=_objective(fit.values, fit.index, fit.theta, groups.correct, groups.total, priors=()),
      likelihood=Likelihood.JOINT,
      # the free item parameters and the abilities of the examinees kept, less the constraints of the scale: one for
      # the Rasch model's mean difficulty, two for the 2PL's mean and standard deviation of the abilities
      n_parameters=int(fit.index.max()) + 1 + kept.shape[0] - (1 if rasch else 2),
      item_parameters=numpy.where(items, 1 if rasch else 2, 0),
      cycles=fit.cycles,
      converged=fit.failure is None,
      change=fit.change,
      grid=None,
    )

  def _cycle(self, groups: _Groups, fit: _Fit, *, rasch: bool) -> _Fit:
    """Run cycles of an item and an ability step until the rule is met, the cycles run out, or a slope has no estimate.

    Returns
    -------
    _Fit
        The item parameters and the abilities on the identified scale, the cycles run, the largest change of an item
        parameter in the last cycle, and None if the rule was met, or else why the cycles stopped short of it.
    """
    free = fit.index >= 0
    values, theta, change = fit.values, fit.theta, math.inf
    for cycle in range(1, self._max_cycles + 1):
      separated = numpy.array([], dtype=int) if rasch else groups.separated(theta)
      if separated.size:
        failure = (
          f"the abilities separate the correct and incorrect responses to the items in rows {separated.tolist()} of "
          "the items kept, so their slopes have no finite estimate"
        )
        return _Fit(fit.index, values, theta, cycle - 1, change, failure)
      stepped = _m_step(values, fit.index, theta, groups.correct, groups.total, priors=())
      change = float(numpy.max(numpy.abs(stepped[free] - values[free])))
      negative = numpy.flatnonzero(stepped[:, 0] <= 0)
      if negative.size:
        failure = (
          f"the item step gave the items in rows {negative.tolist()} of the items kept a slope that is not positive"
        )
        return _Fit(fit.index, stepped, theta, cycle, change, failure)
      abilities = maximum_likelihood(groups.patterns, _to_items(stepped), tol=_ABILITY_TOLERANCE).theta
      stepped, theta = _identified(stepped, abilities, groups.sizes, rasch=rasch)
      change = float(numpy.max(numpy.abs(stepped[free] - values[free])))
      values = stepped
      if change <= self._tolerance:
        return _Fit(fit.index, values, theta, cycle, change)
    failure = f"the last cycle changed an item parameter by {change:.3g}, above the tolerance {self._tolerance:g}"
    return _Fit(fit.index, values, theta, self._max_cycles, change, failure)


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
  if rasch:
    index[:, 1] = numpy.arange(n_items)
    return _Fit(index, values, theta)
  index[:, 0] = numpy.arange(n_items)
  index[:, 1] = numpy.arange(n_items, 2 * n_items)
  if numpy.all(scores == scores[0]):
    msg = (
      f"every examinee kept has {int(scores[0])} correct, so the 2PL's abilities, standardized to fix its scale, start "
      "with no spread"
    )
    raise ValueError(msg)
  values, theta = _identified(values, theta, groups.sizes, rasch=False)
  return _Fit(index, values, theta)


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


def _every_item(kept: FloatArray, items: BoolArray, fill: list[float]) -> FloatArray:
  """Return rows for every column of the responses: those of the items kept, and ``fill`` for the others."""
  rows = numpy.tile(fill, (items.size, 1))
  rows[items] = kept
  return rows
