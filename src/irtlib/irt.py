r"""The item response functions of the 1PL to 4PL models, and the quantities derived from them.

The functions take one item as a row ``[a, b, c, d]`` of shape ``(4,)``, or several as a matrix of such rows, shape
``(n, 4)``: the discrimination :math:`a`, the difficulty :math:`b`, the guessing parameter :math:`c` and the upper
asymptote :math:`d`, with :math:`a > 0` and :math:`0 \leq c < d \leq 1`. A response matrix has one row per examinee and
one column per item, and holds 1 for a correct response, 0 for an incorrect one and NaN where the examinee did not
answer the item.

Under the four-parameter logistic model, an examinee of ability :math:`\theta` answers item :math:`i` correctly with
probability

.. math::

   P_i(\theta) = c_i + (d_i - c_i) F_i(\theta), \quad F_i(\theta) = \frac{1}{1 + e^{-a_i(\theta - b_i)}}

(:cite:t:`Magis2013`, p. 306, Equation 1), with no scaling constant (ADR-0001), and incorrectly with probability
:math:`Q_i = 1 - P_i`. The 3PL fixes :math:`d_i = 1`, and the 2PL also :math:`c_i = 0` (:cite:t:`Magis2012`, p. 3).
The derivatives of :math:`P_i` with respect to the ability are

.. math::

   P_i' = a_i (d_i - c_i) F_i (1 - F_i), \quad P_i'' = a_i P_i' (1 - 2 F_i),

the derivatives of :cite:t:`Barton1981` (p. A2) written with :math:`F_i` and their constant :math:`D` set to 1; the
first is also :cite:t:`Magis2013` (p. 306, Equations 4 and 5).

The information item :math:`i` gives about the ability is :math:`I_i = P_i'^2 / (P_i Q_i)` (:cite:t:`Magis2012`, p. 4,
Equation 4), for the 4PL

.. math::

   I_i(\theta) = \frac{a_i^2 (P_i - c_i)^2 (d_i - P_i)^2}{(d_i - c_i)^2 P_i Q_i}

(:cite:t:`Magis2013`, p. 307, Equation 6), and the test information :math:`I(\theta)` is the sum of the information of
its items (:cite:t:`Magis2012`, p. 5, Equation 8). The maximum likelihood estimate of the ability is asymptotically
normal, with variance the inverse of the test information (:cite:t:`Warm1989`, p. 445) and standard error
:math:`1 / \sqrt{I(\theta)}` (:cite:t:`Magis2012`, p. 4, Equation 3); :py:func:`confidence_interval` is the interval
this normal approximation gives, :math:`\hat\theta \pm z \cdot SE`, with :math:`z` the standard normal's quantile at
:math:`(1 + \text{confidence}) / 2`.

The reliability of the test at :math:`\theta` is :math:`I(\theta) / (I(\theta) + 1 / \sigma^2)`, mirt's
(mirt 1.48, ``R/SingleGroup-methods.R:1474``, the curve ``plot(type = "rxx")`` draws), with the abilities' variance
:math:`\sigma^2 = 1`, as they are :math:`N(0, 1)` (ADR-0008). For abilities :math:`N(0, 1)`, :cite:t:`Bock1982`
(p. 437, Equation 7) define a related reliability of the EAP estimate, one minus its posterior variance.

Since :math:`P_i' = a_i (P_i - c_i)(d_i - P_i) / (d_i - c_i)` (:cite:t:`Magis2013`, p. 306, Equation 5), changing the
variable of integration from :math:`\theta` to :math:`P_i` in Equation 6 and splitting the result into partial
fractions integrates an item's information in closed form:

.. math::

   \int_{\theta_L}^{\theta_U} I_i(\theta)\, d\theta = \frac{a_i}{d_i - c_i} \Big[ P_i - c_i d_i \log P_i +
   (1 - c_i)(1 - d_i) \log(1 - P_i) \Big]_{P_i(\theta_L)}^{P_i(\theta_U)},

derived here rather than printed in a source; it agrees with mirt 1.48's ``areainfo``, which integrates numerically
(``R/areainfo.R:49-66``).

An item is most informative at

.. math::

   \theta^*_i = b_i + \frac{1}{a_i} \log \frac{x^*_i - c_i}{d_i - x^*_i}, \quad
   x^*_i = 2 \sqrt{\frac{-u_i}{3}} \cos\left\{\frac{1}{3} \arccos\left(-\frac{v_i}{2}
   \sqrt{\frac{27}{-u_i^3}}\right) + \frac{4 \pi}{3}\right\} + \frac{1}{2},

with :math:`u_i = -3/4 + (c_i + d_i - 2 c_i d_i)/2` and :math:`v_i = (c_i + d_i - 1)/4`, where :math:`x^*_i` is the
probability of a correct response at :math:`\theta^*_i` (:cite:t:`Magis2013`, pp. 308-309, Equations 13, 18 and 19).
With :math:`d_i = 1` it is :math:`\theta^*_i = b_i + \log[(1 + \sqrt{1 + 8 c_i}) / 2] / a_i` (p. 310, Equation 23),
and with :math:`c_i + d_i = 1` it is :math:`b_i` (pp. 309-310). Magis assumes :math:`0 < c_i < 0.5` and
:math:`0.5 < d_i < 1` (p. 306); :py:func:`max_info_theta` takes Equation 19 for every item with :math:`d_i < 1`.

The log-likelihood of examinee :math:`j`'s responses :math:`u_{ij}` is
:math:`\log L_j(\theta) = \sum_i [u_{ij} \log P_i(\theta) + (1 - u_{ij}) \log Q_i(\theta)]` over the items :math:`i`
the examinee answered (:cite:t:`Magis2012`, p. 4, Equation 2; :cite:t:`Barton1981`, p. A1).

:py:func:`theta_to_scale` takes abilities to scores on a reporting scale along the line through
:math:`(\theta_{min}, \text{scale}_{min})` and :math:`(\theta_{max}, \text{scale}_{max})`, and :py:func:`scale_to_theta`
takes them back.

Calibration solves for each item's parameters, and reports their standard errors, from the derivatives of
:math:`P_i` with respect to those parameters, which every method that estimates items shares. It estimates them in
mirt's parameters (ADR-0007): the slope :math:`\lambda = a` and the intercept :math:`\zeta = -ab`, so that
:math:`a(\theta - b) = \lambda\theta + \zeta`, and the logits :math:`\gamma` of :math:`c` and :math:`\delta` of
:math:`d`. Every array of these derivatives has the four on its last axis, in that order: slope, intercept, logit of
:math:`c`, logit of :math:`d`. By the chain rule,

.. math::

   \frac{\partial P}{\partial \lambda} = (d - c)\,\theta\,F(1 - F), \quad
   \frac{\partial P}{\partial \zeta} = (d - c)\,F(1 - F), \quad
   \frac{\partial P}{\partial \gamma} = c(1 - c)(1 - F), \quad
   \frac{\partial P}{\partial \delta} = d(1 - d)\,F,

from the derivatives in :math:`a`, :math:`b` and :math:`c` of :cite:t:`Barton1981` (p. A3) and the derivative of the
logistic function. A parameter at its bound, :math:`c = 0` or :math:`d = 1`, has derivative 0.

For an item answered by :math:`N_q` examinees at ability :math:`X_q`, :math:`r_q` of them correctly, the
log-likelihood is :math:`\sum_q [r_q \log P(X_q) + (N_q - r_q)\log Q(X_q)]`, the grouped form of :cite:t:`Baker1987`
(p. 112, Equation 3). Its derivative in a parameter :math:`\chi` is
:math:`\sum_q [r_q - N_q P(X_q)]\,(\partial P / \partial\chi) / (PQ)`, the form of :cite:t:`Barton1981` (p. A2)
summed within each group; with :math:`d = 1` the derivative in :math:`\gamma` is the equation for the logit of
:math:`c` of :cite:t:`Mislevy1986` (p. 184, Equation 22). The expected information of one response is
:math:`(\partial P / \partial\chi)(\partial P / \partial\phi) / (PQ)`, Barton and Lord's second derivative (p. A2)
without its terms in :math:`u - P`, whose expected value is 0. For the 2PL in these parameters the observed
information equals the expected one :cite:p:`Baker1987` (p. 116).

Every value stays finite at any finite distance from the difficulty, because nothing is computed by subtracting from 1.
:math:`F` and :math:`1 - F` both come from the logit; :math:`P = c + (d - c)F` and :math:`Q = (1 - d) + (d - c)(1 - F)`;
:math:`\log P` and :math:`\log Q` combine :math:`\log c` or :math:`\log(1 - d)` with :math:`\log(d - c)` and
:math:`\log F` or :math:`\log(1 - F)`. The ratios :math:`(d - c)F / P`, :math:`(d - c)(1 - F) / Q`, :math:`c / P` and
:math:`(1 - d) / Q` each lie between 0 and 1. The first is 0 / 0 only where :math:`c = 0` and :math:`F` has underflowed
to 0, the second only where :math:`d = 1` and :math:`1 - F` has, and both take their limit there, 1. With
:math:`P - c = (d - c)F` and :math:`d - P = (d - c)(1 - F)` (:cite:t:`Magis2013`, p. 306, Equation 3), the item
information is :math:`a^2 F (1 - F)` times the first two ratios.
"""

import math
from collections.abc import Callable
from typing import NamedTuple, overload

import numpy
import numpy.typing as npt
from scipy import stats
from scipy.special import expit, log_expit

from ._responses import responses_and_items
from .typing import BoolArray, FloatArray, IntArray

THETA_MIN_TYPICAL = -4.0
"""The lower end of the typical ability range, the default ``theta_min`` of :py:func:`theta_to_scale`."""

THETA_MAX_TYPICAL = 4.0
"""The upper end of the typical ability range, the default ``theta_max`` of :py:func:`theta_to_scale`."""

THETA_MIN_EXTENDED = -6.0
"""The lower end of the extended ability range, which the default quadrature grid spans."""

THETA_MAX_EXTENDED = 6.0
"""The upper end of the extended ability range, which the default quadrature grid spans."""


def _split(items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return the a, b, c and d of a row or matrix of items, each of shape ``items.shape[:-1]``."""
  return items[..., 0], items[..., 1], items[..., 2], items[..., 3]


def _abilities(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  """Return the abilities shaped to meet every item: ``numpy.shape(theta) + (1,) * (items.ndim - 1)``."""
  thetas = numpy.asarray(theta, dtype=float)
  return thetas.reshape(thetas.shape + (1,) * (items.ndim - 1))


def _logit(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  """Return a (theta - b) for every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``."""
  a, b, _, _ = _split(items)
  return a * (_abilities(theta, items) - b)


def _logistic(theta: float | FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray]:
  """Return F and 1 - F for every ability against every item, both from the logit."""
  z = _logit(theta, items)
  return expit(z), expit(-z)


class _Curve(NamedTuple):
  """The response function at every ability against every item, with its ratios (module docstring)."""

  logistic: FloatArray
  """F."""
  complement: FloatArray
  """1 - F."""
  p: FloatArray
  q: FloatArray
  floor_ratio: FloatArray
  """(d - c) F / P, 1 where P has underflowed to 0."""
  ceiling_ratio: FloatArray
  """(d - c)(1 - F) / Q, 1 where Q has underflowed to 0."""


def _curve(theta: float | FloatArray, items: FloatArray) -> _Curve:
  """Return the response function and its ratios for every ability against every item."""
  logistic, complement = _logistic(theta, items)
  _, _, c, d = _split(items)
  above_floor = (d - c) * logistic
  below_ceiling = (d - c) * complement
  p = c + above_floor
  q = (1 - d) + below_ceiling
  with numpy.errstate(divide="ignore", invalid="ignore"):
    floor_ratio = numpy.where(p > 0, above_floor / p, 1.0)
    ceiling_ratio = numpy.where(q > 0, below_ceiling / q, 1.0)
  return _Curve(logistic, complement, p, q, floor_ratio, ceiling_ratio)


def icc(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the probability of a correct response to each item, its item characteristic curve.

  The probability rises with ``theta`` from the guessing parameter ``c`` to the upper asymptote ``d``, and equals
  ``(c + d) / 2`` at the difficulty ``b``.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray or numpy.float64
      Every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``: a float ``theta`` gives
      ``(n,)`` for ``n`` items and ``k`` abilities give ``(k, n)``. A float ``theta`` with a single item of shape
      ``(4,)`` gives a NumPy scalar, a ``numpy.float64``.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.2, 1.0]])
  >>> icc(0.0, items).round(4)
  array([0.5   , 0.3459])
  >>> icc(numpy.array([-1.0, 0.0, 1.0]), items).shape
  (3, 2)
  """
  _, _, c, d = _split(items)
  return c + (d - c) * expit(_logit(theta, items))


def icc_derivatives(theta: float | FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray]:
  r"""Return the first and second derivatives of each item's characteristic curve with respect to the ability.

  Both are finite at any finite ability, and tend to 0 far from the difficulty.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray]
      The first and the second derivative of :py:func:`icc`, every ability against every item, each of shape
      ``numpy.shape(theta) + items.shape[:-1]``.

  Examples
  --------
  >>> import numpy
  >>> first, second = icc_derivatives(0.0, numpy.array([[2.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.2, 0.9]]))
  >>> first.round(4), second.round(4)
  (array([0.5   , 0.1376]), array([0.    , 0.0636]))
  """
  a, _, c, d = _split(items)
  logistic, complement = _logistic(theta, items)
  first = a * (d - c) * logistic * complement
  return first, a * first * (complement - logistic)


def information(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the Fisher information each item gives about the ability.

  An item is most informative near its difficulty, at :py:func:`max_info_theta`. Far from the difficulty the
  information tends to 0 and stays finite.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      Every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``: a float ``theta`` gives
      ``(n,)`` for ``n`` items and ``k`` abilities give ``(k, n)``.

  Examples
  --------
  >>> import numpy
  >>> information(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  array([0.25, 1.  ])
  """
  a, _, _, _ = _split(items)
  curve = _curve(theta, items)
  return a**2 * curve.logistic * curve.complement * curve.floor_ratio * curve.ceiling_ratio


@overload
def total_information(theta: float, items: FloatArray) -> float: ...
@overload
def total_information(theta: FloatArray, items: FloatArray) -> FloatArray: ...
def total_information(theta: float | FloatArray, items: FloatArray) -> float | FloatArray:
  r"""Return the test information of a set of items, the sum of their :py:func:`information`.

  Items without information, or no items at all, give 0.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  float or numpy.ndarray
      One value per ability: a float for a float ``theta``, otherwise an array of the shape of ``theta``.

  Examples
  --------
  >>> import numpy
  >>> total_information(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  1.25
  """
  values = numpy.sum(information(theta, numpy.atleast_2d(items)), axis=-1)
  return values if numpy.ndim(theta) else float(values)


def var(theta: float, items: FloatArray) -> float:
  r"""Return the variance of the ability estimate at ``theta``, the inverse of the :py:func:`total_information`.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The variance, infinite when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> var(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  0.8
  """
  test_information = total_information(theta, items)
  return math.inf if test_information == 0 else 1 / test_information


def see(theta: float, items: FloatArray) -> float:
  r"""Return the standard error of estimation at ``theta``, the square root of the :py:func:`var`.

  It is in the units of the ability scale.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The standard error, infinite when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> see(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0]]))
  2.0
  """
  return math.sqrt(var(theta, items))


def confidence_interval(theta: float, items: FloatArray, confidence: float = 0.95) -> tuple[float, float]:
  r"""Return the confidence interval of an ability estimate, centred on it, from its standard error :py:func:`see`.

  Parameters
  ----------
  theta : float
      The estimated ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.
  confidence : float, optional
      The confidence level, strictly between 0 and 1. Default is 0.95.

  Returns
  -------
  tuple[float, float]
      The lower and the upper bound of the interval, infinite when the items give no information at ``theta``.

  Raises
  ------
  ValueError
      If ``confidence`` is not strictly between 0 and 1.

  Examples
  --------
  >>> import numpy
  >>> lower, upper = confidence_interval(0.5, numpy.array([[2.0, 0.5, 0.0, 1.0]] * 4))
  >>> round(lower, 4), round(upper, 4)
  (-0.48, 1.48)
  """
  if not 0 < confidence < 1:
    msg = f"Confidence level must be between 0 and 1, got {confidence}"
    raise ValueError(msg)
  margin = float(stats.norm.ppf((1 + confidence) / 2)) * see(theta, items)
  return (theta - margin, theta + margin)


def _linear_map(scale_min: float, scale_max: float, theta_min: float, theta_max: float) -> tuple[float, float]:
  """Return the line that takes ``[theta_min, theta_max]`` onto ``[scale_min, scale_max]``.

  Returns
  -------
  tuple[float, float]
      The slope and the intercept of the scores as a function of the ability.

  Raises
  ------
  ValueError
      If ``theta_max`` is not greater than ``theta_min``, or ``scale_max`` is not greater than ``scale_min``.
  """
  if theta_max <= theta_min:
    msg = f"theta_max ({theta_max}) must be greater than theta_min ({theta_min})"
    raise ValueError(msg)
  if scale_max <= scale_min:
    msg = f"scale_max ({scale_max}) must be greater than scale_min ({scale_min})"
    raise ValueError(msg)
  slope = (scale_max - scale_min) / (theta_max - theta_min)
  return slope, scale_min - slope * theta_min


def theta_to_scale(
  theta: float | FloatArray,
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = THETA_MIN_TYPICAL,
  theta_max: float = THETA_MAX_TYPICAL,
) -> float | FloatArray:
  r"""Convert abilities to scores on a reporting scale, such as 0 to 100 or 200 to 800.

  ``theta_min`` becomes ``scale_min`` and ``theta_max`` becomes ``scale_max``, and abilities between or beyond them
  follow the same line. :py:func:`scale_to_theta` is its inverse.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  scale_min : float, optional
      Score of ``theta_min``. Default is 0.
  scale_max : float, optional
      Score of ``theta_max``. Default is 100.
  theta_min : float, optional
      Ability mapped to ``scale_min``. Default is :py:data:`THETA_MIN_TYPICAL`, -4.
  theta_max : float, optional
      Ability mapped to ``scale_max``. Default is :py:data:`THETA_MAX_TYPICAL`, 4.

  Returns
  -------
  float or numpy.ndarray
      The scores, a float for a float ``theta`` and an array of its shape otherwise.

  Raises
  ------
  ValueError
      If ``theta_max`` is not greater than ``theta_min``, or ``scale_max`` is not greater than ``scale_min``.

  Examples
  --------
  >>> theta_to_scale(2.0)
  75.0
  >>> theta_to_scale(0.0, scale_min=200, scale_max=800)
  500.0
  """
  slope, intercept = _linear_map(scale_min, scale_max, theta_min, theta_max)
  return slope * theta + intercept


def scale_to_theta(
  score: float | FloatArray,
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = THETA_MIN_TYPICAL,
  theta_max: float = THETA_MAX_TYPICAL,
) -> float | FloatArray:
  r"""Convert scores on a reporting scale back to abilities, the inverse of :py:func:`theta_to_scale`.

  With the same four bounds, a score converted to an ability and back is the same score.

  Parameters
  ----------
  score : float or numpy.ndarray
      A score, or an array of scores.
  scale_min : float, optional
      Score of ``theta_min``. Default is 0.
  scale_max : float, optional
      Score of ``theta_max``. Default is 100.
  theta_min : float, optional
      Ability mapped to ``scale_min``. Default is :py:data:`THETA_MIN_TYPICAL`, -4.
  theta_max : float, optional
      Ability mapped to ``scale_max``. Default is :py:data:`THETA_MAX_TYPICAL`, 4.

  Returns
  -------
  float or numpy.ndarray
      The abilities, a float for a float ``score`` and an array of its shape otherwise.

  Raises
  ------
  ValueError
      If ``theta_max`` is not greater than ``theta_min``, or ``scale_max`` is not greater than ``scale_min``.

  Examples
  --------
  >>> scale_to_theta(75.0)
  2.0
  >>> scale_to_theta(500.0, scale_min=200, scale_max=800)
  0.0
  """
  slope, intercept = _linear_map(scale_min, scale_max, theta_min, theta_max)
  return (score - intercept) / slope


def reliability(theta: float, items: FloatArray) -> float:
  r"""Return the reliability of the test at ``theta``, from its :py:func:`total_information`, for abilities N(0, 1).

  Values close to 1 mean a precise test.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The reliability, between 0 and 1; 0 when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> reliability(0.0, numpy.array([[2.0, 0.0, 0.0, 1.0]] * 4))
  0.8
  """
  test_information = total_information(theta, items)
  return test_information / (test_information + 1)


def max_info_theta(items: FloatArray) -> FloatArray:
  r"""Return the ability at which each item gives the most information.

  Parameters
  ----------
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      A new array of shape ``items.shape[:-1]``, one ability per item.

  Examples
  --------
  >>> import numpy
  >>> max_info_theta(numpy.array([[1.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.2, 1.0]])).round(4)
  array([0.5   , 0.2671])
  """
  a, b, c, d = _split(items)
  three_pl = b + numpy.log((1 + numpy.sqrt(1 + 8 * c)) / 2) / a
  u = -(3 / 4) + (c + d - 2 * c * d) / 2
  v = (c + d - 1) / 4
  x_star = 2 * numpy.sqrt(-u / 3) * numpy.cos(numpy.arccos(-(v / 2) * numpy.sqrt(27 / -(u**3))) / 3 + 4 * math.pi / 3)
  x_star += 0.5
  four_pl = b + numpy.log((x_star - c) / (d - x_star)) / a
  return numpy.where(d == 1, three_pl, four_pl)


def info_integral(lower: float, upper: float, items: FloatArray) -> FloatArray:
  r"""Return the integral of each item's :py:func:`information` between two abilities.

  Parameters
  ----------
  lower : float
      Lower integration bound, possibly ``-inf``.
  upper : float
      Upper integration bound, possibly ``inf``.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      The integrals over ``[lower, upper]``, of shape ``items.shape[:-1]``; negated when ``lower`` is above
      ``upper``.

  Examples
  --------
  >>> import numpy
  >>> info_integral(-numpy.inf, numpy.inf, numpy.array([[0.7, -1.0, 0.0, 1.0], [1.3, 0.0, 0.0, 1.0]]))
  array([0.7, 1.3])
  """
  a, _, c, d = _split(items)
  log_p_weight, log_q_weight = c * d, (1 - c) * (1 - d)

  def antiderivative(theta: float) -> FloatArray:
    p = icc(theta, items)
    # a zero weight drops its logarithm, which is -inf at an infinite bound
    with numpy.errstate(divide="ignore", invalid="ignore"):
      log_p_term = numpy.where(log_p_weight > 0, log_p_weight * numpy.log(p), 0.0)
      log_q_term = numpy.where(log_q_weight > 0, log_q_weight * numpy.log1p(-p), 0.0)
    return a / (d - c) * (p - log_p_term + log_q_term)

  return antiderivative(upper) - antiderivative(lower)


def log_response_probabilities(theta: float | FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray]:
  r"""Return the log-probabilities of a correct and of an incorrect response to each item.

  They stay finite at any finite distance from the difficulty. Far below the difficulty the first tends to
  ``log(c)``, far above it the second to ``log(1 - d)``, and without guessing or slipping they keep falling linearly
  in ``theta``.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  tuple[numpy.ndarray, numpy.ndarray]
      The logarithms of :py:func:`icc` and of its complement, every ability against every item, each of shape
      ``numpy.shape(theta) + items.shape[:-1]``.

  Examples
  --------
  >>> import numpy
  >>> log_p, log_q = log_response_probabilities(0.0, numpy.array([[1.0, 0.0, 0.2, 1.0]]))
  >>> numpy.exp(log_p).round(4), numpy.exp(log_q).round(4)
  (array([0.6]), array([0.4]))
  """
  z = _logit(theta, items)
  log_c, log_1md, log_dmc = _log_asymptotes(items)
  return _log_probability(z, log_c, log_dmc), _log_probability(-z, log_1md, log_dmc)


def _log_asymptotes(items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return log c, log(1 - d) and log(d - c) of every item, which do not depend on the ability."""
  _, _, c, d = _split(items)
  # c = 0 and d = 1 give log(0) = -inf, which logaddexp handles exactly
  with numpy.errstate(divide="ignore"):
    return numpy.log(c), numpy.log1p(-d), numpy.log(d - c)


def _log_probability(z: FloatArray, log_floor: FloatArray, log_dmc: FloatArray) -> FloatArray:
  """Return log P from the logit and log c, or log Q from the negated logit and log(1 - d)."""
  return numpy.logaddexp(log_floor, log_dmc + log_expit(z))


def log_likelihood(theta: float | FloatArray, responses: npt.ArrayLike, items: FloatArray) -> FloatArray:
  r"""Return the log-likelihood of each examinee's responses at each ability.

  Each examinee's log-likelihood sums the log-probabilities of :py:func:`log_response_probabilities` over the items
  the examinee answered, and an examinee who answered no item has log-likelihood 0.
  :py:func:`log_likelihood_function` computes the same values for many abilities, one call at a time.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities, each with a finite logit ``a (theta - b)`` for every item.
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items, a matrix of shape ``(n, 4)``, one row per column of ``responses``.

  Returns
  -------
  numpy.ndarray
      Every examinee at every ability, of shape ``(n_examinees,) + numpy.shape(theta)``.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``items`` is not an ``(n, 4)`` matrix, ``responses`` is not a matrix with one column per item, or a response
      is not 0, 1 or NaN.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
  >>> log_likelihood(0.0, numpy.array([[1.0, 0.0], [1.0, numpy.nan]]), items).round(4)
  array([-1.3863, -0.6931])
  """
  return log_likelihood_function(responses, items)(theta)


def log_likelihood_function(responses: npt.ArrayLike, items: FloatArray) -> Callable[[float | FloatArray], FloatArray]:
  r"""Return the log-likelihood of each examinee's responses as a function of the ability.

  The returned function gives, for any ``theta``, the value of :py:func:`log_likelihood`. The responses and items are
  checked here, once, and the terms that do not depend on the ability are computed here too. Changing ``responses``
  or ``items`` afterwards does not change the function.

  Parameters
  ----------
  responses : array_like
      The response matrix, one row per examinee and one column per item: 1 or True for a correct response, 0 or
      False for an incorrect one, and NaN for a missing one.
  items : numpy.ndarray
      The items, a matrix of shape ``(n, 4)``, one row per column of ``responses``.

  Returns
  -------
  Callable[[float | numpy.ndarray], numpy.ndarray]
      The log-likelihood of every examinee at every ability ``theta`` whose logits ``a (theta - b)`` are finite, of
      shape ``(n_examinees,) + numpy.shape(theta)``.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If ``items`` is not an ``(n, 4)`` matrix, ``responses`` is not a matrix with one column per item, or a response
      is not 0, 1 or NaN.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
  >>> log_lik = log_likelihood_function(numpy.array([[True, False]]), items)
  >>> log_lik(0.0).round(4), log_lik(numpy.array([0.0, 1.0])).round(4)
  (array([-1.3863]), array([[-1.3863, -1.6265]]))
  """
  # copies: later changes to the caller's arrays cannot reach the function
  matrix, items = responses_and_items(responses, items)
  if matrix.shape[0] == 1:
    # one examinee selects one term per item it answered, so its log-likelihood is their sum
    answered = numpy.flatnonzero(~numpy.isnan(matrix[0]))
    log_terms = _log_terms(items, answered, matrix[0, answered] == 1)
    return lambda theta: log_terms(theta).sum(axis=-1)[numpy.newaxis]
  # one term per item that someone answered correctly, and one per item that someone answered incorrectly
  right = numpy.flatnonzero((matrix == 1).any(axis=0))
  wrong = numpy.flatnonzero((matrix == 0).any(axis=0))
  log_terms = _log_terms(items, numpy.concatenate([right, wrong]), numpy.arange(right.size + wrong.size) < right.size)
  # which terms each examinee's responses select, one row per examinee
  selected = numpy.concatenate([matrix[:, right] == 1, matrix[:, wrong] == 0], axis=1).astype(float)
  return lambda theta: numpy.moveaxis(log_terms(theta) @ selected.T, -1, 0)


class _LogTerms(NamedTuple):
  """The log P or log Q of each of some items as a function of ability, each written as the log P of a signed item.

  log Q is log P's formula at the negated logit with 1 - d in place of c, and negating a negates the logit exactly, so
  one formula computes every term.
  """

  items: FloatArray
  """The items, with the discrimination negated where the term is log Q."""
  floor: FloatArray
  """c where the term is log P, 1 - d where it is log Q."""
  dmc: FloatArray
  """d - c."""
  log_floor: FloatArray
  log_dmc: FloatArray

  def __call__(self, theta: float | FloatArray) -> FloatArray:
    """Return each term at each ability, of shape ``numpy.shape(theta) + (n_terms,)``."""
    return _log_probability(_logit(theta, self.items), self.log_floor, self.log_dmc)


def _log_terms(items: FloatArray, terms: IntArray, correct: BoolArray) -> _LogTerms:
  """Return the log P, where ``correct`` holds, or the log Q of each item ``items[terms]`` as a function of ability."""
  signed_items = items[terms]
  _, _, c, d = _split(signed_items)
  log_c, log_1md, log_dmc = _log_asymptotes(signed_items)
  floor, log_floor = numpy.where(correct, c, 1.0 - d), numpy.where(correct, log_c, log_1md)
  signed_items[:, 0] = numpy.where(correct, signed_items[:, 0], -signed_items[:, 0])
  return _LogTerms(signed_items, floor, d - c, log_floor, log_dmc)


def probability_gradient(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the derivatives of the probability of a correct response in each item's parameters.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``, with
      :math:`0 \leq c < d \leq 1`.

  Returns
  -------
  numpy.ndarray
      Every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1] + (4,)``: the derivatives
      with respect to the slope, the intercept, the logit of ``c`` and the logit of ``d``.

  Examples
  --------
  >>> import numpy
  >>> probability_gradient(0.0, numpy.array([1.0, 0.0, 0.0, 1.0]))
  array([0.  , 0.25, 0.  , 0.  ])
  """
  _, _, c, d = _split(items)
  logistic, complement = _logistic(theta, items)
  thetas = numpy.broadcast_to(_abilities(theta, items), logistic.shape)
  spread = (d - c) * logistic * complement
  return numpy.stack([thetas * spread, spread, c * (1 - c) * complement, d * (1 - d) * logistic], axis=-1)


def _scores(theta: FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return P, Q and the derivatives of P divided by P Q, the last with the four parameters on its last axis."""
  _, _, c, d = _split(items)
  curve = _curve(theta, items)
  thetas = numpy.broadcast_to(_abilities(theta, items), curve.p.shape)
  # The ratios for c and d are needed only where c > 0 and d < 1, where P >= c and Q >= 1 - d are positive.
  with numpy.errstate(divide="ignore", invalid="ignore"):
    location = curve.floor_ratio * curve.ceiling_ratio / (d - c)
    guessing = numpy.where(c > 0, (1 - c) * (c / curve.p) * curve.ceiling_ratio / (d - c), 0.0)
    upper = numpy.where(d < 1, d * ((1 - d) / curve.q) * curve.floor_ratio / (d - c), 0.0)
  return curve.p, curve.q, numpy.stack([thetas * location, location, guessing, upper], axis=-1)


def log_likelihood_gradient(theta: FloatArray, items: FloatArray, correct: FloatArray, total: FloatArray) -> FloatArray:
  r"""Return the gradient of each item's log-likelihood from response counts at given abilities.

  Parameters
  ----------
  theta : numpy.ndarray
      The abilities, shape ``(k,)``, such as the nodes of a quadrature grid.
  items : numpy.ndarray
      A matrix of items ``[a, b, c, d]``, shape ``(n, 4)``, with :math:`0 \leq c < d \leq 1`.
  correct : numpy.ndarray
      The number of correct responses to each item at each ability, shape ``(k, n)``. Counts need not be integers,
      such as the expected counts of an E-step.
  total : numpy.ndarray
      The number of responses to each item at each ability, shape ``(k, n)``.

  Returns
  -------
  numpy.ndarray
      The gradient of each item's log-likelihood, shape ``(n, 4)``, with respect to the slope, the intercept, the
      logit of ``c`` and the logit of ``d``.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0]])
  >>> log_likelihood_gradient(numpy.array([0.0]), items, numpy.array([[3.0]]), numpy.array([[4.0]]))
  array([[0., 1., 0., 0.]])
  """
  p, _, scores = _scores(theta, items)
  return numpy.einsum("kn,knp->np", correct - total * p, scores)


def parameter_information(theta: FloatArray, items: FloatArray, total: FloatArray) -> FloatArray:
  r"""Return each item's expected information matrix from the number of responses at given abilities.

  Parameters
  ----------
  theta : numpy.ndarray
      The abilities, shape ``(k,)``, such as the nodes of a quadrature grid.
  items : numpy.ndarray
      A matrix of items ``[a, b, c, d]``, shape ``(n, 4)``, with :math:`0 \leq c < d \leq 1`.
  total : numpy.ndarray
      The number of responses to each item at each ability, shape ``(k, n)``. Counts need not be integers.

  Returns
  -------
  numpy.ndarray
      One symmetric matrix per item, shape ``(n, 4, 4)``, over the slope, the intercept, the logit of ``c`` and the
      logit of ``d``. A parameter at its bound, ``c = 0`` or ``d = 1``, has a zero row and column.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0]])
  >>> parameter_information(numpy.array([0.0]), items, numpy.array([[4.0]]))[0]
  array([[0., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])
  """
  p, q, scores = _scores(theta, items)
  return numpy.einsum("kn,knp,knr->npr", total * p * q, scores, scores)
