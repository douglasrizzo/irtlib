r"""Derivatives of the item response function with respect to the item parameters.

Calibration solves for each item's parameters, and reports their standard errors, from the derivatives of the
probability of a correct response with respect to those parameters. They are closed forms of the logistic function,
shared by every method that estimates items.

The probability is catsim's :py:func:`catsim.irt.icc`, :math:`P = c + (d - c)F` with
:math:`F = 1 / (1 + e^{-a(\theta - b)})`. Calibration estimates it in mirt's parameters (ADR-0007): the slope
:math:`\lambda = a` and the intercept :math:`\zeta = -ab`, so that :math:`a(\theta - b) = \lambda\theta + \zeta`, and
the logits :math:`\gamma` of :math:`c` and :math:`\delta` of :math:`d`. Every array here has these four on its last
axis, in that order: slope, intercept, logit of :math:`c`, logit of :math:`d`. By the chain rule,

.. math::

   \frac{\partial P}{\partial \lambda} = (d - c)\,\theta\,F(1 - F), \quad
   \frac{\partial P}{\partial \zeta} = (d - c)\,F(1 - F), \quad
   \frac{\partial P}{\partial \gamma} = c(1 - c)(1 - F), \quad
   \frac{\partial P}{\partial \delta} = d(1 - d)\,F,

from the derivatives in :math:`a`, :math:`b` and :math:`c` of :cite:t:`Barton1981` (p. A3) and the derivative of the
logistic function. A parameter at its bound, :math:`c = 0` or :math:`d = 1`, has derivative 0.

For an item answered by :math:`N_q` examinees at ability :math:`X_q`, :math:`r_q` of them correctly, the
log-likelihood is :math:`\sum_q [r_q \log P(X_q) + (N_q - r_q)\log Q(X_q)]` with :math:`Q = 1 - P`, the grouped form
of :cite:t:`Baker1987` (p. 112, Equation 3). Its derivative in a parameter :math:`\chi` is
:math:`\sum_q [r_q - N_q P(X_q)]\,(\partial P / \partial\chi) / (PQ)`, the form of :cite:t:`Barton1981` (p. A2)
summed within each group; with :math:`d = 1` the derivative in :math:`\gamma` is the equation for the logit of
:math:`c` of :cite:t:`Mislevy1986` (p. 184, Equation 22). The expected information of one response is
:math:`(\partial P / \partial\chi)(\partial P / \partial\phi) / (PQ)`, Barton and Lord's second derivative (p. A2)
without its terms in :math:`u - P`, whose expected value is 0. For the 2PL in these parameters the observed
information equals the expected one :cite:p:`Baker1987` (p. 116).

The ratios :math:`(d - c)F / P`, :math:`(d - c)(1 - F) / Q`, :math:`c / P` and :math:`(1 - d) / Q` are formed from
:math:`F` and :math:`1 - F` without subtracting from 1, and each lies between 0 and 1, so every value stays finite at
any distance from the difficulty.
"""

import numpy
from scipy.special import expit

from .typing import FloatArray


def _logistic(
  theta: float | FloatArray, items: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return theta, F, 1 - F, c and d, each of shape ``numpy.shape(theta) + items.shape[:-1]``."""
  a, b, c, d = items[..., 0], items[..., 1], items[..., 2], items[..., 3]
  thetas = numpy.asarray(theta, dtype=float)
  thetas = thetas.reshape(thetas.shape + (1,) * (items.ndim - 1))
  z = a * (thetas - b)
  return numpy.broadcast_to(thetas, z.shape), expit(z), expit(-z), c, d


def probability_gradient(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the derivatives of the probability of a correct response in each item's parameters.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``, with
      :math:`0 \leq c < d \leq 1`, as :py:class:`catsim.ItemBank` validates.

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
  thetas, logistic, complement, c, d = _logistic(theta, items)
  spread = (d - c) * logistic * complement
  return numpy.stack([thetas * spread, spread, c * (1 - c) * complement, d * (1 - d) * logistic], axis=-1)


def _scores(theta: FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return P, Q and the derivatives of P divided by P Q, the last with the four parameters on its last axis."""
  thetas, logistic, complement, c, d = _logistic(theta, items)
  above_floor = (d - c) * logistic
  below_ceiling = (d - c) * complement
  p = c + above_floor
  q = (1 - d) + below_ceiling
  # (d - c) F / P is 0 / 0 only where c = 0 and F has underflowed to 0, and (d - c)(1 - F) / Q only where d = 1 and
  # 1 - F has; the limit of both is 1. The ratios for c and d are needed only where c > 0 and d < 1, where P >= c and
  # Q >= 1 - d are positive.
  with numpy.errstate(divide="ignore", invalid="ignore"):
    floor_ratio = numpy.where(p > 0, above_floor / p, 1.0)
    ceiling_ratio = numpy.where(q > 0, below_ceiling / q, 1.0)
    location = floor_ratio * ceiling_ratio / (d - c)
    guessing = numpy.where(c > 0, (1 - c) * (c / p) * ceiling_ratio / (d - c), 0.0)
    upper = numpy.where(d < 1, d * ((1 - d) / q) * floor_ratio / (d - c), 0.0)
  return p, q, numpy.stack([thetas * location, location, guessing, upper], axis=-1)


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
