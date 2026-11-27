r"""The quadrature grid, the ability priors, and each examinee's posterior over the grid.

Calibration and ability estimation both integrate over ability, and both replace the integral by a sum over the nodes
:math:`X_q`, :math:`q = 1, \dots, Q`, of a grid. Each node carries a mass :math:`A_q`, the probability of a discrete
prior distribution at that node, and the masses sum to 1 (:cite:t:`Bock1982`, p. 433, Equation 6). A grid holds the
nodes and a weight :math:`w_q` for each, and a prior is a log-density :math:`\log \pi`. A node's mass is its weight
times the prior density there, normalized:

.. math::

   A_q = \frac{w_q\,\pi(X_q)}{\sum_{h=1}^{Q} w_h\,\pi(X_h)}.

The equal weights of :py:meth:`QuadratureGrid.uniform` cancel, which leaves the prior density at the nodes, normalized:
the masses :cite:t:`Bock1981` (p. 449) take for a normal prior, and for any other. Those nodes are evenly spaced, as
Bock and Mislevy's are: Gauss-Hermite quadrature, they note, has less advantage for the likelihoods of adaptive testing
(p. 433). The default grid has 121 nodes on the extended ability range :math:`[-6, 6]`, a spacing of 0.1: about the
spacing of the 80 points between :math:`\pm 4` standard deviations that Bock and Mislevy (p. 433) suggest to keep
precision down to a posterior standard deviation of 0.2.

Examinee :math:`j`'s likelihood at a node, :math:`L_j(X_q)`, is the product over the items :math:`i` it answered of
:math:`P_i(X_q)^{u_{ij}} [1 - P_i(X_q)]^{1 - u_{ij}}` (:cite:t:`Bock1982`, p. 433, Equation 3), and its posterior
mass at the node is

.. math::

   \frac{L_j(X_q)\,A_q}{\sum_{h=1}^{Q} L_j(X_h)\,A_h}.

The mean of the posterior over the nodes is the EAP estimate of the examinee's ability, and its variance serves as
the estimate's error variance: the square of the posterior standard deviation (:cite:t:`Bock1982`, p. 432; p. 433,
Equations 4 and 5). An examinee who answered no item has :math:`L_j = 1` at every node, and its posterior is the
masses. Bock and Mislevy apply Sheppard's correction to the posterior variance (p. 433); :py:func:`posterior_variance`
does not.

The log-likelihoods accumulate as sums over the items (:cite:t:`Bock1982`, p. 432), so :py:func:`posterior` takes
each examinee's log-likelihood at the nodes, whatever the response function that gave it. It adds the log of each
node's weight and prior density, and shifts each examinee's row by its maximum before exponentiating, so that the
likelihood of a long response pattern does not underflow.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, override

import numpy

from .irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .typing import FloatArray

type LogPrior = Callable[[FloatArray], FloatArray]
"""A log prior density: maps an array of abilities to their log-densities, elementwise."""

_MIN_NODES = 2
_POSTERIOR_NDIM = 2


def normal_log_prior(mean: float = 0.0, sd: float = 1.0) -> LogPrior:
  """Return the log-density of a normal prior with the given mean and standard deviation.

  The prior pickles, equals any prior this function returns for the same arguments, and its repr is the call that
  builds it.

  Parameters
  ----------
  mean : float, optional
      Mean of the prior. Default is 0.
  sd : float, optional
      Standard deviation of the prior. Default is 1.

  Returns
  -------
  LogPrior
      A vectorized callable mapping abilities to their log-densities under the prior.

  Raises
  ------
  ValueError
      If ``mean`` is not finite, or ``sd`` is not finite and positive.

  Examples
  --------
  >>> normal_log_prior(mean=0.5, sd=2.0)
  normal_log_prior(mean=0.5, sd=2.0)
  """
  if not math.isfinite(mean):
    msg = f"mean must be finite, got {mean}"
    raise ValueError(msg)
  if not (math.isfinite(sd) and sd > 0):
    msg = f"sd must be positive and finite, got {sd}"
    raise ValueError(msg)
  return _NormalLogPrior(float(mean), float(sd))


# The priors are module-level classes rather than closures, so that an object holding one pickles, as one sent to a
# worker process must.
@dataclass(frozen=True, slots=True)
class _NormalLogPrior:
  """The normal log-density that :py:func:`normal_log_prior` returns."""

  mean: float
  sd: float

  def __call__(self, theta: FloatArray) -> FloatArray:
    """Return the log-density at each ability."""
    var = self.sd * self.sd
    return -0.5 * numpy.log(2.0 * numpy.pi * var) - 0.5 * (theta - self.mean) ** 2 / var

  @override
  def __repr__(self) -> str:
    """Return the call to :py:func:`normal_log_prior` that builds this prior."""
    return f"normal_log_prior(mean={self.mean!r}, sd={self.sd!r})"


def uniform_log_prior(low: float = THETA_MIN_EXTENDED, high: float = THETA_MAX_EXTENDED) -> LogPrior:
  """Return the log-density of a uniform prior on ``[low, high]``.

  The log-density is minus the log of the width inside the bounds, the bounds included, and -inf outside them. The
  prior pickles, equals any prior this function returns for the same arguments, and its repr is the call that builds
  it.

  Parameters
  ----------
  low : float, optional
      Lower bound of the prior's support. Default is ``irtlib.irt.THETA_MIN_EXTENDED``.
  high : float, optional
      Upper bound of the prior's support. Default is ``irtlib.irt.THETA_MAX_EXTENDED``.

  Returns
  -------
  LogPrior
      A vectorized callable mapping abilities to their log-densities under the prior.

  Raises
  ------
  ValueError
      If either bound is not finite or ``high <= low``.

  Examples
  --------
  >>> uniform_log_prior()
  uniform_log_prior(low=-6.0, high=6.0)
  """
  if not (math.isfinite(low) and math.isfinite(high)):
    msg = f"low and high must be finite, got low={low}, high={high}"
    raise ValueError(msg)
  if high <= low:
    msg = f"high must be greater than low, got low={low}, high={high}"
    raise ValueError(msg)
  return _UniformLogPrior(float(low), float(high))


@dataclass(frozen=True, slots=True)
class _UniformLogPrior:
  """The uniform log-density that :py:func:`uniform_log_prior` returns."""

  low: float
  high: float

  def __call__(self, theta: FloatArray) -> FloatArray:
    """Return the log-density at each ability: minus the log of the width inside the bounds, -inf outside."""
    inside = (theta >= self.low) & (theta <= self.high)
    return numpy.where(inside, float(numpy.log(1.0 / (self.high - self.low))), -numpy.inf)

  @override
  def __repr__(self) -> str:
    """Return the call to :py:func:`uniform_log_prior` that builds this prior."""
    return f"uniform_log_prior(low={self.low!r}, high={self.high!r})"


# eq=False: generated equality and hashing would compare the arrays, which raises; a grid is compared by identity
@dataclass(frozen=True, slots=True, eq=False)
class QuadratureGrid:
  """The nodes abilities are integrated over, each with a weight.

  A node's mass is proportional to its weight times the prior density there, so the weights take part in the
  posterior itself; equal weights, as :py:meth:`uniform` builds, cancel in the normalization. Two grids are equal only
  if they are the same object, and a grid can be a set member or a dictionary key.

  Parameters
  ----------
  nodes : numpy.ndarray
      One-dimensional array of finite abilities.
  weights : numpy.ndarray
      One-dimensional array of finite, positive weights, one per node.

  Raises
  ------
  ValueError
      If the arrays are not one-dimensional, differ in length, have fewer than two entries, or hold nodes that are
      not finite or weights that are not finite and positive.
  """

  nodes: FloatArray
  weights: FloatArray

  def __post_init__(self) -> None:
    """Validate the nodes and weights."""
    if self.nodes.ndim != 1 or self.weights.ndim != 1:
      msg = "nodes and weights must be one-dimensional"
      raise ValueError(msg)
    if self.nodes.shape != self.weights.shape:
      msg = f"nodes and weights must have the same length, got {self.nodes.size} and {self.weights.size}"
      raise ValueError(msg)
    if self.nodes.size < _MIN_NODES:
      msg = f"a grid needs at least {_MIN_NODES} nodes, got {self.nodes.size}"
      raise ValueError(msg)
    if not numpy.isfinite(self.nodes).all():
      msg = "nodes must be finite"
      raise ValueError(msg)
    if not (numpy.isfinite(self.weights).all() and (self.weights > 0).all()):
      msg = "weights must be finite and positive"
      raise ValueError(msg)

  @classmethod
  def uniform(cls, n_nodes: int = 121, low: float = THETA_MIN_EXTENDED, high: float = THETA_MAX_EXTENDED) -> Self:
    """Return a grid of equally spaced nodes from ``low`` to ``high``, each weighing the spacing.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes, both bounds included. Default is 121, a spacing of 0.1 on the default bounds.
    low : float, optional
        First node. Default is ``irtlib.irt.THETA_MIN_EXTENDED``.
    high : float, optional
        Last node. Default is ``irtlib.irt.THETA_MAX_EXTENDED``.

    Returns
    -------
    QuadratureGrid
        The grid, with every weight equal to the spacing of the nodes.

    Raises
    ------
    ValueError
        If ``n_nodes < 2`` or ``high <= low``.

    Examples
    --------
    >>> grid = QuadratureGrid.uniform()
    >>> grid.nodes.size, float(grid.nodes[0]), float(grid.nodes[-1]), float(grid.weights[0])
    (121, -6.0, 6.0, 0.1)
    """
    if n_nodes < _MIN_NODES:
      msg = f"n_nodes must be at least {_MIN_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if high <= low:
      msg = f"high must be greater than low, got low={low}, high={high}"
      raise ValueError(msg)
    nodes = numpy.linspace(low, high, n_nodes, dtype=float)
    weights = numpy.full(n_nodes, (high - low) / (n_nodes - 1), dtype=float)
    return cls(nodes=nodes, weights=weights)


def _log_masses(grid: QuadratureGrid, log_prior: LogPrior) -> FloatArray:
  """Return the log of each node's weight times the prior density there, unnormalized, after checking it."""
  log_masses = log_prior(grid.nodes) + numpy.log(grid.weights)
  if numpy.isnan(log_masses).any() or numpy.isposinf(log_masses).any():
    msg = "the log-prior is NaN or +inf at some node"
    raise ValueError(msg)
  if not numpy.isfinite(log_masses).any():
    msg = "the prior has no mass at any node of the grid"
    raise ValueError(msg)
  return log_masses


def _normalized(log_masses: FloatArray) -> FloatArray:
  """Return the exponential of log-masses normalized along the last axis, each row shifted by its maximum first."""
  masses = numpy.exp(log_masses - log_masses.max(axis=-1, keepdims=True))
  return masses / masses.sum(axis=-1, keepdims=True)


def prior_masses(grid: QuadratureGrid, log_prior: LogPrior) -> FloatArray:
  """Return the prior's mass at each node of a grid: the node's weight times the prior density there, normalized.

  The masses are the posterior of an examinee who answered no item.

  Parameters
  ----------
  grid : QuadratureGrid
      The nodes and their weights.
  log_prior : LogPrior
      Vectorized log-density of the prior.

  Returns
  -------
  numpy.ndarray
      One mass per node, summing to 1.

  Raises
  ------
  ValueError
      If the log-prior is NaN or +inf at a node, or -inf at every node.

  Examples
  --------
  >>> prior_masses(QuadratureGrid.uniform(4, -1.0, 2.0), uniform_log_prior(-1.0, 2.0))
  array([0.25, 0.25, 0.25, 0.25])
  """
  return _normalized(_log_masses(grid, log_prior))


def posterior(log_likelihood: FloatArray, grid: QuadratureGrid, log_prior: LogPrior) -> FloatArray:
  """Return each examinee's posterior over the nodes of a grid, from its log-likelihood at the nodes.

  An examinee's posterior mass at a node is proportional to the node's weight times the prior density and the
  likelihood there. A row of zeros, the log-likelihood of an examinee who answered no item, gives the prior's masses.

  Parameters
  ----------
  log_likelihood : numpy.ndarray
      Each examinee's log-likelihood at each node, shape ``(examinees, nodes)``; -inf where the likelihood is 0.
  grid : QuadratureGrid
      The nodes and their weights.
  log_prior : LogPrior
      Vectorized log-density of the prior.

  Returns
  -------
  numpy.ndarray
      The posterior probabilities, shape ``(examinees, nodes)``, each row summing to 1.

  Raises
  ------
  ValueError
      If ``log_likelihood`` is not of shape ``(examinees, nodes)``, if the log-prior or the log-likelihood is NaN or
      +inf at a node, or if an examinee's posterior has no mass at any node.

  Examples
  --------
  >>> import numpy
  >>> grid = QuadratureGrid.uniform(3, -1.0, 1.0)
  >>> likelihood = numpy.array([[0.2, 0.5, 0.8], [1.0, 1.0, 1.0]])
  >>> posterior(numpy.log(likelihood), grid, uniform_log_prior(-1.0, 1.0)).round(4)
  array([[0.1333, 0.3333, 0.5333],
         [0.3333, 0.3333, 0.3333]])
  """
  log_likelihood = numpy.asarray(log_likelihood, dtype=float)
  n_nodes = grid.nodes.size
  if log_likelihood.ndim != _POSTERIOR_NDIM or log_likelihood.shape[1] != n_nodes:
    msg = (
      f"log_likelihood must have one row per examinee and one column per node, shape (examinees, {n_nodes}), got "
      f"{log_likelihood.shape}"
    )
    raise ValueError(msg)
  log_masses = _log_masses(grid, log_prior)
  undefined = (numpy.isnan(log_likelihood) | numpy.isposinf(log_likelihood)).any(axis=1)
  if undefined.any():
    msg = f"the log-likelihood is NaN or +inf for the examinees in rows {numpy.flatnonzero(undefined).tolist()}"
    raise ValueError(msg)
  log_joint = log_likelihood + log_masses
  massless = ~numpy.isfinite(log_joint).any(axis=1)
  if massless.any():
    msg = f"the posterior has no mass at any node for the examinees in rows {numpy.flatnonzero(massless).tolist()}"
    raise ValueError(msg)
  return _normalized(log_joint)


def posterior_mean(post: FloatArray, nodes: FloatArray) -> FloatArray:
  """Return the mean of each examinee's posterior over the nodes.

  Parameters
  ----------
  post : numpy.ndarray
      Posterior probabilities, shape ``(examinees, nodes)``, as :py:func:`posterior` returns.
  nodes : numpy.ndarray
      The nodes the probabilities refer to.

  Returns
  -------
  numpy.ndarray
      The probability-weighted mean of the nodes, one per examinee.

  Examples
  --------
  >>> import numpy
  >>> posterior_mean(numpy.array([[0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]), numpy.array([-1.0, 0.0, 1.0]))
  array([0. , 0.5])
  """
  return numpy.sum(post * nodes, axis=1)


def posterior_variance(post: FloatArray, nodes: FloatArray) -> FloatArray:
  """Return the variance of each examinee's posterior over the nodes.

  Parameters
  ----------
  post : numpy.ndarray
      Posterior probabilities, shape ``(examinees, nodes)``, as :py:func:`posterior` returns.
  nodes : numpy.ndarray
      The nodes the probabilities refer to.

  Returns
  -------
  numpy.ndarray
      The probability-weighted mean of the squared distances between the nodes and the examinee's
      :py:func:`posterior_mean`, one per examinee.

  Examples
  --------
  >>> import numpy
  >>> posterior_variance(numpy.array([[0.25, 0.5, 0.25], [0.0, 0.5, 0.5]]), numpy.array([-1.0, 0.0, 1.0]))
  array([0.5 , 0.25])
  """
  mean = posterior_mean(post, nodes)
  return numpy.sum(post * (nodes - mean[:, numpy.newaxis]) ** 2, axis=1)
