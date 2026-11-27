"""Tests of the quadrature grid, the ability priors and the posterior over the grid.

Expected values come from hand-worked cases, the closed forms of the normal and uniform densities, SciPy's normal
log-density, and a trapezoid integral of the posterior on a fine grid.
"""

from __future__ import annotations

import math
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import norm

from irtlib import quadrature
from irtlib.quadrature import (
  QuadratureGrid,
  normal_log_prior,
  posterior,
  posterior_mean,
  posterior_variance,
  prior_masses,
  uniform_log_prior,
)

if TYPE_CHECKING:
  from irtlib.quadrature import LogPrior
  from irtlib.typing import FloatArray

# Seven nodes with unequal weights, under a normal prior that is not centred on a node, so that leaving out the weights,
# the prior or its mean changes every mass.
_NODES = np.linspace(-3.0, 3.0, 7)
_WEIGHTS = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0])
_PRIOR_MEAN, _PRIOR_SD = 0.5, 2.0


def _masses() -> FloatArray:
  """Return the weights times the normal density of ``_PRIOR_MEAN`` and ``_PRIOR_SD`` at ``_NODES``, normalized."""
  unnormalized = _WEIGHTS * np.exp(-((_NODES - _PRIOR_MEAN) ** 2) / (2 * _PRIOR_SD**2))
  return unnormalized / unnormalized.sum()


def _log_likelihood(responses: FloatArray, items: FloatArray, theta: FloatArray) -> FloatArray:
  """Return each examinee's log-likelihood at each ability, shape (examinees, abilities), skipping NaN responses."""
  a, b, c, d = (items[:, [column]] for column in range(4))
  p = c + (d - c) / (1.0 + np.exp(-a * (theta - b)))
  answered = ~np.isnan(responses)
  correct = np.where(answered, responses, 0.0)
  return correct @ np.log(p) + (answered - correct) @ np.log1p(-p)


def _nan_at_first_node(theta: FloatArray) -> FloatArray:
  values = np.zeros_like(theta)
  values[0] = np.nan
  return values


def _positive_infinity_at_first_node(theta: FloatArray) -> FloatArray:
  values = np.zeros_like(theta)
  values[0] = np.inf
  return values


def _zero_everywhere(theta: FloatArray) -> FloatArray:
  return np.full_like(theta, -np.inf)


@pytest.mark.parametrize(("mean", "sd"), [(0.0, 1.0), (0.5, 2.0), (-1.5, 0.3)])
def test_normal_log_prior_is_the_normal_log_density(mean: float, sd: float) -> None:
  """The prior's values are SciPy's normal log-density; fails if the variance, the constant or the mean is wrong."""
  theta = np.array([-6.0, -1.0, 0.0, 0.7, 4.0])
  np.testing.assert_allclose(normal_log_prior(mean, sd)(theta), norm.logpdf(theta, mean, sd), rtol=1e-14)


def test_normal_log_prior_defaults_to_the_standard_normal() -> None:
  """The default prior has mean 0 and standard deviation 1, the log-density -log(2 pi) / 2 - theta^2 / 2."""
  theta = np.array([-2.0, 0.0, 1.0])
  np.testing.assert_allclose(normal_log_prior()(theta), -0.5 * math.log(2 * math.pi) - theta**2 / 2, rtol=1e-14)


def test_uniform_log_prior_is_minus_the_log_width_on_its_bounds_and_inside() -> None:
  """The density is 1 / (high - low) inside the bounds and on them, and 0 outside.

  Fails if a bound is excluded, or the width is wrong.
  """
  theta = np.array([-2.0, -1.0, 0.0, 1.0, 1.5])
  expected = [-np.inf, -math.log(2.0), -math.log(2.0), -math.log(2.0), -np.inf]
  np.testing.assert_allclose(uniform_log_prior(-1.0, 1.0)(theta), expected, rtol=1e-15)


def test_uniform_log_prior_defaults_to_the_extended_ability_range() -> None:
  """The default prior is uniform on [-6, 6]: log-density -log 12 at both bounds, -inf just outside them."""
  theta = np.array([-6.01, -6.0, 6.0, 6.01])
  np.testing.assert_allclose(uniform_log_prior()(theta), [-np.inf, -math.log(12.0), -math.log(12.0), -np.inf])


@pytest.mark.parametrize(
  ("mean", "sd", "match"),
  [
    (0.0, 0.0, "sd must be positive"),
    (0.0, -1.0, "sd must be positive"),
    (0.0, math.nan, "sd must be positive"),
    (0.0, math.inf, "sd must be positive"),
    (math.inf, 1.0, "mean must be finite"),
    (math.nan, 1.0, "mean must be finite"),
  ],
)
def test_normal_log_prior_rejects_an_invalid_mean_or_sd(mean: float, sd: float, match: str) -> None:
  """A mean that is not finite, or a standard deviation that is not finite and positive, is an error."""
  with pytest.raises(ValueError, match=match):
    normal_log_prior(mean, sd)


@pytest.mark.parametrize(
  ("low", "high", "match"),
  [
    (1.0, 1.0, "high must be greater than low"),
    (2.0, 1.0, "high must be greater than low"),
    (math.nan, 1.0, "must be finite"),
    (0.0, math.inf, "must be finite"),
  ],
)
def test_uniform_log_prior_rejects_bounds_that_are_not_a_finite_interval(low: float, high: float, match: str) -> None:
  """Bounds that are not finite, or not increasing, are an error."""
  with pytest.raises(ValueError, match=match):
    uniform_log_prior(low, high)


@pytest.mark.parametrize(
  ("prior", "source"),
  [
    (normal_log_prior(0.5, 2.0), "normal_log_prior(mean=0.5, sd=2.0)"),
    (normal_log_prior(np.float64(0.5), np.float64(2.0)), "normal_log_prior(mean=0.5, sd=2.0)"),
    (uniform_log_prior(-3.0, 3.0), "uniform_log_prior(low=-3.0, high=3.0)"),
    (uniform_log_prior(np.float64(-3.0), np.float64(3.0)), "uniform_log_prior(low=-3.0, high=3.0)"),
  ],
)
def test_prior_repr_is_the_call_that_builds_it(prior: LogPrior, source: str) -> None:
  """A prior's repr is its factory call, which evaluates to an equal prior.

  A closure's repr is an address, and NumPy floats print as ``np.float64(0.5)``, which does not evaluate in irtlib's
  namespace.
  """
  assert repr(prior) == source
  assert eval(source, vars(quadrature)) == prior


@pytest.mark.parametrize("prior", [normal_log_prior(0.5, 2.0), uniform_log_prior(-3.0, 3.0)])
def test_priors_pickle(prior: LogPrior) -> None:
  """A prior survives pickling with its parameters and values, so an object holding one can go to a worker process.

  A closure does not pickle.
  """
  theta = np.linspace(-4.0, 4.0, 9)
  copy = pickle.loads(pickle.dumps(prior))
  assert copy == prior
  np.testing.assert_array_equal(copy(theta), prior(theta))


def test_quadrature_grid_compares_and_hashes_by_identity() -> None:
  """Two grids compare unequal unless they are the same object, and a grid can be a set member or dict key.

  Fails if the dataclass generates equality and hashing from its array fields: comparing two grids of more than one
  node then raises "the truth value of an array ... is ambiguous", and hashing raises "unhashable type".
  """
  grid, twin = QuadratureGrid.uniform(), QuadratureGrid.uniform()
  same = grid

  assert grid == same
  assert grid != twin
  assert len({grid, twin, same}) == 2


def test_uniform_grid_defaults_to_121_nodes_on_the_extended_ability_range() -> None:
  """The default grid has 121 nodes from -6 to 6, a spacing of 0.1, each weighing the spacing."""
  grid = QuadratureGrid.uniform()
  np.testing.assert_allclose(grid.nodes, np.arange(-60, 61) / 10, rtol=0, atol=1e-12)
  np.testing.assert_allclose(grid.weights, np.full(121, 0.1), rtol=1e-15)


@pytest.mark.parametrize(
  ("n_nodes", "low", "high", "nodes", "spacing"),
  [
    (5, -2.0, 2.0, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 1.0),
    (3, 0.0, 0.5, np.array([0.0, 0.25, 0.5]), 0.25),
  ],
)
def test_uniform_grid_spaces_its_nodes_equally_and_weighs_each_by_the_spacing(
  n_nodes: int, low: float, high: float, nodes: FloatArray, spacing: float
) -> None:
  """The nodes run from low to high, both included, and every weight is the spacing.

  Fails if the weight is 1 / n_nodes or (high - low) / n_nodes.
  """
  grid = QuadratureGrid.uniform(n_nodes, low, high)
  np.testing.assert_allclose(grid.nodes, nodes, rtol=0, atol=1e-12)
  np.testing.assert_allclose(grid.weights, np.full(n_nodes, spacing), rtol=1e-15)


@pytest.mark.parametrize(
  ("n_nodes", "low", "high", "match"),
  [
    (1, -6.0, 6.0, "at least 2"),
    (5, 1.0, 1.0, "high must be greater than low"),
    (5, 2.0, 1.0, "high must be greater than low"),
  ],
)
def test_uniform_grid_rejects_fewer_than_two_nodes_or_an_empty_range(
  n_nodes: int, low: float, high: float, match: str
) -> None:
  """A grid of one node, or with bounds that are not increasing, is an error."""
  with pytest.raises(ValueError, match=match):
    QuadratureGrid.uniform(n_nodes, low, high)


@pytest.mark.parametrize(
  ("nodes", "weights", "match"),
  [
    (np.linspace(-1.0, 1.0, 5), np.array([0.3]), "same length"),
    (np.linspace(-1.0, 1.0, 5), np.ones(4), "same length"),
    (np.ones((2, 3)), np.ones((2, 3)), "one-dimensional"),
    (np.ones(3), np.ones((1, 3)), "one-dimensional"),
    (np.array([0.0]), np.array([1.0]), "at least 2"),
    (np.array([0.0, np.nan]), np.ones(2), "nodes must be finite"),
    (np.array([0.0, np.inf]), np.ones(2), "nodes must be finite"),
    (np.array([0.0, 1.0]), np.array([1.0, -1.0]), "weights must be finite and positive"),
    (np.array([0.0, 1.0]), np.array([1.0, 0.0]), "weights must be finite and positive"),
    (np.array([0.0, 1.0]), np.array([1.0, np.inf]), "weights must be finite and positive"),
    (np.array([0.0, 1.0]), np.array([1.0, np.nan]), "weights must be finite and positive"),
  ],
)
def test_quadrature_grid_rejects_invalid_arrays(nodes: FloatArray, weights: FloatArray, match: str) -> None:
  """A grid built directly from arrays rejects shapes, nodes and weights that cannot integrate."""
  with pytest.raises(ValueError, match=match):
    QuadratureGrid(nodes=nodes, weights=weights)


def test_prior_masses_are_the_weights_times_the_prior_density_normalized() -> None:
  """Each node's mass is its weight times the prior density there, divided by their sum over the nodes.

  Fails if the masses leave out the weights or the prior, or are not normalized.
  """
  grid = QuadratureGrid(nodes=_NODES, weights=_WEIGHTS)
  np.testing.assert_allclose(prior_masses(grid, normal_log_prior(_PRIOR_MEAN, _PRIOR_SD)), _masses(), rtol=1e-13)


def test_posterior_of_an_examinee_with_no_responses_is_the_prior_masses() -> None:
  """A row of zeros, the log-likelihood of an examinee who answered no item, has the prior masses as its posterior.

  The rows without responses sit around one with, so a posterior that mixed rows would change them.
  """
  grid = QuadratureGrid(nodes=_NODES, weights=_WEIGHTS)
  log_likelihood = np.zeros((3, _NODES.size))
  log_likelihood[1] = np.log(np.linspace(0.1, 0.9, _NODES.size))

  post = posterior(log_likelihood, grid, normal_log_prior(_PRIOR_MEAN, _PRIOR_SD))

  np.testing.assert_allclose(post[[0, 2]], [_masses(), _masses()], rtol=1e-13)


def test_each_row_is_that_examinees_posterior_alone() -> None:
  """Each row is its examinee's posterior alone: the masses times the likelihood at each node, normalized over the row.

  The last row's log-likelihood is the first's less 1000, below the log of the smallest float at every node, so its
  posterior is the first row's. A posterior that exponentiated before shifting each row by its own maximum, or shifted
  every row by the largest value in the batch, left that row with no mass; one normalized over the batch, or over the
  nodes for all examinees at once, changed every row.
  """
  grid = QuadratureGrid(nodes=_NODES, weights=_WEIGHTS)
  likelihood = np.array([
    [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
    [0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05],
    [0.0, 0.0, 0.3, 1.0, 0.3, 0.0, 0.0],
  ])
  with np.errstate(divide="ignore"):
    log_likelihood = np.log(likelihood)
  log_likelihood = np.vstack([log_likelihood, log_likelihood[0] - 1000.0])
  joint = _masses() * likelihood
  expected = joint / joint.sum(axis=1, keepdims=True)

  post = posterior(log_likelihood, grid, normal_log_prior(_PRIOR_MEAN, _PRIOR_SD))

  np.testing.assert_allclose(post, np.vstack([expected, expected[0]]), rtol=1e-12, atol=1e-300)


def test_posterior_moments_match_a_brute_force_integral() -> None:
  """On the default grid each examinee's posterior mean and variance match a trapezoid integral on 20,001 points.

  The examinees answer three items of the 2PL to 4PL, one with a missing response, and one answers none. Fails if the
  posterior leaves out the prior or the likelihood, or if the mean or the variance is taken over the wrong axis.
  """
  items = np.array([[1.2, -0.5, 0.0, 1.0], [0.8, 0.3, 0.2, 1.0], [1.6, 1.0, 0.1, 0.95]])
  responses = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, np.nan], [np.nan, np.nan, np.nan]])
  grid = QuadratureGrid.uniform()

  post = posterior(_log_likelihood(responses, items, grid.nodes), grid, normal_log_prior())

  fine = np.linspace(-6.0, 6.0, 20001)
  density = np.exp(_log_likelihood(responses, items, fine) - fine**2 / 2)
  density /= np.trapezoid(density, fine, axis=1)[:, np.newaxis]
  mean = np.trapezoid(fine * density, fine, axis=1)
  variance = np.trapezoid((fine - mean[:, np.newaxis]) ** 2 * density, fine, axis=1)
  np.testing.assert_allclose(posterior_mean(post, grid.nodes), mean, rtol=0, atol=1e-7)
  np.testing.assert_allclose(posterior_variance(post, grid.nodes), variance, rtol=1e-6)


def test_posterior_mean_and_variance_are_taken_per_examinee() -> None:
  """Each row's mean and variance over the nodes, worked by hand.

  Rows: symmetric about 0 (mean 0, variance 0.5); skewed (mean -0.2 + 0.5 = 0.3, variance 0.7 - 0.3^2 = 0.61); a point
  mass at 1 (mean 1, variance 0). Fails if either is taken over the wrong axis, or the variance is centred on 0 or
  another row's mean.
  """
  nodes = np.array([-1.0, 0.0, 1.0])
  post = np.array([[0.25, 0.5, 0.25], [0.2, 0.3, 0.5], [0.0, 0.0, 1.0]])

  np.testing.assert_allclose(posterior_mean(post, nodes), [0.0, 0.3, 1.0], rtol=0, atol=1e-15)
  np.testing.assert_allclose(posterior_variance(post, nodes), [0.5, 0.61, 0.0], rtol=0, atol=1e-15)


@pytest.mark.parametrize(
  ("log_prior", "match"),
  [
    (_nan_at_first_node, r"log-prior is NaN or \+inf"),
    (_positive_infinity_at_first_node, r"log-prior is NaN or \+inf"),
    (_zero_everywhere, "prior has no mass at any node"),
  ],
)
def test_a_prior_that_is_nan_infinite_or_zero_everywhere_on_the_grid_is_an_error(
  log_prior: LogPrior, match: str
) -> None:
  """A log-prior that is NaN or +inf at a node, or -inf at every node, leaves no masses: an error, not NaN masses."""
  grid = QuadratureGrid.uniform()

  with pytest.raises(ValueError, match=match):
    prior_masses(grid, log_prior)
  with pytest.raises(ValueError, match=match):
    posterior(np.zeros((2, grid.nodes.size)), grid, log_prior)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_log_likelihood_that_is_nan_or_positive_infinity_is_an_error(value: float) -> None:
  """A NaN or +inf log-likelihood is an error naming the examinee's row, rather than a row of NaN."""
  grid = QuadratureGrid.uniform()
  log_likelihood = np.zeros((3, grid.nodes.size))
  log_likelihood[1, 5] = value

  with pytest.raises(ValueError, match=r"log-likelihood is NaN or \+inf for the examinees in rows \[1\]"):
    posterior(log_likelihood, grid, normal_log_prior())


def test_an_examinee_whose_posterior_has_no_mass_is_an_error() -> None:
  """A likelihood that is 0 wherever the prior has mass is an error naming the examinee's row.

  Neither the prior nor that likelihood is 0 at every node alone, so a check of either alone misses it.
  """
  grid = QuadratureGrid.uniform(5, -2.0, 2.0)
  log_likelihood = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -np.inf, -np.inf, -np.inf, 0.0]])

  with pytest.raises(ValueError, match=r"no mass at any node for the examinees in rows \[1\]"):
    posterior(log_likelihood, grid, uniform_log_prior(-1.0, 1.0))


@pytest.mark.parametrize("shape", [(121,), (2, 120), (2, 121, 1)])
def test_a_log_likelihood_that_is_not_examinees_by_nodes_is_an_error(shape: tuple[int, ...]) -> None:
  """A log-likelihood that is not one row per examinee and one column per node is an error, not a broadcast."""
  with pytest.raises(ValueError, match="one row per examinee and one column per node"):
    posterior(np.zeros(shape), QuadratureGrid.uniform(), normal_log_prior())


def test_an_empty_batch_has_an_empty_posterior_and_moments() -> None:
  """No examinees give a posterior of shape (0, nodes) and moments of shape (0,), rather than an error.

  Fails if a row's maximum or sum is taken over the whole array, a reduction NumPy refuses on an empty array.
  """
  grid = QuadratureGrid.uniform()

  post = posterior(np.zeros((0, grid.nodes.size)), grid, normal_log_prior())

  assert post.shape == (0, grid.nodes.size)
  assert posterior_mean(post, grid.nodes).shape == (0,)
  assert posterior_variance(post, grid.nodes).shape == (0,)
