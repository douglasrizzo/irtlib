"""Tests of the result every calibrator shares: the bank it builds, and its likelihood, grid and abilities.

Expected values are hand-worked: which estimates break which rule of an item bank, and the names of the rest.
"""

import numpy as np
import pytest

from irtlib.ability import AbilityEstimates
from irtlib.calibration import (
  Calibration,
  Likelihood,
  _valid_items,  # noqa: PLC2701 - no calibrator gives estimates of which none is valid
  default_grid,
)
from irtlib.quadrature import QuadratureGrid


def test_estimates_of_which_none_is_valid_give_no_bank() -> None:
  """Estimates of which none is valid in a bank give no bank, with a warning naming each item and its rule.

  A bank of no items cannot be built, so a function that tried would raise rather than warn.
  """
  estimates = np.array([[-0.8, 0.0, 0.0, 1.0], [1.0, np.nan, 0.0, 1.0]])
  message = r"NaN or infinite parameters for 1 \('item_1'\); discrimination <= 0 for 0 \('item_0'\); .*miskeyed"
  with pytest.warns(RuntimeWarning, match=message):
    assert _valid_items(estimates) is None


def test_an_item_left_out_for_its_asymptotes_is_not_called_miskeyed() -> None:
  """An item whose guessing parameter exceeds its upper asymptote is left out, without being called miskeyed.

  Only a slope that is not positive is what a miskeyed item gives.
  """
  estimates = np.array([[1.0, 0.0, 0.3, 0.2], [1.2, 0.5, 0.0, 1.0]])
  with pytest.warns(RuntimeWarning, match=r"pseudo-guessing >= upper asymptote for 0 \('item_0'\)") as caught:
    bank = _valid_items(estimates)
  assert "miskeyed" not in str(caught[0].message)
  assert bank is not None
  assert bank.names == ("item_1",)
  np.testing.assert_array_equal(bank.items, estimates[[1]])


def _calibration(
  likelihood: Likelihood | str, grid: QuadratureGrid | None, abilities: AbilityEstimates | None = None
) -> Calibration:
  """Return a calibration of two items with a likelihood, a grid and abilities."""
  return Calibration(
    estimates=np.tile([1.0, 0.0, 0.0, 1.0], (2, 1)),
    items=None,
    covariances=None,
    log_likelihood=-1.0,
    log_likelihoods=np.array([-1.0]),
    likelihood=likelihood,  # ty: ignore[invalid-argument-type]
    n_parameters=1,
    item_parameters=np.ones(2, dtype=int),
    converged=True,
    change=0.0,
    grid=grid,
    abilities=abilities,
  )


@pytest.mark.parametrize(
  ("likelihood", "grid"),
  [(Likelihood.MARGINAL, None), (Likelihood.CONDITIONAL, default_grid()), (Likelihood.JOINT, default_grid())],
  ids=str,
)
def test_a_grid_comes_with_the_marginal_likelihood_only(likelihood: Likelihood, grid: QuadratureGrid | None) -> None:
  """A calibration of the marginal likelihood without a grid, or of the conditional or joint likelihood with one, is
  refused.

  Model comparison and item fit read a grid as the sign of a marginal log-likelihood, so a mismatch would let them
  compare a conditional or joint one or integrate over nodes the calibration never used.
  """
  with pytest.raises(ValueError, match="a grid exactly when its likelihood is marginal"):
    _calibration(likelihood, grid)


_ABILITIES = AbilityEstimates(np.zeros(1), np.ones(1), np.zeros(1, dtype=int), np.zeros(1, dtype=bool))


@pytest.mark.parametrize(
  ("likelihood", "grid", "abilities"),
  [
    (Likelihood.JOINT, None, None),
    (Likelihood.MARGINAL, default_grid(), _ABILITIES),
    (Likelihood.CONDITIONAL, None, _ABILITIES),
  ],
  ids=str,
)
def test_abilities_come_with_the_joint_likelihood_only(
  likelihood: Likelihood, grid: QuadratureGrid | None, abilities: AbilityEstimates | None
) -> None:
  """A calibration of the joint likelihood without abilities, or of the marginal or conditional likelihood with them,
  is refused.

  Only joint maximum likelihood estimates abilities with the items; abilities in another result would pass for
  estimates the method never made, and a joint result without them would lose the examinee parameters it estimated.
  """
  with pytest.raises(ValueError, match="abilities exactly when its likelihood is joint"):
    _calibration(likelihood, grid, abilities)


def test_a_likelihood_given_by_name_is_read_as_its_member() -> None:
  """A likelihood given as its name, "marginal", with a grid, is accepted and stored as ``Likelihood.MARGINAL``.

  A member equals its name, so a check by identity on the string would refuse the grid of a marginal calibration.
  """
  assert _calibration("marginal", default_grid()).likelihood is Likelihood.MARGINAL
