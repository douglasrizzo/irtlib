"""Comparison of calibrated models: the likelihood-ratio test, AIC, BIC and G²."""

from dataclasses import dataclass

import numpy
from scipy.stats import chi2

from .calibration import Calibration, _check_marginal, _complete_responses
from .mml import marginal_log_likelihood
from .typing import FloatArray


@dataclass(frozen=True, slots=True)
class ChiSquareTest:
  """A statistic referred to a chi-square distribution.

  Parameters
  ----------
  statistic : float
      The statistic.
  df : int
      Its degrees of freedom.
  p_value : float
      The probability of a statistic at least as large under the null model.
  """

  statistic: float
  df: int
  p_value: float


def aic(calibration: Calibration) -> float:
  """Return the Akaike information criterion of a calibration; smaller is better.

  Parameters
  ----------
  calibration : Calibration
      A calibration of the marginal likelihood; one with priors is judged by its log-likelihood without the
      log-prior.

  Returns
  -------
  float
      ``-2 * log_likelihood + 2 * n_parameters``.

  Raises
  ------
  ValueError
      If the calibration's likelihood is not marginal.
  """
  _check_marginal(calibration, "AIC")
  return -2 * calibration.log_likelihood + 2 * calibration.n_parameters


def bic(calibration: Calibration, n_examinees: int) -> float:
  """Return the Bayesian information criterion of a calibration; smaller is better.

  Parameters
  ----------
  calibration : Calibration
      A calibration of the marginal likelihood; one with priors is judged by its log-likelihood without the
      log-prior.
  n_examinees : int
      The number of examinees the calibration estimated its items from.

  Returns
  -------
  float
      ``-2 * log_likelihood + n_parameters * log(n_examinees)``.

  Raises
  ------
  ValueError
      If ``n_examinees`` is less than 1, or the calibration's likelihood is not marginal.
  """
  _check_marginal(calibration, "BIC")
  if n_examinees < 1:
    msg = f"n_examinees must be at least 1, got {n_examinees}"
    raise ValueError(msg)
  return -2 * calibration.log_likelihood + calibration.n_parameters * float(numpy.log(n_examinees))


def likelihood_ratio(restricted: Calibration, general: Calibration) -> ChiSquareTest:
  """Return the likelihood-ratio test of a restricted model against a more general one it is nested in.

  The two must be calibrations of the same responses, the restricted model a special case of the general one, such
  as the 1PL of the 2PL; only the numbers of items and parameters can be checked. The test of a parameter on the
  boundary of its range, such as the 3PL's guessing parameter at 0 against the 2PL, is the ordinary one, whose
  p-value is then conservative. Calibrations with priors are compared by their log-likelihoods without the log-prior,
  and a restricted model that fits better, as when a calibration stopped short, gives a negative statistic and a
  p-value of 1.

  Parameters
  ----------
  restricted : Calibration
      The calibration of the restricted model, of the marginal likelihood.
  general : Calibration
      The calibration of the general model, of the marginal likelihood.

  Returns
  -------
  ChiSquareTest
      Twice the gain in log-likelihood, on the difference in the numbers of parameters.

  Raises
  ------
  ValueError
      If the two calibrations have different numbers of items, the general model does not have more parameters, or a
      calibration's likelihood is not marginal.
  """
  for calibration in (restricted, general):
    _check_marginal(calibration, "the likelihood-ratio test")
  restricted_items, general_items = restricted.estimates.shape[0], general.estimates.shape[0]
  if restricted_items != general_items:
    msg = f"the calibrations have {restricted_items} and {general_items} items"
    raise ValueError(msg)
  df = general.n_parameters - restricted.n_parameters
  if df < 1:
    msg = (
      f"the general model must have more parameters than the restricted one, got {general.n_parameters} and "
      f"{restricted.n_parameters}"
    )
    raise ValueError(msg)
  statistic = 2 * (general.log_likelihood - restricted.log_likelihood)
  return ChiSquareTest(statistic=statistic, df=df, p_value=float(chi2.sf(statistic, df)))


def g2(calibration: Calibration, responses: FloatArray) -> ChiSquareTest:
  """Return the :math:`G^2` statistic of a calibration against the general multinomial over response patterns.

  The patterns' probabilities are those of every item at its estimates, including the items the calibration's bank
  leaves out.

  Parameters
  ----------
  calibration : Calibration
      The calibration to evaluate, of the marginal likelihood.
  responses : numpy.ndarray
      The complete response matrix the calibration estimated its items from: one row per examinee, one column per
      item, 1 or 0.

  Returns
  -------
  ChiSquareTest
      :math:`G^2`, on the number of possible response patterns less one, less the number of estimated parameters.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses fail :py:func:`irtlib.calibration.validate_responses`, have a missing response, or do not
      have one column per item; the model leaves no degrees of freedom; or the calibration's likelihood is not
      marginal.
  """
  grid = _check_marginal(calibration, "G2")
  matrix = _complete_responses(
    calibration, responses, "G2", "the multinomial over patterns is not defined with missing responses"
  )
  n_items = calibration.estimates.shape[0]
  df = 2**n_items - 1 - calibration.n_parameters
  if df < 1:
    msg = (
      f"the model leaves no degrees of freedom: {2**n_items - 1} patterns less {calibration.n_parameters} parameters"
    )
    raise ValueError(msg)
  _, counts = numpy.unique(matrix, axis=0, return_counts=True)
  log_likelihood = marginal_log_likelihood(matrix, calibration.estimates, grid)
  statistic = 2 * (float(numpy.sum(counts * numpy.log(counts / matrix.shape[0]))) - log_likelihood)
  return ChiSquareTest(statistic=statistic, df=df, p_value=float(chi2.sf(statistic, df)))
