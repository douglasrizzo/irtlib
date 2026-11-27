---
title: Compute the standard errors of item estimates from the observed information
kind: technique
status: blocked
priority: 1
effort: M
blocked_by: [discovery/requirements, estimation/mml-em]
raised_by: [research pass]
---

# Compute the standard errors of item estimates from the observed information

## Problem

ADR-0001 commits pyitemcal to standard errors of its item estimates, and the EM algorithm does not
produce them: "standard errors of the estimated item parameters ... are not provided by the EM
solution" (Bock & Aitkin, 1981, p. 447). No card carried over from catsim covered them. Without
them a calibration cannot say how precise its estimates are, and recovery tests cannot set their
tolerances.

## Evidence

- Bock and Aitkin (1981, p. 447) propose inverting an information matrix approximated from sampled
  response patterns; the methods compared since then work on the EM solution directly (below).
- mirt 1.48 computes standard errors only when asked (`SE = FALSE` by default) and then by Oakes's
  identity (`SE.type = "Oakes"`; manual p. 122; source `R/mirt.R:1582-1584`).

## What pyitemcal and catsim have now

- catsim: standard errors of ability estimates (`catsim.irt.see`).
- pyitemcal: nothing; the E-step and the per-item gradients this card builds on are
  [the MML-EM card](mml-em.md) and [the ICC-gradients card](../irt/icc-gradients.md).

## Proposal

Standard errors from the inverse of the observed information matrix at the MML estimate, computed by
Oakes's identity with a central difference, as mirt does by default; reported for the parameters
pyitemcal estimates and for catsim's $a$ and $b$ by the delta method. The open questions are the
requirements card's to settle; the card assumes the recommendation of each until then.

## Tests

- **Parity with mirt 1.48:** `SE = TRUE` with `SE.type = "Oakes"` and with `SE.type = "Louis"` (manual p. 125) on
  `LSAT6` and `LSAT7` and on a simulated matrix with missing entries, for the 2PL and the
  equal-slope 2PL, with fixtures recorded at `TOL = 1e-8` (the MML-EM card's measurement).
- **Two methods agree:** the central-difference Oakes approximation agrees with Louis's exact form;
  Chalmers (2018, p. 13) reports a mean relative difference of the order of $10^{-11}$ on a
  32-item test.
- **Brute force:** for 3 items, where all 8 response patterns can be enumerated, the information
  equals the negative second derivative of the marginal log-likelihood computed by finite
  differences.
- **Coverage** (recovery, ADR-0003): across replications drawn from known parameters, the standard
  deviation of the estimates matches the mean standard error, and about 95% of the intervals
  $\hat\psi \pm 1.96\,\widehat{SE}$, for an item parameter $\psi$, contain the generating value.

## Open Questions

1. **Which method.** The candidates and what the sources found:
   - Oakes with central differences: mirt's default; a mean relative difference from Louis of
     $2.5 \times 10^{-11}$ (maximum $8.3 \times 10^{-10}$) at 3% of its time (Chalmers, 2018, p. 13).
   - Louis's exact form: the reference in Chalmers's comparisons, but the slowest (p. 13).
   - Supplemented EM: needs an unaccelerated, tightly converged EM history and was less accurate
     (Chalmers, 2018, pp. 8-9, 13; Pritikin, 2017, p. 7).
   - The cross-product: cheap, but only asymptotically equivalent and conservative (Chalmers, 2018,
     p. 2).
   - The expected (Fisher) information: "often becomes too computationally demanding" (p. 2); its
     sum runs over all $2^n$ response patterns (Bock & Aitkin, 1981, p. 446).

   Assumed: Oakes with central differences, mirt's default, matched by the parity tests.
2. **Step size and symmetry.** mirt uses $\delta = 10^{-5}$ and symmetrizes the missing-information
   matrix (source `R/utils.R:1769-1770`, `R/EMstep.group.R:575`). Chalmers (2018, p. 17) finds
   symmetrizing improves the central and Richardson estimates. Assumed: as mirt.
3. **When the information matrix cannot be inverted.** mirt warns "Could not invert information
   matrix; model may not be (empirically) identified." and reports no standard errors when the
   inversion fails or gives a negative or missing variance (source `R/utils.R:1945-1959`). Options:
   the same, as a flag on the result; or an error. The requirements card's convergence question
   decides the form.
4. **The parameterization of the reported errors.** mirt reports the intercept-slope parameters and
   converts to $a$, $b$ by the delta method on request (source `R/SingleGroup-methods.R:1902-1915`).
   With intercept $\zeta$ and $b = -\zeta/a$ the Jacobian is a closed form,
   $\partial b/\partial\zeta = -1/a$ and $\partial b/\partial a = \zeta/a^2$ (the card's own algebra). Depends on the ICC-gradients card's first
   open question.
5. **Misspecification.** The sandwich estimator (Zeileis, 2006, p. 2) stays valid under some
   misspecification provided the estimates remain consistent (p. 3); mirt offers it as
   `"sandwich"` (manual p. 125). Assumed: not in the first version.

---

**Module:** decided by the requirements card (planned)
**Reference:** Louis (1982); Chalmers (2018)
**Last validated:** 2026-11-27, against Bock & Aitkin (1981), Louis (1982), Meng & Rubin (1991),
Chalmers (2018), Pritikin (2017) and Zeileis (2006) on the paper shelf, and mirt 1.48 (reference
manual and source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

A maximum likelihood estimate's sampling variance is, in large samples, the inverse of the observed
information, the negative Hessian of the log-likelihood at the estimate (Meng & Rubin, 1991, p. 901,
Equations 2.3.1-2.3.2). MML maximizes a marginal likelihood that EM never differentiates twice.
Each method below recovers the observed information from quantities EM already computes, at
different costs and accuracies.

## Definition

**The missing-information principle.** With observed data $y$ and complete data $x$, the observed
information is the complete-data information minus the missing information,

$$
I_Y = I_X - I_{X \mid Y}
$$

(Louis, 1982, p. 227, Equation 3.3). Louis writes it as the conditional expectation of the
complete-data observed information, less the conditional covariance of the complete-data score
(p. 227, Equation 3.2); in MML the complete data are the responses with the abilities on the grid,
and the conditional expectations are the E-step's posterior weights.

**Oakes's identity.** With $Q(\psi' \mid \psi)$ the expected complete-data log-likelihood of the
E-step at $\psi$ (here $Q$ is Dempster et al.'s function, not the number of quadrature nodes),

$$
\frac{\partial^2 L(\psi; y)}{\partial\psi\,\partial\psi^\top}
= \left\{ \frac{\partial^2 Q(\psi' \mid \psi)}{\partial\psi'\,\partial\psi'^\top}
+ \frac{\partial^2 Q(\psi' \mid \psi)}{\partial\psi'\,\partial\psi^\top} \right\}_{\psi' = \psi}
$$

(Chalmers, 2018, p. 3, Equation 1, who attributes the identity to Oakes). The first term is the complete-data Hessian,
the M-step's own; the second differentiates the M-step gradient with respect to the parameters of
the E-step. It holds at any $\psi$, not only at the maximum (Chalmers, 2018, p. 7).

**Numerical approximation.** The second term is approximated by recomputing the E-step at perturbed
parameters and differencing the complete-data gradient: forward differences have $O(\Delta)$ error,
central differences $O(\Delta^2)$, and Richardson extrapolation refines central differences over
successively smaller steps (Chalmers, 2018, pp. 5-6). Central differences cost $2p$ E-steps for $p$
parameters (p. 7).

**Supplemented EM.** The rate matrix $DM$ of EM's map satisfies $DM = I_{om}I_{oc}^{-1}$, so
$V = I_{oc}^{-1}(I - DM)^{-1}$ (Meng & Rubin, 1991, p. 901, Equations 2.2.1, 2.4.5-2.4.6), with $DM$
estimated from the EM iteration history (p. 902, Section 3.3).

**The sandwich.** $S = B\,M\,B$, with $B$ the bread from the Hessian and $M$ the variance of the
estimating functions (Zeileis, 2006, p. 2, Equations 4-6). In a correctly specified ML model the bread
alone suffices (p. 3).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - method
     - Open Question 1
     - How the observed information is computed; mirt 1.48 defaults to Oakes's identity with central
       differences.
   * - step
     - Open Question 2
     - The difference step; mirt 1.48 uses 1e-5.
```

## Behavior and Comparison

Pritikin (2017, p. 7) finds Oakes better than every supplemented-EM variant in both accuracy and speed;
the Oakes approximation in that study used forward differences. Chalmers (2018, p. 17) finds the forward
difference "visibly less accurate" than the central difference and Richardson extrapolation in every
condition, and those two practically equivalent. On a 32-item test his timings were 15.856 s for
Louis, 7.128 s for supplemented EM, 1.912 s for Richardson and 0.478 s for central differences
(p. 13).

mirt 1.48 computes these with `SE = TRUE`:

- It approximates Oakes's identity by central differences with $\delta = 10^{-5}$ and symmetrizes the
  missing information (source `R/utils.R:1769-1770, 1786`; `R/EMstep.group.R:575`).
- It zeroes the missing information of the latent mean and variance (`R/EMstep.group.R:578-581`),
  which concerns the `"Rasch"` form, where the variance is estimated.
- It reports no standard errors, with a warning, when the information cannot be inverted
  (`R/utils.R:1945-1959`).
- It converts to $a$, $b$ by the delta method in `coef(..., IRTpars = TRUE)`
  (`R/SingleGroup-methods.R:1902-1915`).

## Guarantees

1. Every reported standard error is the square root of a diagonal element of the inverse observed
   information at the reported estimate.
2. When the information matrix cannot be inverted, the result says so and reports no standard errors.

## References

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2018). Numerical approximation of the observed information matrix with Oakes'
identity. *British Journal of Mathematical and Statistical Psychology, 71*(3), 415-436.
https://doi.org/10.1111/bmsp.12127. Open copy:
https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/bmsp.12127. Pages cited are those of the
Early View version, 1-22.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *Journal of
the Royal Statistical Society: Series B, 44*(2), 226-233.
https://doi.org/10.1111/j.2517-6161.1982.tb01203.x. Not openly available; Monroe and Cai (2013,
p. 14) describe the method.

Meng, X.-L., & Rubin, D. B. (1991). Using EM to obtain asymptotic variance-covariance matrices: The
SEM algorithm. *Journal of the American Statistical Association, 86*(416), 899-909.
https://doi.org/10.1080/01621459.1991.10475130. Not openly available; Pritikin (2016, pp. 4-7)
describes the method.

Monroe, S., & Cai, L. (2013). *Estimation of a Ramsay-curve item response theory model by the
Metropolis-Hastings Robbins-Monro algorithm* (CRESST Report 834). University of California, Los
Angeles. Open copy: https://files.eric.ed.gov/fulltext/ED555725.pdf

Pritikin, J. N. (2016). *A computational note on the application of the supplemented EM algorithm to
item response models* (arXiv:1605.00860). https://arxiv.org/abs/1605.00860. Open access.

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response
models in an expectation-maximization framework. *Cogent Psychology, 4*(1), 1279435.
https://doi.org/10.1080/23311908.2017.1279435. Open access.

Zeileis, A. (2006). Object-oriented computation of sandwich estimators. *Journal of Statistical
Software, 16*(9), 1-16. https://doi.org/10.18637/jss.v016.i09. Open access.

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`). The quantities it needs come from the
MML-EM E-step and M-step: the posterior weights, the complete-data gradient and Hessian per item.
