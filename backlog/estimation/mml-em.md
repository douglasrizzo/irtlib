---
title: Calibrate the 1PL and 2PL by marginal maximum likelihood with the EM algorithm
kind: technique
status: blocked
priority: 1
effort: M
blocked_by: [discovery/requirements, irt/icc-gradients]
raised_by: [moved from catsim's backlog]
---

# Calibrate the 1PL and 2PL by marginal maximum likelihood with the EM algorithm

## Problem

pyitemcal estimates no item parameters yet. Marginal maximum likelihood (MML) with the EM algorithm
of Bock and Aitkin (1981) is the method mirt uses by default and the one the other estimation cards
build on: the 3PL and 4PL cards extend its M-step, and the standard-errors card works on its
solution. The 1PL and 2PL come first because their M-step is the simplest.

## Evidence

- The research pass (2026-11-27) checked the card carried over from catsim against Bock and Aitkin
  (1981), Baker (1987), Thissen (1982) and mirt 1.48, and found:
  - its E-step applied the $N(0, 1)$ density twice, through Gauss-Hermite weights already rescaled to
    the normal (Bock & Aitkin, 1981, p. 446) and again through $g(X_q)$;
  - its bounds $a \in [0.1, 5]$ and $b \in [-6, 6]$ and its 40 Gauss-Hermite nodes had no source;
    mirt leaves the slope and intercept unbounded and uses 61 equally spaced nodes (Behavior and
    Comparison);
  - its "1PL" fixed every slope at 1 without saying how the scale is identified (below);
  - its tests expected the gradient to vanish at the generating parameters, which holds only at the
    maximum likelihood estimate.
- A measurement on mirt 1.48 (2026-11-27), on its `LSAT6` data, which are Thissen's (1982, p. 179,
  Table 1) 1,000 examinees: the two equivalent forms of the 1PL below converge at different speeds,
  and mirt's default stopping rule halts the slower one short of the maximum.

  | mirt call | `TOL` | EM cycles | $\hat a$ or $\hat\sigma$ | $\hat b_1$ | log-likelihood |
  |---|---|---|---|---|---|
  | 2PL, slopes constrained equal | 1e-4 (default) | 12 | 0.75541 | −3.6146 | −2466.937614 |
  | `itemtype = "Rasch"` (slopes 1, variance free) | 1e-4 (default) | 80 | 0.75618 | −3.6111 | −2466.937714 |
  | 2PL, slopes constrained equal | 1e-8 | 37 | 0.75513 | −3.6153 | −2466.937600 |
  | `itemtype = "Rasch"` | 1e-8 | 168 | 0.75514 | −3.6152 | −2466.937600 |

  For the Rasch form, $\hat b = -\hat d/\hat\sigma$. The equal-slope fit reproduces Thissen's (1982,
  p. 180, Table 2) estimates to the three decimals printed.

## What pyitemcal and catsim have now

- catsim: `catsim.estimation.QuadratureGrid` (any nodes and positive weights; `uniform()` gives
  equally spaced nodes with equal weights, 121 on $[-6, 6]$ by default), `normal_log_prior`, and a
  posterior over a grid for one examinee's responses; `catsim.irt.icc` and
  `log_response_probabilities`.
- pyitemcal: nothing; the derivatives with respect to item parameters are
  [the ICC-gradients card](../irt/icc-gradients.md).

## Proposal

MML-EM for the 2PL and for the 1PL with a common, estimated slope, from a response matrix with
missing entries, on an equally spaced grid with a normal prior, with standard errors from
[the standard-errors card](standard-errors.md). The open questions below are the choices the
requirements card settles; the card assumes the recommendation of each until then.

## Tests

- **Published values:**
  - Thissen (1982, p. 180, Table 2), LSAT-6 (Table 1, p. 179), 1PL with a common slope and $N(0, 1)$:
    $\hat a = 0.755$, $\hat b = (-3.615, -1.322, -0.318, -1.730, -2.780)$, to the three decimals printed.
  - Bock and Aitkin (1981, p. 455, Table 3), LSAT-6 and LSAT-7, the same model with difficulties
    centred to sum to zero, labelled "10-pt. normal prior": the MML column. Their scale "is free and
    is determined by the logit function" (p. 454). Thissen's (1982, p. 180) "Standardized MML"
    column equals $\hat a(\hat b_i - \bar b)$ (the card's arithmetic), and for LSAT-6 it agrees with
    Bock and Aitkin's column within 0.001, so the comparison is with $\hat a(\hat b_i - \bar b)$.
- **Parity with mirt 1.48** (fixtures recorded with `TOL = 1e-8`, since the default stops short,
  as measured above): 2PL and equal-slope 2PL on `LSAT6` and `LSAT7`, and on a simulated matrix with
  missing entries; estimates, log-likelihood, and the log-likelihood at fixed parameters.
- **Recovery** (ADR-0003): responses drawn from known 1PL and 2PL parameters with a fixed seed;
  every estimate within a multiple of its standard error, set by the sample size.
- **Monotone likelihood:** without acceleration, the marginal log-likelihood never decreases between
  iterations (Dempster, Laird & Rubin, 1977, p. 7, Theorem 1).
- **Missing responses:** on a 3-item, 4-examinee matrix with one missing entry, the marginal
  log-likelihood at fixed parameters equals the brute-force sum over the grid of the product over
  answered items only.

## Open Questions

1. **How the 1PL fixes its scale.**
   - Options: a common slope estimated with $N(0, 1)$ (Thissen, 1982, p. 175); slopes fixed at 1
     with the variance estimated (mirt's `"Rasch"`, mirt 1.48 manual, p. 123); slopes fixed at 1
     with $N(0, 1)$, a more restrictive model (mirt's `"1PL"`, manual pp. 123, 133).
   - The first two are the same model, because $a(\theta - b)$ with $\theta \sim N(0, 1)$ is
     $a\theta - ab$ with $a\theta \sim N(0, a^2)$ (the card's own step); they reach the same maximum
     in the measurement (Evidence). Thissen (1982, p. 175) makes the related observation that "the
     square of the estimated slope is the same as the variance of the population distribution for
     conventionally standardized CML estimates".
   - The common slope converged in about a fifth of the EM cycles in the measurement, and acceleration
     helps less when the variance is estimated (Beisemann, Wartlick & Doebler, 2020, p. 232).
   - Assumed: the common slope, with the fixed-slope model expressed as a fixed parameter
     (requirements question 8).
2. **The grid.**
   - Gauss-Hermite and equally spaced points both appear in Bock and Aitkin (1981, pp. 445, 449);
     Thissen (1982, p. 179) finds 11 points on $[-5, 5]$ adequate for the 1PL; Woodruff and Hanson
     (1996, p. 21) give about 20 as the minimum for the 2PL and 3PL; mirt uses 61 on $[-6, 6]$.
   - Assumed: mirt's grid, 61 equally spaced nodes on $[-6, 6]$ (`QuadratureGrid.uniform(61)`), with
     `normal_log_prior()` giving the nodes their normal masses, so parity needs no setting.
3. **The stopping rule.**
   - Bock and Aitkin (1981, p. 447) stop when the estimates are "stable to the required number of
     places"; mirt stops when every parameter changes by less than `TOL` = 1e-4.
   - EM converges linearly, at the rate of the largest eigenvalue of its Jacobian (Dempster et al.,
     1977, p. 10). A small change per cycle therefore does not bound the distance to the maximum when
     that rate is close to 1, as the measurement shows.
   - Options: a parameter-change rule with a tighter tolerance; a log-likelihood-change rule; or a
     rule on the gradient of the marginal log-likelihood, which is zero at the maximum.
   - Not assumed: the requirements card decides, with measurements.
4. **Acceleration.**
   - Ramsay (1975) writes a fixed-point iteration $X = G(X)$ as $X = \theta X + (1 - \theta)G(X)$,
     where $0 < \theta < 1$ damps oscillation and $\theta < 0$ accelerates slow convergence (p. 340,
     Equation 5). His rule sets
     $\theta = 1 - (1 - \theta')\lVert\Delta X^{(k-2)}\rVert / \lVert\Delta^2 X^{(k-2)}\rVert$ from the
     first and second differences of the iterates, $\theta'$ being the value used over the previous
     three iterations (p. 341, Equation 10), and generalizes Aitken's $\delta^2$ to many parameters
     (p. 344). His program updates $\theta$ every third iteration, holds it in between, and sets no
     bound on it (p. 345; appendix, p. 358).
   - mirt applies it by default (manual p. 128) with four departures (source
     `R/EMstep.group.R:166, 287-291, 347-357`):
     - every third cycle it extrapolates once from the latest EM step, instead of holding $\theta$;
     - it computes $\theta$ afresh with the latest first difference in the numerator, which under
       Ramsay's model of a constant contraction (pp. 340-341, Equations 6-7) gives a $\theta$ larger by
       exactly 1, a shorter step (the card's own algebra);
     - it floors $\theta$ at $-5$;
     - it skips the step while the latest rise of the log-likelihood exceeds $\ln 100$ (it
       accelerates only when `Mrate`, $e$ to the minus that rise, is above `.01`).

     Thissen (1982, p. 181) reports the acceleration doubling or tripling the speed, but in its exact
     form it "frequently leads to over-acceleration and divergence", so he limits the parameter to
     about −2 to −4. The measurement below agrees: Ramsay's unbounded rule diverged on a slow case,
     and mirt's bounded variant cut the cycles by 1.7 to 2.7 times without changing the estimates.
   - SQUAREM is fully documented (Du & Varadhan, 2020, pp. 3-4). Beisemann et al. (2020, p. 232) find
     acceleration pays most for long tests and when the variance is fixed.
   - Assumed: no acceleration in the first version, measured before adding any (YAGNI). The fixed
     point does not depend on it, so parity holds either way.
5. **The M-step solver and parameterization.** Newton or Fisher scoring per item in the
   intercept-slope form (Baker, 1987, p. 114, Equation 8), or a general optimizer (mirt uses BFGS
   without bounds: manual p. 126; source `R/utils.R:1848`, `R/06-LoadPars.R:364-365`). See the ICC-gradients card's first open question.
6. **Starting values.** Bock and Aitkin (1981, p. 458) name the inverse normal of the item's
   proportion correct as a suitable starting intercept, which mirt uses (Behavior and Comparison); Thissen (1982, p. 178) started at $b = 0$,
   $a = 1$. Assumed: mirt's, for comparable iteration counts.
7. **Items no estimate exists for.** An item answered correctly by everyone or by no one has no
   finite estimate, and MML does not prevent diverging slopes (Baker, 1987, p. 133, quoting
   Swaminathan & Gifford). mirt's manual advises removing items with a mean above .95, or using
   priors (p. 132). Options: an error, a warning with the item flagged, or priors (the 3PL card).
8. **The latent distribution.** Bock and Aitkin (1981, p. 449, Equation 15) estimate it as a
   histogram on the grid; mirt offers it as `dentype = "empiricalhist"` (manual p. 126). Assumed:
   out of the first version; a card of its own if wanted.

---

**Module:** decided by the requirements card (planned)
**Reference:** Bock & Aitkin (1981); Thissen (1982)
**Last validated:** 2026-11-27, against Bock & Aitkin (1981), Baker (1987), Thissen (1982),
Dempster, Laird & Rubin (1977), Ramsay (1975), Beisemann, Wartlick & Doebler (2020), Du & Varadhan
(2020), Woodruff & Hanson (1996) and Engelen (1987) on the paper shelf, and mirt 1.48 (reference
manual and source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

Calibration has to estimate item parameters without knowing the examinees' abilities. Estimating
each ability jointly with the items is not generally possible, "because the number of parameters
increases with the number of subjects and standard limit theorems do not apply" (Bock & Aitkin,
1981, p. 443). Marginal maximum likelihood instead integrates the abilities out over a population
distribution and maximizes the likelihood of the item parameters alone (p. 444). Bock and Aitkin
reformulated its likelihood equations into a solution feasible "for both small and large numbers of
items" (p. 444), in which each cycle solves one small problem per item (p. 447).

For the 1PL, MML and conditional maximum likelihood give nearly the same estimates when the model
fits; neither has the bias of joint estimation for this model (Bock & Aitkin, 1981, p. 454), and MML
keeps examinees with every item right or every item wrong (Engelen, 1987, p. 25).

## Definition

Let examinee $j$'s responses be $u_{ij} \in \{0, 1\}$ for the items $i$ they answered, and $P_i$
catsim's response function with $c = 0$ and $d = 1$. Ability has density $g$, approximated on nodes
$X_q$ with prior masses $A_q$ that sum to 1, $q = 1, \dots, Q$ (Bock & Aitkin, 1981, pp. 445,
449). The marginal probability of examinee $j$'s responses is

$$
\tilde P_j = \sum_{q=1}^{Q} L_j(X_q)\,A_q, \qquad
L_j(X_q) = \prod_{i \text{ answered by } j} P_i(X_q)^{u_{ij}}\bigl(1 - P_i(X_q)\bigr)^{1 - u_{ij}},
$$

and the marginal log-likelihood is $\sum_j \log \tilde P_j$ (Bock & Aitkin, 1981, pp. 445-446,
Equations 6 and 8, written per examinee rather than per response pattern). $A_q$ carries the prior
once: with Gauss-Hermite points it is the rescaled weight (p. 446), with equally spaced points the
normalized density (p. 449). Bock and Aitkin's model is the normal ogive; they state that the logit
model works the same way (p. 448), and Baker (1987, p. 132) gives it in logistic form.

**E-step.** With the current item parameters, accumulate for each item $i$ and node $q$

$$
\bar r_{iq} = \sum_{j \text{ answered } i} \frac{u_{ij}\,L_j(X_q)\,A_q}{\tilde P_j}, \qquad
\bar N_{iq} = \sum_{j \text{ answered } i} \frac{L_j(X_q)\,A_q}{\tilde P_j},
$$

the expected number of correct responses and the expected number of examinees answering item $i$ at
$X_q$ (Baker, 1987, p. 132, Equations 42-44; Bock & Aitkin, 1981, p. 447).

Departure: both sources define the counts over complete response patterns, with one expected
sample size per node shared by every item. With missing responses, the product and the sums run
over answered items only, so the expected sample size is per item. mirt treats a missing response
as contributing no factor (mirt 1.48 source, `R/04-PrepData.R:107`). Beisemann et al. (2020,
p. 211) count the complete-data sample size per item, the persons at a level "attempting item $j$"
(Equation 5), though their expected count (Equation 6) carries no item index.

**M-step.** For each item, maximize

$$
\ell_i = \sum_q \left[\bar r_{iq}\log P_i(X_q) + (\bar N_{iq} - \bar r_{iq})\log\bigl(1 - P_i(X_q)\bigr)\right],
$$

whose likelihood equations in the intercept-slope form are $\sum_q [\bar r_{iq} - \bar N_{iq}P_i(X_q)]
= 0$ and $\sum_q [\bar r_{iq} - \bar N_{iq}P_i(X_q)]X_q = 0$ (Baker, 1987, p. 132, Equations 45-46;
Equation 46's $P_j$ is a misprint for $P_i$). The prior mass enters only through $\bar r$ and $\bar N$.
Bock and Aitkin's (1981, p. 447) description of the M-step says to weight each term by its
prior mass as well; their Equation 12 (p. 446) and definitions (ii)-(iii) (p. 447) apply it once, which this card
follows.

**The 1PL with a common slope.** All items share one slope $a$ and ability is $N(0, 1)$ (Thissen,
1982, pp. 175-176, Equation 1). The location equations stay per item; the slope's equation sums over
all items (p. 180, Equations 8-9). Thissen solves each location, then the slope, by one-dimensional
Newton steps within each cycle (p. 181).

**Convergence.** Each cycle that does not decrease the expected complete-data log-likelihood does not
decrease the marginal likelihood (Dempster et al., 1977, p. 7, Theorem 1). Convergence is linear,
at the rate of the largest eigenvalue of the algorithm's Jacobian at the maximum (p. 10), and becomes
very slow near the solution (Bock & Aitkin, 1981, p. 447). Bock and Aitkin "substitute the expected
values of the missing data ... into the log likelihood", where Dempster et al. take the expected
log-likelihood; both reduce to the same algorithm "when the model is within the exponential family
and the log likelihood is linear in the sufficient statistics" (p. 448). Beisemann et al. (2020,
p. 211) obtain the same expected counts from the posterior expected complete-data log-likelihood of
the 1PL and 2PL.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``model``
     - none
     - 1PL (common slope) or 2PL, as catsim's ``IrtModel``.
   * - ``grid``
     - Open Question 2
     - Nodes and prior masses of the ability distribution; mirt 1.48 uses 61 equally spaced nodes on
       [-6, 6] (manual pp. 127, 129).
   * - ``tolerance``, ``max_iter``
     - Open Question 3
     - When to stop; mirt 1.48 stops when every parameter changes by less than 1e-4, or after 500
       cycles (manual pp. 127, 129).
   * - starting values
     - Open Question 6
     - Where the iterations start.
```

## Behavior and Comparison

The estimates depend little on the shape of the assumed ability distribution: normal, rectangular
and empirical priors give nearly the same LSAT estimates (Bock & Aitkin, 1981, p. 449; Table 1,
p. 451). A flat prior can make the posterior heavy-tailed when many examinees answer every item
right or every item wrong (p. 450). EM converges slowly near the solution (p. 447), and MML does not
protect against slope estimates diverging (Baker, 1987, p. 133). An item answered correctly by every
examinee, or by none, has no finite estimate.

mirt 1.48 implements the method in `mirt(data, 1, itemtype = "2PL")` with `method = "EM"`:

- It estimates the intercept-slope form, which its manual writes $a_1\theta + d$ with $d$ the
  intercept, not catsim's upper asymptote (manual p. 133); slope and intercept have no bounds
  (source `R/06-LoadPars.R:364-365`).
- Its grid is 61 equally spaced points on $[-6, 6]$, the normal density at them divided by its sum
  (source `R/03-estimation.R:663`, `R/utils.R:588-589`).
- It solves each M-step with BFGS (manual p. 126; source `R/utils.R:1848`), through R's `optim`
  with that optimizer's own tolerance unless `control` sets one (manual p. 129), and at most
  `max(ceiling(Mrate * 50), 15)` iterations (source `R/EMstep.utils.R:91-93`). The M-step is
  therefore inexact, which the measurement below shows limits mirt's accuracy at tight tolerances.
- It stops when every parameter changes by less than `TOL` = 1e-4, or after 500 cycles (manual
  pp. 127, 129; source `R/utils.R:2929-2941`), and applies Ramsay acceleration every third cycle
  (manual p. 128; source `R/EMstep.group.R:347-357`; the fourth open question).
- It starts slopes at 0.851 and intercepts at the inverse normal of the proportion correct, rescaled
  (source `R/05-model.elements.R:42-64`).
- Its `"1PL"` fixes every slope at 1 with $N(0, 1)$; its `"Rasch"` fixes every slope at 1 and
  estimates the variance (manual pp. 123, 133). The common-slope 1PL is the 2PL with the slopes constrained
  equal.

Where pyitemcal's defaults differ, each difference gets a setting that reproduces mirt for the parity
tests (ADR-0003); which differences there are is the requirements card's to decide.

**Measured: acceleration** (2026-11-27). A scratch NumPy 2PL following this card, with an exact
M-step (Fisher scoring to $10^{-12}$), ran Ramsay's rule as published and as mirt applies it (the
fourth open question), each stopped by mirt's rule on the unaccelerated EM step at `1e-6`. The table
gives cycles and the largest error against plain EM converged to `1e-13`:

| Acceleration | `LSAT7`, 1,000 × 5 | Simulated, 2,000 × 20, 10% missing | Simulated, 300 × 4, 30% missing |
|---|---|---|---|
| None | 76, $6.1 \times 10^{-6}$ | 38, $3.7 \times 10^{-6}$ | 19,555, $2.9 \times 10^{-3}$ |
| mirt's variant of Ramsay's rule | 28, $2.7 \times 10^{-6}$ | 22, $2.1 \times 10^{-6}$ | 7,327, $2.9 \times 10^{-3}$ |
| Ramsay's rule as published | 26, $5.0 \times 10^{-7}$ | 17, $7.4 \times 10^{-8}$ | diverged |

On the slow case Ramsay's rule set $\theta = -5.5$ at cycle 3; the log-likelihood then fell, and the
estimates were no longer finite at cycle 10. mirt's variant converged to the same estimates as plain
EM, within $1.4 \times 10^{-11}$ when both ran to `1e-13`. In mirt 1.48 on `LSAT7`, with its default M-step, Ramsay acceleration cut the cycles
from 48 to 28 at `TOL = 1e-4` but only from 100 to 96 at `1e-6`, where both fits stopped
$1.5 \times 10^{-5}$ from mirt's estimate at `1e-12`. With `control = list(reltol = 1e-14)` the
counts became 46 to 16 and 77 to 31, close to the scratch EM's, and the errors at `1e-6` fell to
$6.7 \times 10^{-6}$ and $2.1 \times 10^{-6}$. mirt's estimate at `TOL = 1e-12` is then within
$4.2 \times 10^{-7}$ of the scratch EM's, against $1.4 \times 10^{-6}$ with the default M-step.

## Guarantees

1. Without acceleration, the marginal log-likelihood never decreases from one iteration to the next.
2. An examinee's missing response changes nothing but that response's own factor: an examinee who
   answered no item leaves the estimates unchanged.
3. The result says whether the stopping rule was met.
4. The 1PL's estimates are on the scale where ability has mean 0 and variance 1.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Beisemann, M., Wartlick, O., & Doebler, P. (2020). Comparison of recent acceleration techniques for
the EM algorithm in one- and two-parameter logistic IRT models. *Psych, 2*(4), 209-252.
https://doi.org/10.3390/psych2040018. Open access.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via
the EM algorithm. *Journal of the Royal Statistical Society: Series B, 39*(1), 1-22.
https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. Not openly available; Woodruff and Hanson
(1996, pp. 3-8) describe the algorithm.

Du, Y., & Varadhan, R. (2020). SQUAREM: An R package for off-the-shelf acceleration of EM, MM and
other EM-like monotone algorithms. *Journal of Statistical Software, 92*(7), 1-41.
https://doi.org/10.18637/jss.v092.i07. Open access.

Engelen, R. J. H. (1987). *A review of different estimation procedures in the Rasch model* (Research
Report 87-6). University of Twente. Open copy: https://files.eric.ed.gov/fulltext/ED310126.pdf

Ramsay, J. O. (1975). Solving implicit equations in psychometric data analysis. *Psychometrika,
40*(3), 337-360. https://doi.org/10.1007/BF02291762. Not openly available.

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.

Woodruff, D. J., & Hanson, B. A. (1996). *Estimation of item response models using the EM algorithm
for finite mixtures* (ACT Research Report 96-6). ACT. Open copy:
https://www.act.org/content/dam/act/unsecured/documents/ACT_RR96-06.pdf

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`), which settles the open questions above
against the calibrator contract.
