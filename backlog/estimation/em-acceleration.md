---
title: Measure EM acceleration on the 3PL and 4PL and add it if it pays
kind: feature
status: blocked
priority: 3
effort: S
blocked_by: [estimation/mml-em-4pl]
raised_by: [ADR-0007]
---

# Measure EM acceleration on the 3PL and 4PL and add it if it pays

## Problem

`pyitemcal.mml` runs plain EM, which converges linearly and slowly near the solution (Dempster, Laird
& Rubin, 1977, p. 10; Bock & Aitkin, 1981, p. 447). ADR-0007 leaves acceleration out of the first
version, since the 2PL fits took under 0.1 seconds, and has it measured again on the 3PL and 4PL,
whose fits took hundreds to thousands of cycles in mirt (the 3PL and 4PL cards).

## Evidence

- Ramsay (1975) writes a fixed-point iteration $X = G(X)$ as $X = \theta X + (1 - \theta)G(X)$, where
  $0 < \theta < 1$ damps oscillation and $\theta < 0$ accelerates slow convergence (p. 340,
  Equation 5). His rule sets
  $\theta = 1 - (1 - \theta')\lVert\Delta X^{(k-2)}\rVert / \lVert\Delta^2 X^{(k-2)}\rVert$ from the
  first and second differences of the iterates, $\theta'$ being the value used over the previous three
  iterations (p. 341, Equation 10), and generalizes Aitken's $\delta^2$ to many parameters (p. 344).
  His program updates $\theta$ every third iteration, holds it in between, and sets no bound on it
  (p. 345; appendix, p. 358).
- mirt 1.48 applies it by default (manual p. 128) with four departures (source
  `R/EMstep.group.R:166, 287-291, 347-357`):
  - every third cycle it extrapolates once from the latest EM step, instead of holding $\theta$;
  - it computes $\theta$ afresh with the latest first difference in the numerator, which under
    Ramsay's model of a constant contraction (pp. 340-341, Equations 6-7) gives a $\theta$ larger by
    exactly 1, a shorter step (the card's own algebra);
  - it floors $\theta$ at $-5$;
  - it skips the step while the latest rise of the log-likelihood exceeds $\ln 100$ (it accelerates
    only when `Mrate`, $e$ to the minus that rise, is above `.01`).
- Thissen (1982, p. 181) reports the acceleration doubling or tripling the speed, but in its exact
  form it "frequently leads to over-acceleration and divergence", so he limits the parameter to about
  −2 to −4.
- SQUAREM is fully documented (Du & Varadhan, 2020, pp. 3-4). Beisemann et al. (2020, p. 232) find
  acceleration pays most for long tests and when the variance is fixed.
- A measurement (2026-11-27) on a scratch NumPy 2PL EM with an exact M-step (Fisher scoring to
  $10^{-12}$), each run stopped by mirt's rule on the unaccelerated EM step at `1e-6`; cycles and the
  largest error against plain EM converged to `1e-13`:

  | Acceleration | `LSAT7`, 1,000 × 5 | Simulated, 2,000 × 20, 10% missing | Simulated, 300 × 4, 30% missing |
  |---|---|---|---|
  | None | 76, $6.1 \times 10^{-6}$ | 38, $3.7 \times 10^{-6}$ | 19,555, $2.9 \times 10^{-3}$ |
  | mirt's variant of Ramsay's rule | 28, $2.7 \times 10^{-6}$ | 22, $2.1 \times 10^{-6}$ | 7,327, $2.9 \times 10^{-3}$ |
  | Ramsay's rule as published | 26, $5.0 \times 10^{-7}$ | 17, $7.4 \times 10^{-8}$ | diverged |

  On the slow case Ramsay's rule set $\theta = -5.5$ at cycle 3; the log-likelihood then fell, and
  the estimates were no longer finite at cycle 10. mirt's variant converged to the same estimates as
  plain EM, within $1.4 \times 10^{-11}$ when both ran to `1e-13`. In mirt itself, with its default
  BFGS M-step, Ramsay acceleration cut `LSAT7`'s cycles from 48 to 28 at `TOL = 1e-4` but only from
  100 to 96 at `1e-6`; with `control = list(reltol = 1e-14)` the counts became 46 to 16 and 77 to 31,
  close to the scratch EM's.

## What pyitemcal and catsim have now

`pyitemcal.mml.MMLCalibrator` runs plain EM, with no acceleration (ADR-0007). catsim has no EM.

## Proposal

When the 3PL and 4PL are implemented, measure the cycles and time of their fits with and without
mirt's bounded variant of Ramsay's rule, and with SQUAREM; add the one with a measured upside as an
option, defaulting to it only if the gain is large (`AGENTS.md`, choosing techniques). Ramsay's
unbounded rule is not a candidate: it diverged on the slow case.

## Tests

- Accelerated and plain EM reach the same estimates within the parity tolerance on the fixtures.
- With acceleration on, the log-likelihood at the end of each accelerated cycle is not below the
  previous cycle's, or the step is rejected.

## References

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

Ramsay, J. O. (1975). Solving implicit equations in psychometric data analysis. *Psychometrika,
40*(3), 337-360. https://doi.org/10.1007/BF02291762. Not openly available.

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.
