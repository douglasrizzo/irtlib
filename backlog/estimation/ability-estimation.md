---
title: Estimate the abilities of many examinees by ML, MAP, EAP and WLE
kind: feature
status: blocked
priority: 2
effort: L
blocked_by: [irt/response-functions]
raised_by: [ADR-0004 as decided on 2026-11-27]
---

# Estimate the abilities of many examinees by ML, MAP, EAP and WLE

## Problem

ADR-0001 makes ability estimation irtlib's: the mathematics of catsim's four estimators moves here,
as functions over a response matrix, and catsim keeps thin adapters for one examinee's session.
irtlib's mean-squares card needs EAP abilities for every examinee (`fit/mean-squares`).

## What irtlib and catsim have now

References are to catsim's `dev` at commit `97b89f5`.

| catsim | Mathematics, which moves | CAT policy, which stays |
|---|---|---|
| `EAPEstimator` (`estimation/bayesian.py:323-385`) | posterior mean and posterior standard deviation over the grid | its defaults (121 nodes, N(0, 1)), `evaluations`, `repr` |
| `MAPEstimator` (394-524) | a 201-point search of log-likelihood plus log-prior; the middle of three or more tied maxima; a bounded refinement kept only if no worse; the standard error from the information less the prior's curvature | ignoring the session's estimate, `evaluations` |
| `NumericalSearchEstimator` (`estimation/numerical.py:56-135`) | a bounded maximization of the log-likelihood, with the Fisher standard error | Dodd's step for constant patterns (`_dodd`, 14-53), the empty session |
| `WarmLikelihoodEstimator` (`estimation/wle.py:31-180`) | the root of Warm's modified score: a 121-point scan, `brentq` on each sign change, bound candidates, ranking by the integral of the score | the empty session, the warning's `stacklevel=3` |
| `BaseEstimator._check_tol`, `_check_bounds` (`estimation/base.py:44-91`) | copied as private checks | the `estimate(context)` contract |

- Tests: catsim's estimator tests (`tests/test_estimation.py`) are brute-force references (dense
  grids, an independent modified score, an enumerated bias) and behaviour tests, all built through
  a `StepContext`; every one is rewritten for matrix calls.
- catR parity: 49 of the 55 cases of `tests/parity/catr/estimation.R` are mathematics (ML 6, Bayes
  modal 17, EAP 16, WL 10), with their departures `ml/two_stationary_points`, `ml/rising_at_bound` and
  `wl/three_wl_roots`. The 6 Dodd and constant-pattern cases stay with catsim.
- mirt 1.48's `fscores` implements EAP, MAP, ML and WLE, so ADR-0003 asks for mirt cases too.
- catsim's times per CAT step for 20 items (a scratch script, 2026-11-27): EAP 57 µs, ML 74 µs,
  WLE 142 µs, MAP 152 µs.

## Proposal

1. An ability-estimation module beside `fit` and `mml` in the import-linter layers, above `irt`
   and the quadrature module, with one function per
   estimator taking items, a response matrix (NaN for missing) and the estimator's settings, and
   returning each examinee's estimate, its standard error and the number of objective evaluations,
   which catsim's adapters report.
2. The method's formulas and their sources go in the module docstring (`sphinx/AGENTS.md`); catsim's
   technique pages for these estimators hold its current derivations and citations, to re-cite from
   the shelf.
3. A warning raised inside these functions points at irtlib's caller, not at catsim's call depth.
4. catR's `estimation.R` cases and their departures come with the module, and new mirt `fscores`
   cases go in `tests/parity/mirt/`.
5. The estimators join the benchmarks of `maintenance/drop-catsim`, timed against catsim's; if
   those benchmarks do not exist yet, this card adds them.

## Tests

- The brute-force references of catsim's estimator tests, for many examinees at once, with missing
  responses; a row's estimate equals that examinee's estimate alone.
- catR and mirt parity as above.

## Open Questions

1. **A constant pattern under maximum likelihood**, all correct or all incorrect, has no finite
   estimate. mirt 1.48 gives +inf or −inf with no standard error (its source on the shelf,
   `R/fscores.internal.R:397-405`);
   catR's default returns the search bound (case `ml_constant_default`); catsim applies Dodd's step,
   which stays with catsim. The card assumes mirt's rule, with catR's bound recorded as a strict
   expected failure, until the maintainer decides.

## Sources

Bock and Mislevy (1982) for EAP, Magis and Raîche (2012) for catR's estimators and Warm (1985) for
the WLE are on the shelf. Warm (1989), which catsim cites, is with the maintainer to fetch.
