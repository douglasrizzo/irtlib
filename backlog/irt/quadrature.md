---
title: Bring the quadrature grid, the ability priors and the posterior from catsim into irtlib
kind: feature
status: ready
priority: 2
effort: S
blocked_by: []
raised_by: [ADR-0004 as decided on 2026-11-27]
---

# Bring the quadrature grid, the ability priors and the posterior from catsim into irtlib

## Problem

Calibration and ability estimation both integrate over ability on a grid, under a prior. irtlib
imports `catsim.estimation.QuadratureGrid` and `normal_log_prior`, and builds the grid's normalized
prior masses, and each examinee's posterior over the nodes, privately in two places.

## What irtlib and catsim have now

References are to catsim's `dev` at commit `97b89f5`.

- catsim `src/catsim/estimation/bayesian.py`:
  - `LogPrior` (27), `normal_log_prior` (47) and `uniform_log_prior` (98), with the private
    dataclasses that make them picklable and give them a readable `repr` (80-144);
  - `QuadratureGrid` (148-230), validated nodes and weights, with `uniform(n_nodes=121, low=-6,
    high=6)`;
  - `posterior` (233-282): log-prior plus log-likelihood plus log-weights at each node, shifted by
    its maximum and normalized, with errors for NaN, +inf and a posterior with no mass; for one
    examinee's Boolean responses;
  - `posterior_mean` (285) and `posterior_variance` (303).
- irtlib, the same quantities privately: `mml._log_masses`, `_log_joint` and `_posterior`
  (`src/irtlib/mml.py:216-227, 517-520`), and the masses again in `fit.s_x2`
  (`src/irtlib/fit.py:104-105`). For one 20-item examinee on 121 nodes they agree with catsim's
  `posterior` to a relative `1.5e-14`; the batched form takes 142 µs, catsim's 54 µs (a scratch
  script, 2026-11-27, which the benchmarks of `maintenance/drop-catsim` replace).

## Proposal

1. A module above `irt` and below `calibration` and `mml` in the import-linter layers (ADR-0006),
   added to the contract by this card, with
   `LogPrior`, `normal_log_prior`, `uniform_log_prior`, `QuadratureGrid`, a function giving the
   grid's normalized prior masses, and the posterior for many examinees. The posterior takes the
   log-likelihood at the nodes, shape (examinees, nodes), rather than responses and items, so it
   needs no response function and serves both the E-step and ability estimation; its mean and
   variance are taken per examinee.
2. Keep catsim's names, defaults and checks; `LogPrior` uses the Python 3.12 `type` statement. The
   ability range defaults come from `irt`'s `THETA_*_EXTENDED` constants.
3. Leave `mml` and `fit` on their private versions: `maintenance/drop-catsim` switches them.

## Tests

- catsim's behaviour tests of `TestBayesianHelpers` (`tests/test_estimation.py:386-620`), rewritten
  for many examinees: the checks of priors and grids, a posterior that sums to 1, one that is stable
  over many responses, and the brute-force integral (504).
- The posterior of examinees with no responses is the normalized prior masses; a row of the batched
  posterior equals the posterior computed for that examinee alone.
- Pickling and `repr` of the priors.

## Open Questions

1. **Two default grids.** `QuadratureGrid.uniform()` defaults to catsim's 121 nodes, and calibration
   to mirt's 61 (`calibration.default_grid`, ADR-0007). The card keeps both: they serve different
   defaults of different functions.
