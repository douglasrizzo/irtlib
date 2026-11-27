---
title: Bring the ability estimators within 1.2 times catsim's time for one examinee
kind: feature
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [timing of the ability-estimation PR on 2026-11-27]
---

# Bring the ability estimators within 1.2 times catsim's time for one examinee

## Problem

catsim calls an ability estimator once per step of every simulated adaptive test, and will call
irtlib's once it builds on irtlib (ADR-0004). irtlib's four estimators take 1.3 to 2.5 times
catsim's time for one examinee, above the maintainer's threshold of 1.2 times, beyond which a
slowdown gets a card for a faster version (ADR-0004).

## Evidence

`benchmarks/catsim_baseline.py`, three runs on 2026-11-27 (Apple M4 Pro, Python 3.13.6, NumPy
2.5.5, SciPy 1.18.1, catsim at `97b89f5`): one examinee's 20 responses, 14 correct, to the first
items of a random 4PL bank of 500, against catsim's estimators on a step's context built
beforehand, with the range of each time over the runs. Every estimate and standard error agreed
with catsim's: EAP's and the WLE's within a relative 1e-12, MAP's and ML's within the search
tolerance of 1e-6.

| Estimator | catsim | irtlib | Ratio |
|---|---|---|---|
| EAP | 57.9-60.0 µs | 88.8-92.3 µs | 1.48-1.59 |
| MAP | 148.0-153.3 µs | 194.5-197.3 µs | 1.27-1.33 |
| ML, bounds (-6, 6) | 71.0-72.2 µs | 166.0-167.3 µs | 2.30-2.36 |
| WLE | 135.0-138.4 µs | 243.7-250.7 µs | 1.80-1.81 |

All four are above 1.2. EAP's time is mostly that of `irt.log_likelihood` and `quadrature.posterior`
on the grid, 1.44 to 1.53 times catsim's in the same runs (`irt/one-examinee-speed`). The ML
estimate without bounds was timed by the scratch script of the ability-estimation PR only: 191.5 µs
against catsim's 78.0 µs, 2.45 times.

## What irtlib and catsim have now

- `irtlib.ability.expected_a_posteriori` checks the response matrix, builds
  `irt.log_likelihood_function` and calls `quadrature.posterior`, whose checks run on every call.
- `maximum_likelihood` searches 201 points and refines the best, the search of
  `maximum_a_posteriori`, where catsim's `NumericalSearchEstimator` runs Brent's method alone, with
  about 11 evaluations. On 1,139 random 3PL and 4PL patterns Brent's search alone missed the global
  maximum on [-6, 6] 41 times, and the grid search none.
- `weighted_likelihood` takes P and Q from their logarithms, `irt.log_response_probabilities`, where
  catsim's computes them from the logistic directly.
- Every estimator checks its arguments and builds one log-likelihood function per examinee.

## Proposal

Profile each estimator on the benchmark's shapes, and make it faster where the profile points,
without changing its estimates beyond rounding: cheaper checks on the one examinee of a CAT step, a
lighter path through the posterior for EAP, fewer search points where the likelihood has one
maximum (the 1PL and 2PL, Open Question 1), and P and Q from the logistic in the WLE where neither
rounds off.

## Tests

- The estimators' tests and parity cases pass unchanged.
- `benchmarks/catsim_baseline.py` shows each estimator within 1.2 times catsim's time, or the card
  records why one cannot be.

## Open Questions

1. **The ML search on the 1PL and 2PL.** With every $c = 0$, the second derivative of the
   log-likelihood is $-I$ (Warm, 1989, p. 449, in a model with $d = 1$), so the likelihood has at
   most one maximum and Brent's search alone would do there. Whether to branch on the model, or to
   keep one search for every model, is the maintainer's call once the benchmarks measure the cost.

## References

- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
  *Psychometrika, 54*(3), 427-450. doi:10.1007/BF02294627. Not openly available; Warm's (1985)
  dissertation, https://hdl.handle.net/11244/5356, derives the same results.
