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
slowdown gets a card for a faster version (ADR-0004; `maintenance/drop-catsim`).

## Evidence

A scratch script, 2026-11-27, on the ability-estimation PR: one examinee's 20 responses, 14
correct, on a random 4PL bank, as in one step of an adaptive test; the best of 7 runs of 200 calls,
catsim at its pinned commit. Every estimate agreed with catsim's to 1e-6.

| Estimator | catsim | irtlib | Ratio |
|---|---|---|---|
| EAP | 57.9 µs | 92.5 µs | 1.60 |
| MAP | 144.6 µs | 191.0 µs | 1.32 |
| ML, bounds (-6, 6) | 75.6 µs | 166.8 µs | 2.21 |
| ML, no bounds | 78.0 µs | 191.5 µs | 2.45 |
| WLE | 136.3 µs | 253.3 µs | 1.86 |

The benchmarks of `maintenance/drop-catsim` time the estimators again, against catsim's, and
replace these numbers.

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
- The benchmarks of `maintenance/drop-catsim` show each estimator within 1.2 times catsim's time,
  or the card records why one cannot be.

## Open Questions

1. **The ML search on the 1PL and 2PL.** With every $c = 0$, the second derivative of the
   log-likelihood is $-I$ (Warm, 1989, p. 449, in a model with $d = 1$), so the likelihood has at
   most one maximum and Brent's search alone would do there. Whether to branch on the model, or to
   keep one search for every model, is the maintainer's call once the benchmarks measure the cost.

## References

- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
  *Psychometrika, 54*(3), 427-450. doi:10.1007/BF02294627. Not openly available; Warm's (1985)
  dissertation, https://hdl.handle.net/11244/5356, derives the same results.
