---
title: Make the posterior mean and variance of one examinee as fast as catsim's
kind: fix
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [benchmarks against catsim on 2026-11-27]
---

# Make the posterior mean and variance of one examinee as fast as catsim's

## Problem

catsim's EAP estimator takes the mean and the variance of one examinee's posterior at every step of
a simulated adaptive test. On that shape, irtlib's `quadrature.posterior_mean` and
`posterior_variance` take 1.3 to 1.4 times catsim's time, above the 1.2 times the maintainer set for
a faster version (ADR-0004). They sum along an axis of a one-row matrix with `numpy.sum`, where
catsim sums a vector.

This card first covered the log-likelihood and the posterior of one examinee too. Those run within
1.06 times catsim's time since `irt.log_likelihood_function` sums one examinee's terms without the
selection matrix and `quadrature.posterior` checks its rows only when the log-likelihood is not
finite at every node. The card's proposal then left the two moments as they were, since the
posterior's path was under 1.2 and the moments take 0.4 to 0.8 µs more than catsim's, against about
60 µs for the posterior.

## Evidence

`benchmarks/catsim_baseline.py`, three runs before and three after those two changes, on 2026-11-27
(Apple M4 Pro, Python 3.13.6, NumPy 2.5.5, SciPy 1.18.1, catsim at `97b89f5`), the range of the
ratios of irtlib's time to catsim's:

| Function | Shape | Before | After |
|---|---|---|---|
| `irt.log_likelihood` | 20 responses, 121 nodes | 1.24-1.34 | 1.03-1.06 |
| `irt.log_likelihood_function`, a call | 20 responses, 1 ability | 1.16-1.18 | 0.70-0.74 |
| `quadrature.posterior`, with `irt.log_likelihood` | 20 responses, 121 nodes | 1.28-1.31 | 1.04-1.06 |
| `quadrature.posterior_mean` | 121 nodes | 1.39-1.41 | 1.35-1.38 |
| `quadrature.posterior_variance` | 121 nodes | 1.31-1.32 | 1.30-1.34 |

After the changes, `posterior_mean` takes 1.31-1.52 µs against catsim's 0.95-1.12 µs, and
`posterior_variance` 3.21-3.64 µs against 2.39-2.81 µs. The runs before measured lower ratios for
the log-likelihood than the card's first runs on the same day (1.50-1.53), on a machine shared with
other test runs.

A scratch script on the same day timed the same products summed with the arrays' `.sum(axis=1)`
method in place of `numpy.sum(..., axis=1)`: the same values, bit for bit, in 0.81 µs against
catsim's 0.97 µs for the mean, and in 2.23 µs against 2.60 µs for the variance.

## Proposal

Sum with the arrays' `.sum` method in `posterior_mean` and `posterior_variance`, and keep the change
if the benchmarks show both under 1.2 times catsim's time.

## Tests

- The benchmarks' agreement checks: the values stay within a relative 1e-12 of catsim's.
- The existing tests of `quadrature.posterior_mean` and `posterior_variance`, which cover many
  examinees and an empty batch.
