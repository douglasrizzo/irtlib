---
title: Make the log-likelihood and the posterior of one examinee as fast as catsim's
kind: fix
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [benchmarks against catsim on 2026-11-27]
---

# Make the log-likelihood and the posterior of one examinee as fast as catsim's

## Problem

catsim calls the log-likelihood and the posterior once or more per step of every simulated adaptive
test, for one examinee. On that shape, irtlib's versions take 1.3 to 1.4 times catsim's time, above
the 1.2 times the maintainer set for a faster version (ADR-0004). catsim's simulations will feel it
once catsim builds on irtlib.

## Evidence

`benchmarks/catsim_baseline.py`, three runs on 2026-11-27 (Apple M4 Pro, Python 3.13.6, NumPy
2.5.5, SciPy 1.18.1, catsim at `97b89f5`), the range of the ratios of irtlib's time to catsim's:

| Function | Shape | catsim | irtlib | Ratio |
|---|---|---|---|---|
| `irt.log_likelihood` | 20 responses, 121 nodes | 46-53 µs | 63-69 µs | 1.29-1.44 |
| `quadrature.posterior`, with `irt.log_likelihood` | 20 responses, 121 nodes | 54-62 µs | 74-82 µs | 1.32-1.37 |
| `quadrature.posterior_mean` | 121 nodes | 0.92-0.96 µs | 1.29-1.31 µs | 1.36-1.40 |
| `quadrature.posterior_variance` | 121 nodes | 2.36-2.43 µs | 3.14-3.19 µs | 1.31-1.34 |
| `irt.log_likelihood_function`, a call | 20 responses, 1 ability | 3.17-3.26 µs | 3.73-3.92 µs | 1.18-1.21 |

For the log-likelihood, building the function takes 12.7 µs against catsim's 6.7 µs, and its call at
the 121 nodes 43.7 µs against 38.5 µs (a scratch script on the same day). The build checks the
response matrix (1.8 µs) and builds a float matrix that selects each examinee's terms; the call
multiplies by that matrix, where catsim sums one term per item. The posterior then checks the
log-likelihood for NaN and +inf and for rows with no mass (10.5 µs from a given log-likelihood).
`posterior_mean` and `posterior_variance` sum along an axis of a one-row matrix, where catsim sums
a vector.

The response functions and information run at 1.00 to 1.18 times catsim's time on one ability
against 500 items, under the threshold; `information` and its sums are the slowest of them.

## Proposal

Keep the batched contract (ADR-0004), and make the case of one row cheap:

1. In `log_likelihood_function`, select each examinee's terms without the float matrix when there
   is one examinee, as catsim does: one signed term per answered item, summed.
2. In `posterior`, run the checks on the rows only when the log-joint is not finite everywhere,
   which a single `numpy.isfinite(...).all()` decides.
3. Leave `posterior_mean` and `posterior_variance` as they are unless the first two leave the
   posterior's path above 1.2 times: their 0.3 to 0.8 µs are small against the posterior's.

Measure each change with the benchmarks, and keep it only where it brings the ratio under 1.2.

## Tests

- The benchmarks' agreement checks: the values stay within a relative 1e-12 of catsim's.
- The existing tests of `irt.log_likelihood` and `quadrature.posterior`, which cover one examinee,
  many examinees, missing responses and the error rows.
