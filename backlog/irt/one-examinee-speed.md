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
test, for one examinee. On that shape, irtlib's versions take 1.3 to 1.5 times catsim's time, above
the 1.2 times the maintainer set for a faster version (ADR-0004). catsim's simulations will feel it
once catsim builds on irtlib, and so do irtlib's EAP estimate, which calls both
(`estimation/faster-ability-estimation`).

## Evidence

`benchmarks/catsim_baseline.py`, three runs on 2026-11-27 (Apple M4 Pro, Python 3.13.6, NumPy
2.5.5, SciPy 1.18.1, catsim at `97b89f5`), the range of the ratios of irtlib's time to catsim's:

| Function | Shape | catsim | irtlib | Ratio |
|---|---|---|---|---|
| `irt.log_likelihood` | 20 responses, 121 nodes | 44.5-44.7 µs | 67.0-68.0 µs | 1.50-1.53 |
| `quadrature.posterior`, with `irt.log_likelihood` | 20 responses, 121 nodes | 51.3-53.7 µs | 77.2-78.5 µs | 1.44-1.53 |
| `quadrature.posterior_mean` | 121 nodes | 0.85-0.91 µs | 1.19-1.25 µs | 1.35-1.39 |
| `quadrature.posterior_variance` | 121 nodes | 2.15-2.28 µs | 2.89-2.98 µs | 1.29-1.34 |
| `irt.log_likelihood_function`, a call | 20 responses, 1 ability | 2.88-3.06 µs | 3.49-3.60 µs | 1.16-1.21 |

For the log-likelihood, building the function takes 12.2 µs against catsim's 6.4 µs, and its call at
the 121 nodes 45.5 µs against 37.3 µs (a scratch script on the same day). The build checks the
response matrix (1.7 µs) and builds a float matrix that selects each examinee's terms; the call
multiplies by that matrix, where catsim sums one term per item. The posterior then checks the
log-likelihood for NaN and +inf and for rows with no mass (10.3 µs from a given log-likelihood).
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
