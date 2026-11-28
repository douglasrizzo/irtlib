---
title: Bring a call of the normal prior's log-density within 1.2 times catsim's time
kind: fix
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [benchmarks against catsim on 2026-11-27]
---

# Bring a call of the normal prior's log-density within 1.2 times catsim's time

## Problem

The posterior, EAP and MAP evaluate the prior's log-density at every call, and catsim's estimators
call them once or more per step of a simulated adaptive test. A call of the prior that
`quadrature.normal_log_prior` returns takes 1.5 times catsim's time on the grid's 121 nodes, above
the 1.2 times the maintainer set for a faster version (ADR-0004). The posterior and MAP inherit the
difference, though both stay under 1.2.

## Evidence

- `benchmarks/catsim_baseline.py` on 2026-11-27, after #27 (Apple M4 Pro, Python 3.13.6, NumPy
  2.5.5, SciPy 1.18.1, catsim at `97b89f5`), one run on a machine shared with other test runs:
  `quadrature.normal_log_prior, a call`, 121 nodes, 2.49 µs against catsim's 1.66 µs, 1.50 times.
  Before #27 the same row was 0.99 to 1.02 in nine runs that day.
- A scratch script on the same day, at the 121 nodes:

  | Call | Time |
  |---|---|
  | catsim's prior | 1.61 µs |
  | irtlib's prior | 2.47 µs |
  | irtlib's formula without its `numpy.errstate` | 1.84 µs |
  | `with numpy.errstate(over="ignore"): pass` | 0.50 µs |

  The context manager takes about 0.5 µs of the 0.86 µs difference. The rest is the formula: on the
  array, a subtraction, a division, two products and a subtraction, where catsim's takes a
  subtraction, a power, a product, a division and a subtraction.

## What irtlib and catsim have now

- `irtlib.quadrature._NormalLogPrior.__call__` computes `-0.5 * deviation * deviation - log(sd) -
  0.5 * log(2 pi)` from the standardized deviation `(theta - mean) / sd`, under
  `numpy.errstate(over="ignore")`, so that a deviation too large to square gives `-inf` without a
  warning. The log-density is finite for every finite, positive `sd`.
- catsim's prior computes `-0.5 * log(2 pi var) - 0.5 * (theta - mean) ** 2 / var`, which overflows
  for some standard deviations (#27).

## Proposal

Measure, with the benchmarks, each of these, and keep the cheapest that brings the call under 1.2
times catsim's time without giving up the overflow safety:

1. compute the constant `-log(sd) - 0.5 * log(2 pi)` once, when the prior is built, as a field left
   out of equality and the repr;
2. take the error state only when an overflow is possible, which needs a check of the deviations'
   magnitude that may cost what it saves.

If neither does, the card asks the maintainer whether the guard's 0.5 µs is worth keeping.

## Tests

- The existing tests of `normal_log_prior`: SciPy's normal log-density, and the closed forms of a
  very wide and a very narrow prior.
- The benchmarks' agreement check against catsim's prior, within a relative 1e-12.
