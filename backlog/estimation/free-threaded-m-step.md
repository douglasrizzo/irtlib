---
title: Run the M-step's item updates in threads on free-threaded Python
kind: feature
status: blocked
priority: 4
effort: S
blocked_by: [estimation/mml-em]
raised_by: [Python-floor measurement of 2026-11-26]
---

# Run the M-step's item updates in threads on free-threaded Python

## Problem

In marginal maximum likelihood with the EM algorithm, the M-step maximizes each item's expected
log-likelihood separately, one SciPy optimization per item. On a standard (GIL) build of Python
those optimizations run one after another, whatever the number of cores.

## Evidence

A benchmark of one 2PL EM iteration, made when pyitemcal's Python floor was chosen (ADR-0002), on
2026-11-26, on an Apple M4 Pro: 5,000 examinees, 40 items, 61 quadrature nodes, 30% missing
responses, and a per-item L-BFGS-B M-step with analytic gradients. Medians of five interleaved runs:

| Python | M-step, serial | M-step, 4 threads |
|---|---|---|
| 3.14 (GIL) | 9.65 ms | 9.67 ms |
| 3.14t (free-threaded) | 10.51 ms | 4.01 ms |
| 3.15t (free-threaded) | 9.92 ms | 3.27 ms |

A whole iteration (E-step plus M-step) took 12.8 ms on 3.14 and 7.1 ms on 3.14t with threads. The
free-threaded builds pay 8-12% on serial code; 3.13t paid 21-29% and has no wheels for NumPy 2.5 or
SciPy 1.18. The benchmark script is not in the repository; the measurement is repeated on the
implemented M-step (Proposal).

## Proposal

- Run the per-item M-step updates through a `concurrent.futures.ThreadPoolExecutor` when the
  interpreter is free-threaded, with a `max_workers` setting, and serially otherwise.
- Add a CI test entry on the newest free-threaded Python.
- Measure again on the implemented M-step before merging, and report the speedup (AGENTS.md: a
  change to speed is reported with measurements).

## Tests

- The threaded and serial M-steps give identical estimates on the same data, on a free-threaded
  build.
- With one worker, the threaded path calls the optimizer in item order.
