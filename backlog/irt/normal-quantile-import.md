---
title: Compute the normal quantile in irtlib.irt with scipy.special instead of importing scipy.stats
kind: maintenance
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [catsim's import-time card on 2026-11-27]
---

# Compute the normal quantile in irtlib.irt with scipy.special instead of importing scipy.stats

## Problem

`import irtlib` takes about 0.46 seconds, and nearly half of it is `scipy.stats`, which `irtlib.irt`
imports for one thing: the standard normal quantile of `confidence_interval`. Every script, test run,
worker process and package that imports irtlib pays it, catsim's included: catsim's own card,
`maintenance/normal-quantile-import`, removes its two imports of `scipy.stats`, and `import catsim`
still loads it through `irtlib.ItemBank`.

## Evidence

Measured on 2026-11-27 at `9c0349a`, CPython 3.13.6 on an Apple M4 Pro, NumPy 2.5.5 and SciPy
1.18.1, timing each import in turn in a fresh process, three runs within 0.01 s of each other:

| Import | Seconds |
|---|---|
| `numpy`, `scipy.special` and `scipy.optimize` | 0.245 |
| `scipy.stats`, after them | 0.21 |
| the rest of `irtlib` | 0.006 |
| total | 0.46 |

## What irtlib has now

`irtlib/__init__.py` imports `irtlib.bank`, which imports `irtlib.irt`, whose module-level
`from scipy import stats` serves `confidence_interval` alone (`stats.norm.ppf((1 + confidence) / 2)`).
`irtlib.mml` (`norm.ppf`), `irtlib.fit` and `irtlib.comparison` (`chi2.sf`) also import
`scipy.stats`, but `import irtlib` does not load them.

## Proposal

In `irtlib.irt`, replace `stats.norm.ppf(p)` with `scipy.special.ndtri(p)`, the inverse of the
standard normal distribution function, from `scipy.special`, which the module already imports
(`expit`, `log_expit`), and drop `from scipy import stats`. `irtlib.mml` can use `ndtri` too; the
chi-squared tails stay in `scipy.stats`.

## Tests

- After `import irtlib` in a fresh interpreter, `"scipy.stats"` is not in `sys.modules`. The test
  fails on the current code.
- `confidence_interval` at `confidence=0.95` uses the 97.5% quantile of the standard normal,
  1.959963984540054, to the last digit.
