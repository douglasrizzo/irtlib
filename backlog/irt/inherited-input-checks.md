---
title: Reject the out-of-range inputs that the code moved from catsim accepts
kind: fix
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [reviews of the moves on 2026-11-27]
---

# Reject the out-of-range inputs that the code moved from catsim accepts

## Problem

Two functions moved from catsim, unchanged, accept inputs that give a wrong result or a misleading
error, as catsim's originals do (found by the correctness reviews of the moves, 2026-11-27):

- `ItemBank.generate(n_items, model, corr, rng)` does not check `corr`. With `corr=2` it draws from a
  covariance matrix that is not positive semi-definite, and NumPy only warns, so a bank comes back
  from an invalid distribution. With `corr=nan` it raises `LinAlgError` ("SVD did not converge")
  rather than a `ValueError`.
- `normal_log_prior(mean, sd)` accepts any finite, positive `sd`, but its variance `sd * sd`
  underflows to 0 for `sd=1e-200` and overflows to infinity for `sd=1e200`. A grid's masses then
  raise "the log-prior is NaN or +inf", or "no mass at any node", which do not name the cause.

## Proposal

- `generate` raises `ValueError` unless `corr` is finite and between -1 and 1.
- `normal_log_prior` raises `ValueError` when `sd * sd` is 0 or infinite, naming `sd`.

## Tests

- `generate` with `corr` of 2, -2 and NaN raises `ValueError`, and with -1 and 1 builds a bank.
- `normal_log_prior` with `sd` of 1e-200 and 1e200 raises `ValueError` naming `sd`.
