---
title: Share the complete-response check of item fit and G²
kind: maintenance
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [mean-squares PR on 2026-11-27]
---

# Share the complete-response check of item fit and G²

## Why this card exists

Reviewing the mean squares PR (#28) found the same response checks written twice, once in
`irtlib.fit` and once in `irtlib.comparison`. Sharing them meant moving a helper between modules,
outside that PR's concern.

## Problem

Three functions refuse missing responses and responses with another number of items than the
calibration, each with its own copy of the same checks: `irtlib.fit.s_x2` and
`irtlib.fit.mean_squares` share `fit._complete_responses`, and `irtlib.comparison.g2` repeats it.
`comparison` and `fit` sit side by side in the import-linter layers, so `g2` cannot import the
helper from `fit`.

## What irtlib and catsim have now

- `src/irtlib/fit.py`: `_complete_responses(calibration, responses, statistic)`, which validates the
  responses, refuses a missing one naming the statistic, and checks the number of items.
- `src/irtlib/comparison.py`: `g2` runs `validate_responses`, refuses a missing response with its own
  reason (the multinomial over patterns is not defined with missing responses), and checks the
  number of items with the same message as `fit`.

## Proposal

Move the helper into `irtlib.calibration`, beside `validate_responses`, with the reason for refusing
missing responses as an argument, and call it from `s_x2`, `mean_squares` and `g2`.

## Tests

- The existing refusal tests of `tests/test_fit.py` and `tests/test_comparison.py` keep passing
  with their messages.
