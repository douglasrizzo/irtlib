---
title: Refuse items whose parameters are not finite in the ability estimators
kind: fix
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [joint maximum likelihood abilities PR on 2026-11-28]
---

# Refuse items whose parameters are not finite in the ability estimators

## Why this card exists

Writing the docstring that tells a JML user how to estimate abilities at the corrected
difficulties showed that `irtlib.ability.maximum_likelihood` accepts a calibration's estimates with
the NaN rows of the items left out. It returns -6 for every examinee, with no error. That behavior
predates the JML PR, and changing it is outside that PR's concern.

## Problem

The ability estimators take items with a NaN or infinite parameter, and most of them return
something wrong without an error. A caller who passes a calibration's `estimates`, where an item
with no estimate has NaN, gets an answer that looks plausible, or an error that does not name the
item.

## Evidence

Measured on 2026-11-28 at `0fad683`. The items are `[1, b, 0, 1]` and `[1, 0.5, 0, 1]`, with one or
two examinees answering `10` and `01`, through both the one-examinee and the many-examinee paths:

| First item | `maximum_likelihood` | `maximum_a_posteriori` | `expected_a_posteriori` | `weighted_likelihood` |
|---|---|---|---|---|
| `b` NaN, or `a` NaN | -6.0 for every examinee | -6.0 for every examinee | `ValueError`: the log-likelihood is NaN or +inf, naming the rows | `ValueError`: "attempt to get argmax of an empty sequence" |
| `b` infinite | -inf and 42.0 | 0.5 and 0.5 | `ValueError`, naming the rows | -0.599 and 1.599 |

`irtlib.ItemBank` refuses every parameter that is not finite, with the items named
(`irtlib.bank._violations`).

## What irtlib and catsim have now

- `src/irtlib/_responses.py`: `responses_and_items` checks the items' shape and that there is one
  per column of the responses, not their values.
- `src/irtlib/ability.py`: every estimator calls `responses_and_items`.
- catsim passes the items of an `ItemBank`, which are always finite, so this does not reach it.

## Proposal

`responses_and_items` refuses items with a parameter that is not finite, naming their rows, as
`ItemBank` does. Every ability estimator then raises the same `ValueError` before it searches.

## Tests

- Each of the four estimators raises a `ValueError` naming the row of an item with a NaN `b`, a NaN
  `a`, or an infinite `b`, for one examinee and for several. Today ML and MAP return -6, and the
  WLE fails with an error that names no item.

## Open Questions

None.
