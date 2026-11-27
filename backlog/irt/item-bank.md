---
title: Bring the model names, the item bank and response drawing from catsim into irtlib
kind: feature
status: blocked
priority: 2
effort: M
blocked_by: [irt/response-functions]
raised_by: [ADR-0004 as decided on 2026-11-27]
---

# Bring the model names, the item bank and response drawing from catsim into irtlib

## Problem

A calibration produces an item bank and a CAT consumes it; ADR-0001 makes the bank format,
with its validation, its CSV file and random banks, irtlib's. irtlib imports `catsim.IrtModel` and
`ItemBank`, and its recovery tests draw responses inline (`tests/test_mml.py:58, 155, 260, 370`,
`tests/test_fit.py:76`).

## What irtlib and catsim have now

References are to catsim's `dev` at commit `97b89f5`.

- catsim `src/catsim/item_bank.py` (863 lines) moves whole: `IrtModel` (44-61); `ItemBank` (64-571)
  with names, read-only arrays, cached abilities and values of maximum information, `model`,
  `get_items`, `icc`, `information`, `total_information`, `generate` (383-477), `from_csv` and
  `to_csv` (479-535); and its private parsing and validation (574-863). It needs `irt` and the
  private `checks._named` (28-30), which comes along.
- `generate` cites Barrada, Olea, Ponsoda and Abad (2010, p. 445) for the distributions of its
  banks. The page gives "a ~ N(1.2, .25), b ~ N(0, 1) and c ~ N(.25, .02)" and names no model; it
  does not say whether the second number is a standard deviation or a variance, and it does not
  truncate a or clip c (shelf notes, 2026-11-27). The upper asymptote's U(0.94, 1) has no source.
- Response drawing: catsim's `simulation._rows` draws a row as
  `generator.random(n_items) < item_bank.icc(theta)` (614-615), per examinee from its own stream.
- Tests: `tests/test_item_bank.py` (752 lines), behavioural with hand-built inputs, including sample
  moments of `generate` against the documented distributions; `tests/test_item_bank_properties.py`,
  structural invariants and two comparisons of catsim with itself.

## Proposal

1. A bank module with `IrtModel`, `ItemBank` and their helpers, between `calibration` and `irt` in
   the import-linter layers, beside the quadrature module, keeping catsim's names, CSV format
   and validation: a bank is an item set for use, so it refuses $a \le 0$ and anything outside
   $0 \le c < d \le 1$ (maintainer, 2026-11-27). `compat.override` becomes `typing.override`; the
   text about exposure, catsim's ADR numbers and the `.. plot::` directive leave.
2. `generate` states what the source gives and labels the rest as irtlib's choices: reading the
   second numbers as standard deviations, the truncation of a, the clipping of c and the upper
   asymptote's distribution.
3. A drawing module of its own (ADR-0006), with a function that draws a response matrix from
   abilities and items with an `rng` argument (ADR-0005), drawing `rng.random(...) < icc(theta,
   items)` in catsim's order, so catsim can draw through it without changing its seeded results.
4. Export `IrtModel` and `ItemBank` from the package root, as catsim does.
5. Leave `calibration` and `mml` on catsim's `ItemBank`: `maintenance/drop-catsim` switches them,
   since a catsim bank never equals an irtlib one.

## Tests

- catsim's bank tests, rewritten to irtlib's rules; the CSV round trip; a bank's cached values
  against `irt`'s functions computed directly.
- Drawing: the proportion correct of many draws at one ability and item within a binomial bound of
  `icc`; the same `rng` seed gives the same matrix.
