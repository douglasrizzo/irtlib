---
title: Generate banks of a chosen shape, and document the scaling constant D where users look
kind: decision
status: needs-decision
priority: 3
effort: S
blocked_by: []
raised_by: [catsim's research walkthrough of 2026-11-26]
---

# Generate banks of a chosen shape, and document the scaling constant D where users look

catsim's backlog raised this card from all three replications of its research walkthrough
(Wang & Vispoel, 1998; Babcock & Weiss, 2012; Chang & Ying, 1999). It moved here with the item bank
(ADR-0004).

## Problem

1. **Fixed distributions.** `ItemBank.generate(n_items, model, corr, rng)` draws a ~ N(1.2, 0.25)
   truncated, b ~ N(0, 1), c ~ N(0.25, 0.02) and d ~ U(0.94, 1). Simulation studies vary the bank's
   shape, so all three replications built their item matrices with NumPy:
   - Wang & Vispoel: a "good" bank with b ~ U(−3, 3) and a "poor" bank.
   - Babcock & Weiss: flat banks with b ~ U(−3.5, 3.5) and peaked banks with b ~ N(0, 0.75), with a
     fixed c = 0.2.
   - Chang & Ying: a ~ lognormal, because the default a range of 0.40 to 1.86 is narrow for a study
     that stratifies by a.
   Building the matrix was easy in every case (`ItemBank(numpy.column_stack([a, b, c, d]))`). The gap
   is that nothing shows how.
2. **D = 1.7.** irtlib's logistic functions have no scaling constant (ADR-0001), so parameters
   calibrated with D = 1.7 must have a multiplied by 1.7. The Item Response Functions page says the
   functions have no constant; the `ItemBank` constructor's docstring, where a user builds a bank,
   does not, and only catsim's getting-started notebook and stratified selector page give the
   rescaling.

## What irtlib and catsim have now

- `irtlib.bank.ItemBank.generate` draws the fixed distributions above.
- catsim's documentation builds banks with `ItemBank.generate` and from CSV files, and gives the
  rescaling by 1.7 in notebook 01 and on the stratified selector page.

## Proposal

- Pick one:
  - Let `generate` take the distributions, such as a callable or frozen SciPy distribution per
    parameter.
  - Document a recipe for building the matrix yourself, in the `ItemBank` docstring, which catsim's
    guides link.

  The recipe is the smaller change and may be enough, since all three replications managed alone
  (YAGNI). Decide with the maintainer.
- State the absent D in the `ItemBank` constructor's docstring, as part of what the parameters mean,
  with the rescaling once; catsim's pages then link to it (one home per topic).
- The alternative is a `D` argument: on `ItemBank`, which would rescale a once, or on every IRT
  function, as catR has. Prefer the documentation: the item matrix stays the one thing that defines
  an item (ADR-0009), and every caller would otherwise have to pass the same D.

## Tests

- If `generate` takes distributions: a seeded bank drawn from given distributions has the parameters
  those distributions give for the same generator stream.
