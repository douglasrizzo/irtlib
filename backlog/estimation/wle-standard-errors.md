---
title: Choose the WLE's standard error (Warm's corrected SE, the sandwich SE)
kind: technique
status: needs-research
priority: 3
effort: S
blocked_by: []
raised_by: [catsim's Wang and Vispoel (1998) walkthrough]
---

# Choose the WLE's standard error (Warm's corrected SE, the sandwich SE)

catsim's backlog raised this card from its replication of Wang & Vispoel (1998). It moved here with
the ability estimators (ADR-0004): catsim's `WarmLikelihoodEstimator` reports the standard error
that `irtlib.ability.weighted_likelihood` computes.

## Problem

`weighted_likelihood` reports only the plain Fisher SE, the inverse square root of the test
information at the estimate. catsim's WLE technique page discusses Warm's corrected SE and the
sandwich SE, but neither can be chosen. A study that compares SE formulas, as Wang & Wang (2001)
did, cannot be run.

## Proposal (to settle in research)

- A `se=` option on `weighted_likelihood`, with a `StrEnum` of the formulas (AGENTS.md: closed
  option sets are `StrEnum`s), which catsim's `WarmLikelihoodEstimator` then passes through.
- Check each formula against its source: Warm (1989); Magis (2016) for the corrected SE. catsim's
  page cites Magis's Equation 14. Say where irtlib departs from the source.

## Tests

- Each formula on a hand-worked short test, against a value computed from the source's equation.
- The default stays the Fisher SE, so seeded results don't change unless the option is set.
