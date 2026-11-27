---
title: Should the item response plots move from catsim into irtlib?
kind: decision
status: needs-decision
priority: 4
effort: S
blocked_by: []
raised_by: [maintainer during the rescope of 2026-11-27]
---

# Should the item response plots move from catsim into irtlib?

## Problem

ADR-0001 puts what is meaningful for a fixed-form test in irtlib. Some of catsim's plots draw item
response quantities that a fixed-form test has, while others draw adaptive sessions and simulation
studies. The maintainer deferred the decision until the response functions are in irtlib
(2026-11-27).

## What irtlib and catsim have now

- irtlib draws nothing, and has no plotting dependency (ADR-0002).
- catsim's `catsim.plot` (`src/catsim/plot.py`, 1,093 lines) needs the optional `plot` extra,
  matplotlib 3.10.7 or later (catsim's `pyproject.toml:52`). By the line of ADR-0001, its functions fall in two groups:
  - item response quantities: `item_curve`, `bank_information`, `expected_score`,
    `item_person_map` and `likelihood`;
  - adaptive testing: `item_exposure`, `session_progress`, `metric_by_ability`, `metric_by_length`,
    `recovery` and `length_distribution`.

## Proposal

Options, to weigh now that the response functions are in `irtlib.irt`:

1. Move the item response plots into irtlib behind an optional extra with matplotlib, which
   changes ADR-0002.
2. Leave every plot in catsim, drawn from irtlib's quantities once catsim adopts irtlib.

No recommendation yet: the choice depends on whether irtlib's readers need plots without catsim.
