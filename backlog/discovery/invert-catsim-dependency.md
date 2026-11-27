---
title: Should the shared item response kernel move from catsim into pyitemcal?
kind: decision
status: blocked
priority: 3
effort: M
blocked_by: [discovery/requirements]
raised_by: [maintainer during setup]
---

# Should the shared item response kernel move from catsim into pyitemcal?

## Problem

pyitemcal depends on catsim for the item response kernel (ADR-0004). Calibration produces the item
banks a CAT consumes, so the kernel may belong with the producer, with catsim depending on
pyitemcal. The direction is cheap to change before pyitemcal has users and expensive after.

## What pyitemcal and catsim have now

The kernel is in catsim: `catsim.irt` (`icc`, `log_response_probabilities`, `information`),
`QuadratureGrid` and the priors in `catsim.estimation`, `IrtModel`, and `ItemBank` with its
validation and CSV format. pyitemcal imports exactly those names (ADR-0004).

## Proposal

Decide once the requirements card has shaped pyitemcal's own kernel, weighing:

- what pyitemcal's batched, masked functions look like next to catsim's one-examinee ones, and
  whether catsim would use them;
- that catsim would gain a dependency, which changes catsim's ADR-0021 and ADR-0001;
- whether `ItemBank`, with its names and validation, belongs with the calibrator that produces banks
  or with the CAT engine that consumes them;
- the release coupling: whichever package is depended on must release before the other can use a
  change.

Until this is decided, ADR-0004 stands.

## Open Questions

1. Whether the inversion moves the whole kernel or only part of it, such as the response functions
   without `ItemBank`.
