---
title: Which abilities should the Rasch mean squares of a conditional calibration use?
kind: decision
status: needs-decision
priority: 3
effort: XS
blocked_by: []
raised_by: [conditional maximum likelihood PR on 2026-11-27]
---

# Which abilities should the Rasch mean squares of a conditional calibration use?

## Problem

The weighted and unweighted mean squares are the Rasch model's item fit, and conditional maximum
likelihood is its calibration method, yet the two do not meet. The mean-squares card decides on EAP
abilities on the calibration's grid, as mirt does (`backlog/fit/mean-squares.md`, Open Question 1),
and a calibration by `irtlib.cml` has no grid: it integrates over no abilities, so its
`grid` is None (ADR-0006).

## Evidence

- `irtlib.calibration.Calibration.grid` is None for a conditional calibration, and S-X² refuses one
  with an error (ADR-0006).
- Wright and Masters compute the residuals from "estimates of the person and item parameters",
  which "usually come from the same data" (1982, p. 101, Section 5.4.4).

## What irtlib and catsim have now

- `irtlib.ability` estimates abilities by ML, MAP, EAP and WLE from responses and items, on a grid
  the caller passes for EAP, with no calibration needed (ADR-0010).
- The mean squares are being implemented from `backlog/fit/mean-squares.md`, in parallel with
  conditional maximum likelihood (2026-11-27).

## Proposal

Options, once the mean squares land:

1. Refuse a conditional calibration, as S-X² does: the simplest, but it leaves the Rasch estimator
   without the Rasch fit statistics.
2. Take the abilities as an argument, estimated by the caller from the calibration's items, so that
   a conditional calibration can be scored by maximum likelihood or WLE, as Wright and Masters
   estimate persons from the same data.
3. EAP on irtlib's default grid and normal prior for a calibration without a grid: an ability
   distribution the conditional calibration never assumed.

Recommendation: option 2, with EAP on the calibration's grid as the default for a marginal
calibration.

## Open Questions

1. Which option the maintainer prefers.

## References

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf
