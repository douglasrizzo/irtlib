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
likelihood is its calibration method, yet the two do not meet. `irtlib.fit.mean_squares` evaluates
the residuals at EAP abilities on the calibration's grid (ADR-0006), and a calibration by
`irtlib.cml` has no grid: it integrates over no abilities, so its `grid` is None, and
`mean_squares` refuses it with an error, as S-X² does.

## Evidence

- `irtlib.calibration.Calibration.grid` is None for a conditional calibration, and item fit and
  model comparison refuse one (ADR-0006).
- Wright and Masters compute the residuals from "estimates of the person and item parameters",
  which "usually come from the same data" (1982, p. 101, Section 5.4.4).
- EAP abilities lower the mean squares on data that fit the model, by more than the factor Wright
  and Masters give ([discovery/mean-squares-eap.md](../discovery/mean-squares-eap.md)).

## What irtlib and catsim have now

- `irtlib.fit.mean_squares(calibration, responses)` refuses a calibration whose likelihood is not
  marginal.
- `irtlib.ability` estimates abilities by ML, MAP, EAP and WLE from responses and items, on a grid
  the caller passes for EAP, with no calibration needed (ADR-0010).

## Proposal

Options:

1. Keep refusing a conditional calibration, as now: the simplest, but it leaves the Rasch estimator
   without the Rasch fit statistics.
2. Take the abilities as an argument, estimated by the caller from the calibration's items, so that
   a conditional calibration can be scored by maximum likelihood or WLE, as Wright and Masters
   estimate persons from the same data.
3. EAP on irtlib's default grid and normal prior for a calibration without a grid: an ability
   distribution the conditional calibration never assumed.

Recommendation: option 2, with EAP on the calibration's grid as the default for a marginal
calibration, decided together with `discovery/mean-squares-eap.md`, which asks which abilities
lower the mean squares least.

## Open Questions

1. Which option the maintainer prefers.

## References

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf
