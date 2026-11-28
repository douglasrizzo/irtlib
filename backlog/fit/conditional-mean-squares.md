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

## Why this card exists

The conditional maximum likelihood PR (#30) gave conditional calibrations no grid. With no grid,
`irtlib.fit.mean_squares` refuses them, so the Rasch model's own estimator cannot be checked with
the Rasch model's own fit statistics. Which abilities to use instead is a design choice for the
maintainer.

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
- Two R packages score a CML fit's infit and outfit (sources read on 2026-11-28):
  - **eRm 1.0-10** uses abilities. `itemfit` takes the maximum likelihood abilities of
    `person.parameter` at the CML difficulties, and leaves out the examinees with extreme scores
    (source `R/itemfit.ppar.R:6-10, 21-28`; `R/person.parameter.eRm.R:104-149`).
  - **iarm 0.4.3** uses none. `out_infit` compares each response with its expectation "under the
    conditional distribution of responses given the total score", from an eRm fit, "to avoid
    bias" (source `R/Item-Fit.R:118-138, 153`). It cites Christensen, Kreiner and Mesbah (2013,
    pp. 86-90) and Kreiner and Christensen (2011), neither on the shelf.
  - Those expectations are the probabilities $\pi_{ri}$ of a correct response given $r$ correct,
    which `irtlib.cml` already computes in its likelihood equations.
  - Müller (2020), by iarm's author, is open (doi:10.1186/s40488-020-00108-7) and not on the
    shelf either.

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
4. Conditional mean squares, as iarm computes them: residuals against $\pi_{ri}$, with no ability
   estimate at all. This needs its sources on the shelf first, for the residuals' variance and
   standardization.

Recommendation:
- option 2 now, with EAP on the calibration's grid as the default for a marginal calibration,
  decided together with `discovery/mean-squares-eap.md`, which asks which abilities lower the mean
  squares least;
- option 4 once its sources are read, since a conditional calibration then needs no abilities.

## Open Questions

1. Which option the maintainer prefers.

## References

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf

Not yet on the shelf, for option 4:

- Christensen, K. B., Kreiner, S., & Mesbah, M. (Eds.). (2013). *Rasch models in health*. ISTE and
  Wiley. doi:10.1002/9781118574454. Not openly available.
- Kreiner, S., & Christensen, K. B. (2011). Exact evaluation of bias in Rasch model residuals.
  *Advances in Mathematics Research, 12*, 19-40, as iarm 0.4.3 cites it (`R/Item-Fit.R:138`). No
  DOI found.
- Müller, M. (2020). Item fit statistics for Rasch analysis: can we trust them? *Journal of
  Statistical Distributions and Applications, 7*, 5. doi:10.1186/s40488-020-00108-7. Open access.

Packages read: eRm 1.0-10 and iarm 0.4.3, CRAN source packages.
<https://cran.r-project.org/package=eRm>, <https://cran.r-project.org/package=iarm>
