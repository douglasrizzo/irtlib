---
title: Evaluate Rasch item fit with the weighted and unweighted mean squares
kind: technique
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [moved from catsim's backlog]
---

# Evaluate Rasch item fit with the weighted and unweighted mean squares

## Problem

`irtlib.fit` computes S-X², the item-fit statistic for every model. Rasch practitioners also
expect the weighted and unweighted mean squares of Wright and Masters (1982), which compare each
response with its expectation under estimated abilities. irtlib has neither them nor the
ability estimates they need.

## Evidence

- The research pass (2026-11-27) checked the card carried over from catsim:
  - its interpretation table (below 0.7 overfit, 0.7-1.3 good, above 1.3 underfit, above 2.0
    degraded) has no source on the shelf; Wright and Masters (1982, p. 101) standardize the weighted
    mean square instead;
  - its point-biserial rules ("$\le 0$ ... flagged for removal", "$< 0.15$ weak") have no source,
    and "there is no clear basis for determining what magnitude item point-biserial establishes item
    acceptability" (Wright & Stone, 1979, p. 26);
  - its mean-square formulas match Wright and Masters (1982, p. 99), who define them for the Rasch
    family of models (p. 10).

## What irtlib and catsim have now

- irtlib: S-X² (`irtlib.fit.s_x2`); no ability estimates for many examinees at once, which
  ADR-0004 assigns to irtlib.
- catsim: EAP for one examinee at a time (`catsim.estimation.EAPEstimator`).

## Proposal

EAP abilities of every examinee on the calibration's grid, as a function of a calibration and the
responses (ADR-0006), then the weighted and unweighted mean squares for the Rasch model from them.

## Tests

- **Parity with mirt 1.48:** `itemfit(fit_stats = "infit")` on Rasch fits of `LSAT6` and `LSAT7`,
  and `fscores(method = "EAP")` for the abilities.
- **Mean squares:** a hand-worked Rasch case for the weighted and unweighted mean squares of
  Wright and Masters (1982, p. 99, Equations 5.4.6-5.4.7).

## Open Questions

1. **Which abilities the mean squares use.** Wright and Masters use estimates of the person and item
   parameters from the same data (1982, p. 101, Section 5.4.4); mirt uses EAP ability estimates by
   default (source `R/itemfit.R:259-268`). **Decided** (ADR-0006): EAP on the calibration's grid,
   as mirt.

---

**Module:** the fit layer of ADR-0006, beside `irtlib.fit`'s S-X² (planned)
**Reference:** Wright & Masters (1982)
**Last validated:** 2026-11-27, against Wright & Masters (1982) and Wright & Stone (1979) on the paper
shelf, and mirt 1.48 (reference manual and source).
**Implementation guidance:** not yet written; written against ADR-0006 to ADR-0008 when implemented.

## Motivation

The mean squares compare each examinee's response to an item with its expectation at the examinee's
estimated ability, and average the squared standardized residuals over examinees, weighted or not
(Wright & Masters, 1982, pp. 98-99).

## Definition

With $z_{ij}$ the standardized residual of examinee $j$ on item $i$ and $W_{ij}$ the variance of the
response (Wright & Masters, 1982, p. 98, Equation 5.4.3), the unweighted mean square is the mean of
$z_{ij}^2$ and the weighted mean square is $\sum_j z_{ij}^2 W_{ij} / \sum_j W_{ij}$ (p. 99,
Equations 5.4.6-5.4.7). Their expectation is 1, and the weighted mean square is standardized by a
cube-root transformation (p. 100, Table 5.4a; p. 101).

## Behavior and Comparison

mirt 1.48 computes the mean squares as `fit_stats = "infit"` from EAP ability estimates (source
`R/itemfit.R:259-268`, `:576-580`).

## Guarantees

1. On data from a fitting Rasch model, both mean squares are near their expectation of 1.

## References

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf

Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Best%20Test%20Design.pdf

## Implementation Guidance

Follows ADR-0006: abilities and item fit are functions of a calibration result and the responses.
