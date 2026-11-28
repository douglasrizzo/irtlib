---
title: Find how EAP abilities lower the mean squares on data that fit the Rasch model
kind: research
status: needs-research
priority: 4
effort: S
blocked_by: []
raised_by: [mean-squares PR on 2026-11-27]
---

# Find how EAP abilities lower the mean squares on data that fit the Rasch model

## Why this card exists

The mean squares PR (#28) measured infit and outfit well below 1 on data drawn from the Rasch
model, more than the sources explain. It kept mirt's EAP abilities, and on 2026-11-27 the
maintainer chose a research card over a change.

## Problem

`irtlib.fit.mean_squares` evaluates the outfit and infit, Wright and Masters' unweighted and
weighted mean squares, at EAP abilities on the calibration's grid (ADR-0006), as mirt 1.48 does by
default (source `R/itemfit.R:268, 535-538`). Their expectation is 1 when the data fit the model
(Wright & Masters, 1982, p. 100, Table 5.4a; p. 101). On data drawn from the Rasch model, the
mean squares at EAP abilities fall below 1, and their standardized values below 0, by more than
the one factor the sources on the shelf give. A researcher reading an outfit of 0.8 on a short
test cannot tell misfit from this lowering.

## Evidence

- Wright and Masters estimate the person and item parameters from the same data as the residuals.
  They say this lowers the expected mean square by the factor $(L - 1)(N - 1)/(NL)$ for $L$ items
  and $N$ persons. They have "not mastered the statistical details of these effects well enough to
  provide useful corrections", and they find fit statistics computed from estimates "entirely
  satisfactory" in practice (1982, p. 101). Their estimates are not EAP estimates.
- Wright and Stone (1979, p. 79) divide the item's sum of squared standardized residuals by
  $N - 1$, not $N$.
- Measured on 2026-11-27 (irtlib, averages over the items and 10 replications). The responses were
  drawn from the Rasch model with slope 1, difficulties equally spaced on [-2, 2] and abilities
  N(0, 1), then calibrated as a 1PL:

  | Items | Examinees | Outfit | Infit | $(L - 1)/L$ |
  |---|---|---|---|---|
  | 20 | 2,000 | 0.893 | 0.939 | 0.950 |
  | 30 | 2,000 | 0.921 | 0.956 | 0.967 |
  | 40 | 2,000 | 0.939 | 0.965 | 0.975 |
  | 20 | 5,000 | 0.892 | 0.938 | 0.950 |

  With 40 items, the standardized infit averaged -1.58 (standard deviation 0.99) and the
  standardized outfit -1.74 (0.98), against the 0 and 1 of Table 5.4a.
- On LSAT6 and LSAT7, five items each, the outfit is 0.71 to 0.83 under both irtlib and mirt 1.48
  (`tests/parity/mirt/item_fit.json`, the `mean_squares` cases).
- mirt's own example notes that ML abilities agree better with the eRm package (source
  `R/itemfit.R:198-199`).
- Other R packages choose differently (sources read on 2026-11-28):
  - **TAM 4.3-25** has three ways to get the abilities (`man/msq.itemfit.Rd`):
    - `msq.itemfit` evaluates "the individual posterior" instead of a point estimate;
    - `tam.fit` simulates from it;
    - `msq.itemfitWLE` uses weighted likelihood estimates (`R/msq.itemfitWLE.R:1-24`).
  - **eRm 1.0-10** uses maximum likelihood abilities, leaving out the examinees with extreme
    scores (`R/itemfit.ppar.R:6-10, 21-28`).
  - **iarm 0.4.3** uses no ability estimate: it compares responses with their expectation given
    the total score, "to avoid bias" ([fit/conditional-mean-squares](../fit/conditional-mean-squares.md)).
  - No source on the shelf measures any of these against EAP abilities. Müller (2020), open at
    doi:10.1186/s40488-020-00108-7, is a candidate to read.

## What irtlib and catsim have now

- `irtlib.fit.mean_squares` takes EAP abilities on the calibration's grid. Its docstring says that
  estimating the abilities from the same responses lowers the expectation, and gives no number.
- catsim has no mean squares.

## Proposal

Look for a published measurement or correction of the mean squares' expectation and variance
when the abilities are EAP estimates, or other shrunken estimates, together with the items'
estimates. Failing that, look for a source for another choice of abilities, with its measured
effect on the expectation. The candidates are:
- maximum likelihood estimates;
- weighted likelihood estimates;
- TAM's posterior-weighted residuals;
- the conditional residuals of iarm, which need no abilities. Bring
what is found to the maintainer. A change to the abilities or a correction would change ADR-0006's
decision and the mirt parity settings.

## Tests

- Whatever correction or choice of abilities follows from a source has a test that reproduces the
  source's published values, and the guarantee test of `tests/test_fit.py` tightens its bands to
  match.

## Open Questions

1. Does a correction belong in irtlib, or a documented expectation below 1 with its size? The card
   assumes documentation until a source is found.

## References

- Müller, M. (2020). Item fit statistics for Rasch analysis: can we trust them? *Journal of
  Statistical Distributions and Applications, 7*, 5. doi:10.1186/s40488-020-00108-7. Open access;
  not yet on the shelf.
- Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
  https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf
- Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press. Open copy:
  https://www.rasch.org/BTD_RSA/pdf/Best%20Test%20Design.pdf
- eRm 1.0-10, iarm 0.4.3 and TAM 4.3-25, CRAN source packages.
  <https://cran.r-project.org/package=eRm>, <https://cran.r-project.org/package=iarm>,
  <https://cran.r-project.org/package=TAM>
