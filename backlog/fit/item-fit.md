---
title: Evaluate item fit with S-X² and the mean-square statistics
kind: technique
status: blocked
priority: 2
effort: M
blocked_by: [discovery/requirements, estimation/mml-em]
raised_by: [moved from catsim's backlog]
---

# Evaluate item fit with S-X² and the mean-square statistics

## Problem

ADR-0001 commits pyitemcal to evaluating "the fit of a calibrated model to the responses, item by
item". A calibration that converged can still describe some items badly, and "a complete analysis
must include an evaluation of how well the data fit" (Wright & Stone, 1979, p. 66).

## Evidence

- The research pass (2026-11-27) checked the card carried over from catsim:
  - its interpretation table (below 0.7 overfit, 0.7-1.3 good, above 1.3 underfit, above 2.0
    degraded) has no source on the shelf; Wright and Masters (1982, p. 101) standardize the mean
    squares instead;
  - its point-biserial rules ("$\le 0$ ... flagged for removal", "$< 0.15$ weak") have no source,
    and "there is no clear basis for determining what magnitude item point-biserial establishes item
    acceptability" (Wright & Stone, 1979, p. 26);
  - its mean-square formulas match Wright and Masters (1982, p. 99), who define them for the Rasch
    family of models (p. 10).
- A measurement on mirt 1.48 (2026-11-27), `LSAT7`, 2PL at `TOL = 1e-8`: `itemfit()` gives S-X²
  of 4.751, 14.453, 1.273, 5.236 and 0.941, each on 2 degrees of freedom. With one response set
  missing, it stops: "Only X2, G2, PV_Q1, PV_Q1*, infit, X2*, and X2*_df can be computed with
  missing data" (source `R/itemfit.R:473-475`).

## What pyitemcal and catsim have now

Nothing for item fit in either package; the expected counts and the quadrature are the MML-EM
card's.

## Proposal

S-X² (Orlando & Thissen, 2000) as the item-fit statistic for every model, computed on the MML-EM
card's grid, and the weighted and unweighted mean squares (Wright & Masters, 1982) for the Rasch
model. Yen's $Q_1$ is left out: with the true model it rejected 95% of items on a 10-item test
(Orlando & Thissen, 2000, p. 57, Table 1).

## Tests

- **Parity with mirt 1.48:** `itemfit(fit_stats = "S_X2")` on the `LSAT7` 2PL (values above) and on
  simulated 1PL to 3PL data; the statistics, degrees of freedom and collapsed tables.
- **The recursion:** for 4 items, the summed-score likelihoods equal the sums over all response
  patterns with that score.
- **Type I error:** on data from the fitted model, S-X² rejects at about the nominal rate;
  Orlando and Thissen (2000, p. 57, Table 1) found .04 to .07 at the .05 level.
- **Mean squares:** a hand-worked Rasch case for the weighted and unweighted mean squares of
  Wright and Masters (1982, p. 99, Equations 5.4.6-5.4.7).

## Open Questions

1. **Missing responses.** S-X² is defined on complete response patterns, and mirt does not compute
   it with missing data (above). Zhang and Wang (2022, pp. 708-709) adapt it to missing responses,
   and report false-positive rates far above the nominal level for the unadapted indices (p. 714,
   Table 3). Options: complete data only; or their adaptation. Assumed: complete data only, with an
   error naming the alternative.
2. **Collapsing.** Orlando and Thissen (2000, p. 56) collapse from both ends of the score range
   toward the middle until every expected count is at least 1; mirt's collapsing differs in its
   details (source `R/utils.R:2410`). Assumed: Orlando and Thissen's rule, with a setting for
   mirt's in the parity tests.
3. **Which abilities the mean squares use.** Wright and Masters use the Rasch model's estimates; mirt
   uses EAP estimates by default (source `R/itemfit.R:259-268`). Assumed: EAP, as mirt.

---

**Module:** decided by the requirements card (planned)
**Reference:** Orlando & Thissen (2000)
**Last validated:** 2026-11-27, against Orlando & Thissen (2000), Yen (1981), Wright & Masters (1982),
Wright & Stone (1979) and Zhang & Wang (2022) on the paper shelf, and mirt 1.48 (reference manual and
source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

Item fit compares, item by item, the responses a calibrated model predicts with the responses
observed. Statistics that group examinees by their estimated ability, such as Yen's (1981, p. 246)
$Q_1$, inherit the error in those estimates; Orlando and Thissen (2000) group examinees by their
number-correct score instead, which is observed, and compute the expected proportions from the model
itself.

## Definition

**S-X².** For item $i$ and number-correct score $k = 1, \dots, n - 1$, with $N_k$ examinees at score
$k$ and $O_{ik}$ the observed proportion of them answering item $i$ correctly,

$$
S\text{-}X^2_i = \sum_{k=1}^{n-1} N_k \frac{(O_{ik} - E_{ik})^2}{E_{ik}(1 - E_{ik})}
$$

(Orlando & Thissen, 2000, p. 54, Equation 13). The expected proportion $E_{ik}$ is the integral over
ability of $P_i$ times the likelihood of a score of $k - 1$ on the other items, divided by the
integral of the likelihood of a score of $k$, both weighted by the ability density (p. 54,
Equation 12); the summed-score likelihoods come from a recursion over items (p. 53, Equations 6-11,
after Lord and Wingersky). Orlando and Thissen integrate on equally spaced points from −4.5 to 4.5
(p. 54). Scores with small expected counts are collapsed toward the middle until every expected count
is at least 1, and the degrees of freedom are the number of rows after collapsing less the number of
item parameters estimated (p. 56).

**Mean squares** (Rasch model). With $z_{ni}$ the standardized residual of person $n$ on item $i$ and
$W_{ni}$ its variance, the unweighted mean square is the mean of $z_{ni}^2$ and the weighted mean
square is $\sum_n z_{ni}^2 W_{ni} / \sum_n W_{ni}$ (Wright & Masters, 1982, p. 99, Equations
5.4.6-5.4.7). Their expectation is 1, and they are standardized by a cube-root transformation
(p. 100, Table 5.4a; p. 101).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``min_expected``
     - 1
     - Smallest expected count a score group may have after collapsing (Orlando & Thissen, 2000,
       p. 56).
```

## Behavior and Comparison

On data from the fitted model, S-X² rejected at .04 to .07 for 10, 40 and 80 items, where $Q_1$
rejected at .95 to .96 for 10 items (Orlando & Thissen, 2000, p. 57, Table 1). Its power against
fitting the 1PL to 3PL data was .52 to .58 (same table).

mirt 1.48 computes S-X² by default in `itemfit()` with `mincell = 1`, and the mean squares as
`fit_stats = "infit"` from EAP ability estimates (source `R/itemfit.R:259-268`, `:576-580`). On
`LSAT7` it gives the values above.

## Guarantees

1. Every statistic comes with its degrees of freedom after collapsing.
2. An item whose collapsed table leaves no degrees of freedom is reported as such, not as a value.

## References

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response
theory models. *Applied Psychological Measurement, 24*(1), 50-64.
https://doi.org/10.1177/01466216000241003. Not openly available; Kang and Chen (2007, pp. 4-5)
describe the method.

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf

Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Best%20Test%20Design.pdf

Yen, W. M. (1981). Using simulation results to choose a latent trait model. *Applied Psychological
Measurement, 5*(2), 245-262. https://doi.org/10.1177/014662168100500212. Open copy:
https://hdl.handle.net/11299/100381

Zhang, X., & Wang, C. (2022). Modified item-fit indices for dichotomous IRT models with missing data.
*Applied Psychological Measurement, 46*(8), 705-719. https://doi.org/10.1177/01466216221125176. Open
copy: https://files.eric.ed.gov/fulltext/ED628568.pdf

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`). The summed-score recursion and the
expected proportions use the MML-EM card's grid and response probabilities.
