---
title: Compare calibrated models by likelihood ratio, AIC and BIC
kind: technique
status: blocked
priority: 3
effort: S
blocked_by: [discovery/requirements, estimation/mml-em]
raised_by: [research pass]
---

# Compare calibrated models by likelihood ratio, AIC and BIC

## Problem

Choosing between the 1PL, 2PL, 3PL and 4PL for a set of items is the first decision of a calibration
study. The marginal log-likelihood and the number of parameters that MML-EM reports are enough to
compare nested models by a likelihood-ratio test and any models by information criteria; the card
records the definitions, their sources and how they behave.

## Evidence

- The research pass (2026-11-27) found no card for model comparison, which its starting list
  named.
- A measurement on mirt 1.48 (2026-11-27), `LSAT7` at `TOL = 1e-8`: Rasch against 2PL, the
  likelihood-ratio statistic is 12.192 on 4 degrees of freedom, $p = .016$; the 2PL's
  log-likelihood is −2658.805 with 10 parameters, so its AIC is 5337.610 and its BIC 5386.688 for
  1,000 examinees. The Rasch model's $G^2$ against the multinomial is 21.7981 on `LSAT6` and 43.8921
  on `LSAT7`, each on 25 degrees of freedom.

## What pyitemcal and catsim have now

Nothing; the log-likelihood comes from [the MML-EM card](../estimation/mml-em.md).

## Proposal

Report each calibration's maximized log-likelihood and number of estimated parameters, and provide
the likelihood-ratio test of nested models, AIC, BIC, and $G^2$ against the general multinomial for
complete data.

## Tests

- **Published values:** the Rasch $G^2$ of Bock and Aitkin (1981, p. 455, Table 3), 21.80 for
  LSAT-6 and 43.90 for LSAT-7 on 25 degrees of freedom, which mirt reproduces (Evidence); and
  Thissen's (1982, p. 178) $G^2 = 21.8$ for the 1PL and 21.2 for the 2PL on LSAT-6, a difference of
  0.6 on 4 degrees of freedom.
- **Parity with mirt 1.48:** `anova()` on `LSAT7`, the values above.
- **Closed forms:** AIC and BIC from a hand-computed log-likelihood.

## Open Questions

1. **Testing a parameter on its boundary.** The likelihood-ratio test of the 3PL against the 2PL tests
   $c = 0$, on the boundary of its range; mirt offers a 50:50 mixture of chi-square distributions for
   such tests (`anova(..., bounded = TRUE)`, manual p. 6; source `R/SingleGroup-methods.R:611-616`).
   No source on the shelf derives it. Assumed: the ordinary test, with the boundary stated, until a
   source is read.
2. **Which criteria.** mirt also reports a sample-size adjusted BIC and Hannan-Quinn's criterion
   (source `R/03-estimation.R:1053-1056`). Assumed: AIC and BIC only (YAGNI).
3. **$G^2$ with missing responses.** The multinomial over response patterns is not defined when
   responses are missing. Assumed: reported for complete data only.

---

**Module:** decided by the requirements card (planned)
**Reference:** Akaike (1974); Schwarz (1978)
**Last validated:** 2026-11-27, against Akaike (1974), Schwarz (1978), Vrieze (2012), Bock & Aitkin
(1981), Thissen (1982) and Robitzsch (2022) on the paper shelf, and mirt 1.48 (reference manual and
source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

A model with more parameters always fits at least as well; the question is whether the improvement
outweighs the parameters spent. Likelihood-ratio tests answer it for nested models; information
criteria rank any set of models, and differ in what they reward.

## Definition

- **Likelihood ratio.** For nested models, twice the difference of their maximized log-likelihoods,
  referred to a chi-square distribution with the difference in the number of parameters; Thissen
  (1982, p. 178) compares the 1PL and 2PL on LSAT-6 this way.
- **AIC** $= -2\log(\text{maximum likelihood}) + 2k$, for $k$ estimated parameters (Akaike, 1974,
  p. 719).
- **BIC.** Schwarz (1978, p. 461) chooses "the model for which $\log M_j(X_1, \dots, X_n) - \frac12 k_j
  \log n$ is largest"; multiplied by −2, it is $-2\log L + k\log n$ (the card's arithmetic), with $n$
  the number of examinees.
- **$G^2$ against the general multinomial** over the observed response patterns, on $s - 2n$
  degrees of freedom for $s$ observed patterns and $n$ items in the 2PL (Bock & Aitkin, 1981, p. 450,
  Equation 16).

## Parameters

None: each statistic is computed from fitted models.

## Behavior and Comparison

BIC is consistent: it selects the true model as the sample grows when that model is among the
candidates, and AIC is not; AIC is efficient when no candidate is true (Vrieze, 2012, Pages 10-11).
For choosing between the 4PL and related guessing models, Robitzsch (2022, p. 11) concludes that "AIC
should be preferred over BIC".

mirt 1.48 reports AIC, the sample-size adjusted BIC, Hannan-Quinn's criterion and BIC (source
`R/03-estimation.R:1053-1056`), and its `anova()` gives the likelihood-ratio statistic, twice the
difference of the log-likelihoods (`R/SingleGroup-methods.R:611-616`).

## Guarantees

1. Every comparison names the models, their log-likelihoods and their numbers of parameters.
2. The likelihood-ratio test refuses models that are not nested.

## References

Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on
Automatic Control, 19*(6), 716-723. https://doi.org/10.1109/TAC.1974.1100705. Not openly available;
Vrieze (2012, Pages 5-6) describes the criterion.

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Robitzsch, A. (2022). Four-parameter guessing model and related item response models. *Mathematical
and Computational Applications, 27*(6), 95. https://doi.org/10.3390/mca27060095. Open access.

Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics, 6*(2), 461-464.
https://doi.org/10.1214/aos/1176344136. Open copy:
https://projecteuclid.org/journals/annals-of-statistics/volume-6/issue-2/Estimating-the-Dimension-of-a-Model/10.1214/aos/1176344136.pdf

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.

Vrieze, S. I. (2012). Model selection and psychological theory: A discussion of the differences
between the Akaike information criterion (AIC) and the Bayesian information criterion (BIC).
*Psychological Methods, 17*(2), 228-243. https://doi.org/10.1037/a0027127. Open copy (author
manuscript, cited by its page numbers): https://pmc.ncbi.nlm.nih.gov/articles/PMC3366160/

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`), which decides what a calibration
result reports.
