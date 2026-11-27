---
title: Calibrate the 4PL by Bayes modal estimation
kind: technique
status: blocked
priority: 3
effort: S
blocked_by: [discovery/requirements, estimation/mml-em-3pl]
raised_by: [moved from catsim's backlog]
---

# Calibrate the 4PL by Bayes modal estimation

## Problem

The 4PL adds an upper asymptote $d < 1$: examinees of high ability who still miss an item, as by a
clerical error (Barton & Lord, 1981, p. 2). catsim's model has $d$, so a bank calibrated by pyitemcal
can carry it, but estimating it needs many examinees of high ability, and maximum likelihood alone
gives boundary and diverging estimates.

## Evidence

- The research pass (2026-11-27) checked the card carried over from catsim against Loken and Rulison
  (2010), Culpepper (2016), Waller and Feuerstahler (2017), Battauz (2020) and Robitzsch (2022):
  - "a Beta prior on $d$ (e.g., Beta(20, 2)) is essentially mandatory" and the bounds
    $d \in [0.8, 1.0]$ have no source; the priors the sources use are below;
  - "recovery requires $N > 5000$ and $I > 50$": the sample size has support (Waller & Feuerstahler,
    2017, p. 19), the test length none; Culpepper (2016, Section 5) finds $N \geq 2500$ enough for
    educational items;
  - "the 4PL is rarely used in operational testing" is supported (Loken & Rulison, 2010, p. 510;
    Robitzsch, 2022, p. 3).
- A measurement on mirt 1.48 (2026-11-27): the 4PL on `LSAT7` (1,000 examinees, 5 items), with normal
  priors of standard deviation .5 on logit $c$ and logit $d$ centred at logit .2 and logit .8, did not
  converge in 2,000 EM cycles, with slopes between 2.5 and 22.4.

## What pyitemcal and catsim have now

- catsim: the 4PL response function (`catsim.irt.icc`).
- pyitemcal: nothing; the 3PL and the priors are [the 3PL card](mml-em-3pl.md).

## Proposal

The 4PL by the 3PL card's Bayes modal MML, with $d$ estimated on the logit scale and a prior on it.
The open questions are the requirements card's to settle.

## Tests

- **Parity with mirt 1.48:** the 4PL with `norm` priors on logit $c$ and logit $d$ (manual p. 155),
  on simulated data of at least 5,000 examinees; fixtures at `TOL = 1e-8`.
- **Recovery** (ADR-0003): Waller and Feuerstahler's (2017, p. 9) design, tests of 20, 40 and 60
  items at $N$ from 1,000 to 20,000, against their bias and RMSE (Table 4).
- **Published values:** Battauz (2020, pp. 275-276, Table 5), the ISRD-2 data, 18 items, $N = 3247$,
  maximum likelihood without priors and with the 1.7 constant absent, as in catsim.

## Open Questions

1. **The prior on $d$.** The sources use Beta(17, 5) on $d$ (Loken & Rulison, 2010, p. 513), a
   uniform prior between .7 and 1 with "very similar results" (p. 520), uniform priors when the sample
   is adequate (Culpepper, 2016, Section 5), and logit $d \sim N(1.8, 1)$ (Waller & Feuerstahler,
   2017, p. 4). Assumed: a normal prior on logit $d$, the 3PL card's form.
2. **A penalty instead of a prior.** Battauz (2020) shrinks each item's parameters toward their common
   mean; the penalty equals a normal log-density (p. 277), and it cut the RMSE of $a$ from 39.30 to
   11.59 at $n = 500$ in the first simulation setting (p. 273, Table 1). Assumed: priors only
   (YAGNI), since the 3PL card's priors cover the same ground.
3. **Fixing $d$ on some items.** mirt's `upper` sets only the starting value under `"4PL"` (source
   `R/06-LoadPars.R:85-91`); fixing is the requirements card's question 8.

---

**Module:** decided by the requirements card (planned)
**Reference:** Waller & Feuerstahler (2017); Loken & Rulison (2010)
**Last validated:** 2026-11-27, against Barton & Lord (1981), Loken & Rulison (2010), Culpepper
(2016), Waller & Feuerstahler (2017), Battauz (2020) and Robitzsch (2022) on the paper shelf, and
mirt 1.48 (reference manual and source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

Barton and Lord (1981, p. 2) introduced the upper asymptote for high-ability students making a
clerical error on an easy item. They fixed it at a common value for all items (p. 3) and found "no
compelling reason to urge the use of this model" (p. 6). Interest returned with Bayesian estimation:
Loken and Rulison (2010, p. 523) fitted it by Markov chain Monte Carlo, though they "do not mean to
imply that fitting a 4PM necessarily requires a Bayesian approach", and Waller and Feuerstahler
(2017) by Bayes modal estimation. It is "not yet widely applied in the operational practice of
educational studies" (Robitzsch, 2022, p. 3).

## Definition

The 4PL is catsim's response function, $P = c + (d - c)F$. Bayes modal estimation is the 3PL card's,
with $d$ estimated as logit $d$ and a prior on it, and its M-step uses the derivative with respect to
$d$ from [the ICC-gradients card](../irt/icc-gradients.md).

Several sources parameterize the upper asymptote differently: Robitzsch (2022) and Culpepper (2016)
estimate a slipping probability, one minus the upper asymptote, and Loken and Rulison (2010),
Culpepper (2016) and Waller and Feuerstahler (2017) include the 1.7 constant. Their values convert
before they compare with catsim's.

## Parameters

As the 3PL card, with a prior on $d$ (Open Question 1).

## Behavior and Comparison

- **Without priors**, maximum likelihood puts many upper asymptotes at the boundary: 7 of Battauz's
  (2020, pp. 275-276, Table 5) 18 items have $\hat d = 1.00$, and one item's slope is 55.90; with no
  priors, 18 of Robitzsch's (2022, p. 9) 70 items have a slipping estimate of zero.
- **With Bayes modal estimation**, Waller and Feuerstahler (2017, p. 19) conclude that "at least
  $N = 5{,}000$" examinees are needed when accurate $d$ estimates matter; with 1,000 examinees, most
  of their simulated data sets failed (p. 6).
- **By Markov chain Monte Carlo** on the normal-ogive 4PL, Culpepper (2016, Section 5) finds
  $N \geq 2500$ enough for educational items and up to 10,000 for psychopathology items.

mirt 1.48 fits the 4PL with `itemtype = "4PL"`: it stores the upper asymptote as a logit (manual
p. 132; source `R/06-LoadPars.R:24`), starts it at .85 (`R/04-PrepData.R:71`), and sets no prior and
no bound by default (`R/06-LoadPars.R:361-365`). `upper` sets only the starting value under
`"4PL"` (`R/06-LoadPars.R:85-91`).

## Guarantees

1. Every estimate satisfies $0 < c$ and $d < 1$.
2. A 4PL with every $d$ fixed at 1 gives the 3PL's estimates.

## References

Barton, M. A., & Lord, F. M. (1981). *An upper asymptote for the three-parameter logistic
item-response model* (Research Report RR-81-20). Educational Testing Service.
https://doi.org/10.1002/j.2333-8504.1981.tb01255.x. Open copy:
https://files.eric.ed.gov/fulltext/ED207996.pdf

Battauz, M. (2020). Regularized estimation of the four-parameter logistic model. *Psych, 2*(4),
269-278. https://doi.org/10.3390/psych2040020. Open access.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Culpepper, S. A. (2016). Revisiting the 4-parameter item response model: Bayesian estimation and
application. *Psychometrika, 81*(4), 1142-1163. https://doi.org/10.1007/s11336-015-9477-6. Not
openly available; Zhang, Lu, Du and Zhang (2020) describe the method. Cited by section: the shelf copy
is the online version, without page numbers.

Harwell, M. R., & Baker, F. B. (1991). The use of prior distributions in marginalized Bayesian item
parameter estimation: A didactic. *Applied Psychological Measurement, 15*(4), 375-389.
https://doi.org/10.1177/014662169101500409. Open copy: https://hdl.handle.net/11299/114468

Loken, E., & Rulison, K. L. (2010). Estimation of a four-parameter item response theory model.
*British Journal of Mathematical and Statistical Psychology, 63*(3), 509-525.
https://doi.org/10.1348/000711009X474502. Open copy (accepted manuscript):
https://libres.uncg.edu/ir/uncg/f/K_Rulison_Estimation_2010.pdf

Robitzsch, A. (2022). Four-parameter guessing model and related item response models. *Mathematical and
Computational Applications, 27*(6), 95. https://doi.org/10.3390/mca27060095. Open access.

Waller, N. G., & Feuerstahler, L. (2017). Bayesian modal estimation of the four-parameter item response
model in real, realistic, and idealized data sets. *Multivariate Behavioral Research, 52*(3), 350-370.
https://doi.org/10.1080/00273171.2017.1292893. Not openly available; Harwell and Baker (1991) describe
the Bayes modal estimation it applies. Cited by the online version's pages, 1-21.

Zhang, J., Lu, J., Du, H., & Zhang, Z. (2020). Gibbs-slice sampling algorithm for estimating the
four-parameter logistic model. *Frontiers in Psychology, 11*, 2121.
https://doi.org/10.3389/fpsyg.2020.02121. Open access.

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`) and the 3PL card.
