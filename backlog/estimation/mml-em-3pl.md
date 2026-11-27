---
title: Calibrate the 3PL by Bayes modal estimation with priors on the item parameters
kind: technique
status: blocked
priority: 2
effort: M
blocked_by: [discovery/requirements, estimation/mml-em]
raised_by: [moved from catsim's backlog]
---

# Calibrate the 3PL by Bayes modal estimation with priors on the item parameters

## Problem

The 3PL adds each item's guessing parameter $c$, and with it estimates that maximum likelihood alone
often leaves at a boundary or unstable. The remedy the sources and BILOG use is Bayes modal
estimation: the EM algorithm of [the MML-EM card](mml-em.md), with prior densities on the item
parameters added to the M-step (Mislevy, 1986, p. 182). Priors serve the 2PL too, where they keep
slopes finite (Baker, 1987, pp. 130-131), so the mechanism is the card's and the 3PL is its first use.

## Evidence

- The research pass (2026-11-27) checked the card carried over from catsim:
  - its bounds $c \in [0, 0.35]$, $a \in [0.1, 5]$ and $b \in [-6, 6]$ have no source; mirt sets no
    bounds (source `R/06-LoadPars.R:364-365`), and estimating $c$ on the logit scale keeps it inside
    $(0, 1)$ without one (Mislevy, 1986, p. 183);
  - its Beta(5, 17) prior on $c$ is PC-BILOG's default for five-option items (Harwell & Baker, 1991,
    p. 383, Equation 27), not Mislevy's (1986) prior, which is normal on the logit of $c$ (p. 183);
  - "c drifts toward 0 or its upper bound" is not stated in any source read; the likelihood is
    "nearly flat along one or more dimensions" of $(a, b, c)$ jointly (Mislevy, 1986, p. 189).
- A measurement on mirt 1.48 (2026-11-27), `LSAT7` (1,000 examinees, 5 items), `TOL = 1e-6`,
  `technical = list(NCYCLES = 2000)`:

  | Fit | EM cycles | Estimates of $c$ | log-likelihood |
  |---|---|---|---|
  | 3PL, no prior | 637 | 0.000 for 4 items, .298 for item 2 | −2658.0547 |
  | 3PL, logit $c \sim N(\text{logit}\,.2,\ .5^2)$ | 130 | .175 to .212 | −2659.1705 |

## What pyitemcal and catsim have now

- catsim: `catsim.estimation.LogPrior`, `normal_log_prior` and `uniform_log_prior`, priors on ability
  that ADR-0004 lets pyitemcal reuse.
- pyitemcal: nothing; the 2PL is [the MML-EM card](mml-em.md), the derivatives
  [the ICC-gradients card](../irt/icc-gradients.md).

## Proposal

The 3PL by MML-EM, with optional priors on the item parameters in the M-step, estimated on the
transformed scale $(\log a, b, \operatorname{logit} c)$ (Mislevy, 1986, p. 183). The open questions
are the requirements card's to settle; the card assumes the recommendation of each until then.

## Tests

- **Parity with mirt 1.48:** the 3PL with a `norm` prior on logit $c$ (manual p. 155), and the 2PL
  with a `lnorm` prior on $a$, on simulated data with missing entries; fixtures at `TOL = 1e-8`.
  Fixtures do not use mirt's `expbeta` prior. On the logit scale, the derivative of the
  Beta($\alpha_\beta$, $\beta_\beta$) log-density of $c$ is
  $c(1 - c)[(\alpha_\beta - 1)/c - (\beta_\beta - 1)/(1 - c)]$ (the card's chain rule); mirt 1.48's
  `expbeta` evaluates the beta log-density of $c$ (source `R/utils.R:1488-1497`) and adds
  $(\alpha_\beta - 1)/c - (\beta_\beta - 1)/(1 - c)$ to the gradient with respect to logit $c$
  (`R/utils.R:1435-1450`), so the two differ by that factor.
- **Prior gradients:** each prior's gradient and Hessian agree with finite differences of its log
  density on the scale the parameter is estimated on.
- **Flat priors:** with every prior flat, the estimates equal the maximum likelihood estimates.
- **Beta parameterization:** Beta($mp + 1$, $m(1 - p) + 1$) has mode $p$ (closed form); Harwell and
  Baker's (1991, pp. 383, 386) default Beta(5, 17) is $m = 20$, $p = .2$.
- **Recovery** (ADR-0003): 3PL responses from known parameters with a fixed seed; with the priors
  recommended below, every estimate within a multiple of its standard error.

## Open Questions

1. **The form of the prior on $c$.** A normal prior on logit $c$ (Mislevy, 1986, p. 183), which mirt's
   manual recommends for its logit-scale guessing parameter (p. 132); or a beta prior on $c$ itself,
   which allows $c = 0$ (Mislevy, 1986, p. 183) and is PC-BILOG's (Harwell & Baker, 1991, p. 383).
   Assumed: normal on logit $c$, as mirt.
2. **Default priors.** mirt sets no default prior (source `R/06-LoadPars.R:361-362`); BILOG's defaults are normal
   on $(\log a, b, \operatorname{logit} c)$ with variances .25, 4.00 and .25 (Mislevy, 1986, p. 190);
   PC-BILOG's are $\log a \sim N(0, .5^2)$, $b \sim N(0, 2^2)$ and $c \sim$ Beta(5, 17) (Harwell &
   Baker, 1991, p. 383). The measurement shows the 3PL without a prior putting four of five $c$'s at
   0. Assumed: no default prior, as mirt, with the 3PL's documentation recommending one; the
   requirements card decides, with a recovery measurement.
3. **Floating prior means.** BILOG can estimate the priors' means from the items (Mislevy, 1986,
   p. 190); the PC-BILOG manual advises against it for small data sets (Harwell & Baker, 1991,
   p. 386). Assumed: fixed hyperparameters (YAGNI).
4. **How a caller passes priors**, on which parameters and scales: the requirements card's
   question 6.
5. **Starting value of $c$.** mirt starts at .15 (source `R/04-PrepData.R:70`). Assumed: as mirt.

---

**Module:** decided by the requirements card (planned)
**Reference:** Mislevy (1986)
**Last validated:** 2026-11-27, against Mislevy (1986), Harwell & Baker (1991) and Baker (1987) on
the paper shelf, and mirt 1.48 (reference manual and source).
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

On a multiple-choice item, examinees of low ability still answer correctly by chance, which the 3PL
models with a lower asymptote $c$. Its likelihood is poorly determined: discrepant triples $(a, b, c)$
trace similar curves over the abilities where examinees are, so the likelihood is "nearly flat along
one or more dimensions", and Bayesian item estimation was motivated by wanting "reasonable" estimates,
"in particular, finite ones" (Mislevy, 1986, p. 189). Priors on the item parameters give them.

## Definition

The 3PL is catsim's response function with $d = 1$, $P = c + (1 - c)F$. For an item's parameters
$\xi$ with prior density $p(\xi)$, Bayes modal estimation maximizes the log-likelihood plus
$\log p(\xi)$. The EM algorithm carries over unchanged except that the M-step's likelihood equations
gain the derivatives of the log prior (Mislevy, 1986, p. 182, Equation 14). With the expected counts
$\bar r_{iq}$ and $\bar N_{iq}$ of the MML-EM card, the M-step equation for $c$ is
$(1 - c)^{-1}\sum_q [\bar r_{iq} - \bar N_{iq}P_i(X_q)]/P_i(X_q) = 0$ (Harwell & Baker, 1991,
p. 383, Equation 26), and those for the intercept and slope are Baker's (1987, p. 116) Equations
14-15 with the expected counts in place of the observed ones; each prior adds its term
(Harwell & Baker, 1991, p. 383). Estimated on the logit scale $\gamma$ that this card proposes,
the equation for $c$ becomes $c\sum_q [\bar r_{iq} - \bar N_{iq}P_i(X_q)]/P_i(X_q) = 0$ (Mislevy,
1986, p. 184, Equation 22), the equation above multiplied by $dc/d\gamma = c(1 - c)$.

Mislevy (1986, p. 183) estimates $\alpha = \log a$, $\beta = b$ and $\gamma = \operatorname{logit} c$,
so that $a = e^\alpha$ and $c = 1/(1 + e^{-\gamma})$ stay in range, with normal priors on the three,
and notes that a beta prior on untransformed $c$ may be preferable when $c = 0$ should be possible.
A beta prior Beta($\alpha_\beta$, $\beta_\beta$) is set by a weight $m$ and a location $p$ as
$\alpha_\beta = mp + 1$ and $\beta_\beta = m(1 - p) + 1$ (Harwell & Baker, 1991, p. 386); they call
$p$ the mean, but with these parameters $p$ is the mode (the card's arithmetic: the mode of
Beta($\alpha_\beta$, $\beta_\beta$) is $(\alpha_\beta - 1)/(\alpha_\beta + \beta_\beta - 2)$).

The posterior precision of the estimates is the information plus the negative second derivative of
the log prior (Mislevy, 1986, p. 182, Equations 15-17), so the standard errors of
[the standard-errors card](standard-errors.md) carry over with the prior's term added.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - priors on ``a``, ``b``, ``c``
     - Open Question 2
     - A prior density per parameter and item, on the scale the parameter is estimated on.
```

## Behavior and Comparison

Under BILOG's default logit-scale variance of .25, "if the logit average of lower asymptotes were
.25", most c's would lie between .15 and .35 (Mislevy, 1986, p. 190); the range is the prior's
spread, not a bound. Priors on $a$ and
$c$ keep the slopes from diverging and $c$ plausible, the Heywood problem of joint and marginal
estimation (Baker, 1987, pp. 130-131, 133).

mirt 1.48 fits the 3PL with `itemtype = "3PL"`:

- It stores the guessing parameter as a logit (manual p. 132; source `R/06-LoadPars.R:23`), starts it
  at .15 (`R/04-PrepData.R:70`), and sets no prior and no bound by default
  (`R/06-LoadPars.R:361-365`).
- Priors are given in the model syntax as `PRIOR = (items, parameter, type, value1, value2)`, with
  types `norm`, `lnorm`, `beta` and `expbeta` (manual p. 155); for the guessing parameter its manual
  recommends `norm`, on the logit scale (p. 132).
- It solves the M-step with BFGS, or with nlminb when a parameter has a finite bound (source
  `R/EMstep.group.R:118-122`).
- `guess` fixes the guessing parameter for an item type that does not estimate it, such as the 2PL,
  and only sets its starting value under `"3PL"` (source `R/06-LoadPars.R:85-91`); its manual
  describes `guess` as fixed (p. 124).

## Guarantees

1. Every estimate of $c$ lies strictly between 0 and 1.
2. With no priors, the estimates are maximum likelihood estimates.
3. A prior changes only the M-step, so the E-step's expected counts mean the same with or without it.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Harwell, M. R., & Baker, F. B. (1991). The use of prior distributions in marginalized Bayesian item
parameter estimation: A didactic. *Applied Psychological Measurement, 15*(4), 375-389.
https://doi.org/10.1177/014662169101500409. Open copy: https://hdl.handle.net/11299/114468

Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika, 51*(2),
177-195. https://doi.org/10.1007/BF02293979. Not openly available; Harwell and Baker (1991,
pp. 377-387) describe the method.

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`). The prior terms enter only the
M-step's objective, gradient and Hessian, so the 2PL, 3PL and 4PL share one implementation of them.
