---
title: Decide the calibrator contract and the shape of pyitemcal's API
kind: decision
status: needs-decision
priority: 1
effort: L
blocked_by: []
raised_by: [setup of pyitemcal]
---

# Decide the calibrator contract and the shape of pyitemcal's API

## Problem

Every method needs the same answers first: how a caller passes responses, what a calibration
returns, how it stops, and how a method author adds a method. Deciding them method by method would
give each method a different shape. The research pass has rebuilt the method cards from their
sources ([the literature review](literature-review.md)); this card turns what it found into the
decisions the maintainer makes before implementation starts, following the guidance for technique
choices in `AGENTS.md`.

## Evidence

- The research pass (2026-11-26 to 2026-11-27) rebuilt every technique card in `irt/`, `estimation/`
  and `fit/` from its sources and added the standard-errors and model-comparison cards; each card's
  open questions state the choices below. The feature card `estimation/free-threaded-m-step.md` came
  from a measurement, not from catsim, and was not part of the pass.
- Measurements on mirt 1.48 and on a scratch NumPy MML-EM for the 2PL that follows
  [the MML-EM card](../estimation/mml-em.md) (2026-11-27; the scratch code is not in the
  repository; it uses Fisher scoring and no acceleration):
  - on `LSAT7`, the scratch 2PL and mirt, both converged to `1e-12`, agree within
    $8 \times 10^{-7}$ for the intercepts and $1.4 \times 10^{-6}$ for the slopes;
  - mirt itself, at its default `TOL = 1e-4`, stopped after 28 cycles $1.7 \times 10^{-3}$ from its
    estimate converged to `1e-12`;
  - stopping rules in the scratch EM, against its estimates converged to `1e-12`; every rule follows
    the same path and stops at a different cycle:

    | Rule | Tolerance | `LSAT7`, 1,000 × 5: cycles, largest error | Simulated, 2,000 × 20, 10% missing | Simulated, 300 × 4, 30% missing (slow) |
    |---|---|---|---|---|
    | every parameter change below (mirt's rule) | `1e-4` | 46, $5.7 \times 10^{-4}$ | 19, $3.2 \times 10^{-4}$ | 6,421, $2.8 \times 10^{-1}$ |
    | every parameter change below (mirt's rule) | `1e-6` | 76, $6.1 \times 10^{-6}$ | 38, $3.7 \times 10^{-6}$ | 19,555, $2.9 \times 10^{-3}$ |
    | largest gradient of the marginal log-likelihood below | `1e-5` | 88, $1.0 \times 10^{-6}$ | 51, $1.7 \times 10^{-7}$ | 9,622, $8.9 \times 10^{-2}$ |

  - quadrature: with 41, 61 or 121 equally spaced nodes on $[-6, 6]$ the estimates agree within
    $10^{-8}$; with 21 they differ by up to $1.8 \times 10^{-4}$;
  - the 2PL fits of `LSAT7` and of the 2,000 × 20 simulation took under 0.1 seconds each; the 3PL
    without a prior took 637 cycles on `LSAT7`, and a 4PL with priors did not converge in 2,000
    ([the 3PL](../estimation/mml-em-3pl.md) and [4PL](../estimation/mml-em-4pl.md) cards).

## Proposal

Each question gives what the sources and mirt do, the options where there is a real choice, and a
recommendation, which states where it departs from mirt or from a card. Lasting answers go into
ADRs; the method cards' implementation guidance is then written against them.

1. **Calibrators.** Options: functions; one class per model; one class per method holding its
   configuration, with the model as a setting. mirt has one entry point with the model as an
   argument (manual p. 123). The 2PL, 3PL and 4PL share the E-step, and priors change only the M-step
   (the 3PL card). **Recommendation:** one class per method (MML-EM first), configured at
   construction, the model one of catsim's `IrtModel` values, and a method that takes the responses
   and returns a result.
2. **The response matrix.** One row per examinee, one column per item (the documentation charter).
   Options for missing responses: NaN in a float array, a separate mask, a sentinel. mirt takes NA
   (manual p. 123) and treats it as contributing nothing (the MML-EM card). **Recommendation:** a float
   array of 0, 1 and NaN, checked where it enters for its values, its shape, and items no examinee
   answered; an examinee who answered nothing is allowed, since it changes no estimate (the MML-EM
   card's second guarantee).
3. **The result.** An immutable value holding the estimates in catsim's $(a, b, c, d)$ as a validated
   `ItemBank`; their standard errors in the same parameterization, or none with a flag when the
   information matrix cannot be inverted (the standard-errors card, third question); the maximized
   log-likelihood and the number of estimated parameters (the model-comparison card); and the cycles
   run, whether the stopping rule was met, and its final value. Ability estimates for many examinees,
   which ADR-0004 assigns to pyitemcal and the mean squares of the item-fit card need, are a function of
   a result and the responses (EAP on the calibration's grid), not part of the result.
   **Recommendation:** as listed; whether MCMC returns draws or summaries is decided with that card,
   last in the build order.
4. **Stopping.** mirt stops when every parameter changes by less than `1e-4`, or after 500 cycles
   (manual pp. 127, 129). Its default tolerance stopped mirt $1.7 \times 10^{-3}$ short on `LSAT7`, and
   declared convergence of a 4PL whose slopes were still moving. A stopping rule on the gradient did
   as well as mirt's rule at a tighter tolerance on well-behaved data, but stopped 30 times further
   from the maximum on the slow case, so it is not better. **Recommendation:** mirt's rule with a
   default tolerance of `1e-6`: on the measured cases it cut the error about a hundredfold for 1.6 to
   3 times the cycles. Keep mirt's 500 cycles; when they run out, return the result flagged as
   not converged, with a warning. Departure from mirt: the tolerance only.
5. **Scale identification.** MML: ability normal (manual p. 122, `dentype = "Gaussian"`) with
   variance 1 (p. 131) and mean 0 (mirt's parameter table, `pars = "values"`, the MML-EM card's
   source notes). 1PL: a common slope estimated with $N(0, 1)$, equivalent to mirt's
   `"Rasch"` and faster to converge (the MML-EM card's measurement). CML: difficulties summing to
   zero. JML: LOGIST's and BICAL's conventions (the JML card). **Recommendation:** as listed.
6. **Priors on item parameters.** Options: Mislevy's normal priors on $\log a$, $b$ and
   $\operatorname{logit} c$ (Mislevy, 1986, p. 183), as the 3PL card proposes; or mirt's priors on its
   own parameters, a lognormal density on $a$, normal on the intercept, and normal on the logits of
   $c$ and $d$ (manual p. 155). The two give different modes: mirt's lognormal includes the $-1/a$
   term of the density of $a$ (source `R/utils.R:1422-1433`), which a normal prior on $\log a$ does
   not have, so the same hyperparameters give different estimates. mirt sets no default prior; without
   one, its 3PL put four of five guessing estimates at 0 on `LSAT7` (the 3PL card).
   **Recommendation:** mirt's priors and scales, one prior per parameter for all items (the 3PL card
   allows per-item priors; YAGNI), and no default prior, with the 3PL's documentation recommending
   one. Departure from the 3PL card: mirt's scales instead of Mislevy's, for parity.
7. **Starting values.** mirt starts slopes at 0.851 and intercepts at the inverse normal of the
   proportion correct (the MML-EM card); Bock and Aitkin (1981, p. 458) name the inverse normal of the
   proportion correct for the intercepts. With missing responses, mirt's proportion counts a missing
   response as correct (source `R/05-model.elements.R:50` with `R/04-PrepData.R:107`).
   **Recommendation:** mirt's formula on the answered responses only, with the caller able to pass
   starting items. The starting point does not change the estimates; on the simulated data with 10%
   missing, the answered-only start converged at `1e-6` in 38 cycles against mirt's 50 (scratch EM).
   Departure from mirt: the proportion over answered responses.
8. **Models and fixed parameters.** The 1PL (common slope), 2PL, 3PL and 4PL as catsim's `IrtModel`.
   mirt's `guess` and `upper` set only starting values under the 3PL and 4PL (the 3PL card).
   **Recommendation:** a parameter can be fixed at one value for all items, which expresses mirt's
   `"1PL"` (slopes fixed at 1, the MML-EM card's first question) and the 4PL card's second guarantee
   ($d$ fixed at 1); per-item fixing waits for a need (YAGNI).
9. **Item fit.** A function of a calibration result and the responses, returning S-X² with its degrees
   of freedom per item, and the weighted and unweighted mean squares for the Rasch model, from EAP
   abilities ([the item-fit card](../fit/item-fit.md)). **Recommendation:** as described.
10. **Names.** catsim's one-name-per-concept table for every shared name. **Recommendation:** as
    `AGENTS.md` requires.
11. **Modules and layers.** **Recommendation:** item response derivatives, then estimation (the
    shared E-step and M-step, then each method), then standard errors, ability estimation and fit, as
    the import-linter layers; the drawing of responses for recovery tests in a module of its own.
12. **The grid.** 61 equally spaced nodes on $[-6, 6]$, mirt's (manual pp. 127, 129); 41 or more gave
    the same estimates within $10^{-8}$ in the measurement. **Recommendation:** catsim's
    `QuadratureGrid.uniform(61)` with `normal_log_prior()`, as the default.
13. **The M-step's parameters and solver.** Options for the parameters: mirt's slope, intercept and
    logits of $c$ and $d$ (manual pp. 132-133), or Mislevy's $(\log a, b, \operatorname{logit} c)$, as
    the 3PL card proposes. Options for the solver: Newton or Fisher scoring per item (Baker, 1987,
    p. 114), which the ICC-gradients card assumes, or a general optimizer as mirt's BFGS (manual
    p. 126); the solver does not change the maximum. In the intercept-slope form the 2PL's Hessian is
    the information (Baker, 1987, p. 116), and the scratch EM's Fisher scoring there reached mirt's
    estimates. **Recommendation:** mirt's parameters, which its priors (question 6) are defined on,
    and Fisher scoring per item with step halving, converting estimates and standard errors to
    $(a, b, c, d)$ at the end. Departure: the solver, which leaves the estimates unchanged; the 3PL
    card's scales and the free-threaded card's L-BFGS-B sketch are revised to match once this card is
    decided.
14. **Standard errors.** Oakes's identity with central differences, mirt's default, at a mean
    relative difference of $2.5 \times 10^{-11}$ from Louis's exact form and 3% of its time
    ([the standard-errors card](../estimation/standard-errors.md)). **Recommendation:** as mirt.
15. **Acceleration.** None in the first version: the fixed point does not depend on it, and the 2PL
    fits measured took under 0.1 seconds (performance is not a requirement, `AGENTS.md`). The 3PL
    without a prior and the 4PL took hundreds to thousands of cycles, so acceleration is measured again
    when they are implemented. **Recommendation:** none, for now.
16. **Missing responses by method.** MML-EM handles them (its card). Ordinary CML "will likely fail"
    with data missing at random, though it applies when they are missing completely at random
    (Robitzsch, 2021, p. 827, in the CML card), and S-X² "cannot handle incomplete data directly"
    (the item-fit card). The JML card does not address them. **Recommendation:** CML and S-X² refuse
    missing responses with an error naming the alternative; JML's handling is decided in its card.
17. **Items with no finite estimate.** An item answered correctly by every examinee or by none has no
    finite maximum likelihood estimate; mirt's manual advises removing such items or using priors
    (p. 132), and the JML card removes and reports them. **Recommendation:** MML-EM raises an error
    listing them unless priors are given, which keep their estimates finite; JML removes and reports
    them, as its card says.
18. **Parity tolerances** (for [the parity-harness card](mirt-parity-harness.md)). Fixtures recorded
    with mirt at `TOL = 1e-8`, because its default stops short; estimates compared within `1e-5`,
    which the measured 2PL agreement of about $10^{-6}$ leaves room for; closed forms within a relative
    `1e-10`. The 3PL and 4PL fixtures need cases that converge at that tolerance, which `LSAT7` does
    not give for the 4PL. **Recommendation:** as listed, with the 3PL and 4PL tolerances confirmed by
    their fixtures.

## Tests

The decision is done when each question has an answer, every lasting answer is in an ADR, and every
method card's implementation guidance follows them.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Harwell, M. R., & Baker, F. B. (1991). The use of prior distributions in marginalized Bayesian item
parameter estimation: A didactic. *Applied Psychological Measurement, 15*(4), 375-389.
https://doi.org/10.1177/014662169101500409. Open copy: https://hdl.handle.net/11299/114468

Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika, 51*(2),
177-195. https://doi.org/10.1007/BF02293979. Not openly available; Harwell and Baker (1991,
pp. 377-387) describe the method.

Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
*Stats, 4*(4), 814-836. https://doi.org/10.3390/stats4040048. Open access.
