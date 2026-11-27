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
give each method a different shape. The research pass has rebuilt every method card from its
sources ([the literature review](literature-review.md)); this card turns what it found into the
decisions the maintainer makes before implementation starts.

## Evidence

- The research pass (2026-11-26 to 2026-11-27) rewrote every card in `irt/`, `estimation/` and
  `fit/`, with each card's open questions stating the choices below.
- Measurements on mirt 1.48 and on a scratch NumPy MML-EM for the 2PL that follows
  [the MML-EM card](../estimation/mml-em.md) (2026-11-27; the scratch code is not in the
  repository):
  - the scratch 2PL reproduces mirt's estimates on `LSAT7`, both converged to `1e-12`, within
    $8 \times 10^{-7}$ for the intercepts and $1.4 \times 10^{-6}$ for the slopes;
  - stopping rules, against the estimates converged to `1e-12`:

    | Rule | Tolerance | `LSAT7`: cycles, largest error | Simulated, 2,000 × 20, 10% missing: cycles, largest error |
    |---|---|---|---|
    | every parameter change below (mirt's rule) | `1e-4` (mirt's default) | 46, $5.7 \times 10^{-4}$ | 19, $3.2 \times 10^{-4}$ |
    | every parameter change below | `1e-6` | 76, $6.1 \times 10^{-6}$ | 38, $3.7 \times 10^{-6}$ |
    | log-likelihood change below | `1e-8` | 60, $6.9 \times 10^{-5}$ | 35, $7.5 \times 10^{-6}$ |
    | largest gradient of the marginal log-likelihood below | `1e-5` | 88, $1.0 \times 10^{-6}$ | 51, $1.7 \times 10^{-7}$ |

  - quadrature: with 41, 61 or 121 equally spaced nodes on $[-6, 6]$ the estimates agree within
    $10^{-8}$; with 21 they differ by up to $1.8 \times 10^{-4}$;
  - every fit above took under 0.1 seconds;
  - mirt's default rule reported convergence of a 4PL on `LSAT7` at cycle 85 with slopes up to 22.2,
    where at `1e-6` it did not converge in 2,000 cycles ([the 4PL card](../estimation/mml-em-4pl.md)).

## The maintainer's guidance for technique choices

Given on 2026-11-27, for this card and every implementation choice after it:

- Compare and contrast what the papers teach with what the R packages implement; drift from them only
  if the alternative has a measurable upside.
- Weigh the trade-offs of adding dependencies, matching mirt, speed and the like; scratch
  implementations may decide between techniques.
- YAGNI and DRY: look for the parts techniques can share.

## Proposal

Each question gives the options, what the sources and mirt do, and a recommendation. Lasting answers
go into ADRs; the method cards' implementation guidance is then written against them.

1. **Calibrators.** Options: functions; one class per model; one class per method holding its
   configuration, with the model as a setting. mirt has one entry point with the model as an argument
   (manual p. 123). The 2PL, 3PL and 4PL share the E-step, and priors change only the M-step
   ([the 3PL card](../estimation/mml-em-3pl.md)), so a class per model would repeat them.
   **Recommendation:** one class per method (MML-EM first), configured at construction, the model one
   of catsim's `IrtModel` values, and a method that takes the responses and returns a result.
2. **The response matrix.** One row per examinee, one column per item (the documentation charter's
   convention). Options for missing responses: NaN in a float array, a separate mask, a sentinel.
   mirt takes NA (manual p. 123) and treats it as contributing nothing (the MML-EM card).
   **Recommendation:** a float array of 0, 1 and NaN, checked where it enters: values, shape, and
   items or examinees with no responses.
3. **The result.** An immutable value holding:
   - the estimates in catsim's $(a, b, c, d)$, as a validated `ItemBank`;
   - their standard errors in the same parameterization;
   - the maximized log-likelihood and the number of estimated parameters, which model comparison
     needs ([the model-comparison card](../fit/model-comparison.md));
   - the cycles run, whether the stopping rule was met, and its final value.

   Ability estimates stay out (YAGNI; catsim estimates abilities). **Recommendation:** as listed.
4. **Stopping.** mirt stops when every parameter changes by less than `1e-4`; the measurements above
   show that rule leaving errors about five times the tolerance, and declaring convergence on a 4PL
   whose slopes were still moving. The gradient of the marginal log-likelihood is zero at the maximum,
   costs nothing beyond the E-step, and reached errors near $10^{-6}$ at `1e-5` in about twice mirt's
   default cycles. **Recommendation:** stop when the largest gradient is below `1e-5`, with mirt's rule
   available as a setting for the parity tests; when the cycle limit is reached, return the result
   flagged as not converged, with a warning.
5. **Scale identification.** MML: ability $N(0, 1)$, as mirt (manual p. 131). 1PL: a common slope
   estimated with $N(0, 1)$, equivalent to mirt's `"Rasch"` and faster to converge (the MML-EM card's
   measurement). CML: difficulties summing to zero. JML: LOGIST's and BICAL's conventions (the JML
   card). **Recommendation:** as listed.
6. **Priors on item parameters.** Normal priors on $\log a$, $b$, $\operatorname{logit} c$ and
   $\operatorname{logit} d$ (Mislevy's scales; mirt's `lnorm` and `norm`), one per parameter for all
   items; per-item priors only when a need appears (YAGNI). mirt sets no default prior; without one,
   its 3PL put four of five guessing estimates at 0 on `LSAT7` (the 3PL card). **Recommendation:** no
   default priors, as mirt, with the 3PL's documentation recommending one; reuse catsim's `LogPrior`
   for the ability distribution only.
7. **Starting values.** mirt's: slopes 0.851, intercepts from the inverse normal of the proportion
   correct (the MML-EM card), which Bock and Aitkin also name. **Recommendation:** mirt's, with the
   caller able to pass starting items.
8. **Models and fixed parameters.** The 1PL (common slope), 2PL, 3PL and 4PL as catsim's `IrtModel`.
   Fixing a parameter at a value: mirt's `guess` and `upper` set only starting values under the 3PL and
   4PL (the 3PL card). **Recommendation:** no fixed parameters in the first version (YAGNI); the 3PL
   with $d = 1$ and the 2PL with $c = 0$ cover the common cases.
9. **Item fit.** A function of a calibration result and the responses, returning S-X² with its degrees
   of freedom per item ([the item-fit card](../fit/item-fit.md)), in a module that depends on the
   estimation modules. **Recommendation:** as described.
10. **Names.** catsim's one-name-per-concept table for every shared name. **Recommendation:** as
    `AGENTS.md` requires.
11. **Modules and layers.** **Recommendation:** item response derivatives, then estimation (shared
    E-step and M-step, then each method), then standard errors and fit, as the import-linter layers;
    the drawing of responses for recovery tests in a module of its own.
12. **The grid.** 61 equally spaced nodes on $[-6, 6]$, mirt's (manual pp. 127, 129); 41 or more gave
    the same estimates within $10^{-8}$ in the measurement. **Recommendation:** catsim's
    `QuadratureGrid.uniform(61)` with `normal_log_prior()`, as the default.
13. **The M-step's parameterization.** In the intercept-slope form the 2PL's Hessian is the information
    (Baker, 1987, p. 116), mirt estimates in it (manual p. 133), and the scratch M-step converged in it
    to mirt's estimates. **Recommendation:** estimate in the intercept-slope form with logits for $c$
    and $d$, and convert the estimates and standard errors to $(a, b, c, d)$ at the end
    ([the ICC-gradients card](../irt/icc-gradients.md), first open question).
14. **Standard errors.** Oakes's identity with central differences, mirt's default, at a mean
    relative difference of $2.5 \times 10^{-11}$ from Louis's exact form and 3% of its time
    ([the standard-errors card](../estimation/standard-errors.md)). **Recommendation:** as mirt.
15. **Acceleration.** None in the first version: the fixed point does not depend on it, and every fit
    measured took under 0.1 seconds (performance is not a requirement, `AGENTS.md`).
    **Recommendation:** none.
16. **Missing responses by method.** MML-EM and JML handle them; ordinary CML and S-X² are defined on
    complete data (the CML and item-fit cards). **Recommendation:** those two raise an error naming the
    alternative when responses are missing.
17. **Items with no finite estimate.** An item answered correctly by every examinee or by none has no
    finite estimate. **Recommendation:** an error listing the items, checked where the responses
    enter, so the caller removes them deliberately.
18. **Parity tolerances** (for [the parity-harness card](mirt-parity-harness.md)). Fixtures recorded
    with mirt at a tight tolerance (`TOL = 1e-8`), because its default stops short (the MML-EM card);
    estimates compared within `1e-5`, which the measured agreement of about $10^{-6}$ leaves room for;
    closed forms within a relative `1e-10`. **Recommendation:** as listed.

## Tests

The decision is done when each question has an answer, every lasting answer is in an ADR, and every
method card's implementation guidance follows them.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt
