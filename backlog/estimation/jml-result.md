---
title: Return joint maximum likelihood's abilities, and let its correction scale the standard errors
kind: feature
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [maintainer on 2026-11-28 after the joint maximum likelihood PR]
---

# Return joint maximum likelihood's abilities, and let its correction scale the standard errors

## Why this card exists

The joint maximum likelihood PR (#33) took three positions and left each for the maintainer:

- it left the abilities out of the result, as ADR-0006 does for every method;
- it reported the standard errors of the estimates before the Rasch correction;
- it put JML's 1PL on the Rasch scale.

On 2026-11-28 the maintainer decided all three. JML estimates both the item and the examinee
parameters, so its result carries both: the joint abilities of the last cycle, in a new field of
`Calibration`. A boolean setting chooses whether the correction also scales the standard errors.
JML's 1PL stays on the Rasch scale, documented against MML's.

## Problem

- Every cycle of `irtlib.jml.JMLCalibrator` estimates each examinee's ability, and `calibrate`
  discards them. A user who wants JML's examinee parameters must estimate them again with
  `irtlib.ability.maximum_likelihood`, and gets estimates at the corrected difficulties rather than
  the joint ones.
- With the correction on, the difficulties are multiplied by $(M - 1)/M$ and their standard errors
  are not, with no way to have both on one footing.
- JML's 1PL (slopes 1, difficulties with mean 0) and MML's 1PL (one estimated slope, abilities
  $N(0, 1)$) report the same kind of fit on different scales. Neither docstring says so.

## Evidence

Read on 2026-11-28:

- **TAM 4.3-25.** `tam.jml` returns the abilities of its cycles, `theta`, and WLE abilities
  computed after convergence, `WLE` with `errorWLE`. Both are at the difficulties before
  `bias = TRUE` multiplies them by $(I - 1)/I$. The standard errors `errorP` are not multiplied
  (source `R/tam_jml_version2.R:216, 258-283, 300-301`).
- **sirt 4.2-133.** `rasch.jml` returns the abilities of its cycles, with $\pm\infty$ for extreme
  scores under `method = "MLE"`. It takes their standard errors at the corrected difficulties, and
  reports the corrected difficulties beside the uncorrected ones with one set of standard errors
  (source `R/rasch.jml.R:209-220, 237, 247`).
- **immer 1.5-13.** `immer_jml` returns `theta` and `theta_se` (source `R/immer_jml.R:312-313, 335`).
- Under JML, the Rasch model with abilities free to spread and a common slope with abilities
  standardized give the same joint likelihood, since $\theta' = a\theta$ and $b' = ab$ leave every
  logit $a(\theta - b) = \theta' - b'$ unchanged. The 1PL's two scales are one fit.

## What irtlib and catsim have now

- `src/irtlib/jml.py`:
  - `_Fit.theta` holds each group's ability at the last complete cycle, on the identified scale.
  - `_Groups.of` computes each examinee's group (`numpy.unique`'s `inverse`) and does not keep it.
  - The ability step keeps only `maximum_likelihood(...).theta` and drops its standard errors and
    evaluation counts.
  - `correct` is the only setting of the correction.
- `src/irtlib/calibration.py`: `Calibration` has `grid`, set exactly when the likelihood is
  marginal, enforced in `__post_init__`. It has no field for abilities.
- `src/irtlib/ability.py`: `AbilityEstimates(theta, standard_errors, evaluations, from_fallback)`.
- `pyproject.toml`: `ability` and `calibration` share the layer `_m_step | ability | calibration`,
  so neither may import the other.

## Proposal

Decided by the maintainer on 2026-11-28:

1. **`Calibration.abilities: AbilityEstimates | None`,** set exactly when the likelihood is joint
   and None otherwise. `__post_init__` enforces it, as it does `grid` for the marginal likelihood.
   - **The abilities:** the joint ones of the last complete cycle, on the identified scale and before
     any correction, so that they go with the uncorrected difficulties and the reported
     log-likelihood, as TAM's and sirt's do. There is one row per examinee, from the examinee's
     group.
   - **Their standard errors:** those of the last ability step, divided by the scale's standard
     deviation when the 2PL's abilities are standardized.
   - **Examinees left out** for every kept item right or wrong: $+\infty$ or $-\infty$ with an
     infinite standard error, as `maximum_likelihood` gives them without bounds.
   - **A calibration that stopped early** returns the last complete cycle's abilities.
   - **Abilities at the corrected difficulties** are one call away:
     `maximum_likelihood(responses, calibration.estimates)`. The docstring says so.
2. **`JMLCalibrator(..., correct_errors: bool = False)`.**
   - With False, the standard errors are those of the estimates before the correction, as TAM and
     sirt report them.
   - With True, the difficulties' standard errors are multiplied by $(M - 1)/M$ as well.
   - True raises a `ValueError` when no correction is applied: with `correct=False`, or for the 2PL.
3. **`JMLCalibrator`'s docstring** says that its 1PL fixes the slopes at 1 and the difficulties'
   mean at 0. This differs from `MMLCalibrator`'s 1PL, an estimated slope with abilities $N(0, 1)$,
   so results of the two compare only after rescaling.
4. **The layers.** `calibration` must import `AbilityEstimates`, so `ability` moves to the layer
   below. It imports nothing from `calibration`, so the new layers are `_m_step | calibration`
   above `ability`.
5. **ADR-0006.** Abilities are functions of the responses and the items, except a joint
   calibration's, which estimates them. Its layers sentence also changes. It must stay within 600
   words.

## Tests

- The abilities equal those of the brute-force maximum of the joint likelihood that
  `tests/test_jml.py` already writes out, for the Rasch model and the 2PL, with their standard
  errors $I(\hat\theta)^{-1/2}$ at the returned items.
- The two-item closed form:
  - every examinee kept has ability 0;
  - those left out have $\pm\infty$;
  - with `correct_errors=True` the standard errors are exactly half the uncorrected ones,
    $(M - 1)/M = 1/2$.
- `Calibration` refuses a joint likelihood without abilities, and abilities with any other
  likelihood.
- `correct_errors=True` raises with `correct=False` and with the 2PL.

## Open Questions

None: the maintainer decided the design on 2026-11-28.
