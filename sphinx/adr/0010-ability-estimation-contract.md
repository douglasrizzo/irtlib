# ADR-0010: The Ability Estimation Contract

## Status

Accepted 2026-11-27

## Context

ADR-0001 makes ability estimation irtlib's, and ADR-0004 moves catsim's estimators here: EAP, the
Bayes modal (MAP) estimate, maximum likelihood (ML) and Warm's weighted likelihood (WLE). catsim
runs them as strategy objects, one examinee per adaptive step. Calibration studies, item fit and
fixed-form scoring estimate many examinees at once, with missing responses.

Maximum likelihood has no finite estimate for a constant pattern, all correct or all incorrect.
Under the 3PL and 4PL a mixed pattern can lack one too, since the likelihood tends to a positive
limit at either end of the ability scale. The reference implementations differ there: mirt 1.48
reports an infinite estimate with no standard error (source `R/fscores.internal.R:397-405`), and
catR 3.17 by default the end of its search range, $[-4, 4]$ (source `R/thetaEst.R:1-10`,
`range` and `constantPatt = NULL`). catsim, a
CAT, steps toward the bank's hardest or easiest item, a rule that needs the session's current
estimate. The maintainer decided the behaviour below on 2026-11-27.

## Decision

- **Functions.** One function per estimator, taking `(responses, items, *, settings)`: a response
  matrix with one row per examinee and NaN for a missing response, the items, and the estimator's
  settings as keyword arguments with catsim's defaults. An estimator keeps no state.
- **The result.** An immutable value with, per examinee, the estimate, its standard error, the
  number of objective evaluations, and whether a fallback (below) gave the estimate.
- **Number-correct scoring.** The EAP by number correct is a table, one estimate per number correct
  from 0 to the number of items, from the items and the prior alone: every examinee who answers
  every item gets the entry of its number correct, so the function takes no response matrix.
- **Maximum likelihood** is the maximizer over the extended real line: $+\infty$ or $-\infty$
  wherever the likelihood's supremum is reached only in the limit, with a standard error of
  $+\infty$, since the information tends to 0 there. An examinee who answered nothing has no
  maximizer, a NaN estimate, and an infinite standard error; so under WLE.
- **Mitigation, at the caller's choice:** either `bounds`, which gives the maximizer within them, so
  every estimate is finite and a constant pattern gets the bound it points to; or `fallback`,
  another estimator, with its own settings, applied to the examinees without a finite estimate.
  Both together are an error.
- **What stays with catsim:** rules that depend on an adaptive session, such as Dodd's step or a
  starting estimate. catsim's estimators adapt these functions to one examinee's session.

## Consequences

- Fixed-form scoring, item fit and calibration studies estimate every examinee in one call.
- An infinite estimate is reported as such, so no arbitrary range enters a result unless the
  caller sets one.
- The parity tests record mirt's missing standard error and catR's default bound as departures, or
  as parity cases of the matching option.
- catsim keeps its behaviour by passing its bounds, or by applying its own rule before the call.

## Alternatives Considered

- Estimator classes, as catsim's: they hold only settings, which keyword arguments carry.
- A finite default, as catR's search range: an arbitrary bound enters every result.
- A missing standard error for an infinite estimate, as mirt's: inconsistent with the inverse
  square root of the information, which grows without bound.
- A fallback chosen by name, as catR's `constantPatt`: the fallback's prior and grid change its
  estimates, and a name cannot carry them.
