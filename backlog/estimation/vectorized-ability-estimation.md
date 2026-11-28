---
title: Estimate the abilities of many examinees in one vectorized search, by ML, MAP and WLE
kind: feature
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [joint maximum likelihood PR on 2026-11-27]
---

# Estimate the abilities of many examinees in one vectorized search, by ML, MAP and WLE

## Why this card exists

The joint maximum likelihood PR (#33) found that a 2PL calibration spends almost all its time in
the ability step, one maximum likelihood search per response pattern per cycle. The maintainer then
asked on 2026-11-28 what estimating many examinees at once would take. A scratch prototype
(Evidence) answered it: 29 to 49 times faster, with the same estimates.

## Problem

`irtlib.ability` takes many examinees at once (ADR-0010), but its maximum likelihood, MAP and WLE
estimators search one examinee at a time in a Python loop. This is slow wherever many examinees are
scored:

- scoring a calibration sample;
- the ability step of every cycle of `irtlib.jml`;
- simulation studies.

A 2PL joint calibration takes 1.6 s on 20 items and 400 examinees and 4.8 s on 50 items and 1,000,
against 47 ms and 209 ms for `irtlib.mml` on the same data, almost all of it in the ability step
(the joint maximum likelihood PR, 2026-11-27).

## Evidence

A scratch prototype on 2026-11-28 (Apple M4 Pro, Python 3.13.6, NumPy 2.5.5, SciPy 1.18.1) runs
the search `maximum_likelihood` then had with `bounds=(-6, 6)`, before its one-examinee search
moved to Newton's method on 101 points:

1. it evaluates 201 equally spaced abilities for every examinee in one matrix product;
2. it refines each examinee's best point between its neighbours in a single call of
   `scipy.optimize.elementwise.find_minimum`.

The data were random 4PL items (slopes uniform on [0.8, 2], difficulties N(0, 1), $c$ uniform on
[0, 0.25], $d$ uniform on [0.9, 1]), normal abilities and 10% of responses missing. The table gives
the best of three runs:

| Items | Examinees | `maximum_likelihood` | Prototype | Ratio | Largest difference in $\hat\theta$ |
|---|---|---|---|---|---|
| 20 | 400 | 56.3 ms | 2.0 ms | 29 | $5.9 \times 10^{-7}$ |
| 20 | 5,000 | 706.4 ms | 14.4 ms | 49 | $6.9 \times 10^{-7}$ |
| 50 | 5,000 | 1,152.5 ms | 28.1 ms | 41 | $8.5 \times 10^{-7}$ |

The refinement converged for every examinee in at most 10 iterations. The differences are within
the search tolerance of $10^{-6}$.

How other packages estimate many examinees:

- **mirt 1.48** runs one `nlm` search per unique response pattern (source
  `R/fscores.internal.R:406-410, 1072-1076`). It loops over the patterns with `apply`, or spreads
  them over a cluster with `parApply` when `mirtCluster` is set (`R/utils.R:3025-3034`).
- **TAM 4.3-25** updates every person's ability at once. Each iteration of `tam.mml.wle2` takes one
  Newton step for all persons, until the largest step is below `convM` (source
  `R/tam.mml.wle2.R:5, 103-107`; `R/tam_mml_wle_update_theta.R`). A step longer than 0.95 times
  that person's previous step is divided by twice the ceiling of their ratio, which leaves it at
  most half that bound (`R/tam_trim_increment.R`). With `WLE=FALSE` the same loop gives
  maximum likelihood estimates.
- **PP 1.0.0** solves one person at a time in compiled C++. It uses a safeguarded Newton method on
  a bracket of the estimating equation, bisecting whenever the Newton step leaves the bracket
  (source `src/core.cpp:190-231, 240-276`).
- **girth 0.8.0** (Python) runs one `scipy.optimize.fminbound` search per examinee on [-6, 6]
  (source `estimation/unidimensional/dichotomous/ability_estimation.py:45-47, 94-104`).
- **catR 3.17** and catsim estimate one examinee per call (catR source `R/thetaEst.R:1`).

## What irtlib and catsim have now

- `src/irtlib/ability.py`:
  - `_answered` yields one examinee at a time.
  - `_maximum` evaluates 101 points and refines the best between its neighbours by a bracketed
    Newton's method, or skips the points where the log-likelihood is concave (the 1PL and 2PL).
    Under a prior other than the normal, MAP evaluates 201 points and refines the best with
    `scipy.optimize.minimize_scalar`.
  - Without bounds, the search moves a range end outward while the bounds on the log-likelihood
    leave room beyond it. `maximum_likelihood`, `maximum_a_posteriori` and `weighted_likelihood`
    (with `scipy.optimize.brentq`) all build one objective per examinee.
- `expected_a_posteriori` already evaluates every examinee on the grid at once, through
  `irt.log_likelihood_function` and `quadrature.posterior`.
- `src/irtlib/jml.py`: `_Groups` estimates each response pattern once, which saves the repeated
  patterns only.
- catsim estimates one examinee per step of an adaptive test, a path `benchmarks/catsim_baseline.py`
  times against catsim's own estimators.

## Proposal

What it takes:

1. **The grid for every examinee at once.** One product of the response matrix with the log
   probabilities at the search points, as the prototype does, with missing responses as zero
   weights. Its memory is examinees × 101 floats: 4 MB for 5,000 examinees.
2. **The refinement for every examinee at once.**
   - ML and MAP: `scipy.optimize.elementwise.find_minimum` (Chandrupatla's bracketing
     minimization), with each examinee's best point and its two neighbours as the bracket the
     function requires, or a vectorized form of the one-examinee search's bracketed Newton's method
     on the derivatives it already computes. It is in SciPy 1.16.0, irtlib's lowest supported version (checked
     2026-11-28), so no dependency changes.
   - WLE: `scipy.optimize.elementwise.find_root` replaces `brentq`.
   - The objective takes an array of abilities, one per examinee, with the rows' indices as an
     argument the function subsets alongside them.
3. **Every behaviour of today's per-examinee search, vectorized by masks:**
   - the rule that keeps a refinement only if it is no worse than the best point;
   - flat maxima;
   - responses all correct or all incorrect, and unanswered rows;
   - the unbounded search's moving range ends, repeated for the rows whose best point lies at an
     end, which are few;
   - the fallback estimator;
   - the evaluation counts, from the refinement's count for each element.
4. **One examinee stays fast.** The vectorized path's fixed cost may slow catsim's single-examinee
   calls. Measure it with `benchmarks/catsim_baseline.py`, and keep a one-row path if it does.

TAM's damped Newton step for all persons is the alternative to point 2. It assumes one maximum,
which the 1PL and 2PL have and the 3PL and 4PL need not. The grid followed by a bracketed
refinement keeps irtlib's guard against a local maximum. On the 1PL and 2PL the one-examinee
search already skips the points, the log-likelihood being concave, and the vectorized search can
too.

EAP needs none of this: it already scores every examinee on the grid at once.

Where irtlib would use the vectorized estimators:

- **ML:** the ability step of every cycle of `irtlib.jml`, where most of a 2PL calibration's time
  goes.
- **MAP and WLE:** the ability steps of penalized JML and of JML with Warm abilities, should
  [estimation/jml-variants](jml-variants.md) add them.
- **ML or WLE:** the mean squares of a conditional calibration, should
  [fit/conditional-mean-squares](../fit/conditional-mean-squares.md) choose them.
- **All three:** scoring a calibration sample and simulation studies, through `irtlib.ability`
  directly.

## Tests

- The ability tests and parity cases pass unchanged. Every estimate stays within the search
  tolerance of today's, on random 1PL to 4PL banks with missing responses.
- The unbounded search, flat maxima, extreme and unanswered rows, and the fallback keep their
  current tests.
- `benchmarks/catsim_baseline.py` shows no single-examinee estimator slower than before.

## Open Questions

1. Whether to deduplicate response patterns before the search, as mirt does and `irtlib.jml`
   does. With the search vectorized it saves less. The card assumes no deduplication unless the
   measurements show a gain.

## References

- mirt 1.48, CRAN source package. <https://cran.r-project.org/package=mirt>
- TAM 4.3-25, CRAN source package. <https://cran.r-project.org/package=TAM>
- PP 1.0.0, CRAN source package. <https://cran.r-project.org/package=PP>
- girth 0.8.0, source. <https://github.com/eribean/girth>
- catR 3.17, CRAN source package. <https://cran.r-project.org/package=catR>
