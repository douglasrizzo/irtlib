---
title: Estimate the abilities of many examinees in one vectorized search, by the WLE
kind: feature
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [joint maximum likelihood PR on 2026-11-27]
---

# Estimate the abilities of many examinees in one vectorized search, by the WLE

## Why this card exists

The joint maximum likelihood PR (#33) found that a 2PL calibration spends almost all its time in
the ability step, one maximum likelihood search per response pattern per cycle. The maintainer then
asked on 2026-11-28 what estimating many examinees at once would take. A scratch prototype
(Evidence) answered it: 29 to 49 times faster, with the same estimates.

The maximum likelihood and MAP searches, under a normal prior, search every row of a matrix at once since the
batched-search PR of 2026-11-28 (15 to 28 times faster on 2,000 to 5,000 examinees, and a 2PL joint calibration 4 to 6
times faster). What remains is the WLE, whose root search is `scipy.optimize.brentq` per examinee, and MAP under a
prior other than the normal, whose refinement is SciPy's bounded search per examinee.

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

- `src/irtlib/_search.py`: `_Rows` holds a matrix's responses signed for the searches, and `_maxima` searches every row
  at once with the arithmetic of `_Examinee`'s one-row search: the grid of each row's points in chunks, the bracketed
  Newton's method of `_newton_rows`, and the concave search of the 1PL and 2PL. `irtlib.ability.maximum_likelihood`
  and `maximum_a_posteriori`, under a normal prior, call it for two or more rows.
- `src/irtlib/ability.py`: `weighted_likelihood` loops over the examinees: a scan of Warm's modified score at 121
  points, `scipy.optimize.brentq` on each sign change, and `scipy.integrate.quad` to rank several roots. MAP under any
  other prior loops too, with SciPy's bounded search.
- catsim estimates one examinee per step of an adaptive test, which keeps the one-row search.

## Proposal

1. **The WLE's scan for every examinee at once,** through `_Rows`: the modified score's sums over each row's items at
   the 121 points, as `_Examinee.modified_score` computes them.
2. **Its roots for every examinee at once,** with the arithmetic of `brentq`, so that a row's estimate stays the same
   alone and in a matrix; `scipy.optimize.elementwise.find_root` is the alternative, at the cost of that equality.
3. **The ranking of several roots** stays one row at a time, as few rows have more than one.
4. **MAP under other priors** stays one row at a time unless a use appears.

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
