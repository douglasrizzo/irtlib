---
title: Keep the 2PL's joint maximum likelihood slopes finite on short tests
kind: feature
status: blocked
priority: 4
effort: S
blocked_by: []
raised_by: [maintainer on 2026-11-28 after the joint maximum likelihood PR]
---

# Keep the 2PL's joint maximum likelihood slopes finite on short tests

## Why this card exists

The joint maximum likelihood PR (#33) stops a 2PL calibration with a `ConvergenceWarning` once a
slope has no finite estimate. On 2026-11-28 the maintainer decided to keep that behaviour for
now. They asked that the alternatives stay on record, in case a change of behaviour is warranted
later.

**Blocked by** either of:

- a user who needs the 2PL by JML on short tests;
- LOGIST's user's guide (Wingersky, Barton & Lord, 1982) on the shelf, for the default limits of
  option 2.

## Problem

Under the 2PL, JML's slopes can grow without bound, and the maximizing discriminations "are
sometimes infinite" (Mislevy & Stocking, 1989, p. 62). `irtlib.jml.JMLCalibrator` then stops. It
stops before an item step when the abilities separate an item's correct responses from its
incorrect ones, which leaves no finite slope (Baker, 1987, p. 117), and after an item step that
gives a slope that is not positive.

On short tests this is the usual outcome (measured 2026-11-28, slopes uniform on [0.7, 2],
difficulties uniform on [-1.5, 1.5], abilities normal):

| Data | Calibrations that stopped |
|---|---|
| LSAT-6 and LSAT-7, 5 items × 1,000 | both |
| 8 items × 300 | 50 of 50 |
| 10 items × 500 | 19 of 20 |
| 15 items × 500 | 13 of 20 |
| 20 items × 500 | 3 of 20 |
| 20 and 30 items × 1,000 | none of 10 each |

LOGIST's authors advise "at least 20 items per person and at least 800 to 1,000 examinees
responding to each item" (Mislevy & Stocking, 1989, p. 61).

## What irtlib and catsim have now

- `src/irtlib/jml.py`: `JMLCalibrator._cycle` stops on `_Groups.separated` or on a slope that is
  not positive. The result holds the last complete cycle, and the module docstring states the
  measurements above.
- `src/irtlib/_m_step.py`: the item step takes priors (`_prior_terms`), mirt's lognormal on the
  slope among them (ADR-0008). `irtlib.jml` passes none.
- catsim has no calibration.

## Proposal

The options, the first decided for now:

1. **Stop with a warning,** as now.
2. **Slope limits, as LOGIST imposes them** (Baker, 1987, p. 113). Mislevy and Stocking describe
   them (1989, pp. 63-64):
   - limits on $a$ and $c$ are "equivalent to specifying uniform prior distributions on the
     allowable intervals";
   - an estimate past a limit takes the limit, and the item's other parameters maximize the
     likelihood with it fixed;
   - the manual shows how to choose the limits from a partial run of the program.

   This would be a setting of `JMLCalibrator`. Its default values are in LOGIST's user's guide, not
   on the shelf.
3. **A slope prior in the item step,** through the priors `irtlib._m_step` already takes. Mislevy
   and Stocking write that "additional information or structure is required to obtain estimates
   that may be less likely (i.e., do not maximize the likelihood function), but more satisfactory"
   (1989, p. 62). The result would no longer be a joint maximum likelihood estimate, and its
   `log_likelihood` would need a log-posterior beside it.
4. **LOGIST's scale,** which standardizes only the abilities between -3 and 3 (Mislevy & Stocking,
   1989, p. 60) where irtlib standardizes every examinee kept. This is independent of options 1 to
   3.

## Tests

- Option 2: an estimate at a limit, and the item's other parameters maximizing the joint likelihood
  with it fixed, checked by brute force. The short tests above converge.
- Option 3: the estimates maximize the joint log-likelihood plus the log-prior, checked by brute
  force.

## Open Questions

1. Which option, once the card is unblocked. The card assumes option 1.

## References

- Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
  three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
  doi:10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398
- Mislevy, R. J., & Stocking, M. L. (1989). A consumer's guide to LOGIST and BILOG. *Applied
  Psychological Measurement, 13*(1), 57-75. doi:10.1177/014662168901300106. Open copy:
  https://hdl.handle.net/11299/107035
- Wingersky, M. S., Barton, M. A., & Lord, F. M. (1982). *LOGIST user's guide*. Educational Testing
  Service, as Mislevy and Stocking (1989) cite it. Not on the shelf; no open copy found.
