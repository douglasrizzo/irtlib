---
title: Let joint maximum likelihood calibrate responses with missing entries
kind: feature
status: needs-research
priority: 4
effort: S
blocked_by: []
raised_by: [joint maximum likelihood PR on 2026-11-27]
---

# Let joint maximum likelihood calibrate responses with missing entries

## Problem

`irtlib.jml.JMLCalibrator` refuses a response matrix with a missing response and names
`irtlib.mml.MMLCalibrator` instead (ADR-0006), because its sources write joint maximum likelihood
for complete responses: an examinee's number correct out of every item (Wright & Douglas, 1977,
p. 282). Yet with responses missing at random, ordinary conditional maximum likelihood "will likely
fail" and marginal or joint maximum likelihood "will usually be preferred" (Robitzsch, 2021, p. 827),
and responses missing by design "can be ignored for the purpose of maximum likelihood and Bayesian
estimation of item and examinee parameters" (Mislevy & Stocking, 1989, p. 66). A comparison study on
incomplete designs cannot include joint maximum likelihood.

## What irtlib and catsim have now

- `src/irtlib/jml.py`: `JMLCalibrator.calibrate` refuses NaN; `_Groups.of` groups the examinees by
  number correct for the Rasch model and by response pattern for the 2PL.
- The steps it alternates already take missing responses: `irtlib.ability.maximum_likelihood` sums
  the log-likelihood over the items an examinee answered, and the item step of `irtlib._m_step`
  takes a count of responses per item and ability.
- `irtlib.calibration._edited` leaves out examinees and items of complete responses only.

## Proposal

Sum the joint likelihood over the answered responses, as the steps already do, once the research
below settles:

1. which examinees and items have no finite estimate when responses are missing: an examinee with
   every answered item right or wrong, an item answered correctly by every examinee who answered it
   or by none, and whether a split like Haberman's (2004, p. 11, Theorem 1) has a counterpart;
2. whether the Rasch correction $(M - 1)/M$ has a counterpart when examinees answer different numbers
   of items: Wright and Douglas (1977, pp. 286-288) derive it for every examinee answering all $M$;
3. the Rasch model's groups: examinees with the same number correct out of different items do not
   share an ability, so the groups become the response patterns, as for the 2PL.

## Tests

- With responses missing, the estimates and the log-likelihood are the maximum of the joint
  likelihood written out over the answered responses and maximized by SciPy, as
  `tests/test_jml.py` checks for complete ones.
- A matrix with an item missing for half the examinees recovers its generating parameters.

## Open Questions

1. Whether the correction applies with missing responses: off, or with $M$ each examinee's number of
   items. The card assumes off until a source is found.
