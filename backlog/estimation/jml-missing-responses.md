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

## Why this card exists

The joint maximum likelihood PR (#33) refused missing responses, because its sources write JML for
complete ones. Both of its steps already handle missing responses. What is missing is research:
which examinees and items to leave out, and whether the correction still applies.

## Problem

`irtlib.jml.JMLCalibrator` refuses a response matrix with a missing response and names
`irtlib.mml.MMLCalibrator` instead (ADR-0006), because its sources write joint maximum likelihood
for complete responses: an examinee's number correct out of every item (Wright & Douglas, 1977,
p. 282). Yet with responses missing at random, ordinary conditional maximum likelihood "will likely
fail" and marginal or joint maximum likelihood "will usually be preferred" (Robitzsch, 2021, p. 827),
and responses missing by design "can be ignored for the purpose of maximum likelihood and Bayesian
estimation of item and examinee parameters" (Mislevy & Stocking, 1989, p. 66). A comparison study on
incomplete designs cannot include joint maximum likelihood.

## Evidence

Three R packages run JML on responses with missing entries. Each weights its sums by a response
indicator, and each takes the correction's number of items its own way. No source on the shelf
gives that number for missing responses. Read on 2026-11-28:

- **TAM 4.3-25**, `tam.jml`. The indicator is `resp.ind` (source
  `R/tam_jml_version2.R:85-88`). The factor is $(I - 1)/I$, with $I$ the test's number of items
  (`R/tam_jml_version2.R:30, 276-278`).
- **sirt 4.2-133**, `rasch.jml`. The indicator is `dat.resp`, in the gradient and the standard
  errors (source `R/rasch.jml.R:280, 296, 308-309`). The factor is $(I - 1)/I$, with $I$ the
  test's number of items (`R/rasch.jml.R:241, 247`).
- **immer 1.5-13**, `immer_jml`. The indicator is `dat_resp` (source `R/immer_jml.R:40-41`). The
  factor is $(\bar I - 1)/\bar I$, with $\bar I$ the mean number of items the examinees answered
  (`R/immer_jml.R:43-44`).

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

1. Whether the correction applies with missing responses, and with which $M$. The options:
   - off;
   - the test's number of items, as TAM and sirt take it;
   - the mean number answered, as immer takes it;
   - each examinee's number of items.

   The card assumes off until a source is found. Following one of the packages would rest on a
   reference implementation, which needs the maintainer's approval (`backlog/README.md`, "Citation
   convention").

## References

- Haberman, S. J. (2004). *Joint and conditional maximum likelihood estimation for the Rasch model
  for binary responses* (Research Report RR-04-20). Educational Testing Service.
  doi:10.1002/j.2333-8504.2004.tb01947.x. Open access.
- Mislevy, R. J., & Stocking, M. L. (1989). A consumer's guide to LOGIST and BILOG. *Applied
  Psychological Measurement, 13*(1), 57-75. doi:10.1177/014662168901300106. Open copy:
  https://hdl.handle.net/11299/107035
- Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
  *Stats, 4*(4), 814-836. doi:10.3390/stats4040048. Open access.
- Wright, B. D., & Douglas, G. A. (1977). Best procedures for sample-free item analysis. *Applied
  Psychological Measurement, 1*(2), 281-295. doi:10.1177/014662167700100216. Open copy:
  https://hdl.handle.net/11299/98553
- immer 1.5-13, sirt 4.2-133 and TAM 4.3-25, CRAN source packages.
  <https://cran.r-project.org/package=immer>, <https://cran.r-project.org/package=sirt>,
  <https://cran.r-project.org/package=TAM>
