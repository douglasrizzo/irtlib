---
title: Let a package that estimates one examinee at a time give the ability estimators' warnings as its own
kind: decision
status: needs-decision
priority: 3
effort: S
blocked_by: []
raised_by: [review of catsim's estimators on irtlib on 2026-11-27]
---

# Let a package that estimates one examinee at a time give the ability estimators' warnings as its own

## Problem

catsim estimates one examinee per call, a session's responses as a matrix of one row, and its
`WarmLikelihoodEstimator` warns when a ranking integral does not converge. Through
`irtlib.ability.weighted_likelihood` the warning:

- starts with "examinee in row 0: ", a row the caller never passed;
- points at catsim's own call into irtlib, the first frame outside irtlib, rather than at the code
  that called catsim, so a researcher cannot tell which of their calls it came from.

catsim cannot fix either on its side without cost. Recording the warnings with
`warnings.catch_warnings` and giving them again resets Python's registry of warnings already shown
on every call, so under the default filters the warning printed at every step of a simulation, 3
times for 3 estimates where once was the rule (catsim's review, 2026-11-27), and the global filters
it swaps are not thread-safe. catsim lets irtlib's warning through unchanged.

## What irtlib and catsim have now

- `irtlib/ability.py` prefixes a warning about one examinee with `_ROW` ("examinee in row {}: "),
  renumbers rows when the fallback of `maximum_likelihood` warns, and warns with
  `skip_file_prefixes=(_PACKAGE,)`, irtlib's own directory.
- catsim's `WarmLikelihoodEstimator.estimate` calls `weighted_likelihood([responses], items, ...)`.
  Before it built on irtlib, its own estimator warned with `stacklevel=3`, at the caller's line.

## Options

1. **A `skip_file_prefixes` argument** on the estimators, added to irtlib's own directory, so a
   package passes its directory and the warning points past it. Small, and the warning keeps its row.
2. **The diagnostics in the result.** `AbilityEstimates` carries each row's messages, such as a tuple
   of strings per row, and the estimators warn only when asked to; a package gives them in its own
   words. More API, and a result that is no longer just numbers.
3. **Leave it.** catsim's users see the row and catsim's line.

Recommendation: option 1, without the row prefix when the response matrix has one row.

## Tests

- A caller that passes its own directory gets the warning at the line that called it, and only once
  under the default filters across repeated calls from that line.
- A one-row matrix's warning has no row prefix; a matrix of several rows keeps it.
