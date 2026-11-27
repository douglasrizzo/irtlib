---
title: Bring the response functions and their quantities from catsim into irtlib.irt
kind: feature
status: ready
priority: 2
effort: M
blocked_by: []
raised_by: [ADR-0004 as decided on 2026-11-27]
---

# Bring the response functions and their quantities from catsim into irtlib.irt

## Problem

ADR-0001 makes the response functions and the quantities derived from them irtlib's subject, and
ADR-0004 moves catsim's implementation here. Today irtlib imports `catsim.irt.icc` and
`log_response_probabilities`, and its own `irtlib.irt` repeats catsim's intermediate steps
privately.

## What irtlib and catsim have now

References are to catsim's `dev` at commit `97b89f5`.

- catsim `src/catsim/irt.py` (781 lines). Everything moves except `validate_response_vector`
  (756-781), catsim's check of Boolean responses (its ADR-0009), which stays:
  - the constants `THETA_MIN_TYPICAL`, `THETA_MAX_TYPICAL`, `THETA_MIN_EXTENDED` and
    `THETA_MAX_EXTENDED` (19-48), which arrive first, with the test helpers;
  - `icc` (66), `information` (105), `total_information` (151), `var` (187), `see` (217),
    `confidence_interval` (246), `theta_to_scale` (345), `scale_to_theta` (400), `reliability`
    (452), `max_info_theta` (482), `info_integral` (540), `log_response_probabilities` (588) and their
    private helpers;
  - `log_likelihood` and `log_likelihood_function` (644-753) split: the sum and its closure, which
    computes the terms that do not depend on the ability once, move; the Boolean check stays.
- Four copies of the same intermediates: catsim's `_split_params` and `_logit` (54-63), the ratios
  of `information` (141-147), WLE's `_response_curves` (`estimation/wle.py:153-163`), and irtlib's
  `_logistic` and `_scores` (`src/irtlib/irt.py:44-99`). Their guards for 0/0 differ.
- catR parity: catsim's `tests/parity/catr/irt.R` and `irt.json` (catR's `Pi` and `Ii` on 1PL to
  4PL banks at seven abilities), read by `tests/test_parity_catr.py::TestItemResponseFunctions`.
- Tests: `tests/test_irt.py` (521 lines) is mostly hand-worked and closed-form values, which port;
  `tests/test_irt_properties.py` is invariants and comparisons of catsim with itself.

## Proposal

1. Move every name above into `irtlib.irt`, keeping catsim's names (ADR-0004). Merge the copies
   into one set of private helpers, with one rule for 0/0. Add the derivatives of P with respect to
   the ability (P′ and P″, as WLE's `_response_curves` computes them), so ability estimation needs no
   edit of this module.
2. `log_likelihood` takes a response matrix, examinees by items, with NaN for a missing response,
   and returns one value per examinee and ability. It keeps the closure form for searches that
   evaluate it many times.
3. A response-matrix check for every function that takes responses, in a private module below
   `irt`. It takes 0, 1 or NaN in a two-dimensional float matrix, as `calibration.validate_responses`
   does, and allows an examinee with no answers and an item nobody answered, which ability
   estimation needs. It rejects dtypes other than Boolean, integer and float, and names the position
   of a bad value, as catsim's `simulation._response_matrix` does (358-416). Calibration's stricter
   rule, that every item is answered, stays in `calibration.py`. The module goes in the
   import-linter layers between `irt` and `typing`.
4. The formulas, with their sources, go in the module docstring; the function docstrings keep to
   their call contracts (`sphinx/AGENTS.md`, "Where a formula lives"). catsim's ADR numbers and
   links to catsim pages leave the text. The skipped doctest of `confidence_interval` prints a wrong
   value (`[-0.234, 1.234]`, against `(-2.297, 3.297)` when run); it runs.
5. Bring the catR harness: `generate.R` trimmed to `irt.R` (and later `estimation.R`), the library
   variable `IRTLIB_CATR_LIB`, one `json.R` shared with the mirt harness, `TestData` and
   `TestItemResponseFunctions`.
6. Leave `mml`, `fit` and `calibration` importing from catsim: `maintenance/drop-catsim` switches
   every module to the moved names in one pull request.

## Tests

- The hand-worked and closed-form tests of catsim's `test_irt.py`, rewritten to irtlib's rules.
- catR parity for `icc` and `information` at a relative `1e-10`.
- The batched `log_likelihood` against a brute-force sum over each examinee's answered items, with
  missing responses.
- Property tests for invariants: P between c and d and non-decreasing in the ability, scale
  conversions that round-trip. A comparison with catsim's original is a migration check only.

## Sources

The claims cite pages read on the shelf. Magis and Raîche (2012) and Warm (1985) come from catsim's
shelf. Magis (2013, the 4PL information and the ability of maximum information) and Thissen (2000,
reliability) are with the maintainer to fetch. catsim cites De Ayala (2009) for the information,
variance and log-likelihood; a source on the shelf that supports the same claim may take its place.
