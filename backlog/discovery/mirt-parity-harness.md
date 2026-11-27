---
title: Build the parity harness that checks pyitemcal against mirt
kind: feature
status: blocked
priority: 1
effort: M
blocked_by: [discovery/requirements]
raised_by: [ADR-0003]
---

# Build the parity harness that checks pyitemcal against mirt

## Problem

ADR-0003 makes mirt pyitemcal's reference implementation, checked by parity tests that need no R
when they run. The first calibration method needs the harness in place to be checked at all.

## What pyitemcal and catsim have now

pyitemcal has nothing. catsim's catR harness is the model: R scripts and JSON fixtures in
`tests/parity/catr/` (`generate.R` pins `CATR_VERSION`, installs into `~/.cache/catsim-parity/R-lib`
unless `CATSIM_CATR_LIB` overrides it, and writes one JSON file per family), and
`tests/test_parity_catr.py`, with deliberate departures as strict expected failures.

## Proposal

- `tests/parity/mirt/generate.R` pins the mirt version, installs it into
  `~/.cache/pyitemcal-parity/R-lib` unless `PYITEMCAL_MIRT_LIB` overrides it, and runs every case
  file beside it.
- Each case fixes a response matrix (generated once with a recorded seed and stored in the fixture,
  so R and NumPy need not share random numbers), the model and every setting mirt uses (quadrature
  points, tolerance, priors, standard-error method), and records mirt's estimates, standard errors,
  log-likelihood and iteration count, with the R and mirt versions.
- `tests/test_parity_mirt.py` reads the fixtures and compares pyitemcal's outputs, with each
  departure as a strict expected failure naming its reason.
- The pretty-format-json hook already skips `tests/parity/mirt/*.json`, so the 17 significant digits
  R writes survive.
- `CONTRIBUTING.md` gets the parity section: how to install R and mirt, and the one command that
  regenerates the fixtures.

## Tests

- A fixture regenerated with the same mirt version is byte-identical.
- Perturbing one of mirt's recorded estimates beyond the tolerance fails the comparison.

## Open Questions

1. The tolerances: the requirements card's question 18 recommends fixtures recorded with mirt at
   `TOL = 1e-8`, estimates compared within `1e-5` and closed forms within a relative $10^{-10}$, as
   catsim's catR tests do.
2. The standard-error method the fixtures record: Oakes's identity with central differences, mirt's
   default, with `SE.type = "Louis"` as a cross-check
   ([the standard-errors card](../estimation/standard-errors.md)).
