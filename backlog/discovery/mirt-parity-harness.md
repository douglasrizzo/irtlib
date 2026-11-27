---
title: Build the parity harness that checks pyitemcal against mirt
kind: feature
status: ready
priority: 1
effort: M
blocked_by: []
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

Both are decided, accepted by the maintainer on 2026-11-27 with ADR-0006 to ADR-0008.

1. **Decided:** the tolerances, which ADR-0003 leaves to this card. Fixtures recorded with mirt at `TOL = 1e-8`,
   because its default stops short (ADR-0007); estimates compared within `1e-5`, which the measured
   2PL agreement of about $10^{-6}$ leaves room for; closed forms within a relative $10^{-10}$, as
   catsim's catR tests do. The 3PL and 4PL tolerances are confirmed by their fixtures, which need
   cases that converge at `TOL = 1e-8`: `LSAT7` does not give one for the 4PL. mirt's M-step is
   inexact by default, so the fixtures also set `control = list(reltol = 1e-14)`: on `LSAT7` at `TOL = 1e-8` it brought mirt's 2PL estimate
   within $7.1 \times 10^{-7}$ of an exact-M-step EM's converged estimate, against
   $2.3 \times 10^{-6}$ without it (measured 2026-11-27 with the scratch EM of the MML-EM card's
   acceleration measurement).
2. **Decided:** the standard-error method the fixtures record, Oakes's identity with central differences, mirt's
   default (ADR-0007), with `SE.type = "Louis"` as a cross-check
   ([the standard-errors card](../estimation/standard-errors.md)).
