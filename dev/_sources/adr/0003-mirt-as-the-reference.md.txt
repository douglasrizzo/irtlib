# ADR-0003: Parity Tests Check irtlib Against mirt

## Status

Accepted 2026-11-26

## Context

A calibration method can be implemented, converge and still be wrong: a wrongly scaled quadrature, a
gradient off by a factor or a prior applied twice all give plausible estimates. An independent
implementation of the same method is a strong check. mirt, a published and widely used R package,
implements marginal maximum likelihood estimation of the 1PL to 4PL models with priors, standard
errors of the item estimates and item-fit statistics, and calibration researchers rely on it.

A check against generating parameters is a second, independent one: responses drawn from known
parameters should give back those parameters, within the sampling error of the sample size. It
needs no other software, and it catches what agreement with mirt cannot, such as a departure both
packages share.

## Decision

- mirt is irtlib's reference implementation: every method mirt implements is checked against it
  by parity tests in three parts:
  - R scripts in `tests/parity/mirt/` run a pinned mirt version on fixed response matrices and write
    the matrices, the settings and mirt's outputs, with the R and mirt versions, to JSON files in the
    repository;
  - a pytest module compares irtlib's outputs with those files in every test run, with no R
    installed;
  - one documented command regenerates the files, when the cases change or the pinned mirt version
    moves. It needs R and mirt.
- The response functions and the ability estimators that come from catsim (ADR-0004) keep
  catsim's parity cases against the R package catR, in `tests/parity/catr/`, built in the same three
  parts: catR is their second reference implementation.
- Every method also has parameter-recovery tests: responses drawn from known parameters, with a
  fixed seed, and the estimates compared with the generating values within a tolerance that the
  sample size justifies.
- irtlib keeps its own defaults. Every departure from mirt either has a setting that reproduces
  mirt, which the parity tests use, or is stated where the method is documented, with its reason,
  and kept in the parity tests as a strict expected failure with that reason, so a departure that
  closes fails the run until it is removed from the list.
- mirt's estimates are recorded at `TOL = 1e-8` with an exact M-step
  (`control = list(reltol = 1e-14)`), since its defaults stop short of the maximum (ADR-0007).
  Estimates are compared within `1e-5`, which the measured 2PL agreement of about $10^{-6}$ leaves
  room for; closed forms and sums on the same grid within a relative $10^{-10}$. Each tolerance is a
  constant of the parity test module, with its reason.

## Consequences

- A claim that irtlib agrees with mirt is backed by a test, and a change that breaks agreement
  fails it.
- A new method that mirt implements comes with its parity cases and its recovery tests; one that mirt
  lacks comes with recovery tests and the published values its sources give.
- The parity tests follow one mirt version; moving to another is a deliberate change of the fixed
  outputs.
- No document says what mirt or another package lacks.

## Alternatives Considered

- Running mirt in CI: R and its packages on every run, slower and less reliable, to check outputs
  that change only with the cases or the pinned version.
- Recovery tests alone: they cannot tell a method's own bias from an implementation error.
- Several reference packages: two references can disagree, and each doubles the fixtures.
- mirt's defaults everywhere: a default that its sources show to be poorly founded need not be
  copied, as long as a setting reproduces mirt for the parity tests.
