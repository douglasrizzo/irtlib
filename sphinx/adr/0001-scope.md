# ADR-0001: pyitemcal Calibrates Unidimensional Dichotomous Items

## Status

Accepted 2026-11-26

## Context

catsim computes and evaluates computerized adaptive tests and takes item parameters as an input,
from whatever software calibrated them (catsim's ADR-0001). Its backlog held a literature review and
design cards for calibrating items, parked as outside its scope: calibration is a field of
estimation with its own literature and software. A package that estimates the parameters catsim
consumes, in Python and with catsim's conventions, can grow that work without widening catsim.

Calibration also reaches well past what catsim's models need: polytomous and multidimensional
models, linking and equating separate calibrations onto one scale, and calibrating new items online
while a CAT administers them. Each is a body of methods of its own.

## Decision

- pyitemcal estimates the item parameters of the unidimensional dichotomous logistic models, 1PL to
  4PL, from a matrix of responses that may have missing entries, with the standard errors of the
  estimates.
- It evaluates the fit of a calibrated model to the responses, item by item.
- Its estimates use catsim's parameterization, the $a$, $b$, $c$ and $d$ of the four-parameter
  logistic model with no scaling constant, so a calibrated bank is one catsim reads (ADR-0004).
- Which estimation methods it implements, and in what order, follows from a review of their
  primary sources (`backlog/discovery/literature-review.md`).
- Outside its scope: polytomous models, multidimensional models, linking and equating, online
  calibration of items seeded into adaptive tests, serving or storing tests, and a graphical or web
  interface.

## Consequences

- Its readers are, in order: researchers who calibrate item banks or compare calibration methods,
  method authors who add an estimation method, and contributors.
- Every design question is settled for one model family: dichotomous responses and one ability per
  examinee.
- A calibrated bank goes straight into catsim's simulations, and a study can generate items in
  catsim, calibrate them here and simulate tests with the estimates.
- The out-of-scope work stays as parked cards in `backlog/scope/`; bringing any of it into scope
  changes this ADR first.

## Alternatives Considered

- Calibration inside catsim: catsim's ADR-0001 keeps catsim to computing and evaluating adaptive
  tests, and calibration is a different field of estimation.
- Polytomous or multidimensional models from the start: each multiplies the parameterizations,
  the estimation methods and the parity cases before one model family works end to end.
- Online calibration from the start: it needs both a working calibrator and catsim's engine, so it
  builds on this scope rather than replacing it.
