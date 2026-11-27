# ADR-0001: irtlib Implements Item Response Theory for Unidimensional Dichotomous Items

## Status

Accepted 2026-11-26; scope widened 2026-11-27

## Context

The package began as pyitemcal, a calibrator importing the response functions, quadrature grid,
priors and item bank of catsim, a computerized adaptive testing (CAT) package that also holds the
ability estimators. A calibration library depending on a CAT library inverts their relation: a CAT is
a sequential design built on an item response model, and calibrating items and estimating abilities
are both inference on that model. In R, mirtCAT builds on mirt, which holds the models and their
estimation.

That mathematics is a body of knowledge of its own, and each implementation carries a validation
burden: reference implementations, published values, parity fixtures. Split across two packages, it
is written and checked twice.

## Decision

- irtlib implements item response theory for the unidimensional dichotomous logistic models, 1PL to
  4PL:
  - the response functions and the quantities derived from them, such as information, likelihoods,
    standard errors, reliability and score scales;
  - calibration: the item parameters, with their standard errors, from a response matrix that may
    have missing entries;
  - ability estimation for many examinees at once;
  - the fit of a calibrated model, item by item and between models;
  - the item bank format, with its validation, its CSV file and random banks, and drawing responses
    from the model.
- The line between irtlib and catsim is the fixed-form test. What is meaningful when every examinee
  answers the same items belongs here. What only makes sense when the next item is chosen from
  earlier answers belongs to catsim: item selection, stopping rules, exposure control,
  initialization, sessions, and the simulation and evaluation of adaptive tests. The ability
  estimators split along the same line: their mathematics is irtlib's, and catsim adapts it to one
  examinee's session, together with its rules for an estimate that does not exist yet, such as
  stepping toward the bank's hardest item.
- irtlib owns quantities only catsim uses today, such as the ability of maximum information: one
  owner validates each formula once.
- Estimates use the $a$, $b$, $c$ and $d$ of the four-parameter logistic model with no scaling
  constant.
- The calibration methods and their order follow a review of their primary sources
  (`backlog/discovery/literature-review.md`).
- Outside its scope: polytomous and multidimensional models, linking and equating, online
  calibration of items seeded into adaptive tests, adaptive testing itself, serving or storing
  tests, and a graphical or web interface.

## Consequences

- Its readers are, in order: researchers who calibrate item banks, compare calibration methods or
  score examinees; method authors who add an estimation method; packages built on its item response
  theory, catsim first; and contributors.
- Every design question is settled for one model family: dichotomous responses and one ability per
  examinee.
- catsim depends on irtlib for its item response theory (ADR-0004).
- The out-of-scope work stays as parked cards in `backlog/scope/`; bringing any of it into scope
  changes this ADR first.

## Alternatives Considered

- Calibration only, with the item response functions in catsim: the dependency runs from the
  estimation library to its application, and the shared mathematics is validated in two packages.
- A third package for the shared mathematics: three packages to release in step, where irtlib with
  this scope is that package.
- One package for both: item response theory inside a CAT package, or a CAT engine inside an
  estimation library, each with an audience the other does not need.
- Polytomous or multidimensional models from the start: each multiplies the parameterizations,
  the estimation methods and the parity cases before one model family works end to end.
