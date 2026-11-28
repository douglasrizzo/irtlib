# ADR-0004: catsim Builds on irtlib's Item Response Theory

## Status

Accepted 2026-11-26; inverted 2026-11-27

## Context

The package began importing catsim's response functions, quadrature grid, priors, model names, item
bank and array types, each from catsim's public API, and kept private copies of parts of catsim it
needed in another shape: the logit and the ratios of catsim's information function, catsim's
posterior extended to many examinees with missing responses, and the check of a response matrix that
catsim's simulation makes. ADR-0001 makes item response theory irtlib's subject, so that code
belongs here, and catsim, the adaptive-testing package, builds on it.

catsim's item response code, its response functions and quantities, estimators, priors and item
bank, comes to about 2,600 lines, with parity cases against the R package catR for the response
functions and the ability estimators. catsim is released and supports Python 3.11; irtlib is not
released and requires Python 3.12 (ADR-0002). `catsim.irt` is part of catsim's public API.

## Decision

- irtlib imports nothing from catsim at run time.
- catsim's item response code moves into irtlib in parts, each with its reference cases and its
  tests rewritten to irtlib's rules (`AGENTS.md`): the response functions and their quantities;
  the quadrature grid and priors; the model names, the item bank with its CSV file and random banks,
  and drawing responses; and ability estimation.
- Moved functions take many examinees at once, a response matrix of floats with NaN for a missing
  response, so a call for one examinee is the case of one row.
- Moved code keeps the names catsim's one-name-per-concept table gives each concept (catsim's
  ADR-0023).
- catsim depends on irtlib and removes its copies (catsim's ADR-0024). Until irtlib's first
  release, catsim's development takes irtlib from a commit of its GitHub repository, and catsim's
  next release waits for irtlib's.
- catsim at its last commit with its own copies, `97b89f5`, is a development dependency of
  irtlib's benchmarks, which time each moved function against catsim's original and compare their
  values. A moved function that takes more than 1.2 times catsim's time, on the shapes catsim calls
  it with, gets a card for a faster version in irtlib.

## Consequences

- Each item response quantity has one implementation, validated in the package whose subject it is.
- Importing irtlib imports only NumPy and SciPy.
- catsim's first release built on irtlib waits for irtlib's first release, and needs Python 3.12.
- The names that leave `catsim.irt` and its neighbours are catsim's to retire, in catsim's own
  pull requests.
- catsim runs at irtlib's speed; the benchmarks keep measuring it against catsim's originals.

## Alternatives Considered

- Keeping the import of catsim's functions: the estimation library depends on its application
  (ADR-0001).
- Moving everything in one pull request: too large to review, and the parts depend on each other in
  a fixed order.
- catsim waiting for irtlib's first release: the two copies would drift in the meantime. catsim
  depends on an unpublished irtlib on its development branch only, and releases after irtlib.
- Copying catsim's tests unchanged: a test that compares moved code with catsim's original shows
  the copy is faithful, not that it is correct; correctness rests on reference implementations and
  published values (ADR-0003).
