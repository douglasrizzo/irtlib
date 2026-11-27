# ADR-0004: pyitemcal Builds on catsim's Public Item Response Functions

## Status

Accepted 2026-11-26

## Context

Calibration and adaptive testing share a kernel: the probability of a correct response under the
four-parameter logistic model, its logarithms, item information, quadrature grids over ability,
priors, the model names, the bounds a valid item obeys and the file a bank is stored in. A second
copy of that kernel can drift from catsim's, and a bank that catsim reads differently from how it
was calibrated gives results that are wrong without any error.

catsim's kernel already serves calibration: its response functions broadcast an array of abilities
against an item matrix with no per-call checks. Where the needs differ, catsim's functions take one
examinee at a time: its log-likelihood, posterior and estimators have no batched form and no missing
responses, and catsim has no derivatives with respect to item parameters. catsim requires only NumPy
and SciPy, so depending on it adds no other package.

When pyitemcal was set up, the catsim functions it uses existed only on catsim's `dev` branch: the
last catsim release, 0.21.0, predates them.

## Decision

- pyitemcal imports from catsim's public API only, each name from its public path (catsim's
  ADR-0023), and only:
  - `catsim.irt`: `icc`, `log_response_probabilities` and `information`;
  - `catsim.estimation`: `QuadratureGrid`, `LogPrior`, `normal_log_prior` and `uniform_log_prior`;
  - `catsim`: `IrtModel`, and `ItemBank` with its CSV reading and writing, for validated output;
  - `catsim.typing`: `FloatArray` and `IntArray`.
- pyitemcal implements what calibration needs and catsim lacks: the log-likelihood and posterior
  of a whole response matrix with missing entries, derivatives with respect to item parameters,
  ability estimation for many examinees at once, and drawing responses for recovery tests.
- pyitemcal depends on catsim; catsim does not know pyitemcal. When catsim changes a name pyitemcal
  uses, pyitemcal follows in a pull request of its own.
- Until catsim's next release, pyitemcal takes catsim from its `dev` branch, through uv's sources,
  with the commit pinned in `uv.lock`. From that release on, it declares the range of catsim
  versions it is tested with (`backlog/maintenance/released-catsim.md`).

## Consequences

- The response functions, the parameterization and the bank format are catsim's, so they cannot
  drift apart.
- Importing pyitemcal imports catsim, about 0.4 seconds on the machine measured.
- A catsim change to a name in the list breaks pyitemcal until pyitemcal follows; catsim's own
  tests do not warn of it.
- pyitemcal cannot be published while it depends on an unreleased catsim, since a wheel's metadata
  cannot name a Git branch.
- Whether the kernel should move into pyitemcal, with catsim depending on it, is an open decision
  (`backlog/discovery/invert-catsim-dependency.md`).

## Alternatives Considered

- An independent copy of the kernel, checked against catsim in tests: the copy drifts first and the
  tests catch it after.
- A third package for the kernel: three packages to release in step for about a hundred lines of
  code.
- Moving the kernel into pyitemcal before the requirements show what the kernel must offer: it
  commits to a dependency direction without knowing what it serves.
