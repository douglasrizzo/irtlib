---
title: Depend on a released catsim, and follow its releases
kind: maintenance
status: blocked
priority: 2
effort: S
blocked_by: []
raised_by: [ADR-0004]
---

# Depend on a released catsim, and follow its releases

## Problem

pyitemcal takes catsim from its `dev` branch (`[tool.uv.sources]` in `pyproject.toml`), because the
catsim names it uses (ADR-0004) are not in catsim 0.21.0, the last release. A wheel's metadata
cannot name a Git branch, so a published pyitemcal would install catsim 0.21.0, which lacks them.
The card is blocked until catsim's next release.

## Evidence

On 2026-11-26, catsim 0.21.0 (released 2026-04-08) had none of `irt.log_response_probabilities`,
`QuadratureGrid`, `LogPrior`, `IrtModel`, `ItemBank.to_csv` or `catsim.typing`; catsim's `dev` was
851 commits ahead of the tag, with `pyproject.toml` still at 0.21.0.

## Proposal

1. When catsim releases, declare the range of catsim versions pyitemcal is tested with, such as
   `catsim>=0.22,<0.23`, and remove the Git source from `[tool.uv.sources]`.
2. Make the CI test job's `lowest-direct` entries cover the range's floor.
3. Make sure pyitemcal hears of each later catsim release. Dependabot's uv updates already run
   monthly: check whether they propose a catsim release outside an upper-bounded range (a
   `versioning-strategy` that widens the range). If they do, a Dependabot PR is the signal, and CI
   on it shows whether pyitemcal needs changes. If they do not, add a weekly CI workflow that installs
   the newest catsim release over the lock and runs the tests.
4. Update ADR-0004's last Decision item and the test-wheel job's comment in `.github/workflows/ci.yml`,
   which installs the locked runtime dependencies because the wheel names catsim without a version.

## Tests

- `uv lock` resolves catsim from PyPI.
- The test-wheel job passes when the wheel is installed with its own dependencies.
- A catsim release outside the range produces a PR or a failing scheduled run.
