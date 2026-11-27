---
title: Publish releases to PyPI with trusted publishing
kind: maintenance
status: blocked
priority: 3
effort: M
blocked_by: [maintenance/released-catsim, maintenance/license, maintenance/docs-site]
raised_by: [setup of pyitemcal]
---

# Publish releases to PyPI with trusted publishing

## Problem

pyitemcal builds and tests its wheel in CI but publishes nothing. A release needs a license, a
catsim it can name in its metadata, and documentation to point users to.

## What pyitemcal and catsim have now

catsim's `.github/workflows/ci-cd.yml` is the model:

- the build job checks that the release tag equals both the project version and `__version__` in
  the wheel;
- publishing uses PyPI trusted publishing (`pypa/gh-action-pypi-publish`, `id-token: write`, a `pypi`
  environment), only on a published release, after the quality gates, the wheel test and the
  production docs deployment;
- TestPyPI receives an rc build on the pull request from `dev` to `main`;
- a final job signs the distributions with Sigstore and attaches them to the GitHub release;
- `.github/release.yml` shapes the generated release notes.

## Proposal

Copy that path, adapted, with the version bumped by `uv version` only. Register the trusted publisher
on PyPI and TestPyPI, and create the `pypi` and `testpypi` environments. Add a "Releasing" section to
`CONTRIBUTING.md`. Reserve no name on PyPI before the first real release.

## Tests

- A dev-to-main pull request publishes an rc to TestPyPI, and that rc installs and imports.
- A release whose tag differs from the project version fails before upload.
