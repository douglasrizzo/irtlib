---
title: Report coverage to Codecov
kind: maintenance
status: ready
priority: 4
effort: XS
blocked_by: []
raised_by: [setup of irtlib]
---

# Report coverage to Codecov

## Problem

The coverage gate (`fail_under = 90` in `pyproject.toml`) fails a run below the threshold, but shows
no history and no per-PR diff of what lost coverage.

## Proposal

Once the first module has tests, upload the coverage report from one CI test entry with
`codecov/codecov-action`, pinned by SHA, with a `CODECOV_TOKEN` secret, as catsim's CI does.

## Tests

- A pull request shows Codecov's coverage diff.
