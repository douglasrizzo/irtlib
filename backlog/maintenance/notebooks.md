---
title: Write the tutorial notebooks
kind: docs
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [setup of irtlib]
---

# Write the tutorial notebooks

## Problem

Researchers learn a calibration package by running it on data. Once a method exists, a notebook that
calibrates a bank, reads the result and checks the fit is the fastest way in.

## Proposal

- A first notebook: draw responses from a known bank (catsim's `ItemBank.generate`), calibrate it,
  compare the estimates with the generating parameters, and simulate adaptive tests in catsim with
  the estimated bank.
- Add catsim's notebook tooling: nbstripout and the nbmake pre-commit hook, the Ruff hooks over
  notebooks, `ipykernel` and `nbmake` in the `dev` group, and a CI step that runs every notebook.
- Add the notebooks' rules to `sphinx/AGENTS.md`, adapted from catsim's charter.

## Tests

- Every notebook runs in CI with nbmake, in under a minute.
