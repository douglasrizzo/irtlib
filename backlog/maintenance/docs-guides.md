---
title: Write the guides
kind: docs
status: ready
priority: 3
effort: L
blocked_by: []
raised_by: [maintainer on 2026-11-28]
---

# Write the guides

## Problem

The site has an introduction, the terms and the API reference, but no page that walks a researcher
through a task end to end with code to copy. The landing page's entry paths link the API reference
in the meantime.

## What irtlib and catsim have now

- irtlib: no guide. The landing page's sections, "Calibrate an item bank", "Score examinees" and
  "Build on irtlib", link modules.
- catsim's guides ("Running simulations", "Reproducibility", "Extending catsim", "Embedding catsim")
  and the rules for them in catsim's `sphinx/AGENTS.md`, "Guides": one task end to end, with runnable
  code, what the output means, the choices the reader makes, each linked to the page that explains
  it, and the pitfalls; 800-1,500 words each.

## Proposal

Four guides, under Getting started, each in its own pull request, each linked from its landing-page
section:

1. **Calibrate an item bank:** from a response matrix to a saved bank, with standard errors:
   choosing MML, CML or JML; reading a `Calibration`; the items left out; saving with
   `ItemBank.to_csv`.
2. **Score examinees:** the four estimators on calibrated items, their standard errors, and the
   responses all correct or all incorrect.
3. **Check fit and compare models:** S-X² and the mean squares item by item, then the
   likelihood-ratio test, AIC, BIC and G² between models.
4. **Reproducibility:** the `rng` argument (ADR-0005), seeds, and drawing responses for a
   parameter-recovery study.

A later item, deferred by the maintainer on 2026-11-28 since extending irtlib matters less than
extending catsim: a guide for method authors on adding a calibration method (ADR-0006), the
counterpart of catsim's "Extending catsim".

The first guide adds the guides' rules to `sphinx/AGENTS.md`, adapted from catsim's charter. The
guides may use the figures of `maintenance/docs-figures.md` once it is done.

## Tests

- Every code block runs as written, in order, against the current API.
- Each guide's reader can finish its task from the page alone, following links only for depth.
