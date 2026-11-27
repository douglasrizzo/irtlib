---
title: Build the documentation site
kind: docs
status: blocked
priority: 3
effort: M
blocked_by: [estimation/mml-em]
raised_by: [setup of pyitemcal]
---

# Build the documentation site

## Problem

pyitemcal's documents are the README, the ADRs and the backlog. Researchers need a site with an
introduction, the methods' pages, the API reference and the ADRs, once there is a method to
document.

## What pyitemcal and catsim have now

`sphinx/` holds the ADRs, `references.bib` and the charter (`sphinx/AGENTS.md`). catsim's site is the
model: Sphinx with MyST, numpydoc, sphinxcontrib-bibtex, pydata-sphinx-theme and the plot directive,
built with `-W --keep-going` in CI and deployed to GitHub Pages, `/dev` from the `dev` branch and the
root from releases.

## Proposal

- Add `sphinx/conf.py`, the landing page, an introduction to calibration, a terms-and-notation page
  (replacing the interim terms in `sphinx/AGENTS.md`), a method-page template, the API reference and
  an architecture-decisions page with the ADRs' toctree.
- Add the `docs` dependency group, the build and deploy jobs to CI, and the build job to the
  rulesets' required checks.
- Extend `sphinx/AGENTS.md` with the rules for the new kinds of pages, adapted from catsim's charter.

## Tests

- The site builds with no warnings.
- Every ADR and every `references.bib` entry renders.
