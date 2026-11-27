---
title: Make pyitemcal citable with CITATION.cff and Zenodo
kind: maintenance
status: blocked
priority: 4
effort: XS
blocked_by: [maintenance/pypi-publishing]
raised_by: [setup of pyitemcal]
---

# Make pyitemcal citable with CITATION.cff and Zenodo

## Problem

Researchers who use pyitemcal in a study need a way to cite it, and a DOI for the version they used.

## Proposal

- Add `CITATION.cff`, which GitHub turns into a "Cite this repository" box.
- Enable Zenodo's GitHub integration, so each GitHub release gets a DOI, and add the concept DOI to
  the README and `CITATION.cff`.

## Tests

- `cffconvert --validate` passes on `CITATION.cff`.
- The first release after this card has a Zenodo record.
