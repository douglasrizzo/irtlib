---
title: Calibrate multidimensional item response models
kind: feature
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: [scope set by ADR-0001]
---

# Calibrate multidimensional item response models

## Problem

Tests that measure several abilities at once need multidimensional models, with a discrimination per
dimension and an ability vector per examinee. Their calibration integrates over several dimensions,
which changes the quadrature, the identification constraints and the cost of every method. ADR-0001
keeps irtlib to one ability per examinee, and catsim's models are unidimensional.

## Proposal

Moving this into scope changes ADR-0001, and needs a multidimensional item format that catsim, or
whatever consumes the banks, can read. The research for it is a pass of its own, under the citation
rules of `backlog/README.md`.
