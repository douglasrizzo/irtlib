---
title: Calibrate polytomous item response models
kind: feature
status: parked
priority: 4
effort: XL
blocked_by: []
raised_by: [scope set by ADR-0001]
---

# Calibrate polytomous item response models

## Problem

Items scored in more than two categories (rating scales, partial credit) need polytomous models,
such as the graded response model and the generalized partial credit model. ADR-0001 keeps irtlib
to dichotomous items, and catsim, whose banks irtlib produces, has no polytomous models either.

## Proposal

Moving this into scope changes ADR-0001, and needs a polytomous parameterization that catsim, or
whatever consumes the banks, can read. The research for it is a pass of its own, under the citation
rules of `backlog/README.md`.
