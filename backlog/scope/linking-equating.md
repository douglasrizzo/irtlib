---
title: Link separate calibrations onto one scale
kind: feature
status: parked
priority: 4
effort: L
blocked_by: []
raised_by: [scope set by ADR-0001]
---

# Link separate calibrations onto one scale

## Problem

Each calibration sets its own ability scale, so parameters from two calibrations, of two forms or two
years, are not comparable until they are linked, for example through items the calibrations share.
A bank built from several calibrations needs linking. ADR-0001 leaves linking and equating out of
scope.

## Proposal

Moving this into scope changes ADR-0001. The research covers the linking methods (moment methods and
characteristic-curve methods among them) and concurrent calibration of several groups, under the
citation rules of `backlog/README.md`.
