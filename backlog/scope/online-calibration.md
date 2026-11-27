---
title: Calibrate new items seeded into adaptive tests
kind: feature
status: parked
priority: 4
effort: L
blocked_by: []
raised_by: [scope set by ADR-0001]
---

# Calibrate new items seeded into adaptive tests

## Problem

An operational CAT renews its bank by seeding uncalibrated pretest items into live tests and
calibrating them from the responses, with the operational items' parameters fixed. The responses are
sparse and depend on the ability estimates the CAT made, so the methods differ from calibrating a
complete field test. ADR-0001 leaves online calibration out of scope, though it is the natural bridge
between irtlib and catsim: catsim's simulations can generate the seeded responses.

## Proposal

Moving this into scope changes ADR-0001, and builds on a working calibrator with fixed-parameter
support. The research covers the online calibration methods and pretest-item assignment designs,
under the citation rules of `backlog/README.md`. The calibration would be irtlib's and the adaptive
test catsim's, which builds on irtlib (ADR-0004), so the work would span both packages.
