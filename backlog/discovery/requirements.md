---
title: Decide the calibrator contract and the shape of pyitemcal's API
kind: decision
status: blocked
priority: 1
effort: L
blocked_by: [discovery/research-pass]
raised_by: [setup of pyitemcal]
---

# Decide the calibrator contract and the shape of pyitemcal's API

## Problem

The method cards carry API sketches from catsim's backlog (`MMLEMCalibrator.calibrate`,
`src/catsim/calibration/`), which nobody decided on. Every method needs the same answers first: how
a caller passes responses, what a calibration returns, and how a method author adds a method.
Deciding them method by method would give each method a different shape.

## Proposal

Settle each question below with the maintainer, record the lasting answers in ADRs, and rewrite the
method cards' implementation guidance against them. The research pass may add questions where the
sources leave a choice open.

1. **Calibrators:** classes holding configuration, with a method that takes the data, or functions;
   one class per method or one per model.
2. **The response matrix:** its orientation (examinees as rows), its dtype, and how a missing response
   is marked (NaN in a float array, a separate mask, or a sentinel), and what is checked where it
   enters (catsim's ADR-0009 checks explicit inputs where they enter).
3. **The result:** estimates as an `ItemBank` or a parameter array; standard errors; the
   log-likelihood; iterations and convergence status; ability estimates or posteriors; whether it is
   an immutable value.
4. **Convergence:** which criterion, what happens when it is not met (a warning, an error, or a flag
   on the result).
5. **Scale identification:** the ability distribution's mean and variance for marginal methods, the
   constraints for joint and conditional ones.
6. **Priors:** how a caller sets priors on item parameters, and on the ability distribution,
   reusing catsim's `LogPrior` and `QuadratureGrid` (ADR-0004).
7. **Starting values:** computed, passed by the caller, or both.
8. **Models:** how a caller chooses 1PL to 4PL, using catsim's `IrtModel`, and how fixed parameters
   (a common discrimination, a fixed guessing parameter) are expressed.
9. **Item fit:** a function of a calibration result and the responses, or a method of the result.
10. **Names:** catsim's one-name-per-concept table (`theta`, `items`, `responses`, `rng`, ...) for
    every name both packages share.
11. **Modules and layers:** the package's modules, and the import-linter layers contract they make.

## Tests

The decision is done when each question has an answer, every lasting answer is in an ADR, and every
method card's implementation guidance follows them.
