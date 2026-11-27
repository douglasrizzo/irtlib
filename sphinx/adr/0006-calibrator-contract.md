# ADR-0006: The Calibrator Contract

## Status

Accepted 2026-11-27

## Context

Every calibration method needs the same answers first: how a caller passes responses, what a
calibration returns, and where each part of the code lives. Answered method by method, they would
give each method a different shape. The maintainer accepted the answers below on 2026-11-27, after
the research pass rebuilt the method cards from their sources and before any method existed.

mirt has one entry point with the model as an argument (mirt 1.48 manual, p. 123). The 2PL, 3PL and
4PL share their E-step, and priors change only the M-step (`backlog/estimation/mml-em-3pl.md`).
mirt takes a missing response as `NA` (manual p. 123), and a missing response contributes no factor
to the marginal likelihood (mirt 1.48 source, `R/04-PrepData.R:107`).

## Decision

- **Calibrators.** One class per estimation method, configured at construction, with the model one
  of catsim's `IrtModel` values. A calibrator's method takes the responses and returns a result.
- **Responses.** A float array with one row per examinee and one column per item, holding 0, 1, or
  NaN for a missing response. It is checked where it enters: its values, its shape, and items no
  examinee answered. An examinee who answered no item is accepted, since that examinee changes no
  estimate.
- **The result.** An immutable value holding:
  - the estimates, as a validated catsim `ItemBank` in $(a, b, c, d)$;
  - their standard errors in the same parameterization, or none, with a flag, when the information
    matrix cannot be inverted;
  - the maximized log-likelihood and the number of estimated parameters;
  - the cycles run, whether the stopping rule was met, and the rule's final value.
- **Non-convergence.** When the cycles run out, the calibrator returns the result flagged as not
  converged, with a warning.
- **Ability estimates and item fit** are functions of a result and the responses, not part of the
  result: EAP abilities on the calibration's grid; S-X² with its degrees of freedom per item; and the
  weighted and unweighted mean squares for the Rasch model.
- **Missing responses by method.** A method whose sources do not handle missing responses refuses
  them, with an error naming an alternative: conditional maximum likelihood and S-X².
- **Layers,** lowest first: the derivatives of the item response function; estimation, with the
  shared E-step and M-step below the methods, each method computing its own standard errors from
  them; then ability estimation and fit.
  Drawing responses for recovery tests is a module of its own. The import-linter contract that
  `AGENTS.md` requires enforces the order.

## Consequences

- A method author writes one class against one contract; the E-step, the M-step, the response
  checks and the result are shared.
- Results read and write as catsim banks, so a calibrated bank goes straight into catsim.
- A calibration that stops short is visible in its result and in a warning, and is never mistaken
  for a converged one.
- Whether Markov chain Monte Carlo returns draws or summaries is decided with its card.

## Alternatives Considered

- Plain functions: each call repeats the configuration, and nothing holds a method's settings for
  reuse across data sets.
- One class per model: the models share the E-step and differ only in the M-step's terms, so
  classes per model duplicate the shared work.
- A separate mask for missing responses, or a sentinel value: two arrays to keep in step, or a
  reserved code in the data; NaN marks a missing response in the one array, with no code reserved.
- Raising an error on non-convergence: it discards estimates a caller may want to inspect.
- Abilities and fit inside the result: they need the responses, which the result does not keep.
