# ADR-0006: The Calibrator Contract

## Status

Accepted 2026-11-27; revised 2026-11-27

## Context

Every calibration method needs the same answers first: how a caller passes responses, what a
calibration returns, and where each part of the code lives. Answered method by method, each method
would take a different shape. The maintainer accepted them on 2026-11-27, before any method
existed.

mirt has one entry point with the model as an argument (mirt 1.48 manual, p. 123). The 2PL, 3PL and
4PL share their E-step, and priors change only the M-step (Mislevy, 1986, p. 182).
mirt takes a missing response as `NA` (manual p. 123), and a missing response contributes no factor
to the marginal likelihood (mirt 1.48 source, `R/04-PrepData.R:107`).

## Decision

- **Calibrators.** One class per estimation method, configured at construction, with the model an
  `IrtModel` value. A calibrator's method takes the responses and returns a result.
- **Responses.** A float array with one row per examinee and one column per item, holding 0, 1, or
  NaN for a missing response. It is checked where it enters: values, shape, and items no examinee
  answered. An examinee who answered no item is accepted: it changes no estimate.
- **The result.** An immutable value holding:
  - the estimates in $(a, b, c, d)$, and a bank of the items valid for use, with a warning naming
    any other, such as a miskeyed item;
  - their standard errors in the same parameterization, or none, with a flag, when the information
    matrix cannot be inverted;
  - the maximized log-likelihood and the number of estimated parameters;
  - the cycles run, whether the stopping rule was met, and the rule's final value;
  - the grid abilities were integrated over, which fit and ability estimates reuse.
- **Non-convergence.** When the cycles run out, the calibrator returns the result flagged as not
  converged, with a warning.
- **Ability estimates and item fit** are functions of the responses and the items, a calibration's
  for item fit, not part of the result: EAP abilities; S-X² with its degrees of freedom per item;
  and the Rasch model's weighted and unweighted mean squares.
- **Missing responses by method.** A method whose sources do not handle missing responses refuses
  them, with an error naming an alternative: conditional maximum likelihood and S-X².
- **Layers,** lowest first: the item response functions and their derivatives; the quadrature grid,
  priors and posterior, and the item bank; calibration, with the shared E-step and M-step below the
  methods, each method computing its own standard errors from them; then ability estimation and
  fit. Drawing responses is its own module; the import-linter contract enforces the order.

## Consequences

- A method author writes one class against one contract, sharing the E-step, the M-step, the
  response checks and the result.
- A calibration's bank reads and writes in the bank format (ADR-0001), so it goes straight into
  catsim.
- A calibration that stops short is never mistaken for a converged one.
- Whether Markov chain Monte Carlo returns draws or summaries is its card's decision.

## Alternatives Considered

- Plain functions: each call repeats the configuration.
- One class per model: the models share the E-step and differ only in the M-step's terms.
- A separate mask for missing responses, or a sentinel value: two arrays to keep in step, or a
  reserved code in the data; NaN marks a missing response in the one array, with no code reserved.
- Raising an error on non-convergence: it discards estimates a caller may want to inspect.
- Abilities and fit inside the result: they need the responses, which the result does not keep.
