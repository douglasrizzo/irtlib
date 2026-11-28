# ADR-0006: The Calibrator Contract

## Status

Accepted 2026-11-27; revised 2026-11-27 and 2026-11-28

## Context

Every calibration method needs the same answers: how a caller passes responses, what a calibration
returns, where each part of the code lives. The maintainer accepted them on 2026-11-27, before
any method existed; on 2026-11-28, joint maximum likelihood's abilities and each item's covariances
joined its result.

mirt has one entry point with the model as an argument (mirt 1.48 manual, p. 123). The 2PL, 3PL and
4PL share their E-step, and priors change only the M-step (Mislevy, 1986, p. 182).
mirt takes a missing response as `NA` (manual p. 123), which contributes no factor to the marginal
likelihood (source `R/04-PrepData.R:107`).

## Decision

- **Calibrators.** One class per estimation method, configured at construction, with the model an
  `IrtModel` value where the method fits several. Its method takes the responses and returns a
  result.
- **Responses.** A float array with one row per examinee and one column per item, holding 0, 1, or
  NaN for a missing response. It is checked where it enters: values, shape, and items no examinee
  answered. An examinee who answered no item is accepted: it changes no estimate.
- **The result.** An immutable value holding:
  - the estimates in $(a, b, c, d)$, and a bank of the items valid for use, with a warning naming
    any other, such as a miskeyed item;
  - each item's covariances in the same parameterization, a shared parameter in every item's, and
    the standard errors from them; none, with a flag, if the information cannot be inverted;
  - the maximized log-likelihood, which likelihood it is, and the number of estimated parameters;
  - the cycles run, whether the stopping rule was met, and the rule's final value;
  - for a marginal likelihood only, the grid abilities were integrated over, which fit and ability
    estimates reuse; for a joint one only, its abilities.
- **Comparison.** Model comparison and item fit refuse a calibration whose likelihood is not
  marginal.
- **Non-convergence.** A result whose stopping rule was not met is flagged, with a warning.
- **Ability estimates and item fit**, but for a joint calibration's abilities, are functions of the
  responses and the items, not part of the result.
- **Missing responses by method.** A method whose sources do not handle missing responses refuses
  them, with an error naming an alternative: conditional and joint maximum likelihood, S-X² and the
  mean squares.
- **Layers,** lowest first: the item response functions and their derivatives; the quadrature grid,
  priors and posterior, and the item bank; ability estimation; calibration's checks, result and
  M-step; the methods; then fit. Drawing responses is its own module; import-linter enforces the
  order.

## Consequences

- A method author writes one class against one contract, sharing the E-step, M-step, checks and
  result.
- A calibration's bank, in the bank format (ADR-0001), goes straight into catsim.
- A calibration that stops short is never mistaken for a converged one.
- Whether Markov chain Monte Carlo returns draws or summaries is its card's decision.

## Alternatives Considered

- Plain functions: each call repeats the configuration.
- One class per model: the models share the E-step and differ only in the M-step's terms.
- A separate mask for missing responses, or a sentinel value: two arrays to keep in step, or a
  reserved code in the data.
- Raising an error on non-convergence: it discards estimates a caller may want to inspect.
- Abilities and fit inside every result: they need the responses, which the result does not keep.
- Standard errors alone, too few for a band around an item's curve; covariances across items,
  which nothing reads.
