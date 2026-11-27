# ADR-0007: MML-EM's Numerical Settings Follow mirt, With a Tighter Tolerance

## Status

Accepted 2026-11-27

## Context

MML-EM is pyitemcal's first method; the 3PL, 4PL and standard errors build on it
(`backlog/estimation/mml-em.md`). Its numerical settings decide how close a result is to the maximum
and whether the parity tests can reproduce mirt (ADR-0003). The guidance for choosing techniques
(`AGENTS.md`) departs from mirt only for a measured upside.

A scratch NumPy 2PL following the MML-EM card reached mirt 1.48's estimates on `LSAT7` within
$1.4 \times 10^{-6}$, and measurements on 2026-11-27 showed:

- mirt's default tolerance, every parameter changing by less than $10^{-4}$ between cycles (mirt
  1.48 manual, p. 127), stopped $1.7 \times 10^{-3}$ short of mirt's own converged estimate. At
  $10^{-6}$ the same rule cut the error about a hundredfold, for 1.6 to 3 times the cycles, on
  `LSAT7` and two simulated data sets. A rule on the gradient of the marginal log-likelihood did as
  well on the first two, but stopped 30 times further from the maximum than mirt's rule at $10^{-6}$
  on a slow case.
- Grids of 41 or more equally spaced nodes on $[-6, 6]$ gave the same estimates within $10^{-8}$.
- With 10% of responses missing, starting values computed from the answered responses converged in
  38 cycles, against 50 for mirt's, which counts a missing response as correct.
- The 2PL fits took under 0.1 seconds. Ramsay's acceleration as mirt applies it cut the cycles 1.7 to
  2.7 times without changing the estimates (the MML-EM card).

## Decision

- **Parameters and solver.** The M-step estimates mirt's parameters: the slope, the intercept, and
  the logits of $c$ and $d$ (manual pp. 132-133). Each item is solved by Fisher scoring with step
  halving. Estimates and standard errors are converted to catsim's $(a, b, c, d)$ at the end, the
  standard errors by the delta method.
- **Grid.** 61 equally spaced nodes on $[-6, 6]$ with normal masses, mirt's (manual pp. 127, 129):
  catsim's `QuadratureGrid.uniform(61)` with `normal_log_prior()`.
- **Stopping.** mirt's rule, every parameter changing by less than the tolerance between cycles,
  with a default tolerance of $10^{-6}$, and mirt's 500 cycles (manual p. 129).
- **Starting values.** mirt's formula, computed from the answered responses only; a caller may pass
  starting items.
- **Standard errors.** Oakes's identity, approximated by central differences, mirt's default
  (manual p. 122; `backlog/estimation/standard-errors.md`).
- **Acceleration.** None in the first version; measured again with the 3PL and 4PL, whose fits
  took hundreds to thousands of cycles.

## Consequences

- Departures from mirt: the default tolerance, the starting values over answered responses, and the
  solver. None changes the maximum, which the parity tests compare, with mirt's fixtures recorded
  at a tight tolerance (`backlog/discovery/mirt-parity-harness.md`).
- Priors are defined on the parameters the M-step estimates, so mirt's priors apply unchanged
  (ADR-0008).
- A fit takes up to three times mirt's cycles. Performance is not a requirement (`AGENTS.md`);
  acceleration is the remedy if it becomes one.

## Alternatives Considered

- mirt's tolerance of $10^{-4}$: its error on `LSAT7` was larger than the agreement parity needs.
- A stopping rule on the gradient: no better than mirt's rule at $10^{-6}$, and worse on the slow
  case.
- Mislevy's parameters, $(\log a, b, \operatorname{logit} c)$: mirt's priors are not defined on
  them, so parity with priors would need a second set of densities.
- A general optimizer, as mirt's BFGS: the same maximum, but an inexact M-step limits accuracy at
  tight tolerances (`backlog/estimation/mml-em.md`), and for the 2PL in the intercept-slope form
  the Hessian equals the information Fisher scoring uses (Baker, 1987, p. 116).
- Gauss-Hermite nodes: a grid other than mirt's, which parity would need a setting for.
- Acceleration now: a speed-up the first version does not need.
