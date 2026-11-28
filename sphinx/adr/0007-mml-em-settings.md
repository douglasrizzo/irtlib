# ADR-0007: MML-EM's Numerical Settings Follow mirt, With a Tighter Tolerance and SQUAREM

## Status

Accepted 2026-11-27; acceleration added 2026-11-27

## Context

MML-EM's settings decide how close a result is to the maximum and whether parity tests can
reproduce mirt (ADR-0003); the guidance for choosing techniques (`AGENTS.md`) departs from mirt
only for a measured upside.

Measurements on 2026-11-27, with a scratch 2PL within $1.4 \times 10^{-6}$ of mirt 1.48's
estimates on `LSAT7`, showed:

- mirt's default tolerance, $10^{-4}$ (manual, p. 127), stopped $1.7 \times 10^{-3}$ short of its
  converged estimate; $10^{-6}$ cut the error a hundredfold for 1.6 to 3 times the cycles. A rule
  on the gradient stopped 30 times further away on a slow case.
- Grids of 41 or more equally spaced nodes on $[-6, 6]$ gave the same estimates within $10^{-8}$.
- With 10% of responses missing, starting values from the answered responses took 38 cycles,
  mirt's, which count a missing response as correct, 50.
- An exact BFGS M-step brought mirt's estimate within $4.2 \times 10^{-7}$ of the scratch EM's,
  against $1.4 \times 10^{-6}$ for its default.

The first version ran plain EM. With the 3PL and 4PL, the fixtures' fits took up to 936 cycles,
3.2 seconds, for a 4PL of 5,000 examinees with priors (measured 2026-11-27). Ramsay's rule as mirt
applies it cut that to 346 cycles, SQUAREM to 127 and 1.0 second, and SQUAREM cut every other
case's cycles 2.2 to 4.1 times, to the same estimates.

## Decision

- **Parameters and solver.** The M-step estimates mirt's parameters: the slope, the intercept, and
  the logits of $c$ and $d$ (manual pp. 132-133), by Fisher scoring with step halving over all free
  parameters at once. Estimates and standard errors are converted to the $(a, b, c, d)$ of ADR-0001
  at the end, the standard errors by the delta method.
- **Grid.** 61 equally spaced nodes on $[-6, 6]$ with normal masses, mirt's (manual pp. 127, 129).
- **Stopping.** mirt's rule, every parameter changing by less than the tolerance in a cycle, with a
  default tolerance of $10^{-6}$, and mirt's 500 cycles (manual p. 129).
- **Starting values.** mirt's formula, computed from the answered responses only; a caller may pass
  starting items.
- **Standard errors.** Oakes's identity by central differences, mirt's default (manual p. 122;
  Oakes, 1999, p. 480).
- **Acceleration.** SQUAREM by default (Du & Varadhan, 2020, p. 4, Algorithm 1): an extrapolation
  from every two cycles, kept where its log-posterior is above the second cycle's, then a cycle from
  it. An extrapolation is not a cycle, so the stopping rule measures an EM step.

## Consequences

- Departures from mirt: the tolerance, the starting values, the solver and the acceleration. None
  changes the maximum, which the parity tests compare, accelerated or not (ADR-0003).
- No cycle lowers the log-posterior, and a fit takes fewer cycles than mirt's on every fixture case
  that records them.

## Alternatives Considered

- mirt's tolerance of $10^{-4}$: its error on `LSAT7` exceeds the agreement parity needs.
- A stopping rule on the gradient: worse on the slow case.
- Mislevy's parameters, $(\log a, b, \operatorname{logit} c)$: parity with mirt's priors would
  need a second set of densities.
- A general optimizer, as mirt's BFGS: inexact at tight tolerances, and for the 2PL the Hessian is
  the information Fisher scoring uses (Baker, 1987, p. 116).
- Gauss-Hermite nodes: parity would need a setting for mirt's grid.
- Ramsay's rule as mirt applies it by default (`R/EMstep.group.R:347-357`): 2.7 times SQUAREM's
  cycles on the 4PL, and the log-posterior can fall. As published, it diverged on a slow 2PL.
- SQUAREM keeping every extrapolation: 90 cycles on the 4PL, but one lowered the log-posterior by
  16 on `LSAT6`'s 3PL from a far start.
- Halving a refused steplength, as mirt's SQUAREM does (`R/EMstep.group.R:359-388`): 171 cycles.
