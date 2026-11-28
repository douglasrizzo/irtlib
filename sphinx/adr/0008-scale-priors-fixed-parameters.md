# ADR-0008: Scale, Priors and Fixed Parameters

## Status

Accepted 2026-11-27

## Context

Item estimates mean something only on a scale, which every method fixes its own way. The 3PL and
4PL need priors to keep their estimates finite and plausible, and the same priors serve the 2PL
(Mislevy, 1986, p. 189; Baker, 1987, pp. 130-131). Two families of priors were on the table:
Mislevy's (1986, p. 183) normal priors on $\log a$, $b$ and the logit of $c$; and mirt's, on the
slope, the intercept and the logits of $c$ and $d$ that mirt estimates (mirt 1.48 manual, p. 155).
The two give different estimates from the same hyperparameters: mirt's lognormal prior on the slope
keeps the $-\log a$ term of the lognormal log-density (source `R/utils.R:1422-1433`), which a normal
prior on $\log a$ lacks. Without a prior, mirt's 3PL put four of five guessing estimates at 0 on
`LSAT7` (measured 2026-11-27).

On `LSAT6`, the 1PL with a common estimated slope and the 1PL with slopes fixed at 1 and the ability
variance estimated (mirt's `"Rasch"`) reach the same maximum, in 12 and 80 cycles (mirt 1.48,
measured 2026-11-27). An item answered correctly by all or none has no finite maximum likelihood
estimate; mirt's manual advises removing such items or using priors (p. 132).

## Decision

- **Scale.**
  - Marginal maximum likelihood: ability normal with mean 0 and variance 1, mirt's default
    (manual pp. 122, 131).
  - The 1PL: a common slope estimated with that normal ability distribution.
  - Conditional maximum likelihood: difficulties summing to zero.
  - Joint maximum likelihood: LOGIST's and BICAL's conventions (`backlog/estimation/jml.md`).
- **Priors.** mirt's densities on mirt's parameters: lognormal on the slope, normal on the
  intercept, and normal on the logits of $c$ and $d$. A prior is given per parameter and applies to
  every item. There is no default prior; the 3PL's documentation recommends one.
- **Fixed parameters.** A parameter can be fixed at one value for every item. Slopes fixed at 1 with
  the normal ability distribution give mirt's `"1PL"` (manual pp. 123, 133); $d$ fixed at 1 gives the
  3PL as a special case of the 4PL.
- **Items with no finite estimate.** Marginal maximum likelihood raises an error listing them,
  unless priors keep their estimates finite. Conditional and joint maximum likelihood leave out and
  report those answered by all or none, as their sources do (Gustafsson, 1980, p. 379; Mislevy &
  Stocking, 1989, p. 65; Wright & Douglas, 1977, p. 294); conditional maximum likelihood raises one
  for any other (Haberman, 2004, pp. 37-38).

## Consequences

- Priors use mirt's definitions, so the parity tests cover calibrations with priors.
- A caller who wants Mislevy's priors must translate them: BILOG's defaults are not mirt's.
- The common-slope 1PL converges faster than the `"Rasch"` form and needs no estimated variance;
  the `"Rasch"` form waits for a need.
- A 3PL calibrated without a prior may put guessing estimates at the boundary; its documentation
  recommends a prior.

## Alternatives Considered

- Mislevy's scales: well founded, but the parity tests could not cover priors without a second set
  of densities.
- Default priors, as BILOG's (Mislevy, 1986, p. 190): they change the estimates without the caller
  asking, and mirt sets none (source `R/06-LoadPars.R:361-362`).
- Priors and fixed values per item: no current need; one value per parameter covers mirt's
  `"1PL"` and the 4PL's special cases.
- Warning on items with no finite estimate instead of an error: the estimate that follows is
  meaningless, and the caller can remove the item or give a prior.
- mirt's `"Rasch"` form for the 1PL: the same model, more cycles.
