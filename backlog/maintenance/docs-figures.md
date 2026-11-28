---
title: Draw the documentation's figures
kind: docs
status: blocked
priority: 4
effort: M
blocked_by: [discovery/irt-plots]
raised_by: [maintainer on 2026-11-28]
---

# Draw the documentation's figures

## Problem

The site has no figures. Item response theory is taught with curves, and a page on a calibration or
an estimator says more with one figure than with a paragraph. The maintainer deferred figures on
2026-11-28 until the decision of `discovery/irt-plots` says which package keeps which plots.

## What irtlib and catsim have now

- irtlib draws nothing and has no plotting dependency (ADR-0002); its docs build has no plot
  directive.
- catsim draws its documentation's figures with matplotlib's plot directive, as SVG, in the style of
  its `sphinx/_static/figures.mplstyle`, and the figures call `catsim.plot`. irtlib's docs build
  cannot import catsim (ADR-0004).

## Proposal

Once the plots decision is made, add the plot directive with matplotlib in the `docs` dependency
group, and a copy of catsim's `figures.mplstyle` recolored to irtlib's blues (`#03a9f4`, `#0277bd`)
and slates. The figures, page by page:

- **Introduction:** the characteristic curves of a 1PL, 2PL, 3PL and 4PL item on one axis, with the
  roles of $a$, $b$, $c$ and $d$ marked; one item's curve with its information curve.
- **Item response functions:** curves for several discriminations; the information of 4PL items,
  whose single maximum moves with the asymptotes (Magis, 2013, p. 307; p. 312); test information
  with the standard error on a twin axis.
- **Quadrature and ability priors:** the nodes of the default grid with their normal masses; one
  examinee's posterior over the grid, with the prior.
- **MML:** the observed proportions correct at the nodes (the expected counts' ratio) with the fitted
  curve, the picture of a calibration; the marginal log-likelihood by EM cycle, with and without
  acceleration.
- **JML and CML:** JML against CML difficulties on a short test, showing that JML's Rasch
  difficulties converge to limits a term of order $1/M$ away (Haberman, 2004, p. 13).
- **Ability estimation:** the likelihood and the posterior of one response pattern, with the ML,
  MAP, EAP and WLE estimates marked; the bias of ML and WLE across abilities on a short test.
- **Item fit:** observed against expected proportions correct by score group for an item that fits
  and one that does not.
- **Parameter recovery** (a guide's figure): estimated against generating parameters.

Each figure needs its claims cited, as any page does; the parameter values come from
`ItemBank.generate` with a fixed `rng`.

## Tests

- The site builds with `-W` with every figure, each with a sentence on what to read from it.

## References

- Haberman, S. J. (2004). *Joint and conditional maximum likelihood estimation for the Rasch model
  for binary responses* (RR-04-20). ETS. https://doi.org/10.1002/j.2333-8504.2004.tb01947.x
- Magis, D. (2013). A note on the item information function of the four-parameter logistic model.
  *Applied Psychological Measurement, 37*(4), 304-315. https://doi.org/10.1177/0146621613475471
