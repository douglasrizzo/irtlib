# Expected A Posteriori

:API: {py:func}`irtlib.ability.expected_a_posteriori`
:Sources: {cite:t}`Bock1982`; {cite:t}`Magis2012`

## Motivation

The expected a posteriori (EAP) estimate is the mean of the examinee's posterior, the likelihood of
its responses times a prior density of ability, and its standard error is the posterior standard
deviation. It needs no search, only sums over a grid of abilities, and it is finite whatever the
responses. {doc}`map` takes the posterior's mode instead.

The posterior variance "serves as its error variance" {cite:p}`Bock1982{p. 432}`, and in Bock and
Mislevy's simulations the posterior standard deviation came close to the standard error of
measurement {cite:p}`Bock1982{p. 431}`.

## Definition

The EAP estimate is the mean of the examinee's posterior over the grid of
{doc}`/methods/models/quadrature`, and its standard error the posterior standard deviation
{cite:p}`Bock1982{p. 433, Equations 4 and 5}`; {cite:p}`Magis2012{pp. 5-6, Equations 10 and 11}`.
Bock and Mislevy credit these quadrature formulas to {cite:t}`Bock1981`
{cite:p}`Bock1982{p. 432}`.

## Implementation

The posterior is that of {doc}`/methods/models/quadrature`: the masses of the prior on the grid,
times the likelihood at each node, normalized, with no correction of the variance, where Bock and
Mislevy apply Sheppard's correction {cite:p}`Bock1982{p. 433}`. The nodes are evenly spaced, as
theirs are: Gauss-Hermite quadrature is exact for integrands that are polynomials, and "this class
does not include the likelihood functions in adaptive testing" {cite:p}`Bock1982{p. 433}`. It is finite
whatever the responses, and an examinee who answered no item gets the prior's mean and standard
deviation on the grid. A caller that estimates many matrices with the same grid and prior can pass a
{class}`~irtlib.quadrature.GridPrior` built once instead of the two.

**The grid's accuracy.** The grid is accurate while it resolves the posterior. Against a reference
grid of 2,401 nodes on the same range, the default grid of 121 nodes, 0.1 apart, gives the
posterior mean and standard deviation within 0.41% of the reference standard deviation while the
posterior standard deviation is at least 0.065, which takes a test information of about 236. Below
that it degrades: the error reaches 7.3% at a posterior standard deviation of 0.05 and 48% at 0.035.
A grid of 41 nodes, 0.3 apart, is within 0.37% at 0.2, and reaches 7.9% at 0.15 and 42% at 0.11.
The errors are the worst over the posterior's position between two nodes, for a normal likelihood
under the standard normal prior (measured 2026-11-28). The default's spacing is that of the 80
points between $\pm 4$ standard deviations that Bock and Mislevy suggest to keep precision down to
a posterior standard deviation of 0.2 {cite:p}`Bock1982{p. 433}`. A standard error below about 0.065
needs a finer grid than the default.

:::{dropdown} Comparison with mirt and catR

- **mirt 1.48:** irtlib's default grid and prior are mirt's, 121 nodes on $[-6, 6]$
  (`R/fscores.R:302, 318-319`; `R/fscores.internal.R:231`) with the standard normal density at the
  nodes, normalized (`R/fscores.internal.R:316-317`). mirt gives NA to an examinee who answered no
  item.
- **catR 3.17:** the parity tests first check that catR's recorded EAP estimates are the posterior
  mean and standard deviation written out, on the same nodes and weights, then compare irtlib's with
  them ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 18 38 26

* - Parameter
  - Default
  - Description
  - Default from
* - `grid`
  - 121 nodes on $[-6, 6]$
  - The nodes the posterior is computed on.
  - mirt's (`R/fscores.R:302, 318-319`); a spacing of 0.1, after {cite:t}`Bock1982{p. 433}`
* - `log_prior`
  - the standard normal
  - The prior's log-density.
  - mirt's (`R/fscores.internal.R:316-317`); abilities $N(0, 1)$
    ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
* - `grid_prior`
  - None
  - A {class}`~irtlib.quadrature.GridPrior` that holds a grid and a prior together, in place of the
    two.
  - none
```

## Guarantees

1. Every estimate is finite, whatever the responses, and lies within the grid's nodes.
2. An examinee who answered no item gets the prior's mean and standard deviation on the grid.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.ability.expected_a_posteriori
```
