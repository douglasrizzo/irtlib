# Quadrature and Ability Priors

:API: {py:mod}`irtlib.quadrature`
:Sources: {cite:t}`Bock1982`; {cite:t}`Bock1981`

## Motivation

Marginal maximum likelihood integrates over the abilities of a population, and the expected a
posteriori estimate integrates over one examinee's posterior. Neither integral has a closed form, so
both become sums over the nodes of a grid of abilities. {mod}`irtlib.quadrature` holds that grid, the
ability priors it is weighted by, and each examinee's posterior over it, shared by calibration and
ability estimation.

## Definition

Calibration and ability estimation both replace the integral over ability by a sum over the nodes
$X_q$, $q = 1, \dots, Q$, of a grid. Each node carries a mass $A_q$, the probability of a discrete
prior distribution at that node, and the masses sum to 1 {cite:p}`Bock1982{p. 433, Equation 6}`. A
grid holds the nodes and a weight $w_q$ for each, and a prior is a log-density $\log \pi$. A node's
mass is its weight times the prior density there, normalized:

$$
A_q = \frac{w_q\,\pi(X_q)}{\sum_{h=1}^{Q} w_h\,\pi(X_h)}.
$$

Examinee $j$'s likelihood at a node, $L_j(X_q)$, is the product over the items $i$ it answered of
$P_i(X_q)^{u_{ij}} [1 - P_i(X_q)]^{1 - u_{ij}}$ {cite:p}`Bock1982{p. 433, Equation 3}`, and its
posterior mass at the node is

$$
\frac{L_j(X_q)\,A_q}{\sum_{h=1}^{Q} L_j(X_h)\,A_h}.
$$

The mean of the posterior over the nodes is the EAP estimate of the examinee's ability, and its
variance serves as the estimate's error variance: the square of the posterior standard deviation
{cite:p}`Bock1982{p. 432}`; {cite:p}`Bock1982{p. 433, Equations 4 and 5}`.

## Implementation

**The grid.** The equal weights of {meth}`~irtlib.quadrature.QuadratureGrid.uniform` cancel, which
leaves the prior density at the nodes, normalized: the masses {cite:t}`Bock1981{p. 449}` take for a
normal prior, and for any other. Those nodes are evenly spaced, as Bock and Mislevy's are:
Gauss-Hermite quadrature, they note, has less advantage for the likelihoods of adaptive testing
{cite:p}`Bock1982{p. 433}`. The default grid has 121 nodes on the extended ability range $[-6, 6]$,
a spacing of 0.1: about the spacing of the 80 points between $\pm 4$ standard deviations that Bock
and Mislevy (p. 433) suggest to keep precision down to a posterior standard deviation of 0.2.
Calibration uses its own grid, 61 nodes on the same range
({doc}`ADR-0007 </adr/0007-mml-em-settings>`).

**The posterior.** An examinee who answered no item has $L_j = 1$ at every node, and its posterior
is the masses. The log-likelihoods accumulate as sums over the items
{cite:p}`Bock1982{p. 432}`, so {func}`~irtlib.quadrature.posterior` takes each examinee's
log-likelihood at the nodes, whatever the response function that gave it. It adds the log of each
node's weight and prior density, and shifts each examinee's row by its maximum before
exponentiating, so that the likelihood of a long response pattern does not underflow. A
{class}`~irtlib.quadrature.GridPrior` evaluates and checks a prior on a grid's nodes once, for a
caller that computes many posteriors with the same grid and prior.

**Departure from the source.** Bock and Mislevy apply Sheppard's correction to the posterior
variance {cite:p}`Bock1982{p. 433}`; {func}`~irtlib.quadrature.posterior_variance` does not.

:::{dropdown} Comparison with mirt and catR

- The range $[-6, 6]$ is mirt 1.48's default `theta_lim` {cite:p}`Chalmers2026{p. 129}`.
- The EAP estimates that catR 3.17 records are checked to be the posterior mean and standard
  deviation written out, on the same nodes and weights, within a relative $10^{-10}$, before irtlib
  is compared with them ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 26 14 34 26

* - Parameter
  - Default
  - Description
  - Default from
* - `mean`, `sd` of {func}`~irtlib.quadrature.normal_log_prior`
  - `0.0`, `1.0`
  - The normal prior's mean and standard deviation.
  - abilities $N(0, 1)$, mirt's default ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
* - `low`, `high` of {func}`~irtlib.quadrature.uniform_log_prior`
  - `-6.0`, `6.0`
  - The interval of the uniform prior.
  - the extended ability range, mirt's `theta_lim` {cite:p}`Chalmers2026{p. 129}`
* - `n_nodes` of {meth}`~irtlib.quadrature.QuadratureGrid.uniform`
  - `121`
  - The number of equally spaced nodes.
  - a spacing of 0.1, after {cite:t}`Bock1982{p. 433}`
* - `low`, `high` of {meth}`~irtlib.quadrature.QuadratureGrid.uniform`
  - `-6.0`, `6.0`
  - The first and last node.
  - the extended ability range, mirt's `theta_lim` {cite:p}`Chalmers2026{p. 129}`
```

## Guarantees

1. The masses of a grid under a prior are non-negative and sum to 1; a node outside a uniform
   prior's interval gets mass 0.
2. An examinee who answered no item gets the prior's masses as its posterior.
3. The posterior of a long response pattern does not underflow to zero at every node.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.quadrature
   :members:
```
