# Weighted Likelihood

:API: {py:func}`irtlib.ability.weighted_likelihood`
:Sources: {cite:t}`Warm1989`; {cite:t}`Magis2012`

## Motivation

Warm's weighted likelihood estimate (WLE) removes the first-order bias of the maximum likelihood
estimate {cite:p}`Warm1989{p. 427}`; {cite:p}`Warm1989{p. 431, Theorem}`, without a prior.
Warm finds finite estimates for responses all correct or all incorrect on his tests
{cite:p}`Warm1989{p. 443, Table 1}`, and {cite:t}`Magis2012{p. 11}` propose the estimator for them.
Choose it over {doc}`ml` when the estimate's bias matters, and over {doc}`map` and {doc}`eap` when no
prior should enter the estimate.

## Definition

Warm's estimate is the root of the modified score equation

$$
\sum_i \frac{(u_{ij} - P_i) P_i'}{P_i Q_i} + \frac{J}{2 I} = 0, \quad J = \sum_i \frac{P_i' P_i''}{P_i Q_i}
$$

{cite:p}`Warm1989{p. 431, Equation 10}`; {cite:p}`Warm1989{p. 430}`, the derivative of the log of the
weighted likelihood $w(\theta) L_j(\theta)$, with $\partial \log w / \partial\theta = J / (2 I)$,
which the estimate maximizes (p. 446, Equations A3 and A4). When every $c_i = 0$, $w = I^{1/2}$
(p. 431), in Warm's model, which fixes $d_i = 1$ (p. 428, Equation 1). The estimate's variance is
asymptotically $1 / I$, that of the maximum likelihood estimate (p. 431).

## Implementation

**The standard error** is $1 / \sqrt{I(\hat\theta)}$; {cite:t}`Magis2012{p. 6, Equation 14}` give a
corrected one, which irtlib does not use.

**The search.** It evaluates the modified score at 121 equally spaced abilities within the bounds.
Each change of sign from positive to negative, refined by Brent's method step for step as SciPy
1.18.1's `scipy.optimize.brentq` takes it (`scipy/optimize/Zeros/brentq.c:36-130`), from the values
of the scan at its ends, is a maximum of the weighted likelihood, and so is a bound the score points
out of. Of several, the estimate is the one of largest weighted likelihood, ranked by integrating the
score from the lower bound with `scipy.integrate.quad`. irtlib's estimate lies within the bounds
whatever the responses. An examinee who answered no item has no estimate, NaN, and standard error
$\infty$, as under maximum likelihood ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`).

:::{dropdown} Comparison with mirt and catR

- **mirt 1.48** maximizes $\log L_j + \tfrac12 \log I$ instead (`R/fscores.internal.R:705-738`),
  Warm's estimate when every answered item has $c_i = 0$ and $d_i = 1$, and takes its standard error
  from `nlm`'s numerical Hessian of that weighted log-likelihood (`R/fscores.internal.R:1103-1119`).
- **catR 3.17** uses the same standard error by default (`R/semTheta.R:90-96`). Where Warm's modified
  score has three roots in the range, two maxima of the weighted likelihood and a minimum, catR's
  `uniroot` can converge to the lower maximum, where irtlib returns the one of larger weighted
  likelihood: the parity tests keep that case as an expected failure
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 18 38 26

* - Parameter
  - Default
  - Description
  - Default from
* - `bounds`
  - `(-6.0, 6.0)`
  - The interval the estimate lies within.
  - catsim's ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`)
* - `tol`
  - `1e-6`
  - The root search's tolerance in ability.
  - catsim's ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`)
```

## Guarantees

1. The estimate of every examinee who answered an item lies within the bounds, whatever the
   responses.
2. Where the weighted likelihood has several maxima in the bounds, the estimate is the one of largest
   weighted likelihood among those the scan separates.
3. An examinee who answered no item gets NaN, with standard error $\infty$.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.ability.weighted_likelihood
```
