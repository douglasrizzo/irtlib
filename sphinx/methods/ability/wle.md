# Weighted Likelihood

:API: {py:func}`irtlib.ability.weighted_likelihood`
:Sources: {cite:t}`Warm1985`; {cite:t}`Warm1989`; {cite:t}`Magis2012`

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

Warm's dissertation gives the same steps: the maximum likelihood estimate's bias is $-J / (2 I^2)$,
the expression of Lord (1983a) as {cite:t}`Warm1985{p. 7, Equation 6}` gives it; the weight's
log-derivative $J / (2 I)$ cancels it (p. 66, Equations A.6 and A.7), which gives the modified score
equation (p. 8, Equation 11). In general the weight has no closed form, but when every $c_i = 0$, in
Warm's model, which has no upper asymptote, it is $I^{1/2}$ {cite:p}`Warm1985{p. 9}`, and under the 1PL and the 2PL the WLE is the Bayes modal
estimate under Jeffreys' prior {cite:p}`Magis2012{p. 6}`. When an item has $c_i > 0$ or $d_i < 1$, $J$
is not the derivative of $I$, so the WLE is defined by its equation, not by a likelihood
weighted by $I^{1/2}$.

## Implementation

**The standard error** is $1 / \sqrt{I(\hat\theta)}$, infinite where the information is 0.
{cite:t}`Magis2012{p. 6, Equation 14}` give a corrected one, which adds $(I'J - IJ') / (2 I^2)$ to the
information, and which irtlib does not use. When every item has $c_i = 0$ and $d_i = 1$ the corrected
sum is the curvature of the weighted log-likelihood at its maximum, so it is positive; with a guessing
parameter or an upper asymptote below 1 it can be negative, which leaves no standard error. It was on
13 of 3,920 sets of responses, 0.4% of their probability, on random 3PL tests of 2 to 8 items, and on
none of 4,560 on 2PL tests (measured 2026-11-24).

Against the WLE's spread at a known ability, over every set of responses to 5 and to 10 items and
3,000 simulated ones to 20 and to 40, the standard error $1 / \sqrt{I}$ overestimates the error of
short tests, by 10% under the 2PL and by 28% under the 3PL at 5 items: a stopping rule on the
standard error runs a few items longer than needed. Equation 14 roughly halves that gap. From 20 items
on, the two agree within about 3% (measured 2026-11-24).

**Not the approximation.** Expanding the modified score around the maximum likelihood estimate gives
$\hat\theta_{\mathrm{WLE}} \approx \hat\theta_{\mathrm{ML}} + J / (2 I^2)$, which irtlib does not use: it
fails where the WLE matters most, since the maximum likelihood estimate is infinite for responses all
correct or all incorrect, and with a guessing parameter it can lie where $I$ is near 0.

**The search.** It evaluates the modified score at 121 equally spaced abilities within the bounds,
0.1 apart on the default $[-6, 6]$, so two roots closer than that can fall between the same two
points and go unseen.
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
