# Maximum A Posteriori

:API: {py:func}`irtlib.ability.maximum_a_posteriori`
:Sources: {cite:t}`Samejima1969`; {cite:t}`Magis2012`; {cite:t}`Warm1989`; {cite:t}`Bock1982`

## Motivation

The maximum a posteriori (MAP), or Bayes modal, estimate is the most likely ability given the
responses and a prior density of ability: the mode of the posterior. Under a normal prior the
posterior of responses all correct or all incorrect has a finite mode, and the standard error adds
the prior's precision to the test information. {doc}`eap` takes the posterior's mean instead of its
mode, and {doc}`ml` uses no prior.

{cite:t}`Samejima1969{p. 7, Equations 2-5 and 2-6}` defines the Bayes modal estimator, the ability
that maximizes the prior density times the likelihood; {cite:t}`Bock1982{p. 432}` credit
{cite:t}`Bock1981` with the name MAP, and {cite:t}`Magis2012{p. 4}` attribute the estimator to
Birnbaum (1969).

## Definition

The MAP estimate maximizes the posterior, $\log \pi(\theta) + \log L_j(\theta)$ for a prior density
$\pi$ {cite:p}`Magis2012{p. 5, Equation 5}`; {cite:p}`Warm1989{p. 430}`. For a normal prior of
variance $\sigma^2$ its standard error is $1 / \sqrt{1 / \sigma^2 + I(\hat\theta)}$
{cite:p}`Magis2012{p. 5, Equation 6}`.

**Against the maximum likelihood and EAP estimates.** When every answered item has $c_i = 0$ and
$d_i = 1$ the log-likelihood is concave, so under a normal prior the derivative of the log-posterior
points toward the prior mean at the maximum likelihood estimate and toward that estimate at the
prior mean, and the single mode lies between the two. Under a uniform prior on an interval wide
enough, the MAP and maximum likelihood estimates are equivalent {cite:p}`Magis2012{p. 5}`. As items
accumulate, the posterior approaches normality {cite:p}`Bock1982{p. 434}`, and its mode, the MAP
estimate, and its mean, the {doc}`eap` estimate, draw together.

## Implementation

**The standard error.** For any prior, irtlib takes
$1 / \sqrt{I(\hat\theta) - (\log\pi)''(\hat\theta)}$, with the prior's second derivative a central
difference of step $10^{-4}$: Equation 6 for a normal prior, and for a uniform one, inside its
support, the maximum likelihood standard error. Where the difference is not finite, at the edge of a
bounded prior's support, the prior adds nothing; where the sum is not positive, the standard error is
$\infty$. The general form has no published source; the
[sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list it.

**The search.** It evaluates the log-posterior at equally spaced abilities within the bounds: under a
normal prior whose precision $1 / \sigma^2$ is a finite positive float, as many as the
{doc}`ml` search for the same items, and under any other prior 201, 0.06 apart on $[-6, 6]$. Three or
more tied at the maximum make a flat posterior, and the estimate is the middle one; otherwise the
best is refined between its neighbours, and the refinement is kept only if no worse, so the estimate
stays inside the prior's support and is the highest of the modes the points separate. With a normal
prior $N(\mu, \sigma^2)$ the refinement is the maximum likelihood search's Newton's method, with
$-(\theta - \mu) / \sigma^2$ and $-1 / \sigma^2$ added to the derivatives, and on items that all
have $c_i = 0$ and $d_i = 1$ the log-posterior is concave too, and the search skips the points as the
maximum likelihood search does. With any other prior the refinement is SciPy's bounded search,
`scipy.optimize.minimize_scalar`. An examinee who answered no item gets the prior's mode.

The points come first for two reasons. A prior that is zero on part of the bounds, such as a
uniform prior on $[-1, 1]$ searched over $[-6, 6]$, makes the log-posterior $-\infty$ there, a flat
region that a bounded scalar search cannot cross; and a posterior with several modes, from a prior
with several or from a 3PL likelihood, has local modes where a local search can stop. The prior's
support must hold at least one of the points, which any interval wider than their spacing does.

**The observed information.** When every answered item has $c_i = 0$ and $d_i = 1$, the observed
information is $I$ exactly {cite:p}`Warm1989{p. 449}`, whose model fixes $d_i = 1$
{cite:p}`Warm1989{p. 428, Equation 1}`.

:::{dropdown} Comparison with mirt and catR

- **mirt 1.48** maximizes the same posterior with `nlm` and takes the standard error from `nlm`'s
  numerical Hessian (`R/fscores.internal.R:684-703, 1047-1070`): the observed information plus the
  prior's precision. When every answered item has $c_i = 0$ and $d_i = 1$, the two standard errors
  agree.
- **catR 3.17** computes the same standard errors, Equation 6 for a normal prior and the maximum
  likelihood one inside a uniform prior's support (`R/semTheta.R:43-50, 82-89`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 18 38 26

* - Parameter
  - Default
  - Description
  - Default from
* - `log_prior`
  - the standard normal
  - The prior's log-density, such as {func}`~irtlib.quadrature.normal_log_prior` or
    {func}`~irtlib.quadrature.uniform_log_prior`.
  - abilities $N(0, 1)$ ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
* - `bounds`
  - `(-6.0, 6.0)`
  - The interval the estimate maximizes the posterior within.
  - catsim's ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`)
* - `tol`
  - `1e-6`
  - The search's tolerance in ability.
  - catsim's ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`)
```

## Guarantees

1. Every estimate lies within the bounds, whatever the responses.
2. An examinee who answered no item gets the prior's mode.
3. Where the posterior has several modes, the estimate is the highest the search's points separate.
4. A standard error that the prior and the information leave undefined is $\infty$.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.ability.maximum_a_posteriori
```
