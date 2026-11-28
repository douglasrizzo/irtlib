# Maximum Likelihood

:API: {py:func}`irtlib.ability.maximum_likelihood`
:Sources: {cite:t}`Magis2012`; {cite:t}`Warm1989`; {cite:t}`Chang1996`; {cite:t}`Veerkamp1994`

## Motivation

The maximum likelihood (ML) estimate is the ability at which the examinee's responses are most
likely. It needs no prior, so it adds nothing to what the responses say, and its standard error
comes straight from the test information. Its weakness is responses all correct or all incorrect,
which no finite ability explains best; irtlib reports them as they are, $+\infty$ or $-\infty$, and
lets the caller choose bounds or a fallback estimator. {doc}`map` and {doc}`eap` add a prior, and
{doc}`wle` removes the estimate's first-order bias.

## Definition

The estimate maximizes $L_j(\theta)$, and its standard error is $1 / \sqrt{I(\hat\theta)}$
{cite:p}`Magis2012{p. 4, Equations 2 and 3}`. The likelihood of responses all correct rises with the
ability and that of responses all incorrect falls, so their estimates are $+\infty$ and $-\infty$
{cite:p}`Warm1989{p. 442}`. As $\theta \to -\infty$ each $P_i$ tends to $c_i$, and as
$\theta \to +\infty$ to $d_i$, so the likelihood tends to $\prod_i c_i^{u_{ij}} (1 - c_i)^{1 - u_{ij}}$
and to $\prod_i d_i^{u_{ij}} (1 - d_i)^{1 - u_{ij}}$: under the 3PL and 4PL, responses that are
neither all correct nor all incorrect can also have their supremum at an infinity.

The maximum likelihood estimate is the maximizer over the extended real line: $+\infty$ or $-\infty$
wherever the likelihood comes closest to its supremum only in that limit, with standard error
$\infty$, since the information tends to 0 at either infinity
({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`). An examinee who answered no item has a
constant likelihood, which every ability maximizes: its estimate is NaN, and its standard error
$\infty$, the information being 0.

## Implementation

**Bounds or a fallback.** The caller has two ways to finite numbers, never used together:

- `bounds`: the estimate maximizes the likelihood within them, and responses all correct or all
  incorrect give the bound they point to. Bounds are always set in practice
  {cite:p}`Warm1989{p. 442}`.
- `fallback`: another estimator, called once on the rows whose estimate is not finite, the
  unanswered rows included, whose estimates, standard errors and evaluation counts replace theirs;
  the `from_fallback` field of {class}`~irtlib.ability.AbilityEstimates` marks those rows.

**The search.** Where an item answered has $c_i > 0$ or $d_i < 1$, the likelihood can have several
maxima: under the 3PL the likelihood equation may have several roots {cite:p}`Chang1996{p. 223}`, and
the likelihood several local maxima {cite:p}`Veerkamp1994{p. 4}`. The search evaluates the log-likelihood at equally spaced abilities of the range: 101
when no item answered has a slope $a_i$ above 2.5, 0.12 apart on $[-6, 6]$, and otherwise
$\lceil 40 \max_i a_i \rceil + 1$, 0.3 divided by the steepest slope apart on $[-6, 6]$, since a
steep item can make a peak narrower than 0.12, up to 2,401 points, 0.005 apart, for slopes of 60 and
above, so that a pathological slope cannot exhaust memory. It refines the best between its neighbours
by Newton's method on the log-likelihood's first and second derivatives, from the vertex of the
parabola through the three points, keeping each step inside a bracket that shrinks to the side the
first derivative points to and bisecting the bracket where a step would leave it, and it keeps the
refinement only if no worse. When every item answered has $c_i = 0$ and $d_i = 1$, the second
derivative is $-I$ (see {doc}`map`), so the log-likelihood is concave and has at most one maximum.
The search then skips the points: the estimate is a bound where the first derivative points out of
the range, and otherwise the zero of the first derivative, found by the same bracketed Newton's
method from the middle of the range.

On about 38,000 random mixed patterns of 1 to 60 3PL and 4PL items with slopes from 0.5 to 2.5, half
with responses drawn from the model, 101 and 201 points found the global maximum on $[-6, 6]$ every
time and 81 points missed it once, while Brent's method over the whole range, with no points, missed
it 203 times, by a median of 3.7 in ability. On about 27,000 patterns of 1 to 30 items, half of them
with slopes up to 60, the points spaced by the steepest slope missed it twice, each time by less than
0.001 in log-likelihood, where 101 points missed it 50 times and 201 points 11 (measured 2026-11-28).
The numbers of points have no published source; the
[sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list them.

**Without bounds.** The range starts as $[-6, 6]$. Beyond the lower end of the range, a correct
response's probability is at most its value at the end, since it rises with the ability, and an
incorrect one's at most $1 - c_i$; beyond the upper end, a correct one's is at most $d_i$ and an
incorrect one's at most its value at the end. While these bounds leave room beyond an end for a
log-likelihood above both the best found and the two limits, that end moves out, twice as far each
time, and the new stretch is searched the same way, with as many points as the range, so that their
spacing doubles with each stretch. The estimate is then the infinity of the larger limit, $-\infty$
of two equal ones, if that limit is at least the best value found, within a relative $10^{-12}$, and
the best ability found otherwise. After 64 moves of an end, which only items of a vanishing
discrimination need, the search stops at the best ability found and warns.

:::{dropdown} Comparison with mirt and catR

- **mirt 1.48** also gives $+\infty$ or $-\infty$ to responses all correct or all incorrect
  (`R/fscores.internal.R:397-405`), with no standard error, NA, where irtlib's is $\infty$, and NA to
  both for an examinee who answered no item (`R/fscores.internal.R:94-96, 185`). It searches the other
  responses with R's `nlm` from the EAP estimate, up to an ability of $\pm 20$ (`R/fscores.R:305`;
  `R/fscores.internal.R:687`), takes the standard error from `nlm`'s numerical Hessian, the observed
  information (`R/fscores.internal.R:1072-1101`), which is $I$ for items with $c_i = 0$ and $d_i = 1$,
  and turns an estimate whose standard error exceeds 30 into an infinity
  (`R/fscores.internal.R:1094-1096`), which irtlib does not do.
- **catR 3.17** sets bounds by default: its default `constantPatt = NULL` estimates responses all
  correct or all incorrect by maximum likelihood too {cite:p}`Magis2022{p. 115}`, which returns the end
  of its `range` that the score points to (`R/thetaEst.R:9-10, 100-109`). Where the likelihood has a
  local minimum inside the range, catR's `thetaEst` solves the likelihood equation with `uniroot`
  (`R/thetaEst.R:100-102`), which can converge to that minimum, while irtlib maximizes the
  likelihood: the parity tests keep those cases as expected failures
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 14 42 26

* - Parameter
  - Default
  - Description
  - Default from
* - `bounds`
  - None
  - The interval the estimate maximizes the likelihood within; None searches the extended real line.
  - {doc}`ADR-0010 </adr/0010-ability-estimation-contract>`
* - `fallback`
  - None
  - Another estimator for the examinees without a finite estimate; not with `bounds`.
  - {doc}`ADR-0010 </adr/0010-ability-estimation-contract>`
* - `tol`
  - `1e-6`
  - The search's tolerance in ability.
  - catsim's ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`)
```

## Guarantees

1. Responses all correct get $+\infty$ and responses all incorrect $-\infty$, with standard error
   $\infty$, unless `bounds` or `fallback` is given.
2. With `bounds`, the estimate of every examinee who answered an item lies within them.
3. Where the likelihood has several maxima, the estimate is the highest the search's points
   separate.
4. An examinee who answered no item gets NaN, with standard error $\infty$, unless `fallback` is
   given.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.ability.maximum_likelihood
```
