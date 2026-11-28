# Introduction

{py:mod}`irtlib` is a Python package for item response theory (IRT), the models that relate an
examinee's responses to the items of a test to the examinee's ability and to the items' parameters.
It calibrates items, estimating their parameters from the responses of a sample of examinees; it
estimates the abilities of examinees from their responses to calibrated items; and it checks how
well a calibration fits the responses. This page introduces the models and the methods, and ends
with a map of irtlib's parts. {doc}`terms-and-notation` lists the terms and symbols these pages use.

## What a Test Measures

A test measures an examinee's *ability*, written $\theta$: a characteristic that cannot be observed
directly, such as knowledge of a subject. What can be observed are the examinee's responses to the
test's *items*, each scored correct or incorrect, and item response theory infers the ability from
them as a number on a continuous scale.

The responses of a sample of examinees form a *response matrix*, with one row per examinee and one
column per item. It holds 1 for a correct response, 0 for an incorrect one, and NaN where the
examinee did not answer the item, as when examinees take forms of a test that share only some of
their items. With $N$ examinees and $M$ items, $u_{ij}$ is examinee $j$'s response to item $i$.

## The Logistic Models

Under the four-parameter logistic model (4PL), an examinee of ability $\theta$ answers item $i$
correctly with probability

$$
P_i(\theta) = c_i + \frac{d_i - c_i}{1 + e^{-a_i(\theta - b_i)}}
$$

{cite:p}`Magis2013{p. 306, Equation 1}`, the item's *characteristic curve*. Its four parameters
shape the curve:

- the *discrimination* $a_i$, its slope: how sharply the item separates examinees of lower and
  higher ability;
- the *difficulty* $b_i$, the ability at which the curve is steepest, on the same scale as
  $\theta$;
- the *guessing parameter* $c_i$, the curve's lower asymptote: the probability that an examinee of
  very low ability answers the item correctly;
- the *upper asymptote* $d_i$: the probability that an examinee of very high ability answers the
  item correctly, below 1 for an item that such examinees still miss
  {cite:p}`Barton1981{p. 2}`.

The three-parameter model (3PL) fixes every $d_i = 1$, the two-parameter model (2PL) also every
$c_i = 0$, and the one-parameter model (1PL) also gives every item the same discrimination; the
Rasch model is the 1PL with every discrimination equal to 1 {cite:p}`Magis2012{p. 3}`. irtlib writes
the models without the scaling constant that some programs multiply each $a_i$ by
({doc}`/adr/0001-scope`).

Each response also carries *information* about the ability: the more information a test's items
give at an examinee's ability, the smaller the standard error of the estimate made from their
responses {cite:p}`Magis2012{p. 4, Equation 3}`. The formulas of these quantities are on the
{doc}`/methods/models/item-response-functions` page.

The ability scale has no natural origin or unit. Moving every ability and difficulty by one amount
leaves every probability unchanged, and so does stretching the abilities and difficulties by one
factor while the discriminations shrink by it: the scale is set only up to a linear transformation
{cite:p}`Baker1987{p. 112}`. Each calibration method fixes the scale in its own way, which its
documentation states.

## Calibration

Calibration estimates the items' parameters, with their standard errors, from a response matrix.
The examinees' abilities are unknown too, and the likelihood of the responses depends on them. The
three methods irtlib implements differ in what they do with the abilities.

**Joint maximum likelihood** treats each examinee's ability as one more parameter and maximizes the
likelihood of the responses over the abilities and the item parameters together, with no assumption
about how abilities are distributed in the population {cite:p}`Baker1987{p. 112}`. Each examinee
adds a parameter, so the number of parameters grows with the sample {cite:p}`Bock1981{p. 443}`: for
the Rasch model, as examinees are added to a test of fixed length, the difficulty estimates converge
to limits that differ from the difficulties by a term of order $1/M$
{cite:p}`Haberman2004{p. 13}`. Haberman (p. 35) finds it "difficult to advocate use of JMLE except
for preliminary estimation of parameters". An examinee who answers every item correctly, or none,
has no finite ability estimate, and an item answered correctly by every examinee or by none has no
finite difficulty, so both are left out before estimation
{cite:p}`Wright1977{p. 294}`. irtlib calibrates the Rasch model and the 2PL this way
({mod}`irtlib.jml`).

**Conditional maximum likelihood** applies to the Rasch model alone. There, an examinee's number of
correct responses is a sufficient statistic for its ability, so the likelihood of the responses
given each examinee's number correct holds the item difficulties alone, and "does not depend on the
distribution function" of ability {cite:p}`Haberman2004{p. 35}`. No such statistic eliminates the
ability from the likelihood of the other models {cite:p}`Bock1981{p. 443}`. The estimates are
consistent as the number of examinees grows {cite:p}`Andersen1970{p. 286, Theorem 1}`
({mod}`irtlib.cml`).

**Marginal maximum likelihood** integrates the abilities out over a population distribution and
maximizes the likelihood of the item parameters alone, and Bock and Aitkin's EM algorithm makes it
feasible for any number of items, with one small problem per item in each cycle
{cite:p}`Bock1981{pp. 444, 447}`. The integral becomes a sum over the nodes of a quadrature grid
{cite:p}`Bock1981{p. 445}`. Under the 3PL, different triples $(a_i, b_i, c_i)$ can trace similar
curves where the examinees are, so that the likelihood is "nearly flat along one or more dimensions"
{cite:p}`Mislevy1986{p. 189}`. Priors on the item parameters then make the estimates Bayes modal:
they maximize the likelihood times the prior, and the EM algorithm carries over
{cite:p}`Mislevy1986{p. 182}`. irtlib calibrates the 1PL to the 4PL this way, with or without
priors, and it is the method that takes a response matrix with missing responses
({mod}`irtlib.mml`).

## Ability Estimation

With the items calibrated, each examinee's ability is estimated from its responses, with a standard
error. irtlib has four estimators ({mod}`irtlib.ability`):

- the *maximum likelihood* (ML) estimate maximizes the likelihood of the responses, and its standard
  error is the inverse square root of the test information {cite:p}`Magis2012{p. 4, Equations 2 and
  3}`. Responses all correct give an estimate of $+\infty$, and responses all incorrect one of
  $-\infty$ {cite:p}`Warm1989{p. 442}`;
- the *maximum a posteriori* (MAP), or Bayes modal, estimate maximizes the posterior, the likelihood
  times a prior density of ability {cite:p}`Magis2012{p. 5, Equation 5}`;
- the *expected a posteriori* (EAP) estimate is the mean of the posterior, and its standard error is
  the posterior standard deviation {cite:p}`Bock1982{p. 433}`;
- Warm's *weighted likelihood* (WLE) estimate removes the first-order bias of the maximum likelihood
  estimate {cite:p}`Warm1989{p. 427}`.

## Checking a Calibration

Item fit compares, item by item, the responses a calibration predicts with those observed
({mod}`irtlib.fit`). Orlando and Thissen's S-X² groups examinees by their number-correct score,
which is observed, rather than by an ability estimate, whose groups depend on the model
{cite:p}`Orlando2000{p. 52}`; it applies to every model. The weighted and unweighted mean squares of
the Rasch model compare each response with its expectation at the examinee's estimated ability
{cite:p}`Wright1982{pp. 97-99}`.

A model with more parameters always fits at least as well, and model comparison asks whether the
improvement outweighs the parameters spent ({mod}`irtlib.comparison`): the likelihood-ratio test of
nested models {cite:p}`Orlando2000{p. 51}`, the information criteria AIC
{cite:p}`Akaike1974{p. 719}` and BIC {cite:p}`Schwarz1978{p. 461}`, and $G^2$, which compares the
model with the general multinomial over response patterns {cite:p}`Bock1981{p. 450}`. BIC selects
the true model as the sample grows when it is among the candidates, and AIC does not; AIC is
efficient when no candidate is true {cite:p}`Vrieze2012{Pages 10-11}`.

## irtlib's Parts

- {class}`irtlib.ItemBank` holds the items of a test, one row $[a, b, c, d]$ per item, with their
  names. It reads and writes them as a CSV file, and draws random banks.
- {mod}`irtlib.irt` holds the item response functions and the quantities derived from them:
  information, the standard error, the log-likelihood and their derivatives.
- {mod}`irtlib.quadrature` holds the quadrature grid that calibration and ability estimation
  integrate over, the ability priors, and each examinee's posterior over the grid.
- {mod}`irtlib.mml`, {mod}`irtlib.cml` and {mod}`irtlib.jml` hold the calibrators. Each returns the
  same result, a {class}`~irtlib.calibration.Calibration` ({doc}`/adr/0006-calibrator-contract`).
- {mod}`irtlib.ability` holds the ability estimators, {mod}`irtlib.fit` item and person fit, and
  {mod}`irtlib.comparison` model comparison.
- {mod}`irtlib.drawing` draws responses from the model, for simulation and parameter-recovery
  studies. Every function that draws random numbers takes them from its `rng` argument
  ({doc}`/adr/0005-random-numbers`).

## References

```{bibliography}
:filter: docname in docnames
```
