# Conditional Maximum Likelihood

:API: {py:class}`irtlib.cml.CMLCalibrator`
:Sources: {cite:t}`Andersen1970`; {cite:t}`Andersen1972`; {cite:t}`Gustafsson1980`; {cite:t}`Haberman2004`

## Motivation

In the Rasch model an examinee's number of correct responses is a sufficient statistic for its
ability, so the likelihood of the responses given each examinee's number correct holds the item
difficulties alone, and "does not depend on the distribution function" of ability
{cite:p}`Haberman2004{p. 35}`. Its maximizers, the conditional maximum likelihood (CML) estimates,
are consistent as the number of examinees grows {cite:p}`Andersen1970{p. 286, Theorem 1}`, though
"not in general ... efficient" (p. 284).

Choose it to calibrate the Rasch model without assuming how abilities are distributed in the
population. It needs complete responses; {doc}`mml` calibrates the Rasch model on incomplete ones,
under a normal population.

## Definition

Write $\varepsilon_i = e^{-b_i}$ for item $i$'s easiness, the item parameter of Gustafsson's
multiplicative form of the model {cite:p}`Gustafsson1980{p. 379, Equation 1}`, $s_i$ for its number
of correct responses and $n_r$ for the number of examinees with $r$ correct, out of $M$ items. The
conditional likelihood is

$$
\frac{\prod_i \varepsilon_i^{s_i}}{\prod_{r=1}^{M-1} \gamma_r^{n_r}},
$$

where the elementary symmetric function $\gamma_r$ is the sum, over every set of $r$ items, of the
product of their easiness values (p. 379, Equations 5 and 6). Its logarithm, the conditional
log-likelihood, is $-\sum_i s_i b_i - \sum_r n_r \log \gamma_r$; the marginal log-likelihood adds to
it the log-likelihood of the numbers correct {cite:p}`Haberman2004{pp. 35-36}`, so the two are not
compared ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

The estimates solve the likelihood equations

$$
s_i = \sum_{r=1}^{M-1} n_r \pi_{ri}, \qquad \pi_{ri} = \frac{\varepsilon_i \gamma_{r-1}^{(i)}}{\gamma_r},
$$

where $\gamma_{r-1}^{(i)}$, the derivative of $\gamma_r$ with respect to $\varepsilon_i$, is the
symmetric function of order $r - 1$ of every item but $i$
{cite:p}`Gustafsson1980{p. 380, Equation 7}`, and $\pi_{ri}$ is the probability of a correct response
to item $i$ given $r$ correct {cite:p}`Haberman2004{pp. 15-16}`. The difficulties are identified by
summing to zero ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`), as
{cite:t}`Andersen1972{p. 45}` identifies them; {cite:t}`Gustafsson1980{p. 380}` sets one easiness to
1 instead.

## Implementation

**Examinees and items left out.** For an examinee with no item or every item correct, "both sides
are equal to one" in the conditional likelihood {cite:p}`Engelen1987{p. 11}`, and items answered
correctly by every examinee or by none have no estimate {cite:p}`Gustafsson1980{p. 379}`: both are
left out, the examinees first and then the items, repeatedly until neither is left, since leaving an
item out can leave an examinee with no item or every item correct. The items left out get no
estimate.

**Likelihoods with no maximum.** Even then the conditional likelihood can lack a maximum. It has none
exactly when, for some number correct between 1 and $M - 2$ and some cut of the item totals, the
examinees with at most that many correct answered no item below the cut and the others every item
above it {cite:p}`Haberman2004{pp. 37-38, Theorem 10}`, whose bounds on the item totals, $n_0$ and
$n - n_0$, are 0 and $n$ once the examinees are left out. The likelihood then rises without bound as
the two groups of items move apart, so that under the sum to zero no difficulty has a finite
estimate, and calibration raises an error naming the groups.

**Newton-Raphson.** The equations are solved by Newton-Raphson {cite:p}`Haberman2004{p. 43}`. The
gradient of the conditional log-likelihood with respect to $b_i$ is $\sum_r n_r \pi_{ri} - s_i$, and
its Hessian is $-V$, where $V_{ij} = \sum_{r=1}^{M-1} n_r v_{rij}$ is the conditional covariance of
the item scores given the numbers correct (p. 39), with $v_{rii} = \pi_{ri} - \pi_{ri}^2$ and, for
$i \neq j$, $v_{rij} = \pi_{rij} - \pi_{ri} \pi_{rj}$ (pp. 27-28, 43). Here
$\pi_{rij} = \varepsilon_i \varepsilon_j \gamma_{r-2}^{(i,j)} / \gamma_r$, the probability of
correct responses to both items, where $\gamma_{r-2}^{(i,j)}$, the second derivative of $\gamma_r$
with respect to $\varepsilon_i$ and $\varepsilon_j$, is the symmetric function of order $r - 2$ of
every item but $i$ and $j$ {cite:p}`Gustafsson1980{p. 380}`. $V$ has rank $M - 1$
{cite:p}`Haberman2004{p. 39}`, so a step solves the system in its Moore-Penrose inverse $V^+$:
Haberman's iteration sets the first difficulty to 0 through a matrix $K$ (pp. 26, 43), and for
difficulties that sum to zero, the centred parameters of p. 21, it reads
$b_{t+1} = b_t + V^+ (\sum_r n_r \pi_r - s)$.

**Step halving, a departure.** Each step is halved, up to 30 times, until the conditional
log-likelihood does not fall, which the sources do not do: from the starting values below, full
steps diverged on two items answered 11 to 1.

**Starting values and stopping.** The iterations start from the values of
{cite:t}`Andersen1972{p. 52, Equation 40}`, $b_i = \log((N - s_i) / s_i)$ centred to sum to zero,
computed over the $N$ examinees kept where Andersen counts every examinee. They stop when no
difficulty changes by more than the tolerance in an iteration {cite:p}`Haberman2004{p. 44}`,
counting only a full step: a step that halving shrank below the tolerance can lie far from the
maximum. Without extreme items, Newton-Raphson usually takes 4 to 5 iterations
{cite:p}`Gustafsson1980{p. 383}`.

**The symmetric functions.** They come from the summation algorithm, which adds one item at a time,
$\gamma_r(\varepsilon_1, \dots, \varepsilon_t) = \gamma_r(\varepsilon_1, \dots, \varepsilon_{t-1}) +
\varepsilon_t \gamma_{r-1}(\varepsilon_1, \dots, \varepsilon_{t-1})$ from $\gamma_0 = 1$, with no
subtraction, so that it stays accurate on long tests {cite:p}`Gustafsson1980{p. 381, Equation 10}`.
Run with an item's easiness, or a pair's, set to zero, it gives the first and second derivatives
{cite:p}`Gustafsson1980{p. 382}`; {cite:p}`Baker1996{pp. 182-183}`, so that an iteration costs a
number of operations growing as $M^4$ {cite:p}`Baker1996{p. 190}`. The recursion runs on the
logarithms of the functions, which departs from the sources: those of orders near $M/2$ grow with
the test until they reach "the limit set by the size of the floating point numbers"
{cite:p}`Gustafsson1980{p. 383}`.

**Standard errors.** The standard error of each difficulty is the square root of the diagonal of
$V^+$ at the estimates. {cite:t}`Haberman2004{p. 40}` takes it from $K V^- K^T$ for the first
difficulty set to 0; for difficulties that sum to zero, centring that covariance gives $V^+$.
{cite:t}`Andersen1972{p. 50}` estimates the covariance by the inverse of the information as well.

**Complete responses.** CML needs complete responses: for responses missing at random, "ordinary
CML" "will likely fail" {cite:p}`Robitzsch2021{p. 827}`, and {class}`~irtlib.mml.MMLCalibrator`
calibrates incomplete responses ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 14 42 26

* - Parameter
  - Default
  - Description
  - Default from
* - `tolerance`
  - `1e-5`
  - The stopping rule: the largest change of a difficulty in a full Newton-Raphson step.
  - {cite:t}`Haberman2004{p. 44}`
* - `max_cycles`
  - `100`
  - The most Newton-Raphson iterations before the calibration stops unconverged.
  - irtlib; no source yet
    ([sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md))
```

## Guarantees

1. The difficulties of the items estimated sum to zero.
2. Examinees with no item or every item correct, and items answered correctly by every examinee or
   by none, are left out, repeatedly until neither is left; the items left out have NaN estimates.
3. Responses whose conditional likelihood has no maximum raise an error naming the two groups of
   items.
4. A missing response raises an error that names {class}`~irtlib.mml.MMLCalibrator` as the
   alternative.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.cml
   :members:
```
