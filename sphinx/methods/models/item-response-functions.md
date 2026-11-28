# Item Response Functions

:API: {py:mod}`irtlib.irt`
:Sources: {cite:t}`Magis2013`; {cite:t}`Magis2012`; {cite:t}`Barton1981`; {cite:t}`Baker1987`;
  {cite:t}`Veerkamp1994`; {cite:t}`Barrada2006`; {cite:t}`Orlando2000`; {cite:t}`Thissen1995`

## Motivation

Every method in irtlib rests on the same few quantities of the logistic models: the probability of a
correct response, its derivatives, the information an item gives about the ability, the
likelihood of a set of responses, and the probability of each number correct. {mod}`irtlib.irt` computes them once, for every model from the 1PL
to the 4PL, and the calibrators, the ability estimators and item fit call it, as catsim does. A
package that builds on irtlib can call it too.

## Definition

Under the four-parameter logistic model, an examinee of ability $\theta$ answers item $i$ correctly
with probability

$$
P_i(\theta) = c_i + (d_i - c_i) F_i(\theta), \quad F_i(\theta) = \frac{1}{1 + e^{-a_i(\theta - b_i)}}
$$

{cite:p}`Magis2013{p. 306, Equation 1}`, and incorrectly with probability $Q_i = 1 - P_i$. The 3PL
fixes $d_i = 1$, and the 2PL also $c_i = 0$ {cite:p}`Magis2012{p. 3}`. The derivatives of $P_i$ with
respect to the ability are

$$
P_i' = a_i (d_i - c_i) F_i (1 - F_i), \quad P_i'' = a_i P_i' (1 - 2 F_i),
$$

the derivatives of {cite:t}`Barton1981{p. A2}` written with $F_i$ and their constant $D$ set to 1;
the first is also {cite:t}`Magis2013{p. 306, Equations 4 and 5}`.

**Information.** The information item $i$ gives about the ability is $I_i = P_i'^2 / (P_i Q_i)$
{cite:p}`Magis2012{p. 4, Equation 4}`, for the 4PL

$$
I_i(\theta) = \frac{a_i^2 (P_i - c_i)^2 (d_i - P_i)^2}{(d_i - c_i)^2 P_i Q_i}
$$

{cite:p}`Magis2013{p. 307, Equation 6}`, and the test information $I(\theta)$ is the sum of the
information of its items {cite:p}`Magis2012{p. 5, Equation 8}`.

**Standard error and confidence interval.** The maximum likelihood estimate of the ability is
asymptotically normal, with variance the inverse of the test information
{cite:p}`Warm1989{p. 445}` and standard error $1 / \sqrt{I(\theta)}$
{cite:p}`Magis2012{p. 4, Equation 3}`. {func}`~irtlib.irt.confidence_interval` is the interval this
normal approximation gives, $\hat\theta \pm z \cdot SE$, with $z$ the standard normal's quantile at
$(1 + \text{confidence}) / 2$.

**Reliability.** The reliability of the test at $\theta$ is $I(\theta) / (I(\theta) + 1 / \sigma^2)$,
with the abilities' variance $\sigma^2 = 1$, as they are $N(0, 1)$
({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`). For abilities $N(0, 1)$,
{cite:t}`Bock1982{p. 437, Equation 7}` define a related reliability of the EAP estimate, one minus
its posterior variance.

**The ability of maximum information.** An item is most informative at

$$
\theta^*_i = b_i + \frac{1}{a_i} \log \frac{x^*_i - c_i}{d_i - x^*_i}, \quad
x^*_i = 2 \sqrt{\frac{-u_i}{3}} \cos\left\{\frac{1}{3} \arccos\left(-\frac{v_i}{2}
\sqrt{\frac{27}{-u_i^3}}\right) + \frac{4 \pi}{3}\right\} + \frac{1}{2},
$$

with $u_i = -3/4 + (c_i + d_i - 2 c_i d_i)/2$ and $v_i = (c_i + d_i - 1)/4$, where $x^*_i$ is the
probability of a correct response at $\theta^*_i$ {cite:p}`Magis2013{pp. 308-309, Equations 13, 18
and 19}`. With $d_i = 1$ it is $\theta^*_i = b_i + \log[(1 + \sqrt{1 + 8 c_i}) / 2] / a_i$
(p. 310, Equation 23), and with $c_i + d_i = 1$ it is $b_i$ (pp. 309-310). The 3PL's $\theta^*_i$ is also
Equation 2 of {cite:t}`Barrada2006{p. 157}`, with their constant $D = 1.7$.

**The integral of the information.** Since $P_i' = a_i (P_i - c_i)(d_i - P_i) / (d_i - c_i)$
{cite:p}`Magis2013{p. 306, Equation 5}`, changing the variable of integration from $\theta$ to $P_i$
in the information and splitting the result into partial fractions integrates an item's information
in closed form:

$$
\int_{\theta_L}^{\theta_U} I_i(\theta)\, d\theta = \frac{a_i}{d_i - c_i} \Big[ P_i - c_i d_i \log P_i +
(1 - c_i)(1 - d_i) \log(1 - P_i) \Big]_{P_i(\theta_L)}^{P_i(\theta_U)},
$$

which extends the 3PL closed form of {cite:t}`Veerkamp1994{p. 22, Equation 25}`, to which it reduces
when $d_i = 1$. Under the 2PL the integral over the whole scale is $a_i$ (p. 22, Equation 28).

**The log-likelihood.** The log-likelihood of examinee $j$'s responses $u_{ij}$ is

$$
\log L_j(\theta) = \sum_i [u_{ij} \log P_i(\theta) + (1 - u_{ij}) \log Q_i(\theta)]
$$

over the items $i$ the examinee answered {cite:p}`Magis2012{p. 4, Equation 2}`;
{cite:p}`Barton1981{p. A1}`.

**The probability of each number correct.** The probability $S_k(\theta)$ that an examinee of
ability $\theta$ answers $k$ of $M$ items correctly, $k = 0, \dots, M$, sums the probabilities of
the response patterns with $k$ correct {cite:p}`Thissen1995{p. 40, Equation 1}`. Lord and
Wingersky's recursion builds it one item at a time, without enumerating the patterns
{cite:p}`Thissen1995{p. 42}`. With $S^{(i)}_k$ the probability of $k$ correct on the first $i$
items, it starts from $S^{(1)}_0 = Q_1$ and $S^{(1)}_1 = P_1$, and adding item $i$ gives

$$
S^{(i)}_k(\theta) = P_i(\theta)\, S^{(i-1)}_{k-1}(\theta) + Q_i(\theta)\, S^{(i-1)}_k(\theta),
\quad k = 0, \dots, i,
$$

where a term whose number correct is below 0 or above $i - 1$ is 0
{cite:p}`Orlando2000{p. 53, Equations 6-10}`; {cite:p}`Thissen1995{p. 42, Equation 16}`.
$S_k(\theta)$ is $S^{(M)}_k(\theta)$ (p. 53, Equation 11).

**A reporting scale.** {func}`~irtlib.irt.theta_to_scale` takes abilities to scores on a reporting
scale along the line through $(\theta_{min}, \text{scale}_{min})$ and
$(\theta_{max}, \text{scale}_{max})$, and {func}`~irtlib.irt.scale_to_theta` takes them back.

## Implementation

**Items and shapes.** The functions take one item as a row `[a, b, c, d]` of shape `(4,)`, or
several as a matrix of such rows, shape `(n, 4)`, with $a > 0$ and $0 \leq c < d \leq 1$, and a
response matrix with one row per examinee and one column per item, holding 1, 0 or NaN where the
examinee did not answer the item. The models have no scaling constant
({doc}`ADR-0001 </adr/0001-scope>`).

**The derivatives calibration uses.** Calibration solves for each item's parameters, and reports
their standard errors, from the derivatives of $P_i$ with respect to those parameters, which every
method that estimates items shares. It estimates them in four parameters
({doc}`ADR-0007 </adr/0007-mml-em-settings>`): the slope $\lambda = a$ and the intercept
$\zeta = -ab$, so that $a(\theta - b) = \lambda\theta + \zeta$, and the logits $\gamma$ of $c$ and
$\delta$ of $d$. Every array of these derivatives has the four on its last axis, in that order. By
the chain rule,

$$
\frac{\partial P}{\partial \lambda} = (d - c)\,\theta\,F(1 - F), \quad
\frac{\partial P}{\partial \zeta} = (d - c)\,F(1 - F), \quad
\frac{\partial P}{\partial \gamma} = c(1 - c)(1 - F), \quad
\frac{\partial P}{\partial \delta} = d(1 - d)\,F,
$$

from the derivatives in $a$, $b$ and $c$ of {cite:t}`Barton1981{p. A3}` and the derivative of the
logistic function. A parameter at its bound, $c = 0$ or $d = 1$, has derivative 0.

For an item answered by $N_q$ examinees at ability $X_q$, $r_q$ of them correctly, the
log-likelihood is $\sum_q [r_q \log P(X_q) + (N_q - r_q)\log Q(X_q)]$, the grouped form of
{cite:t}`Baker1987{p. 112, Equation 3}`. Its derivative in a parameter $\chi$ is
$\sum_q [r_q - N_q P(X_q)]\,(\partial P / \partial\chi) / (PQ)$, the form of
{cite:t}`Barton1981{p. A2}` summed within each group; with $d = 1$ the derivative in $\gamma$ is the
equation for the logit of $c$ of {cite:t}`Mislevy1986{p. 184, Equation 22}`. The expected
information of one response is $(\partial P / \partial\chi)(\partial P / \partial\phi) / (PQ)$,
Barton and Lord's second derivative (p. A2) without its terms in $u - P$, whose expected value is 0.
For the 2PL in these parameters the observed information equals the expected one
{cite:p}`Baker1987{p. 116}`.

**Precision far from the difficulty.** Nothing is computed by subtracting from 1. $F$ and $1 - F$
both come from the logit; $P = c + (d - c)F$ and $Q = (1 - d) + (d - c)(1 - F)$; $\log P$ and
$\log Q$ combine $\log c$ or $\log(1 - d)$ with $\log(d - c)$ and $\log F$ or $\log(1 - F)$. The
ratios $(d - c)F / P$, $(d - c)(1 - F) / Q$, $c / P$ and $(1 - d) / Q$ each lie between 0 and 1.
The first is 0 / 0 only where $c = 0$ and $F$ has underflowed to 0, the second only where $d = 1$
and $1 - F$ has, and both take their limit there, 1. With $P - c = (d - c)F$ and
$d - P = (d - c)(1 - F)$ {cite:p}`Magis2013{p. 306, Equation 3}`, the item information is
$a^2 F (1 - F)$ times the first two ratios.

**Departures from the sources.** Magis assumes $0 < c_i < 0.5$ and $0.5 < d_i < 1$
{cite:p}`Magis2013{p. 306}`; {func}`~irtlib.irt.max_info_theta` takes his Equation 19 for every item
with $d_i < 1$, and his Equation 23 for every item with $d_i = 1$. The closed form of the integral of
the information with $d_i < 1$ has no published source; the
[sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list both.

:::{dropdown} Comparison with mirt and catR

- The reliability is mirt 1.48's (`R/SingleGroup-methods.R:1474`), the curve that
  `plot(type = "rxx")` draws; the parity tests check it against mirt's recorded values within a
  relative $10^{-10}$.
- The closed-form integral of the information agrees with mirt 1.48's `areainfo`, which integrates
  numerically (`R/areainfo.R:49-66`), within the same tolerance.
- The parameters calibration solves in are mirt's ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).
- The probabilities and the information agree with catR 3.17's on the recorded abilities and items
  within a relative $10^{-10}$, after the recorded values are checked against Magis's equations
  written out ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

The functions take abilities and items; these have defaults:

```{list-table}
:header-rows: 1
:widths: 26 14 34 26

* - Parameter
  - Default
  - Description
  - Default from
* - `confidence` of {func}`~irtlib.irt.confidence_interval`
  - `0.95`
  - The interval's confidence level, strictly between 0 and 1.
  - catsim's, from which the function moved ({doc}`ADR-0004 </adr/0004-catsim-builds-on-irtlib>`)
* - `scale_min`, `scale_max` of {func}`~irtlib.irt.theta_to_scale` and {func}`~irtlib.irt.scale_to_theta`
  - `0`, `100`
  - The reporting scale's scores at the two abilities below.
  - catsim's ({doc}`ADR-0004 </adr/0004-catsim-builds-on-irtlib>`)
* - `theta_min`, `theta_max` of the same functions
  - `-4`, `4`
  - The abilities that map to `scale_min` and `scale_max`.
  - catsim's ({doc}`ADR-0004 </adr/0004-catsim-builds-on-irtlib>`)
```

## Guarantees

1. Probabilities, information and log-likelihoods are finite at any finite ability, however far
   from an item's difficulty.
2. Items without information, or no items at all, give a test information of 0.
3. A guessing parameter of 0 or an upper asymptote of 1 has derivative 0 in calibration's
   parameters.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.irt
   :members:
```
