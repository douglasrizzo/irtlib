# Marginal Maximum Likelihood

:API: {py:class}`irtlib.mml.MMLCalibrator`; {py:class}`irtlib.mml.Priors`;
  {py:func}`irtlib.mml.marginal_log_likelihood`
:Sources: {cite:t}`Bock1981`; {cite:t}`Mislevy1986`; {cite:t}`Du2020`; {cite:t}`Oakes1999`

## Motivation

Calibration has to estimate item parameters without the examinees' abilities. Marginal maximum
likelihood (MML) integrates the abilities out over a population distribution and maximizes the
likelihood of the item parameters alone; Bock and Aitkin's EM algorithm makes it feasible for any
number of items, with one small problem per item in each cycle {cite:p}`Bock1981{pp. 444, 447}`.

It is irtlib's method for every model from the 1PL to the 4PL, and its only calibration method that
takes a response matrix with missing responses
({doc}`ADR-0006 </adr/0006-calibrator-contract>`). Its estimates assume a normal population of
abilities, where {doc}`cml` assumes nothing about it, and its likelihood is the one that
{doc}`/methods/fit/index` and {doc}`/methods/comparison` take. For the 3PL and 4PL, priors on the
item parameters keep the estimates stable.

## Definition

Ability is normal with mean 0 and variance 1, approximated on nodes $X_q$ with masses $A_q$ that sum
to 1 ({doc}`/methods/models/quadrature`). Examinee $j$'s marginal probability is
$\tilde P_j = \sum_q L_j(X_q) A_q$, where $L_j$ is the product of
$P_i^{u_{ij}} (1 - P_i)^{1 - u_{ij}}$ over the items $i$ examinee $j$ answered, and the marginal
log-likelihood is $\sum_j \log \tilde P_j$ {cite:p}`Bock1981{pp. 445-446, Equations 6 and 8}`.

Each cycle has two steps {cite:p}`Baker1987{p. 132, Equations 42-46}`;
{cite:p}`Bock1981{p. 447}`:

- the E-step computes, for each item and node, the expected number of examinees answering the item
  there, $\bar N_{iq} = \sum_j L_j(X_q) A_q / \tilde P_j$ over the examinees who answered it, and the
  expected number answering it correctly, $\bar r_{iq}$, the same sum weighted by $u_{ij}$;
- the M-step maximizes
  $\sum_q [\bar r_{iq} \log P_i(X_q) + (\bar N_{iq} - \bar r_{iq}) \log Q_i(X_q)]$ in the item
  parameters, the grouped log-likelihood of {doc}`/methods/models/item-response-functions`.

The prior masses enter only through the expected counts. No cycle decreases the marginal likelihood
{cite:p}`Dempster1977{p. 7, Theorem 1}`, and convergence is linear, slow near the solution
{cite:p}`Dempster1977{p. 10}`; {cite:p}`Bock1981{p. 447}`.

With priors on the item parameters, the estimates are Bayes modal: they maximize the log-likelihood
plus the log-prior, and the EM algorithm carries over with the derivatives of the log-prior added to
the M-step {cite:p}`Mislevy1986{p. 182, Equation 14}`.

## Implementation

**Missing responses.** The sums run over the examinees who answered each item, so the expected
sample size $\bar N_{iq}$ is per item; the sources define it over complete response patterns, shared
by every item.

**The grid.** 61 equally spaced nodes on $[-6, 6]$, with the normal density at the nodes, normalized
{cite:p}`Bock1981{p. 449}` ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).

**The M-step.** It solves in the slope, the intercept and the logits of $c$ and $d$, by
Fisher scoring with step halving, on all free parameters at once: the system is one block per item
unless a parameter is shared ({doc}`ADR-0007 </adr/0007-mml-em-settings>`). The 1PL shares one
slope, estimated, with abilities $N(0, 1)$ {cite:p}`Thissen1982{pp. 175-176, Equation 1}`; fixing the
slope at 1 for every item gives the model with slopes 1 and abilities $N(0, 1)$
({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`).

**Stopping.** A calibration stops when no estimated parameter changes by more than its tolerance in
a cycle, an E-step and an M-step ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).

**Priors.** The priors are on these parameters
({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`): lognormal on the slope, whose
log-density keeps the $-\log a$ term, normal on the intercept, and normal on the logit of $c$, which
keeps $c$ inside $(0, 1)$ {cite:p}`Mislevy1986{p. 183}`. In the M-step's scoring information, a
log-prior's negative second derivative counts only where it is positive, so the system stays
positive definite. The 3PL's likelihood is "nearly flat along one or more dimensions"
{cite:p}`Mislevy1986{p. 189}`, which a prior on the logit of $c$ remedies. The 4PL adds an upper
asymptote $d < 1$, for examinees of high ability who still miss an item
{cite:p}`Barton1981{p. 2}`, estimated on its logit with a normal prior there; recovering it takes
thousands of examinees {cite:p}`Waller2017{p. 19}`; {cite:p}`Culpepper2016{Section 5}`.

**Items with no finite estimate.** An item answered correctly by every examinee who answered it, or
by none, has no finite estimate, and calibration raises an error listing such items unless a prior on
the intercept keeps their estimates finite
({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`).

**Acceleration.** An EM cycle is a fixed-point iteration $x_{n+1} = F(x_n)$ of the vector $x$ of
free parameters; the largest eigenvalue of $F$'s Jacobian at the solution, the fraction of missing
information, governs how slowly it converges {cite:p}`Du2020{p. 3}`. By default a calibration
accelerates it with SQUAREM {cite:p}`Du2020{p. 4, Algorithm 1}`. From $x_0$, two cycles give
$x_1 = F(x_0)$ and $x_2 = F(x_1)$; with $r = x_1 - x_0$, $v = x_2 - x_1 - r$ and the steplength
$\alpha = -\lVert r \rVert / \lVert v \rVert$, the authors' default (p. 4), the extrapolation is
$x_0 - 2 \alpha r + \alpha^2 v$. Where its log-posterior is above that of $x_2$, Algorithm 1's test
with $\eta = 0$, one cycle from it, the stabilization step, gives the next $x_0$; elsewhere $x_2$
does. No cycle then lowers the log-posterior, as no cycle of plain EM does
{cite:p}`Dempster1977{p. 6}`. The norms are Euclidean over the free parameters, a shared slope
counted once, and an extrapolation with a parameter that is not finite, or a guessing parameter at or
above its upper asymptote, is not kept. The extrapolation is not a cycle: the stopping rule applies to
every cycle, an EM step with or without acceleration.

**Standard errors.** They are the square roots of the inverse observed information at the
estimates, by Oakes's identity {cite:p}`Oakes1999{p. 480}`: the score of the marginal log-likelihood
is the M-step's gradient at the E-step's own parameters (Equation 5), and its derivative as both move
together is the observed-data Hessian, the sum of the complete-data Hessian and the missing
information (Equation 6). That score, log-prior included, is differenced centrally with step
$10^{-5}$, the E-step recomputed at each shifted point, so the posterior precision takes in the
log-prior's second derivative {cite:p}`Mislevy1986{p. 182, Equations 15-17}`; the result is
symmetrized, which {cite:t}`Chalmers2018{p. 17}` finds improves it
({doc}`ADR-0007 </adr/0007-mml-em-settings>`). The standard errors of $a$, $b = -\zeta / \lambda$,
$c$ and $d$ follow by the delta method.

**Starting values.** The slope 0.851, the intercept $\Phi^{-1}(p_i) \cdot 1.702 / \sqrt{0.75}$, or 2
where $p_i$ is 0 or 1, the guessing parameter 0.15 and the upper asymptote 0.85, where $p_i$ is item
$i$'s proportion correct among the examinees who answered it
({doc}`ADR-0007 </adr/0007-mml-em-settings>`); {cite:t}`Bock1981{p. 458}` name the inverse normal of
the proportion correct as a starting intercept.

:::{dropdown} Comparison with mirt

- **What matches.** `mirt(data, 1, itemtype = ...)` in mirt 1.48, with its default grid, its
  starting values and `SE.type = "Oakes"` {cite:p}`Chalmers2026{p. 122}`; irtlib computes the
  starting values over the answered responses only ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).
- **Without priors** the parity tests find the estimates within $10^{-5}$ of mirt's recorded at
  `TOL = 1e-8`, the log-likelihoods within a relative $10^{-10}$, and the standard errors within a
  relative $10^{-4}$, for the 1PL, the 2PL and fixed slopes, on the LSAT 6 and LSAT 7 data and on
  simulated data ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
- **With priors** the estimates and standard errors agree as closely on the LSAT 7 3PL. On the
  simulated 2PL, 3PL and 4PL, mirt reports convergence where the gradient of the marginal
  log-posterior is still $6.6 \times 10^{-5}$ to $7.4 \times 10^{-4}$ in size; irtlib's estimates bring
  it below $10^{-6}$, at a higher log-posterior, and differ from mirt's by up to
  $1.7 \times 10^{-4}$, and by $1.6 \times 10^{-2}$ on the 4PL, whose likelihood is nearly flat along
  a ridge (measured 2026-11-27). With a lognormal prior on the slope, mirt's Hessian of the prior has
  the opposite sign on its last term (`R/utils.R:1430`), which its standard errors take in; irtlib
  differentiates the lognormal log-density itself. The 4PL's asymptotes' standard errors agree within
  a relative $2 \times 10^{-3}$. The parity tests keep each departure as an expected failure.
- **The parameters and the priors.** The M-step solves in mirt's parameters
  {cite:p}`Chalmers2026{pp. 132-133}`, and the priors are mirt's densities on them
  ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`).
- **The tolerance.** mirt's default `TOL` is $10^{-4}$ for EM {cite:p}`Chalmers2026{p. 127}`;
  irtlib's is $10^{-6}$ ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).
- **Acceleration.** mirt accelerates by Ramsay's rule by default and offers SQUAREM
  {cite:p}`Chalmers2026{pp. 128-129}`; the maximum is the same
  ({doc}`ADR-0007 </adr/0007-mml-em-settings>`).
- **Standard errors.** Chalmers (2018, pp. 5-7), as mirt, differences only the missing information
  and takes the complete-data Hessian analytically; differencing the whole score keeps one path for
  every model and prior.
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 16 40 26

* - Parameter
  - Default
  - Description
  - Default from
* - `model`
  - `"2PL"`
  - `"1PL"` (one estimated slope shared by every item), `"2PL"`, `"3PL"` or `"4PL"`.
  - mirt's default item type {cite:p}`Chalmers2026{p. 123}`
* - `fixed_slope`
  - None
  - A slope every item keeps instead of estimating it; 1 with the 1PL gives slopes 1 and abilities
    $N(0, 1)$.
  - no fixed slope ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
* - `priors`
  - None
  - A {class}`~irtlib.mml.Priors` on the slope, the intercept and the logits of $c$ and $d$, each a
    mean and a standard deviation. Recommended for the 3PL and the 4PL.
  - no default prior ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
* - `tolerance`
  - `1e-6`
  - The stopping rule: the largest change of an estimated parameter in a cycle.
  - {doc}`ADR-0007 </adr/0007-mml-em-settings>`
* - `max_cycles`
  - `500`
  - The most cycles before the calibration stops unconverged.
  - mirt's `NCYCLES` {cite:p}`Chalmers2026{p. 129}`
* - `acceleration`
  - `"squarem"`
  - `"squarem"` {cite:p}`Du2020{p. 4, Algorithm 1}` or `"none"`.
  - {doc}`ADR-0007 </adr/0007-mml-em-settings>`
* - `grid`
  - 61 nodes on $[-6, 6]$
  - The nodes the abilities are integrated over.
  - mirt's `quadpts` and `theta_lim` {cite:p}`Chalmers2026{pp. 127, 129}`
* - `start` of `calibrate`
  - None
  - Items to start from, one per column.
  - mirt's starting values ({doc}`ADR-0007 </adr/0007-mml-em-settings>`)
```

## Guarantees

1. Responses may be missing; an examinee who answered no item changes no estimate.
2. With or without acceleration, a calibration converges to the same maximum.
3. An item answered correctly by every examinee who answered it, or by none, raises an error naming
   it, unless a prior on the intercept is given.
4. An item whose estimates are not valid in an item bank, such as a miskeyed item whose slope is not
   positive, is left out of the result's bank, with a warning naming it.
5. A calibration that does not meet its stopping rule within `max_cycles` says so in its result and
   warns with a {class}`~irtlib.calibration.ConvergenceWarning`.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.mml
   :members:
```
