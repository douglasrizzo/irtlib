# Joint Maximum Likelihood

:API: {py:class}`irtlib.jml.JMLCalibrator`
:Sources: {cite:t}`Wright1977`; {cite:t}`Baker1987`; {cite:t}`Haberman2004`; {cite:t}`Mislevy1989`

## Motivation

Joint maximum likelihood (JML) treats every examinee's ability as a parameter, and maximizes the
likelihood of the responses over the abilities and the item parameters together, with no assumption
about the distribution of ability {cite:p}`Baker1987{p. 112}`. As each examinee adds a parameter, the
item estimates do not converge to the item parameters as examinees are added to a test of fixed
length {cite:p}`Bock1981{p. 443}`; {cite:p}`Baker1987{pp. 117-118}`: for the Rasch model they converge
to limits that differ from the parameters by a term of order $1/M$
{cite:p}`Haberman2004{p. 13}`, and Haberman (p. 35) finds it "difficult to advocate use of JMLE
except for preliminary estimation of parameters".

irtlib has it for comparing calibration methods and for preliminary estimates, for the Rasch model
and the 2PL. It returns the examinees' abilities with the items. {doc}`cml` gives consistent Rasch
difficulties, and {doc}`mml` calibrates every model.

## Definition

With $N$ examinees and $M$ items, the logarithm of the joint likelihood is

$$
\sum_{j=1}^N \sum_{i=1}^M \left[u_{ij} \log P_i(\theta_j) + (1 - u_{ij}) \log Q_i(\theta_j)\right]
$$

{cite:p}`Baker1987{p. 112, Equations 1 and 3}`, for the 2PL (p. 114, Equation 5) or for the Rasch
model, whose slopes are 1 (p. 115, Equation 9; {cite:t}`Wright1977{p. 282, Equations 1-3}`).

Each cycle has two steps, after Birnbaum's paradigm {cite:p}`Baker1987{p. 112}`:

- the item step maximizes, for each item, its grouped log-likelihood with the abilities fixed;
- the ability step estimates each examinee's ability with the items fixed, by maximum likelihood.

The likelihood is the same for abilities and items moved together along a linear transformation of
the ability scale {cite:p}`Baker1987{p. 112}`, so the scale has to be fixed.

## Implementation

**Examinees and items left out.** An examinee with no item or every item correct has no finite
ability estimate, and an item answered correctly by every examinee or by none no finite estimate
either {cite:p}`Wright1977{p. 294}`; {cite:p}`Haberman2004{p. 7}`. Both are left out before
estimation, the examinees first and then the items, repeatedly until neither is left, as Wright and
Douglas's editing does (p. 294), and the items left out get no estimate. For the Rasch model, the
joint likelihood of the responses left can still lack a maximum: it has none exactly when the
examinees with at most some number correct answered none of a group of items and the others all of
the rest {cite:p}`Haberman2004{p. 11, Theorem 1}`, the same split that leaves the conditional
likelihood of {doc}`cml` none once those examinees and items are left out, and calibration raises an
error naming the groups.

**The item step** maximizes each item's grouped log-likelihood by the Fisher scoring of
{doc}`mml`'s M-step, with the estimated abilities in place of the quadrature nodes and the observed
counts in place of the expected ones. For the Rasch model and the 2PL the Hessian is the
information, so the steps are Newton-Raphson's {cite:p}`Baker1987{p. 116}`, those of Wright and
Douglas (p. 284, Equation 11) for the difficulties and of Baker (p. 114, Equation 8) for the 2PL's
intercept and slope.

**The ability step** uses {func}`irtlib.ability.maximum_likelihood`. Under the Rasch model every
examinee with the same number correct has the same estimate {cite:p}`Wright1977{p. 283}`;
{cite:p}`Baker1987{p. 115}`, and under either model every examinee with the same responses, so each is
estimated once.

**The scale.** After each ability step, abilities and items are moved to fix the scale
({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`): for the Rasch model so that the
difficulties have mean 0, as BICAL fixes it {cite:p}`Baker1987{p. 115}` and as Wright and Douglas
recentre them {cite:p}`Wright1977{p. 283}`; {cite:p}`Wright1977{p. 284, step iv}`; for the 2PL so that
the abilities of the examinees kept have mean 0 and standard deviation 1, its divisor the number of
examinees, as LOGIST fixes it {cite:p}`Baker1987{p. 112}`. LOGIST standardizes only the abilities
between -3 and 3 {cite:p}`Mislevy1989{p. 60}`, where irtlib standardizes every examinee kept.

**Starting values and stopping.** The cycles start from Wright and Douglas's starting values
{cite:p}`Wright1977{p. 283, Equation 9}`; {cite:p}`Wright1977{p. 284, Equation 10}`, the abilities
$\log(r / (M - r))$ for $r$ correct and the difficulties $\log((N - s_i) / s_i)$ centred, for $s_i$
correct responses to item $i$; for the 2PL, which they do not cover, the slopes start at 1 and the
abilities are standardized. A calibration stops when no item parameter changes by more than its
tolerance in a cycle, the rule and the defaults of {doc}`mml`
({doc}`ADR-0007 </adr/0007-mml-em-settings>`). Wright and Douglas stop when the mean squared change of
the difficulties falls below $10^{-4}$ {cite:p}`Wright1977{p. 285, Equation 15}`, which stopped 0.0007
and 0.0009 short of the maximum on 20 items and on LSAT-6, where irtlib's rule stops within
$2 \times 10^{-7}$ of it (measured 2026-11-27). The ability step compares likelihoods, which can
leave changes of about $10^{-8}$ from cycle to cycle, as on a 2PL of 20 items and 400 examinees, so a
tolerance below that may not be met (measured 2026-11-28).

**The Rasch model's bias and its correction.** Wright and Douglas write each difficulty estimate as
the conditional one plus a bias, $d_i^* = d_i + a_{ri}$ at score $r$, approximating the bias by the
change in the conditional ability for score $r$ when item $i$ is left out, and its relative bias as
$c_{ri} = a_{ri} / d_i$ {cite:p}`Wright1977{p. 286, Equations 25-27}`. Averaged over scores and items
with $|d_i| \geq 0.5$, they find it close to $1 / (M - 1)$ whatever the spread of the difficulties
(p. 287, Equations 31-34, Table 1), so the estimates spread outward from the conditional ones, and
multiplying the centred difficulties by the correction factor $(M - 1) / M$ (p. 287, Equation 35)
leaves a residual bias of at most 0.012 for 20 items (p. 288, Table 2). The factor is the one "used in
most versions of UCON", and with two items, for which the joint estimates are exactly twice the
conditional ones, it is exact {cite:p}`Wright1977{p. 288}`; {cite:p}`Haberman2004{pp. 17-18}`. It
assumes item difficulties and examinees spread "more or less uniform[ly]" over the scores
{cite:p}`Wright1977{p. 288}`, and it brings the estimates close to the conditional ones, which "is not
the same as" removing their bias from the parameters {cite:p}`Baker1987{p. 118}`. Computed exactly
for 20 items equally spaced, with as many examinees at each score, the relative bias is 0.055 for
difficulties within $\pm 1$, as Wright and Douglas find, but 0.060 within $\pm 2$ and 0.071 within
$\pm 3$, and the largest residual after the correction 0.004, 0.029 and 0.090 (measured 2026-11-27):
the correction leaves more of the bias of the items far from the centre of a wide test. It is applied
once the cycles stop, by default for the Rasch model; Wright and Douglas derive it for the Rasch model
alone.

**Standard errors.** Each item's standard errors are the square roots of the diagonal of the inverse
of its item step's information at the last cycle {cite:p}`Baker1987{p. 119}`, for the Rasch model
$(\sum_r n_r p_{ri} (1 - p_{ri}))^{-1/2}$, with $n_r$ examinees of score $r$ and $p_{ri}$ their
probability of a correct response to item $i$ {cite:p}`Wright1977{p. 285, Equation 16}`; the 2PL's
difficulty takes its standard error by the delta method, as in {doc}`mml`. They treat the abilities
as known, and with the correction they are those of the estimates before it, unless `correct_errors`
multiplies them by its factor too. For a fixed number of items, JML's estimates are not centred on
the parameters and their asymptotic covariance is not the customary one, so "ordinary asymptotic
confidence intervals" for the difficulties cannot be built from them
{cite:p}`Haberman2004{pp. 26-27, Theorem 7}`; {cite:p}`Haberman2004{p. 29}`.

**The 2PL's slopes.** Under the 2PL a slope can grow without bound, as large slopes and the
abilities of the examinees who answer those items correctly feed each other from cycle to cycle, the
Heywood case; LOGIST imposes upper limits on the slopes {cite:p}`Baker1987{p. 113}`, and "the
maximizing values for item discriminations under the 2PL ... are sometimes infinite"
{cite:p}`Mislevy1989{p. 62}`. irtlib imposes no limit. An item whose correct responses the abilities
all put at or above its incorrect ones, or at or below, discriminates perfectly, and its slope has no
finite estimate {cite:p}`Baker1987{p. 117}`, so the cycles stop before such an item step; they stop
too at an item step that gives a slope that is not positive, since the ability step needs positive
slopes. Either way the result holds the last complete cycle and says so. On short tests this is the
usual outcome. On LSAT-6 and LSAT-7, five items and 1,000 examinees, the cycles stopped after 3 and 2
at an item step that gave one slope a value that was not positive and another 3.53 and 7.21. On
random 2PL tests, slopes uniform on $[0.7, 2]$, difficulties on $[-1.5, 1.5]$ and abilities normal,
the cycles stopped on all 50 tests of 8 items and 300 examinees (38 at a separation, 12 at a slope
that was not positive), on 19 of 20 of 10 items and 500 examinees, 13 of 20 of 15 and 500, and 3 of
20 of 20 and 500, and on none of 10 tests each of 20 and 30 items with 1,000 examinees (measured
2026-11-28). The authors of LOGIST advise at least 20 items per examinee and 800 to 1,000 examinees
per item {cite:p}`Mislevy1989{p. 61}`.

**The abilities.** The result holds every examinee's ability
({doc}`ADR-0006 </adr/0006-calibrator-contract>`), from the cycle its item estimates come from and
before any correction, so that the abilities, the uncorrected item estimates and the log-likelihood go
together; each ability's standard error is that of maximum likelihood, the inverse square root of the
test information, there. An examinee left out has the ability $\infty$ or $-\infty$, as maximum
likelihood gives responses all right or all wrong.

**Complete responses.** JML needs complete responses, as its sources write it, an examinee's number
correct out of every item {cite:p}`Wright1977{p. 282}`, and {class}`~irtlib.mml.MMLCalibrator`
calibrates incomplete ones ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

**Against conditional maximum likelihood.** On two SAT tests of 78 and 60 items, the largest
difference between the joint and conditional estimates of the Rasch difficulties was 0.097 and 0.126,
against conditional standard errors of at most 0.008 {cite:p}`Haberman2004{pp. 44-45}`. In
{cite:t}`Robitzsch2021`'s simulations, JML with the correction and extreme scores removed (p. 816)
had a mean absolute bias below 0.025 (p. 821) in 54.2% of the conditions, against 100% for
conditional maximum likelihood (p. 823, Table 2).

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 14 42 26

* - Parameter
  - Default
  - Description
  - Default from
* - `model`
  - `"2PL"`
  - `"2PL"`, with the abilities given mean 0 and standard deviation 1, or `"1PL"`, the Rasch model,
    every slope 1, with the difficulties given mean 0.
  - {class}`~irtlib.mml.MMLCalibrator`'s default
* - `correct`
  - None
  - Whether to multiply the Rasch difficulties by $(M - 1) / M$ once the likelihood is maximized;
    None corrects the Rasch model and not the 2PL.
  - {cite:t}`Wright1977{p. 287, Equation 35}`, for the Rasch model alone
* - `correct_errors`
  - False
  - Whether to multiply the difficulties' standard errors by the same factor.
  - the standard errors of the estimates before the correction
* - `tolerance`
  - `1e-6`
  - The stopping rule: the largest change of an item parameter in a cycle.
  - {doc}`mml`'s ({doc}`ADR-0007 </adr/0007-mml-em-settings>`)
* - `max_cycles`
  - `500`
  - The most cycles before the calibration stops unconverged.
  - {doc}`mml`'s; no source for JML yet
    ([sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md))
```

## Guarantees

1. The Rasch difficulties have mean 0 before the correction; the 2PL's abilities of the examinees
   kept have mean 0 and standard deviation 1.
2. Examinees with no item or every item correct, and items answered correctly by every examinee or
   by none, are left out, repeatedly until neither is left; the items left out have NaN estimates,
   and the examinees left out the ability $\infty$ or $-\infty$.
3. When a 2PL slope has no finite estimate, or an item step gives a slope that is not positive, the
   result holds the last complete cycle and warns with a
   {class}`~irtlib.calibration.ConvergenceWarning`.
4. A missing response raises an error that names {class}`~irtlib.mml.MMLCalibrator` as the
   alternative.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.jml
   :members:
```
