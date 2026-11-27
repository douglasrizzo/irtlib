---
title: Differentiate the item response function with respect to the item parameters
kind: technique
status: blocked
priority: 1
effort: S
blocked_by: [discovery/requirements]
raised_by: [moved from catsim's backlog]
---

# Differentiate the item response function with respect to the item parameters

## Problem

Every calibration method solves likelihood equations in the item parameters: the M-step of
marginal maximum likelihood ([the MML-EM card](../estimation/mml-em.md)) and the item stage of joint
maximum likelihood ([the JML card](../estimation/jml.md)) each need the derivatives of the
probability of a correct response with respect to $a$, $b$, $c$ and $d$, and the information they
carry. catsim differentiates the item response function only with respect to ability.

## Evidence

- ADR-0004 lists "derivatives with respect to item parameters" among what pyitemcal implements and
  catsim lacks.
- The research pass (2026-11-27) checked the card carried over from catsim against its sources. Its
  first derivatives with respect to $a$, $b$ and $c$ match Barton and Lord (1981, p. A3) once their
  scaling constant is set to 1; its derivative with respect to $d$ had no source; two of its tests
  were wrong: the gradient of the log-likelihood vanishes at the maximum likelihood estimate, not at
  the generating parameters, and the expected information is not the second derivative of the
  log-likelihood of an observed sample.

## What pyitemcal and catsim have now

- catsim: `catsim.irt.icc` (the response function, items as rows `[a, b, c, d]`, broadcasting
  abilities against items), `catsim.irt.log_response_probabilities` and `catsim.irt.information`
  (Fisher information about ability). No derivative with respect to item parameters.
- pyitemcal: nothing.

## Proposal

Functions that return, for every ability against every item, the four derivatives of the response
probability below; the gradient of an item's log-likelihood from counts of correct responses at
given abilities (the M-step's data); and the expected information matrix of an item's parameters.
Their signatures and module follow the requirements card.

## Tests

- **Hand-worked values** from the closed form below:

  | Item $[a, b, c, d]$ | $\theta$ | $P$ | $\partial P/\partial a$ | $\partial P/\partial b$ | $\partial P/\partial c$ | $\partial P/\partial d$ |
  |---|---|---|---|---|---|---|
  | $[1, 0, 0, 1]$ | 0 | 0.5 | 0 | −0.25 | 0.5 | 0.5 |
  | $[1.5, 1, 0.2, 1]$ | 0 | 0.3459404190 | −0.1193171617 | −0.1789757425 | 0.8175744762 | 0.1824255238 |
  | $[0.8, −0.5, 0.25, 0.95]$ | 1.2 | 0.8070317884 | 0.1934061794 | −0.0910146727 | 0.2042403023 | 0.7957596977 |

  The second row's $P$ is also the value in `catsim.irt.icc`'s own example.
- **Finite differences:** each derivative agrees with the central difference of `catsim.irt.icc`
  over a grid of abilities and items spanning the bank bounds, to the step's $O(h^2)$ error.
- **Log-likelihood gradient:** the gradient from counts agrees with central differences of the
  grouped log-likelihood below.
- **Expected information:** equals the average over the two outcomes $u = 1$ (weight $P$) and
  $u = 0$ (weight $1 - P$) of the negative second derivative of $u \log P + (1 - u)\log(1 - P)$,
  computed by finite differences.
- **Identity:** $\partial P/\partial c + \partial P/\partial d = 1$ at every ability, since $P$ is
  linear in $c$ and $d$.

## Open Questions

1. **The parameterization inside the M-step.** The sources solve for the intercept and slope,
   $\zeta + \lambda\theta$ (Bock & Aitkin, 1981, p. 445, Equation 3; Baker, 1987, p. 114, Equation 5),
   and mirt estimates in that form (mirt 1.48 manual, p. 133). In it, the 2PL's second derivatives
   contain no data (Baker, 1987, p. 116); in catsim's $a(\theta - b)$ they do (below). Solving in
   $(\zeta, \lambda)$ and converting to $(a, b)$ gives simpler Newton steps and direct parity with
   mirt's estimates; solving in $(a, b)$ avoids the conversion of standard errors. The card assumes
   the requirements card chooses; both need the derivatives here.
2. **Observed or expected information in the M-step's Newton steps.** Bock and Aitkin (1981, p. 446)
   and Baker (1987, p. 116) use the expected information (Fisher scoring) where the second derivatives
   contain data. Assumed: expected, unless the requirements card finds a reason otherwise.

---

**Module:** decided by the requirements card (planned)
**Reference:** Barton & Lord (1981); Baker (1987)
**Last validated:** 2026-11-27, against Barton & Lord (1981), Baker (1987) and Bock & Aitkin (1981)
on the paper shelf, and the mirt 1.48 reference manual.
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

Calibration fits each item's response function to the responses, one item at a time once the
abilities are known or integrated out. The equations it solves and the standard errors it reports
both come from the derivatives of the response probability with respect to the item's own
parameters. They are closed forms of the logistic function, shared by every method that estimates
items.

## Definition

catsim's response function, with $\psi_i(\theta) = 1 / (1 + e^{-a_i(\theta - b_i)})$, is
$P_i(\theta) = c_i + (d_i - c_i)\,\psi_i(\theta)$ (the formula lives in `catsim.irt.icc`'s
docstring). Its derivatives with respect to the item parameters are

$$
\frac{\partial P}{\partial a} = (d - c)(\theta - b)\,\psi(1 - \psi), \qquad
\frac{\partial P}{\partial b} = -(d - c)\,a\,\psi(1 - \psi),
$$

$$
\frac{\partial P}{\partial c} = 1 - \psi, \qquad
\frac{\partial P}{\partial d} = \psi .
$$

Barton and Lord (1981, p. A3) give the first three for $P = c + (\delta - c)/(1 + e^{-DL})$ with
$L = a(\theta - b)$ and $D = 1.7$ (p. A1), writing $\psi(1 - \psi)$ as
$1/(e^{DL} + 2 + e^{-DL})$. This card departs from them in two ways:

- $D = 1$, catsim's parameterization (ADR-0001); Baker (1987, p. 114) recommends abandoning the
  1.702 multiplier.
- Barton and Lord fix one upper asymptote $\delta$, the same for every item (p. 3), and give no
  derivative with respect to it. The derivative with respect to $d$ above is this card's, by differentiating
  their model equation (p. A1) with respect to $\delta$.

For an item answered by $N_q$ examinees at ability $X_q$, $r_{iq}$ of them correctly, the
log-likelihood is

$$
\ell_i = \sum_q \left[ r_{iq} \log P_i(X_q) + (N_q - r_{iq}) \log\bigl(1 - P_i(X_q)\bigr) \right],
$$

the grouped form of Baker (1987, p. 112, Equation 3), and its derivative with respect to a
parameter $\chi$ of item $i$ is

$$
\frac{\partial \ell_i}{\partial \chi} = \sum_q \frac{r_{iq} - N_q P_i(X_q)}{P_i(X_q)\,\bigl(1 - P_i(X_q)\bigr)} \, \frac{\partial P_i(X_q)}{\partial \chi},
$$

the per-response form of Barton and Lord (1981, p. A2), $\sum (u - P)(\partial P/\partial\chi)/(PQ)$,
summed within each group. Written in the intercept-slope form $\zeta + \lambda\theta$, it reduces
to Baker's (1987, p. 114) Equations 6 and 7 when $c = 0$ and $d = 1$, and to his Equations 14-16
(p. 116) for the 3PL when $d = 1$.

Barton and Lord (1981, p. A2) give the second derivative

$$
\frac{\partial^2 \ell}{\partial\chi\,\partial\phi} = \sum \left[ -\frac{\partial P}{\partial\phi}\frac{\partial P}{\partial\chi}\frac{1}{PQ}
+ (u - P)\left( \frac{\partial^2 P/\partial\chi\,\partial\phi}{PQ} + \frac{\partial P}{\partial\chi}\frac{\partial P}{\partial\phi}\frac{P - Q}{P^2Q^2} \right) \right].
$$

Since $E[u] = P$, the terms in $u - P$ have expectation zero, and the expected information of one
response is $(\partial P/\partial\chi)(\partial P/\partial\phi)/(PQ)$. This is the card's own step
from their formula. The observed second derivative equals it only where the $(u - P)$ terms vanish
identically: Baker (1987, p. 116) states that this holds for the one- and two-parameter models, in
the intercept-slope form he uses, and not for the three-parameter model. In catsim's $a(\theta - b)$
form it does not hold even for the 2PL, because $\partial^2 P/\partial a\,\partial b \neq 0$ (Barton
& Lord, 1981, p. A3).

## Parameters

None: the functions take abilities and items.

## Behavior and Comparison

The derivatives are exact closed forms. Where $P$ is close to 0 or 1 the factor $1/(PQ)$ grows, and
for $c = 0$ or $d = 1$ the products $(\partial P/\partial\chi)/(PQ)$ simplify by cancellation; a
numerically careful implementation forms them from $\psi$ rather than dividing.

mirt estimates in the intercept-slope form $a_1\theta + d$ (mirt 1.48 manual, p. 133). Agreement
with mirt is checked at the level of the calibrator ([the MML-EM card](../estimation/mml-em.md)).

## Guarantees

1. Each derivative equals the limit of the difference quotient of `catsim.irt.icc` in that parameter.
2. The derivatives with respect to $c$ and $d$ sum to 1 at every ability.
3. The expected information matrix of an item's parameters is symmetric and positive semidefinite.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Barton, M. A., & Lord, F. M. (1981). *An upper asymptote for the three-parameter logistic
item-response model* (Research Report RR-81-20). Educational Testing Service.
https://doi.org/10.1002/j.2333-8504.1981.tb01255.x. Open copy:
https://files.eric.ed.gov/fulltext/ED207996.pdf

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual]. https://cran.r-project.org/package=mirt

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`), which sets the calibrator contract
these functions serve and the parameterization of Open Question 1.
