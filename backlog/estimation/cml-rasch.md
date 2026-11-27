---
title: Estimate Rasch difficulties by conditional maximum likelihood
kind: technique
status: blocked
priority: 3
effort: S
blocked_by: [discovery/requirements]
raised_by: [moved from catsim's backlog]
---

# Estimate Rasch difficulties by conditional maximum likelihood

## Problem

For the Rasch model, the number of correct responses is a sufficient statistic for ability, so
conditioning on it removes the abilities from the likelihood and leaves the difficulties alone,
and the conditional likelihood "does not depend on the distribution function" of ability (Bock &
Aitkin, 1981, p. 443; Haberman, 2004, p. 35).
Conditional maximum likelihood (CML) is the reference estimator for the Rasch model in the
literature pyitemcal's comparison studies draw on, and the 1PL MML estimates are checked against it
(Bock & Aitkin, 1981, p. 455, Table 3).

## Evidence

The research pass (2026-11-27) checked the card carried over from catsim against Andersen (1970),
Gustafsson (1980), Baker and Harwell (1996), Engelen (1987) and Robitzsch (2021):

- its gradient, $-n_i + \sum_r N_r \gamma_r^{(-i)}/\gamma_r$, lacked the easiness factor and used the
  wrong order of the symmetric function: the likelihood equations are
  $s_i = \sum_r n_r \varepsilon_i \gamma_{r-1}^{(i)}/\gamma_r$ (Gustafsson, 1980, p. 380, Equation 7);
- "consistent and efficient": consistent (Andersen, 1970, p. 286, Theorem 1), but "the c.m.l.e. will
  not in general be efficient" (p. 284);
- "cost $O(I^2)$ per score group": the symmetric functions of all orders together cost
  $k(k - 1)/2$ multiplications by the summation algorithm (Baker & Harwell, 1996, p. 189).

## What pyitemcal and catsim have now

Nothing for conditional estimation in either package.

## Proposal

CML for the Rasch model, solving Gustafsson's (1980) likelihood equations by Newton-Raphson, with
the symmetric functions computed by the summation algorithm, and the difficulties identified by a
sum-to-zero constraint. It follows MML-EM in the build order of
[the literature review](../discovery/literature-review.md).

## Tests

- **Published symmetric functions:** for the LSAT-6 CML estimates, with easiness values 3.5118,
  .6219, .2905, .8450 and 1.8648, the symmetric functions are $\gamma_1, \dots, \gamma_5 = 7.1340$,
  16.9493, 16.7781, 7.0529 and .9997, with every first derivative in the same table (Baker & Harwell,
  1996, pp. 171, 173, Table 3). The table's marginal row $\sum_i \varepsilon_i\gamma_{r-1}^{(i)}$ is
  misprinted for $r = 4$ and 5: by their identity $r\gamma_r = \sum_i \varepsilon_i\gamma_{r-1}^{(i)}$
  it is 28.2116 and 4.9985, not 33.5562 and 7.0529 (the card's arithmetic), so tests use the
  $\gamma_r$ and the first derivatives only.
- **Published estimates:** the LSAT-6 and LSAT-7 CML difficulties of Bock and Aitkin (1981, p. 455,
  Table 3), centred to sum to zero.
- **Closed form:** with two items, $b_2 - b_1 = \log n_{10} - \log n_{01}$, where $n_{10}$ counts
  examinees with item 1 right and item 2 wrong (Andersen, 1970, p. 299).
- **Brute force:** for 4 items, the symmetric functions equal the sums over all subsets of the items.

## Open Questions

1. **Missing responses.** "For missing at random data, LIM (and ordinary CML) will likely fail (see
   [134]), and MML or JML will usually be preferred", while the methods "can also be applied in
   situations in which data is missing completely at random (MCAR)" (Robitzsch, 2021, p. 827);
   ADR-0001 commits pyitemcal to response matrices with missing entries. Options: CML for complete
   data only, with an error otherwise; or a CML over the items each examinee answered, for data
   missing completely at random, which needs a source for its equations. Assumed: complete data
   only.
2. **Identification.** One item's easiness set to 1 (Gustafsson, 1980, p. 380) or difficulties summing
   to zero, as in Bock and Aitkin's (1981, p. 454) comparison. Assumed: sum to zero.

---

**Module:** decided by the requirements card (planned)
**Reference:** Andersen (1970); Gustafsson (1980)
**Last validated:** 2026-11-27, against Andersen (1970), Gustafsson (1980), Baker & Harwell (1996),
Engelen (1987), Bock & Aitkin (1981), Haberman (2004) and Robitzsch (2021) on the paper shelf.
**Implementation guidance:** not yet written; it waits for the requirements card.

## Motivation

The Rasch model's difficulties can be estimated without assuming anything about the abilities, by
conditioning on each examinee's number correct. The conditional estimates are consistent as the
number of examinees grows (Andersen, 1970, p. 286), where joint maximum likelihood is not
([the JML card](jml.md)). For the Rasch model, MML and CML give nearly the same estimates when the
model fits (Bock & Aitkin, 1981, p. 454).

## Definition

Write $\varepsilon_i = e^{-b_i}$ for item $i$'s easiness, $s_i$ for its number of correct responses
and $n_r$ for the number of examinees with $r$ correct, out of $k$ items (Gustafsson's $k$; the
charter's $M$). The conditional likelihood is $\prod_i \varepsilon_i^{s_i} / \prod_r \gamma_r^{n_r}$,
where the elementary symmetric function $\gamma_r$ is the sum over all sets of $r$ items of the
product of their easiness values (Gustafsson, 1980, pp. 379-380, Equations 5-6). The likelihood
equations are

$$
s_i = \sum_{r=1}^{k-1} n_r \frac{\varepsilon_i \gamma_{r-1}^{(i)}}{\gamma_r}, \qquad i = 1, \dots, k,
$$

with $\gamma_{r-1}^{(i)}$ the symmetric function of order $r - 1$ of every item but $i$, which is
also the derivative of $\gamma_r$ with respect to $\varepsilon_i$ (Gustafsson, 1980, p. 380,
Equation 7). For examinees with no item or every item correct, "both sides are equal to one" in the
conditional likelihood (Engelen, 1987, p. 11), so they are left out of the sum. Items answered
correctly by every examinee or by none have no estimate and are left out too (Gustafsson, 1980,
p. 379).

**The symmetric functions.** They follow from $\gamma_r = \varepsilon_i \gamma_{r-1}^{(i)} +
\gamma_r^{(i)}$ (Gustafsson, 1980, p. 380, Equation 8), by adding one item at a time (the summation
algorithm, as Gustafsson names it after Fischer's recursion) or by removing one (the difference
algorithm) (pp. 380-382). The difference algorithm is faster but
"breaks down when there are more than 20-50 items" (p. 381); the summation algorithm is accurate and
slower by a factor of about $k/4$ when it also gives the first derivatives (p. 382). Its cost for all
orders is $k(k - 1)/2$ multiplications (Baker & Harwell, 1996, p. 189).

## Parameters

None beyond the identification constraint (Open Question 2).

## Behavior and Comparison

On LSAT-6, where the Rasch model fits, CML and MML agree closely; on LSAT-7, where it fits poorly,
they differ more (Bock & Aitkin, 1981, pp. 454-455, Table 3). In Robitzsch's (2021, p. 823, Table 2)
simulations, CML had acceptable bias in every condition and ranked 8th of 32 methods on relative
RMSE; JML with an $\varepsilon$ adjustment of 0.24 and several MML variants with flexible ability
distributions ranked above it.

## Guarantees

1. The estimates do not depend on the distribution of ability.
2. Examinees with no item or every item correct do not change the estimates.

## References

Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators. *Journal
of the Royal Statistical Society: Series B, 32*(2), 283-301.
https://doi.org/10.1111/j.2517-6161.1970.tb00842.x. Not openly available; Haberman (2004, §2)
describes conditional estimation.

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Baker, F. B., & Harwell, M. R. (1996). Computing elementary symmetric functions and their
derivatives: A didactic. *Applied Psychological Measurement, 20*(2), 169-192.
https://doi.org/10.1177/014662169602000206. Open copy: https://hdl.handle.net/11299/119090

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Engelen, R. J. H. (1987). *A review of different estimation procedures in the Rasch model* (Research
Report 87-6). University of Twente. Open copy: https://files.eric.ed.gov/fulltext/ED310126.pdf

Gustafsson, J.-E. (1980). A solution of the conditional estimation problem for long tests in the
Rasch model for dichotomous items. *Educational and Psychological Measurement, 40*(2), 377-385.
https://doi.org/10.1177/001316448004000214. Not openly available; Baker and Harwell (1996) describe
the method.

Haberman, S. J. (2004). *Joint and conditional maximum likelihood estimation for the Rasch model for
binary responses* (Research Report RR-04-20). Educational Testing Service.
https://doi.org/10.1002/j.2333-8504.2004.tb01947.x. Open copy:
https://files.eric.ed.gov/fulltext/EJ1110899.pdf

Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
*Stats, 4*(4), 814-836. https://doi.org/10.3390/stats4040048. Open access.

## Implementation Guidance

Waits for the requirements card (`discovery/requirements.md`).
