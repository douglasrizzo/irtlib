---
title: Estimate items by joint maximum likelihood, for comparison studies
kind: technique
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [moved from catsim's backlog]
---

# Estimate items by joint maximum likelihood, for comparison studies

## Problem

Joint maximum likelihood (JML) estimates every examinee's ability alongside the item parameters. It
is the method of LOGIST and of the Rasch programs BICAL and UCON (Baker, 1987, pp. 112, 115, 118), so
researchers comparing calibration methods, the first audience ADR-0001 names, need it as a
reference point. Its estimates are not consistent when the test length is fixed, so pyitemcal offers
it for comparison and preliminary estimates, not as a default.

## Evidence

The research pass (2026-11-27) checked the card carried over from catsim:

- "difficulty variance is underestimated (difficulties shrunk toward the mean)" is the opposite of
  what the sources find for the Rasch model: JML spreads the difficulties outward, the estimate being
  the CML value times $1 + \kappa$ with a relative bias $\kappa > 0$ (Wright & Douglas, 1977,
  pp. 286-287, Equations 26-27, Table 1), and with two items it is exactly twice the conditional
  estimate (p. 288);
- "the Wright-Douglas correction divides each $b_i$ by $(I-1)/I$" has it backwards: for $M$ items,
  the correction multiplies by $(M - 1)/M$ (p. 287, Equation 35, where $M$ is their $L$);
- "fence $\theta$ at $\pm K$ (e.g. $K = 4$)" has no source; LOGIST gives examinees whose ability
  estimate diverges "default boundary values", unnumbered (Mislevy & Stocking, 1989, p. 65);
- "discrimination $a$ is overestimated" rests only on secondary reports (Baker, 1987, p. 118) and is
  dropped.

## What pyitemcal and catsim have now

- catsim: ability estimators for one examinee at a time.
- pyitemcal: nothing; ADR-0004 assigns pyitemcal ability estimation for many examinees at once, and
  the item step's derivatives are `pyitemcal.irt`'s.

## Proposal

JML for the 1PL and 2PL, alternating an ability step and an item step, with examinees and items that
have no finite estimate removed first, and the Rasch correction available. It comes after the methods
of the build order in [the literature review](../discovery/literature-review.md), for comparison
studies.

## Tests

- **Closed form:** with two Rasch items, the JML difference of difficulties $b_2 - b_1$ is exactly
  twice the CML one, $\log n_{10} - \log n_{01}$, where $n_{10}$ counts examinees with item 1 right
  and item 2 wrong (Wright & Douglas, 1977, p. 288; Andersen, 1970, p. 299).
- **Bias in the expected direction:** on Rasch data with $M$ items and roughly uniform scores and
  difficulties, the uncorrected difficulties exceed the conditional ones by a relative amount close to
  $1/(M - 1)$, for items with $|b| \geq .5$ (Wright & Douglas, 1977, pp. 286-288, Equations 31-34,
  Table 1); after the correction, the residual bias for $M = 20$ is within the values of their Table 2
  (p. 288).
- **Editing:** examinees with zero or perfect scores, and items answered by all or none, are removed
  before estimation (Mislevy & Stocking, 1989, p. 65; Wright & Douglas, 1977, p. 294).

## Open Questions

1. **Which JML.** Robitzsch (2021, p. 827) finds penalized JML and JML with an $\varepsilon$
   adjustment better than the usual variants, and than CML and MML in some conditions, and suggests
   them instead. Assumed: the usual JML with the Rasch correction, for comparison with the
   literature; the variants as a later card if wanted (YAGNI).
2. **Identification.** LOGIST fixes the mean and standard deviation of the abilities at 0 and 1, and
   BICAL the mean difficulty at 0 (Baker, 1987, pp. 112, 115). **Decided** (ADR-0008): as LOGIST for
   the 2PL and as BICAL for the Rasch model.

---

**Module:** the estimation layer of ADR-0006, named by the PR that implements it (planned)
**Reference:** Wright & Douglas (1977); Haberman (2004)
**Last validated:** 2026-11-27, against Wright & Douglas (1977), Haberman (2004), Andersen (1970),
Baker (1987), Bock & Aitkin (1981), Mislevy & Stocking (1989) and Robitzsch (2021) on the paper
shelf.
**Implementation guidance:** not yet written; written against ADR-0006 to ADR-0008 when implemented.

## Motivation

JML treats every ability as a parameter and maximizes the likelihood of all of them with the items,
alternating between the two (Baker, 1987, p. 112). It needs no assumption about the distribution of
ability. Its drawback is that the number of parameters grows with the number of examinees, so the
item estimates do not converge to the item parameters as examinees are added to a test of fixed
length (Bock & Aitkin, 1981, p. 443; Baker, 1987, p. 117).

## Definition

The likelihood of the item responses is maximized jointly over the abilities and the item parameters.
Each cycle estimates every examinee's ability with the items fixed, then every item's parameters
with the abilities fixed; the item step is the grouped likelihood of `pyitemcal.irt` with the
estimated abilities as the groups (Baker, 1987, pp. 112-114). The scale is identified by fixing the
abilities' mean and standard deviation, or for the Rasch model the mean difficulty (pp. 112, 115).
Examinees with every item right or wrong have no finite estimate and are removed, as are items
answered by all or none (Mislevy & Stocking, 1989, p. 65; Wright & Douglas, 1977, p. 294).

**Bias and its correction for the Rasch model.** With the number of items $M$ fixed, the JML
difficulties converge to a limit that differs from the parameters by a term of order $1/M$ (Haberman,
2004, p. 13). Wright and Douglas (1977, p. 287, Equations 34-35) find the relative bias close to
$1/(M - 1)$, whatever the spread of the difficulties, and multiply the difficulties, centred at 0 as
BICAL centres them (Baker, 1987, p. 115), by $(M - 1)/M$; it is the factor "used in most versions of
UCON" (Wright & Douglas, 1977, p. 288). The correction
brings the estimates close to the conditional ones, which "is not the same as" removing the bias
with respect to the parameters (Baker, 1987, p. 118).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``correct``
     - on, Rasch model only
     - Multiply the centred difficulties by (M - 1)/M, as UCON does.
```

## Behavior and Comparison

For the Rasch model with 2 items, the JML estimates are exactly twice the conditional ones (Wright &
Douglas, 1977, p. 288). On two SAT tests of 78 and 60 items, the largest difference between JML and
CML difficulties was 0.097 and 0.126, against CML standard errors of at most 0.008 (Haberman, 2004,
pp. 44-45). Haberman (2004, p. 35) finds it "difficult to advocate use of JMLE except for preliminary
estimation". In Robitzsch's (2021, p. 823, Table 2) simulations, JML with the $(M - 1)/M$ correction
and extreme scores removed (his JMLM, p. 816) had acceptable bias, a mean absolute bias below 0.025
(p. 821), in 54.2% of the conditions, against 100% for CML.

## Guarantees

1. Examinees and items without a finite estimate are reported, not estimated.
2. With the correction off, the Rasch estimates are the unconditional maximum likelihood estimates.

## References

Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators. *Journal
of the Royal Statistical Society: Series B, 32*(2), 283-301.
https://doi.org/10.1111/j.2517-6161.1970.tb00842.x. Not openly available; Haberman (2004, §2)
describes conditional estimation.

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Haberman, S. J. (2004). *Joint and conditional maximum likelihood estimation for the Rasch model for
binary responses* (Research Report RR-04-20). Educational Testing Service.
https://doi.org/10.1002/j.2333-8504.2004.tb01947.x. Open copy:
https://files.eric.ed.gov/fulltext/EJ1110899.pdf

Mislevy, R. J., & Stocking, M. L. (1989). A consumer's guide to LOGIST and BILOG. *Applied
Psychological Measurement, 13*(1), 57-75. https://doi.org/10.1177/014662168901300106. Open copy:
https://hdl.handle.net/11299/107035

Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
*Stats, 4*(4), 814-836. https://doi.org/10.3390/stats4040048. Open access.

Wright, B. D., & Douglas, G. A. (1977). Best procedures for sample-free item analysis. *Applied
Psychological Measurement, 1*(2), 281-295. https://doi.org/10.1177/014662167700100216. Open copy:
https://hdl.handle.net/11299/98553

## Implementation Guidance

Follows ADR-0006 to ADR-0008; items with no finite estimate are removed and reported (ADR-0008).
The item step shares the M-step's per-item solver (MML-EM card), with estimated abilities in place
of quadrature nodes.
