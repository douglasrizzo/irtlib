---
title: Should irtlib.jml offer penalized JML, JML with Warm abilities and JMLε?
kind: decision
status: needs-decision
priority: 4
effort: S
blocked_by: []
raised_by: [joint maximum likelihood PR on 2026-11-27]
---

# Should irtlib.jml offer penalized JML, JML with Warm abilities and JMLε?

## Why this card exists

The joint maximum likelihood PR (#33) implemented one variant of JML and asked whether
Robitzsch's (2021) other variants deserve a card. They change only the ability step, and two of
those steps already exist in `irtlib.ability`. On 2026-11-28 the maintainer asked for the card,
pending a decision, with what other packages implement. The variants serve comparison studies of
the Rasch model, the use `irtlib.jml` already has. No user has asked for them.

## Problem

`irtlib.jml.JMLCalibrator` implements one JML variant, JMLM in Robitzsch's terms:

- examinees with extreme scores are left out;
- abilities are estimated by maximum likelihood;
- the Rasch difficulties are multiplied by $(I - 1)/I$ (Robitzsch, 2021, p. 816).

Robitzsch (2021) compares three other variants, which keep the examinees with extreme scores:

- **JMLW** estimates the abilities by Warm's weighted likelihood, which "can be interpreted as a
  Bayesian estimation method with a Jeffrey’s prior for abilities". Its estimates can be
  corrected like JMLM's (p. 816).
- **PJML** adds a ridge penalty $-\lambda\theta_p^2$ to each person's log-likelihood. That is a
  normal prior $N(0, \sigma^2_{prior})$ on the abilities, whose best value "will typically differ"
  between optimizing person or item estimates (p. 816).
- **JMLε** replaces each sufficient statistic $S_p$ by $S^*_p = \varepsilon + (I - 2\varepsilon)/I
  \cdot S_p$, which lies in $[\varepsilon, I - \varepsilon]$, so every ability estimate is finite
  (p. 816, Equation 3).

A comparison study that wants them cannot run them in irtlib.

## Evidence

- Robitzsch (2021) finds that "in many situations (i.e., for long tests), penalized JML and JML with
  ε adjustment resulted in very efficient estimates and might be considered alternatives to JML
  implementations currently used in statistical software" (p. 814).
- His Table 2 (p. 823) gives the percentage of conditions with acceptable bias, and the rank by
  relative RMSE out of 32 methods:

  | Method | Acceptable bias | RMSE rank |
  |---|---|---|
  | CML | 100% | 8 |
  | JMLM (irtlib's) | 54.2% | 21 |
  | JMLW | 35.4% | 22 |
  | PJML, $\sigma_{prior}$ = 1 / 1.5 / 2 | 22.4% / 84.9% / 27.1% | 23 / 9 / 30 |
  | JMLε, ε = 0.24 | 56.3% | 1 |

  So PJML depends strongly on its prior. The ε of 0.24 "turned out to be optimal in preliminary
  simulation studies" (p. 821).
- **The software** (Robitzsch, 2021, p. 821; each package's source read on 2026-11-28):
  - **sirt 4.2-133** has JMLM, JMLW and PJML in `rasch.jml`: `method = "MLE"` or `"WLE"`, and
    `prior_sd` for the penalized version (`man/rasch.jml.Rd`; `R/rasch.jml.R:27, 171-172`).
  - **immer 1.5-13** has JMLε in `immer_jml`: `est_method = "eps_adj"` by default with `eps = 0.3`,
    and also `"jml"` and `"jml_bc"` (`man/immer_jml.Rd`; `R/immer_jml.R:7, 60-62`).
  - **TAM 4.3-25** has JML in `tam.jml`, with maximum likelihood abilities in its cycles and the
    correction on by default (`bias = TRUE`; `R/tam_jml_version2.R:192-196, 276-278`).

## What irtlib and catsim have now

- `src/irtlib/jml.py`: the ability step calls `irtlib.ability.maximum_likelihood` for each group of
  examinees. `calibration._edited` leaves out the extreme examinees first.
- `irtlib.ability` already has the abilities of two variants:
  - `weighted_likelihood`, Warm's estimator, for JMLW;
  - `maximum_a_posteriori` with a normal prior, for PJML.
- catsim has no calibration.

## Proposal

Options:

1. **Keep JMLM alone.** `irtlib.jml` serves comparisons and preliminary estimates, CML and MML
   are irtlib's estimators for the Rasch model, and no user has asked for the variants (YAGNI).
2. **An `ability` setting**, a `StrEnum` of ML, WLE and MAP (with the prior's standard deviation).
   WLE and MAP keep the examinees with extreme scores, so the editing applies to items only. This is
   small, since both estimators exist. Robitzsch describes the correction for JMLM and JMLW only
   (p. 816), so whether it applies would depend on the setting.
3. **Option 2 and JMLε.** JMLε needs an ability step on the modified scores, for the Rasch model
   only.

Recommendation: option 1 until a comparison study needs the variants, then option 2.

## Open Questions

1. Which option the maintainer prefers. The card assumes option 1.

## References

- Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
  *Stats, 4*(4), 814-836. doi:10.3390/stats4040048. Open access.
- sirt 4.2-133, CRAN source package. <https://cran.r-project.org/package=sirt>
- immer 1.5-13, CRAN source package. <https://cran.r-project.org/package=immer>
- TAM 4.3-25, CRAN source package. <https://cran.r-project.org/package=TAM>
