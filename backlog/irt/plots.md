---
title: Draw the item response plots the maintainer chose
kind: feature
status: ready
priority: 3
effort: L
blocked_by: []
raised_by: [maintainer on 2026-11-28 after the plot catalogue]
---

# Draw the item response plots the maintainer chose

## Problem

On 2026-11-28 the maintainer chose to draw in `irtlib.plot` every plot of the catalogue below
(ADR-0011): the item response plots of the literature and of the R packages on the shelf, for the
unidimensional dichotomous 1PL to 4PL. `item_curve` draws one item's characteristic and information
curves. The plots below remain; catsim deletes its copy of a plot once irtlib has it.

## What irtlib and catsim have now

- `irtlib.plot.item_curve`, with the calling conventions of ADR-0011.
- catsim's `catsim.plot` draws `bank_information` (with a second y axis for the standard error),
  `expected_score`, `item_person_map` and `likelihood` (of a CAT session), which move here.
- A plot whose "Needs" column is filled waits for that quantity in irtlib's results.

## Proposal

One pull request per family, each plot with a figure in its docstring and tests on closed forms or
hand-worked cases. A PR that draws a plot deletes its row; the last one deletes the card.

### Item curves

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Characteristic curves of several items (`items_icc`) | The characteristic curves of a few items on one axes, each named. |  |  | Chalmers 2012, p. 28, Fig. 2; Lalor & Rodriguez 2023, Fig. 1b-c; mirt 1.48 manual, p. 194, plot 'trace'; eRm 1.0-10, plotjointICC |
| Information curves of several items (`items_iic`) | The information curves of a few items on one axes. |  |  | Rizopoulos 2006, p. 13, Fig. 1; mirt 1.48 manual, p. 194, plot 'infotrace' |
| Characteristic curve with a confidence envelope (`item_envelope`) | An item's estimated curve with a 95% band from the uncertainty of its parameter estimates. |  |  | mirt 1.48 manual, p. 94, itemplot CE, CEalpha, CEdraws |
| Generating and calibrated curves (`item_recovered`) | The curve an item was simulated from against the curve calibrated from its responses. |  |  | Orlando & Thissen 2000, pp. 61-62, Figs. 2-4 |

### Test functions

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Test information (`test_information`) | The sum of the items' information against ability, for a bank or chosen sets of its items. | `bank_information(se=False)` |  | Chalmers 2012, p. 27, Fig. 1; Rizopoulos 2006, p. 13, Fig. 1; mirt 1.48 manual, p. 194, plot 'info'; TAM 4.3-25, IRT.informationCurves; eRm 1.0-10, plotINFO |
| Test information and standard error (`test_information_se`) | Test information above, its standard error 1/√I(θ) below, sharing the ability axis. | `bank_information(se=True)` |  | mirt 1.48 manual, p. 194, plot 'infoSE'; PP 1.0.0, pp-plots 'precision' |
| Standard error of measurement (`test_se`) | The test's standard error 1/√I(θ) against ability, for one or more item sets. |  |  | mirt 1.48 manual, p. 194, plot 'SE'; Wright & Masters 1982, p. 177, Fig. 8.3f |
| Test characteristic curve (`test_characteristic`) | The expected number correct against ability: the sum of the items' characteristic curves. | `expected_score` |  | mirt 1.48 manual, p. 194, plot 'score'; Wright & Stone 1979, p. 132, Fig. 6.3.1 |
| Conditional reliability (`test_reliability`) | irtlib.irt.reliability against ability: I(θ) / (I(θ) + 1/σ²). |  |  | mirt 1.48 manual, p. 194, plot 'rxx' |
| Relative efficiency of two forms (`relative_efficiency`) | The ratio of two item sets' information at each ability, on a log scale around 1. |  |  | mirt 1.48 manual, p. 94, itemplot 'RE'; p. 93, relative test information; Wright & Masters 1982, p. 177, Fig. 8.3f |
| Information over an interval (`information_area`) | Test information with the area over an ability interval shaded and its share of the total. |  |  | Rizopoulos 2006, p. 13, Fig. 1; mirt 1.48 manual, p. 7, areainfo |

### Examinees on the scale

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Item-person map (`item_person_map`) | Examinees' abilities above the axis and the items' locations below it, on one θ scale. | `item_person_map` |  | Wright & Stone 1979, p. 121, Fig. 5.10.1; p. 205, Fig. 8.8.1; eRm 1.0-10, plotPImap; TAM 4.3-25, IRT.WrightMap |
| Targeting (`targeting`) | The examinees' abilities above, the test information below with a tick at each item's location. |  |  | Wright & Stone 1979, p. 138, Fig. 6.7.1; PP 1.0.0, pp-plots 'precision' |
| Likelihood of a response pattern (`likelihood`) | The likelihood of one examinee's responses across abilities, relative to its maximum, with the ML estimate and, in a simulation, the true ability. | `likelihood(session)` |  | Bock & Mislevy 1982, p. 434, Fig. 1 |
| Prior, likelihood and posterior (`prior_posterior`) | The prior, the likelihood and the posterior of one response pattern, each as a density. |  |  | Bock & Mislevy 1982, p. 434, Fig. 1; Kim & Bolt 2007, p. 41, Fig. 1 |
| Population posterior (`population_posterior`) | The mean of the examinees' posteriors over the quadrature grid, against the prior. |  |  | mirt 1.48 manual, p. 194, plot 'posteriorTheta' |
| Measurement precision of examinees (`precision`) | Each examinee's standard error against their estimate, with the test's curve. |  |  | PP 1.0.0, pp-plots 'precision' |
| Sum score to θ (`sum_score_conversion`) | The EAP estimate of each sum score, with a band of one posterior standard deviation. |  | EAP estimates for sum scores | mirt 1.48 manual, p. 74, fscores method 'EAPsum' |

### Fit

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Empirical characteristic curve (`empirical_icc`) | The model curve with the observed proportion correct in groups of examinees by ability estimate, with 95% intervals. |  |  | Yen 1981, p. 248, ten ability cells; mirt 1.48 manual, p. 85, itemfit empirical.plot, empirical.CI; mirt 1.48 manual, p. 94, itemplot empirical_proportions; eRm 1.0-10, plotICC empICC; TAM 4.3-25, plot.tam; iarm 0.4.3, ICCplot |
| S-X² observed and expected (`sx2_item`) | The observed proportion correct at each sum score against the proportion S-X² expects, with the item's statistic. |  | each item's observed and expected proportions by sum score | Orlando & Thissen 2000, p. 54, S-X²; mirt 1.48 manual, p. 85, itemfit S_X2.plot |
| Observed and expected sum scores (`sum_score_fit`) | How many examinees got each sum score, against the number the calibrated model expects. |  | the model's sum-score distribution | mirt 1.48 manual, p. 194, plot 'EAPsum'; source SingleGroup-methods.R:1262-1270 |
| Item fit against location (`pathway_map`) | Each item's standardized infit against its difficulty, with the band from -2 to 2. |  |  | Wright & Masters 1982, pp. 102-103, Fig. 5.4c; eRm 1.0-10, plotPWmap |
| Item estimates from two groups (`invariance`) | Each item's difficulty calibrated by CML on low and on high scorers, with 95% intervals, against the identity line. |  |  | Wright & Masters 1982, pp. 115-116, Fig. 5.9; eRm 1.0-10, plotGOF; Yen 1981, p. 254, Fig. 2 |
| Person response profile (`person_profile`) | One examinee's standardized residual on each item, the items in order of difficulty. |  |  | Wright & Stone 1979, p. 180, Fig. 7.9.3; PP 1.0.0, pp-plots 'profile' |
| Person fit against ability (`person_fit_map`) | Each examinee's standardized fit against their ability estimate, with the band from -2 to 2. |  | a person-fit statistic | PP 1.0.0, pp-plots 'fit'; eRm 1.0-10, plotPWmap |

### Calibration diagnostics

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Parameter recovery (`parameter_recovery`) | Estimated against true item parameters, with 95% intervals from the calibration's standard errors. |  |  | Yen 1981, p. 254, Fig. 2; Wright & Masters 1982, p. 74, Fig. 4.3b |
| Item estimates with intervals (`item_estimates_ci`) | Each item's difficulty estimate with its 95% confidence interval, items sorted by estimate. |  |  | eRm 1.0-10, plotDIF |
| Bias of ability estimators (`estimator_bias`) | The mean error of the ML, WLE and EAP estimates at each true ability, on a fixed 20-item test. |  |  | Warm 1989, p. 433, Fig. 1; Waller & Feuerstahler 2017, Fig. 5 |
| Convergence history (`convergence`) | The marginal log-likelihood after each EM cycle. |  |  | Varadhan & Roland 2004, p. 60, Fig. 5; TAM 4.3-25, plotDevianceTAM |
| MCMC trace (`mcmc_trace`) | The draws of one item parameter from three chains with dispersed starting values. |  | a Markov chain Monte Carlo calibration (card estimation/bayesian-mcmc) | Kim & Bolt 2007, p. 43, Fig. 2; p. 45, Fig. 3; sirt 4.2-133, plot.mcmc.sirt |

### Screening the responses

| Plot | What it draws | catsim | Needs | Sources |
|---|---|---|---|---|
| Sum score distribution (`sum_score_histogram`) | How many examinees got each number correct. |  |  | mirt 1.48 manual, p. 54, empirical_plot 'hist' |
| Classical item statistics (`item_statistics`) | Each item's item-rest correlation against its proportion correct. |  | the proportion correct and item-rest correlation of each item, classical statistics irtlib does not compute | mirt 1.48 manual, p. 54, empirical_plot 'discrim_diff'; p. 55, discrim.cut |
| Proportion correct by rest score (`rest_score_curve`) | The proportion correct at each score on the other items, with a kernel smooth. |  |  | mirt 1.48 manual, p. 54, empirical_plot 'prop' with smooth; sirt 4.2-133, plot.np.dich; eRm 1.0-10, plotICC empICC 'kernel'; iarm 0.4.3, ICCplot |

## Tests

- Each plot draws what its sources draw, checked on closed forms or hand-worked cases.
- `uv run pytest -q`, the strict docs build and pre-commit pass.

## References

Every location above was read on the shelf on 2026-11-28.

- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. Applied Psychological Measurement, 6(4), 431-444. doi:10.1177/014662168200600405
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. Journal of Statistical Software, 48(6), 1-29. doi:10.18637/jss.v048.i06
- Kim, J.-S., & Bolt, D. M. (2007). Estimating item response theory models using Markov chain Monte Carlo methods. Educational Measurement: Issues and Practice, 26(4), 38-51. doi:10.1111/j.1745-3992.2007.00107.x
- Lalor, J. P., & Rodriguez, P. (2023). py-irt: A scalable item response theory library for Python. INFORMS Journal on Computing, 35(1), 5-13. doi:10.1287/ijoc.2022.1250 (figure read in the arXiv preprint 2203.01282v2)
- Magis, D. (2013). A note on the item information function of the four-parameter logistic model. Applied Psychological Measurement, 37(4), 304-315. doi:10.1177/0146621613475471
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. Applied Psychological Measurement, 24(1), 50-64. doi:10.1177/01466216000241003
- Rizopoulos, D. (2006). ltm: An R package for latent variable modeling and item response theory analyses. Journal of Statistical Software, 17(5), 1-25. doi:10.18637/jss.v017.i05
- Varadhan, R., & Roland, C. (2004). Squared extrapolation methods (SQUAREM): A new class of simple and efficient numerical schemes for accelerating the convergence of the EM algorithm (Working Paper 63). Johns Hopkins University, Department of Biostatistics.
- Waller, N. G., & Feuerstahler, L. (2017). Bayesian modal estimation of the four-parameter item response model in real, realistic, and idealized data sets. Multivariate Behavioral Research, 52(3), 350-370. doi:10.1080/00273171.2017.1292893
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. Psychometrika, 54(3), 427-450. doi:10.1007/BF02294627
- Wright, B. D., & Masters, G. N. (1982). Rating scale analysis. MESA Press.
- Wright, B. D., & Stone, M. H. (1979). Best test design. MESA Press.
- Yen, W. M. (1981). Using simulation results to choose a latent trait model. Applied Psychological Measurement, 5(2), 245-262. doi:10.1177/014662168100500212
- R packages, read on the shelf as their manuals and help pages: mirt 1.48 (CRAN reference manual of 2026-10-06), eRm 1.0-10, TAM 4.3-25, sirt 4.2-133, iarm 0.4.3, PP 1.0.0.
