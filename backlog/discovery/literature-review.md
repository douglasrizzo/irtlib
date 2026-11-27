# Literature Review: Item Calibration

> Built from sources on the paper shelf, each claim cited to the location where it was read
> (citation convention in [the backlog README](../README.md)). It covers every method card in `irt/`,
> `estimation/` and `fit/`. Formulas live on the method cards; this review says what the sources
> establish, how the methods compare, how mirt implements them, and in what order pyitemcal should
> build them.

## Marginal maximum likelihood with the EM algorithm

**Implemented:** `pyitemcal.mml`, whose module docstring holds the design and its sources.

**What it is.** Marginal maximum likelihood (MML) integrates the examinees' abilities out over a
population distribution and maximizes the likelihood of the item parameters alone (Bock & Aitkin,
1981, p. 444). Bock and Lieberman's direct solution needs an information matrix summed over all
$2^n$ response patterns, impractical beyond about 12 items (p. 444). Bock and Aitkin's
reformulation alternates an E-step, which turns the data into expected counts of examinees and of
correct responses at each quadrature node, with an M-step, which fits each item separately to those
counts (p. 447). Baker (1987, p. 132) gives the same algorithm for the logistic models.

**Why MML.** Joint estimation of abilities and items is not generally possible, because the number of
parameters grows with the number of examinees and standard limit theorems do not apply (Bock &
Aitkin, 1981, p. 443). Conditional maximum likelihood removes the abilities only for the 1PL (p. 443).
For the 1PL, MML and conditional estimates nearly coincide when the model fits, and neither has
joint estimation's bias for that model (p. 454); MML keeps examinees with every item right or every
item wrong (Engelen, 1987, p. 25).

**Properties.**

- The estimates depend little on the shape of the assumed ability distribution: normal, rectangular
  and empirical priors gave nearly identical LSAT estimates (Bock & Aitkin, 1981, p. 449; Table 1,
  p. 451).
- The likelihood never decreases from one cycle to the next (Dempster, Laird & Rubin, 1977, p. 7,
  Theorem 1), but convergence is linear, at the rate of the largest eigenvalue of the algorithm's
  Jacobian (p. 10), and slow near the solution (Bock & Aitkin, 1981, p. 447).
- MML does not protect against slope estimates diverging (Baker, 1987, p. 133).
- The 1PL with a common slope and a standard normal ability distribution is the same model as the
  1PL with slopes fixed at 1 and the ability variance estimated, since $a\theta$ with
  $\theta \sim N(0, 1)$ has variance $a^2$ (this review's step; mirt reached the same maximum for
  both on LSAT-6, in 2026-11-27's measurement).
  Thissen (1982, p. 175) notes that "the square of the estimated slope is the same as the variance of
  the population distribution for conventionally standardized CML estimates".

**Published values.** Thissen (1982, pp. 179-180, Tables 1-2) gives the LSAT-6 response patterns and
their 1PL MML estimates; Bock and Aitkin (1981, p. 451, Table 1; p. 455, Table 3) give normal-ogive
and 1PL estimates for LSAT-6 and LSAT-7. mirt 1.48 and `pyitemcal.mml` reproduce Thissen's
estimates to the three decimals printed (`tests/test_mml.py`).

**Acceleration.** Ramsay (1975, pp. 340-341, Equations 5 and 10) extrapolates a slow fixed-point
iteration along its last step, by a factor set from the norms of the iterates' first and second
differences and updated every third iteration (p. 345). As Thissen (1982, p. 181) applies it to MML,
it doubles or triples the speed, but in its exact form "frequently leads to over-acceleration and
divergence", so he limits it; mirt bounds it too, and departs from Ramsay's rule in three other ways
(`backlog/estimation/em-acceleration.md`). In that card's measurement, mirt's variant cut the cycles
by 1.7 to 2.7 times without changing the estimates, and Ramsay's unbounded rule diverged on a slow
case. Among the newer
methods compared for the 1PL and 2PL, acceleration paid most for long tests and when the ability
variance was fixed (Beisemann, Wartlick & Doebler, 2020, p. 232). SQUAREM is documented in full by
Du and Varadhan (2020, pp. 3-4).

**mirt.** mirt 1.48 estimates by this method by default, on 61 equally spaced nodes, and stops when
every parameter changes by less than $10^{-4}$; for the 1PL above, that rule stopped the
fixed-slope form short of the maximum (the card's measurement). The card lists its settings, with
their locations.

## Standard errors of item estimates

**Implemented:** the standard errors of `pyitemcal.mml`, whose module docstring holds the design.

The EM algorithm does not give standard errors (Bock & Aitkin, 1981, p. 447). The observed information
is the complete-data information less the missing information (Louis, 1982, p. 227, Equation 3.3;
Meng & Rubin, 1991, p. 901, Equation 2.4.2), and the methods differ in how they obtain the missing
part: Louis's exact form, the supplemented EM algorithm from the EM iteration history (Meng & Rubin,
1991, pp. 901-902), or Oakes's identity, which writes the observed-data Hessian, at any parameter
value, as the complete-data Hessian of the M-step plus the derivative of the M-step gradient with
respect to the E-step's parameters (Oakes, 1999, p. 480, Equation 6), the latter approximated
numerically from E-steps at perturbed parameters (Chalmers, 2018, pp. 3-6). Oakes's approximation beat every supplemented-EM variant in accuracy and
speed (Pritikin, 2017, p. 7); with central differences it matched Louis's exact form to a mean
relative difference of $2.5 \times 10^{-11}$, at 3% of the time (Chalmers, 2018, p. 13). mirt 1.48 uses it by default.

## Derivatives with respect to item parameters

**Implemented:** `pyitemcal.irt`, whose module docstring holds the design and its sources.

Every M-step and every standard error uses the derivatives of the response probability with respect
to the item's parameters. Barton and Lord (1981, pp. A1-A3) give them for the four-parameter model with
a common upper asymptote; Baker (1987, pp. 114, 116) gives the likelihood equations of the 2PL and 3PL
and notes that for the 1PL and 2PL, in the intercept-slope form, the second derivatives contain no
observed data, so "the Hessian is the information matrix".

## The 3PL and priors on item parameters

**Implemented:** the 3PL and priors of `pyitemcal.mml`, whose module docstring holds the design.

The 3PL's likelihood is "nearly flat along one or more dimensions" of $(a, b, c)$, and Bayesian item
estimation was motivated by wanting "reasonable" estimates, "in particular, finite ones" (Mislevy,
1986, p. 189). Bayes modal estimation keeps the EM algorithm and adds the derivatives of log prior
densities to the M-step's equations (p. 182, Equation 14), estimating $\log a$, $b$ and
$\operatorname{logit} c$ (p. 183). BILOG's default priors on these are normal with variances .25,
4.00 and .25 (p. 190); PC-BILOG's are $\log a \sim N(0, .5^2)$, $b \sim N(0, 2^2)$ and
$c \sim$ Beta(5, 17) for five-option items (Harwell & Baker, 1991, p. 383). Priors on the slope also
keep the 2PL's estimates finite (Baker, 1987, pp. 130-131). mirt 1.48 sets no default prior (source
`R/06-LoadPars.R:361-362`) and recommends a normal prior on the logit of its guessing parameter
(manual p. 132). On LSAT-7 without
a prior, mirt put four of five guessing estimates at 0; with a logit-normal prior they lay between
.175 and .212 (the card's measurement).

## The 4PL

**Implemented:** the 4PL of `pyitemcal.mml`, whose module docstring holds the design.

Barton and Lord (1981, p. 2) proposed an upper asymptote below 1 for high-ability examinees who make
a clerical error, fixed it at one value for all items (p. 3), and found "no compelling reason to urge
the use of this model" (p. 6); it is still rarely used (Loken & Rulison, 2010, p. 510; Robitzsch,
2022, p. 3). Without priors, maximum likelihood puts many upper asymptotes at 1 (Battauz, 2020,
pp. 275-276, Table 5; Robitzsch, 2022, pp. 8-9). Bayes modal estimation needs "at least
$N = 5{,}000$ subjects ... for accurate 4PM item calibration" (Waller & Feuerstahler, 2017, p. 19),
and Markov chain Monte
Carlo on the normal-ogive model about 2,500 for educational items (Culpepper, 2016, Section 5).

## Conditional maximum likelihood

**Card:** [estimation/cml-rasch.md](../estimation/cml-rasch.md).

For the Rasch model, conditioning on each examinee's number correct, a sufficient statistic, removes
the abilities from the likelihood (Bock & Aitkin, 1981, p. 443). The conditional estimates are
consistent (Andersen, 1970, p. 286, Theorem 1), though "not in general ... efficient" (p. 284). The
elementary symmetric functions they need are computed by the summation algorithm, since the faster
difference algorithm "breaks down when there are more than 20-50 items" (Gustafsson, 1980,
pp. 381-382). With data missing at random, ordinary CML "will likely fail" and MML or JML is
preferred (Robitzsch, 2021, p. 827), and pyitemcal's response matrices may have missing entries
(ADR-0001).

## Joint maximum likelihood

**Card:** [estimation/jml.md](../estimation/jml.md).

Joint maximum likelihood estimates every ability with the items (Baker, 1987, p. 112). For the Rasch
model with the number of items $M$ fixed, its difficulties converge to a limit that differs from the
parameters by a term of order $1/M$ (Haberman, 2004, p. 13); they are spread outward by about
$1/(M - 1)$ relative to the conditional estimates, which Wright and Douglas correct by multiplying by
$(M - 1)/M$ (Wright & Douglas, 1977, pp. 286-287), bringing them close to the conditional ones
(Baker, 1987, p. 118). Haberman (2004, p. 35) finds it "difficult to advocate use of JMLE except for
preliminary estimation"; in Robitzsch's (2021, p. 827) simulations, penalized JML and JML with an
$\varepsilon$ adjustment "outperformed conditional and marginal maximum likelihood as well as LIM in
many situations in terms of the RMSE".

## Markov chain Monte Carlo, and MH-RM

**Card:** [estimation/bayesian-mcmc.md](../estimation/bayesian-mcmc.md).

Metropolis-within-Gibbs sampling fits the logistic models (Patz & Junker, 1999a, p. 159); its
posterior means matched BILOG's estimates (Patz & Junker, 1999a, p. 163, Table 2; Kim & Bolt, 2007,
p. 46). Its value is the posterior: for highly discriminating items, posterior standard deviations
were 20% to 60% larger than the EM standard errors (Albert, 1992, p. 262). Reported run times are
hours, which Patz and Junker (1999a, p. 170) attribute largely to unoptimized code, and the 3PL mixes
slowly (Patz & Junker, 1999b, p. 353). The Metropolis-Hastings Robbins-Monro algorithm uses such
sampling for maximum likelihood, with a cost that grows linearly in the number of dimensions (Cai,
2010a, pp. 40, 51), but for one dimension EM was faster (pp. 48-49), so it gets no card.

## Item fit

**Implemented:** S-X² in `pyitemcal.fit`, whose module docstring holds the design; the Rasch mean
squares are [fit/mean-squares.md](../fit/mean-squares.md).

Orlando and Thissen's (2000) S-X² groups examinees by their number-correct score and computes the
expected proportions from the model by a recursion over items (pp. 53-54); on data from the fitted
model it rejected at .04 to .07, where their $Q_1$-$X^2$, an index similar to Yen's (1981) $Q_1$
that groups by estimated ability, rejected at .95 to .96 on a 10-item test and .06 to .15 on an
80-item test (Orlando & Thissen, 2000, pp. 55, 57, Table 1). The weighted and unweighted mean squares
of Wright and Masters (1982, p. 99) apply to the Rasch family (p. 10); no source on the shelf gives
interpretation bands for them, such as 0.7 to 1.3, and "there is no clear basis" for a point-biserial
cutoff (Wright & Stone, 1979, p. 26). S-X² "cannot handle incomplete data directly"; on 40-item tests
with 10% or 20% of responses missing, the unadapted index rejected far too often (Zhang & Wang, 2022,
pp. 705, 714, Table 3).

## Model comparison

**Implemented:** `pyitemcal.comparison`, whose module docstring holds the design.

Nested models are compared by $-2$ times the difference of their log-likelihoods, "distributed as
$\chi^2$" (Orlando & Thissen, 2000, p. 51), as Thissen (1982, p. 178) compares the 1PL and 2PL; any
models by AIC (Akaike, 1974, p. 719) or BIC (Schwarz, 1978, p. 461). BIC is consistent when the
true model is a candidate and AIC is efficient when none is (Vrieze, 2012, Pages 10-11); for the 4PL
and related models, "AIC should be preferred over BIC" (Robitzsch, 2022, p. 11). For the Rasch
model, mirt 1.48 gives $G^2$ of 21.80 on LSAT-6 and 43.89 on LSAT-7, against Bock and Aitkin's
(1981, p. 455, Table 3) 21.80 and 43.90 (the card's measurement).

## Build order

1. The derivatives, which the M-steps and the standard errors share.
2. MML-EM for the 1PL and 2PL.
3. Standard errors.
4. Item fit, which ADR-0001 requires, by S-X².
5. Priors on item parameters and the 3PL, one M-step addition that the 2PL shares.
6. Model comparison, from the log-likelihoods the methods above report.
7. The 4PL, for banks that need an upper asymptote, with samples of several thousand.
8. Conditional maximum likelihood, for the Rasch model on complete data.
9. Joint maximum likelihood, for comparison studies.
10. Markov chain Monte Carlo, for posterior distributions of the item parameters.

## Sources

Every source cited here is on the paper shelf and in `sphinx/references.bib`. The carried-over
review's impact and feasibility ratings are dropped: none of them had a source.

## References

Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on
Automatic Control, 19*(6), 716-723. https://doi.org/10.1109/TAC.1974.1100705. Not openly available;
Vrieze (2012, Pages 5-6) describes the criterion.

Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling.
*Journal of Educational Statistics, 17*(3), 251-269. https://doi.org/10.3102/10769986017003251. Not
openly available; Sheng (2008) describes the method.

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

Barton, M. A., & Lord, F. M. (1981). *An upper asymptote for the three-parameter logistic
item-response model* (Research Report RR-81-20). Educational Testing Service.
https://doi.org/10.1002/j.2333-8504.1981.tb01255.x. Open copy:
https://files.eric.ed.gov/fulltext/ED207996.pdf

Battauz, M. (2020). Regularized estimation of the four-parameter logistic model. *Psych, 2*(4),
269-278. https://doi.org/10.3390/psych2040020. Open access.

Beisemann, M., Wartlick, O., & Doebler, P. (2020). Comparison of recent acceleration techniques for
the EM algorithm in one- and two-parameter logistic IRT models. *Psych, 2*(4), 209-252.
https://doi.org/10.3390/psych2040018. Open access.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Cai, L. (2010a). High-dimensional exploratory item factor analysis by a Metropolis-Hastings
Robbins-Monro algorithm. *Psychometrika, 75*(1), 33-57. https://doi.org/10.1007/s11336-009-9136-x.
Open access.

Chalmers, R. P. (2018). Numerical approximation of the observed information matrix with Oakes'
identity. *British Journal of Mathematical and Statistical Psychology, 71*(3), 415-436.
https://doi.org/10.1111/bmsp.12127. Open copy:
https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/bmsp.12127. Pages cited are those of the
Early View version, 1-22.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Culpepper, S. A. (2016). Revisiting the 4-parameter item response model: Bayesian estimation and
application. *Psychometrika, 81*(4), 1142-1163. https://doi.org/10.1007/s11336-015-9477-6. Not
openly available; Zhang, Lu, Du and Zhang (2020) describe the method. Cited by section: the shelf copy
is the online version, without page numbers.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via
the EM algorithm. *Journal of the Royal Statistical Society: Series B, 39*(1), 1-22.
https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. Not openly available; Woodruff and Hanson
(1996, pp. 3-8) describe the algorithm.

Du, Y., & Varadhan, R. (2020). SQUAREM: An R package for off-the-shelf acceleration of EM, MM and
other EM-like monotone algorithms. *Journal of Statistical Software, 92*(7), 1-41.
https://doi.org/10.18637/jss.v092.i07. Open access.

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

Harwell, M. R., & Baker, F. B. (1991). The use of prior distributions in marginalized Bayesian item
parameter estimation: A didactic. *Applied Psychological Measurement, 15*(4), 375-389.
https://doi.org/10.1177/014662169101500409. Open copy: https://hdl.handle.net/11299/114468

Kang, T., & Chen, T. T. (2007). *An investigation of the performance of the generalized S-X²
item-fit index for polytomous IRT models* (ACT Research Report 2007-1). ACT. Open copy:
https://www.act.org/content/dam/act/unsecured/documents/ACT_RR2007-1.pdf

Kim, J.-S., & Bolt, D. M. (2007). Estimating item response theory models using Markov chain Monte
Carlo methods. *Educational Measurement: Issues and Practice, 26*(4), 38-51.
https://doi.org/10.1111/j.1745-3992.2007.00107.x. Open copy:
https://www.ncme.org/wp-content/uploads/2025/10/Module-27-Markov-chain-Monte-Carlo-Estimation-Kim-1.pdf

Loken, E., & Rulison, K. L. (2010). Estimation of a four-parameter item response theory model.
*British Journal of Mathematical and Statistical Psychology, 63*(3), 509-525.
https://doi.org/10.1348/000711009X474502. Open copy (accepted manuscript):
https://web.archive.org/web/20240708235413/https://libres.uncg.edu/ir/uncg/f/K_Rulison_Estimation_2010.pdf
(archived copy of the UNCG repository's file)

Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *Journal of
the Royal Statistical Society: Series B, 44*(2), 226-233.
https://doi.org/10.1111/j.2517-6161.1982.tb01203.x. Not openly available; Monroe and Cai (2013,
p. 14) describe the method.

Meng, X.-L., & Rubin, D. B. (1991). Using EM to obtain asymptotic variance-covariance matrices: The
SEM algorithm. *Journal of the American Statistical Association, 86*(416), 899-909.
https://doi.org/10.1080/01621459.1991.10475130. Not openly available; Pritikin (2016, pp. 4-7)
describes the method.

Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika, 51*(2),
177-195. https://doi.org/10.1007/BF02293979. Not openly available; Harwell and Baker (1991,
pp. 377-387) describe the method.

Monroe, S., & Cai, L. (2013). *Estimation of a Ramsay-curve item response theory model by the
Metropolis-Hastings Robbins-Monro algorithm* (CRESST Report 834). University of California, Los
Angeles. Open copy: https://files.eric.ed.gov/fulltext/ED555725.pdf

Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *Journal of
the Royal Statistical Society: Series B, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188.
Not openly available; Chalmers (2018, pp. 3-6) describes the method.

Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response
theory models. *Applied Psychological Measurement, 24*(1), 50-64.
https://doi.org/10.1177/01466216000241003. Not openly available; Kang and Chen (2007, pp. 4-5)
describe the method.

Patz, R. J., & Junker, B. W. (1999a). A straightforward approach to Markov chain Monte Carlo methods
for item response models. *Journal of Educational and Behavioral Statistics, 24*(2), 146-178.
https://doi.org/10.3102/10769986024002146. Not openly available; Kim and Bolt (2007) describe the
method.

Patz, R. J., & Junker, B. W. (1999b). Applications and extensions of MCMC in IRT: Multiple item types,
missing data, and rated responses. *Journal of Educational and Behavioral Statistics, 24*(4),
342-366. https://doi.org/10.3102/10769986024004342. Not openly available; Kim and Bolt (2007, p. 42)
describe the Metropolis-within-Gibbs sampling it extends, citing this pair of papers with the a and b
labels swapped.

Pritikin, J. N. (2016). *A computational note on the application of the supplemented EM algorithm to
item response models* (arXiv:1605.00860). https://arxiv.org/abs/1605.00860. Open access.

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response
models in an expectation-maximization framework. *Cogent Psychology, 4*(1), 1279435.
https://doi.org/10.1080/23311908.2017.1279435. Open access.

Ramsay, J. O. (1975). Solving implicit equations in psychometric data analysis. *Psychometrika,
40*(3), 337-360. https://doi.org/10.1007/BF02291762. Not openly available.

Robitzsch, A. (2021). A comprehensive simulation study of estimation methods for the Rasch model.
*Stats, 4*(4), 814-836. https://doi.org/10.3390/stats4040048. Open access.

Robitzsch, A. (2022). Four-parameter guessing model and related item response models. *Mathematical and
Computational Applications, 27*(6), 95. https://doi.org/10.3390/mca27060095. Open access.

Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics, 6*(2), 461-464.
https://doi.org/10.1214/aos/1176344136. Open copy:
https://projecteuclid.org/journals/annals-of-statistics/volume-6/issue-2/Estimating-the-Dimension-of-a-Model/10.1214/aos/1176344136.pdf

Sheng, Y. (2008). Markov chain Monte Carlo estimation of normal ogive IRT models in MATLAB. *Journal of
Statistical Software, 25*(8), 1-15. https://doi.org/10.18637/jss.v025.i08. Open access.

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.

Vrieze, S. I. (2012). Model selection and psychological theory: A discussion of the differences
between the Akaike information criterion (AIC) and the Bayesian information criterion (BIC).
*Psychological Methods, 17*(2), 228-243. https://doi.org/10.1037/a0027127. Open copy (author
manuscript, cited by its page numbers): https://pmc.ncbi.nlm.nih.gov/articles/PMC3366160/

Waller, N. G., & Feuerstahler, L. (2017). Bayesian modal estimation of the four-parameter item response
model in real, realistic, and idealized data sets. *Multivariate Behavioral Research, 52*(3), 350-370.
https://doi.org/10.1080/00273171.2017.1292893. Not openly available; Harwell and Baker (1991) describe
the Bayes modal estimation it applies. Cited by the online version's pages, 1-21.

Woodruff, D. J., & Hanson, B. A. (1996). *Estimation of item response models using the EM algorithm
for finite mixtures* (ACT Research Report 96-6). ACT. Open copy:
https://www.act.org/content/dam/act/unsecured/documents/ACT_RR96-06.pdf

Wright, B. D., & Douglas, G. A. (1977). Best procedures for sample-free item analysis. *Applied
Psychological Measurement, 1*(2), 281-295. https://doi.org/10.1177/014662167700100216. Open copy:
https://hdl.handle.net/11299/98553

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Rating%20Scale%20Analysis.pdf

Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press. Open copy:
https://www.rasch.org/BTD_RSA/pdf/Best%20Test%20Design.pdf

Yen, W. M. (1981). Using simulation results to choose a latent trait model. *Applied Psychological
Measurement, 5*(2), 245-262. https://doi.org/10.1177/014662168100500212. Open copy:
https://hdl.handle.net/11299/100381

Zhang, J., Lu, J., Du, H., & Zhang, Z. (2020). Gibbs-slice sampling algorithm for estimating the
four-parameter logistic model. *Frontiers in Psychology, 11*, 2121.
https://doi.org/10.3389/fpsyg.2020.02121. Open access.

Zhang, X., & Wang, C. (2022). Modified item-fit indices for dichotomous IRT models with missing data.
*Applied Psychological Measurement, 46*(8), 705-719. https://doi.org/10.1177/01466216221125176. Open
copy: https://files.eric.ed.gov/fulltext/ED628568.pdf
