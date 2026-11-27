# Literature Review: Item Calibration

> Written by [the research pass](research-pass.md) from sources on the paper shelf, each claim cited
> to the location where it was read (citation convention in [the backlog README](../README.md)).
> It covers the methods reviewed so far; the last section lists the rest, whose cards still carry
> the unverified material carried over from catsim. Formulas live on the method cards; this review
> says what the sources establish, how the methods compare, how mirt implements them, and in what
> order pyitemcal should build them.

## Marginal maximum likelihood with the EM algorithm

**Card:** [estimation/mml-em.md](../estimation/mml-em.md).

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
  $\theta \sim N(0, 1)$ has variance $a^2$ (the MML-EM card's step, confirmed by its measurement).
  Thissen (1982, p. 175) notes that "the square of the estimated slope is the same as the variance of
  the population distribution for conventionally standardized CML estimates".

**Published values.** Thissen (1982, pp. 179-180, Tables 1-2) gives the LSAT-6 response patterns and
their 1PL MML estimates; Bock and Aitkin (1981, p. 451, Table 1; p. 455, Table 3) give normal-ogive
and 1PL estimates for LSAT-6 and LSAT-7. mirt 1.48 reproduces Thissen's estimates to the three
decimals printed (the MML-EM card's measurement).

**Acceleration.** Ramsay's acceleration, as Thissen (1982, p. 181) applies it, doubles or triples the
speed, but in its exact form "frequently leads to over-acceleration and divergence", so he limits it. Among the newer methods compared for the 1PL and 2PL, acceleration
paid most for long tests and when the ability variance was fixed (Beisemann, Wartlick & Doebler, 2020,
p. 232). SQUAREM is documented in full by Du and Varadhan (2020, pp. 3-4).

**mirt.** mirt 1.48 estimates by this method by default, on 61 equally spaced nodes, and stops when
every parameter changes by less than $10^{-4}$; for the 1PL above, that rule stopped the
fixed-slope form short of the maximum (the card's measurement). The card lists its settings, with
their locations.

## Standard errors of item estimates

**Card:** [estimation/standard-errors.md](../estimation/standard-errors.md).

The EM algorithm does not give standard errors (Bock & Aitkin, 1981, p. 447). The observed information
is the complete-data information less the missing information (Louis, 1982, p. 227, Equation 3.3;
Meng & Rubin, 1991, p. 901, Equation 2.4.2), and the methods differ in how they obtain the missing
part: Louis's exact form, the supplemented EM algorithm from the EM iteration history (Meng & Rubin,
1991, pp. 901-902), or Oakes's identity, approximated numerically from E-steps at perturbed parameters
(Chalmers, 2018, pp. 3-6). Oakes's approximation beat every supplemented-EM variant in accuracy and
speed (Pritikin, 2017, p. 7); with central differences it matched Louis's exact form to a mean
relative difference of $2.5 \times 10^{-11}$, at 3% of the time (Chalmers, 2018, p. 13). mirt 1.48 uses it by default.

## Derivatives with respect to item parameters

**Card:** [irt/icc-gradients.md](../irt/icc-gradients.md).

Every M-step and every standard error uses the derivatives of the response probability with respect
to the item's parameters. Barton and Lord (1981, pp. A1-A3) give them for the four-parameter model with
a common upper asymptote; Baker (1987, pp. 114, 116) gives the likelihood equations of the 2PL and 3PL
and notes that for the 1PL and 2PL, in the intercept-slope form, the second derivatives contain no
observed data, so "the Hessian is the information matrix".

## The 3PL and priors on item parameters

**Card:** [estimation/mml-em-3pl.md](../estimation/mml-em-3pl.md).

The 3PL's likelihood is "nearly flat along one or more dimensions" of $(a, b, c)$, and Bayesian item
estimation was motivated by wanting "reasonable" estimates, "in particular, finite ones" (Mislevy,
1986, p. 189). Bayes modal estimation keeps the EM algorithm and adds the derivatives of log prior
densities to the M-step's equations (p. 182, Equation 14), estimating $\log a$, $b$ and
$\operatorname{logit} c$ (p. 183). BILOG's default priors on these are normal with variances .25,
4.00 and .25 (p. 190); PC-BILOG's are $\log a \sim N(0, .5^2)$, $b \sim N(0, 2^2)$ and
$c \sim$ Beta(5, 17) for five-option items (Harwell & Baker, 1991, p. 383). Priors on the slope also
keep the 2PL's estimates finite (Baker, 1987, p. 131). mirt 1.48 sets no default prior and
recommends a normal prior on the logit of its guessing parameter (manual p. 132). On LSAT-7 without
a prior, mirt put four of five guessing estimates at 0; with a logit-normal prior they lay between
.175 and .212 (the card's measurement).

## The 4PL

**Card:** [estimation/mml-em-4pl.md](../estimation/mml-em-4pl.md).

Barton and Lord (1981, p. 2) proposed an upper asymptote below 1 for high-ability examinees who make
a clerical error, fixed it at one value for all items (p. 3), and found "no compelling reason to urge
the use of this model" (p. 6); it is still rarely used (Loken & Rulison, 2010, p. 510; Robitzsch,
2022, p. 3). Without priors, maximum likelihood puts many upper asymptotes at 1 (Battauz, 2020,
pp. 275-276, Table 5; Robitzsch, 2022, p. 9). Bayes modal estimation needs "at least $N = 5{,}000$"
examinees for accurate upper asymptotes (Waller & Feuerstahler, 2017, p. 19), and Markov chain Monte
Carlo on the normal-ogive model about 2,500 for educational items (Culpepper, 2016, Section 5).

## Build order, so far

1. The derivatives, which the M-steps and the standard errors share.
2. MML-EM for the 1PL and 2PL.
3. Standard errors.
4. Priors on item parameters and the 3PL, one M-step addition that the 2PL shares.
5. The 4PL, for banks that need an upper asymptote, with samples of several thousand.

The full order, with the methods below, is set when the pass ends.

## Not yet reviewed

These cards still carry material carried over from catsim, unverified; the review sections replace
it as the pass reaches them:

- joint and conditional maximum likelihood: [jml](../estimation/jml.md),
  [cml-rasch](../estimation/cml-rasch.md);
- Markov chain Monte Carlo and the Metropolis-Hastings Robbins-Monro algorithm:
  [bayesian-mcmc](../estimation/bayesian-mcmc.md);
- item fit and model comparison: [item-fit](../fit/item-fit.md).

The carried-over review's impact and feasibility ratings are dropped: none of them had a source.

## References

Baker, F. B. (1987). Methodology review: Item parameter estimation under the one-, two-, and
three-parameter logistic models. *Applied Psychological Measurement, 11*(2), 111-141.
https://doi.org/10.1177/014662168701100201. Open copy: https://hdl.handle.net/11299/103398

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

Harwell, M. R., & Baker, F. B. (1991). The use of prior distributions in marginalized Bayesian item
parameter estimation: A didactic. *Applied Psychological Measurement, 15*(4), 375-389.
https://doi.org/10.1177/014662169101500409. Open copy: https://hdl.handle.net/11299/114468

Loken, E., & Rulison, K. L. (2010). Estimation of a four-parameter item response theory model.
*British Journal of Mathematical and Statistical Psychology, 63*(3), 509-525.
https://doi.org/10.1348/000711009X474502. Open copy (accepted manuscript):
https://libres.uncg.edu/ir/uncg/f/K_Rulison_Estimation_2010.pdf

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

Pritikin, J. N. (2016). *A computational note on the application of the supplemented EM algorithm to
item response models* (arXiv:1605.00860). https://arxiv.org/abs/1605.00860. Open access.

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response
models in an expectation-maximization framework. *Cogent Psychology, 4*(1), 1279435.
https://doi.org/10.1080/23311908.2017.1279435. Open access.

Robitzsch, A. (2022). Four-parameter guessing model and related item response models. *Mathematical and
Computational Applications, 27*(6), 95. https://doi.org/10.3390/mca27060095. Open access.

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.

Waller, N. G., & Feuerstahler, L. (2017). Bayesian modal estimation of the four-parameter item response
model in real, realistic, and idealized data sets. *Multivariate Behavioral Research, 52*(3), 350-370.
https://doi.org/10.1080/00273171.2017.1292893. Not openly available; Harwell and Baker (1991) describe
the Bayes modal estimation it applies. Cited by the online version's pages, 1-21.

Woodruff, D. J., & Hanson, B. A. (1996). *Estimation of item response models using the EM algorithm
for finite mixtures* (ACT Research Report 96-6). ACT. Open copy:
https://www.act.org/content/dam/act/unsecured/documents/ACT_RR96-06.pdf

Zhang, J., Lu, J., Du, H., & Zhang, Z. (2020). Gibbs-slice sampling algorithm for estimating the
four-parameter logistic model. *Frontiers in Psychology, 11*, 2121.
https://doi.org/10.3389/fpsyg.2020.02121. Open access.
