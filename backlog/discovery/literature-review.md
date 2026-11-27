# Literature Review: Item Calibration

> Written by [the research pass](research-pass.md) from sources on the paper shelf, each claim cited
> to the location where it was read (citation convention in [the backlog README](../README.md)).
> It covers the methods reviewed so far; the last section lists the rest, whose cards still carry
> the unverified material carried over from catsim. Formulas live on the method cards; this review
> says what the sources establish, how the methods compare, how mirt implements them, and in what
> order pyitemcal should build them.

## 1. Marginal maximum likelihood with the EM algorithm

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
  1PL with slopes fixed at 1 and the ability variance estimated: "the square of the estimated slope
  is the same as the variance of the population distribution" (Thissen, 1982, p. 175).

**Published values.** Thissen (1982, pp. 179-180, Tables 1-2) gives the LSAT-6 response patterns and
their 1PL MML estimates; Bock and Aitkin (1981, p. 451, Table 1; p. 455, Table 3) give normal-ogive
and 1PL estimates for LSAT-6 and LSAT-7. mirt 1.48 reproduces Thissen's estimates to the three
decimals printed (the MML-EM card's measurement).

**Acceleration.** Ramsay's acceleration, as Thissen (1982, p. 181) applies it, doubles or triples the
speed but diverges unless limited. Among the newer methods compared for the 1PL and 2PL, acceleration
paid most for long tests and when the ability variance was fixed (Beisemann, Wartlick & Doebler, 2020,
p. 232). SQUAREM is documented in full by Du and Varadhan (2020).

**mirt.** mirt 1.48 estimates by this method by default, on 61 equally spaced nodes, and stops when
every parameter changes by less than $10^{-4}$; for the 1PL above, that rule stopped the
fixed-slope form short of the maximum (the card's measurement). The card lists its settings, with
their locations.

## 2. Standard errors of item estimates

**Card:** [estimation/standard-errors.md](../estimation/standard-errors.md).

The EM algorithm does not give standard errors (Bock & Aitkin, 1981, p. 447). The observed information
is the complete-data information less the missing information (Louis, 1982, p. 227, Equation 3.3;
Meng & Rubin, 1991, p. 901, Equation 2.4.2), and the methods differ in how they obtain the missing
part: Louis's exact form, the supplemented EM algorithm from the EM iteration history (Meng & Rubin,
1991, p. 901), or Oakes's identity, approximated numerically from E-steps at perturbed parameters
(Chalmers, 2018, pp. 3-6). Oakes's approximation beat every supplemented-EM variant in accuracy and
speed (Pritikin, 2017, p. 7); with central differences it matched Louis's exact form to about
$10^{-11}$ at 3% of the time (Chalmers, 2018, p. 13). mirt 1.48 uses it by default.

## 3. Derivatives with respect to item parameters

**Card:** [irt/icc-gradients.md](../irt/icc-gradients.md).

Every M-step and every standard error uses the derivatives of the response probability with respect
to the item's parameters. Barton and Lord (1981, pp. A1-A3) give them for the four-parameter model with
a common upper asymptote; Baker (1987, pp. 114, 116) gives the likelihood equations of the 2PL and 3PL
and notes that for the 1PL and 2PL, in the intercept-slope form, the second derivatives contain no
observed data, so "the Hessian is the information matrix".

## 4. Build order, so far

1. The derivatives (section 3), which the M-step and the standard errors share.
2. MML-EM for the 1PL and 2PL (section 1).
3. Standard errors (section 2).

The full order, with the methods below, is set when the pass ends.

## 5. Not yet reviewed

These cards still carry material carried over from catsim, unverified; the review sections replace
it as the pass reaches them:

- the 3PL and 4PL, and priors on item parameters: [mml-em-3pl](../estimation/mml-em-3pl.md),
  [mml-em-4pl](../estimation/mml-em-4pl.md);
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

Beisemann, M., Wartlick, O., & Doebler, P. (2020). Comparison of recent acceleration techniques for
the EM algorithm in one- and two-parameter logistic IRT models. *Psych, 2*(4), 209-252.
https://doi.org/10.3390/psych2040018. Open access.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters:
Application of an EM algorithm. *Psychometrika, 46*(4), 443-459. https://doi.org/10.1007/BF02293801.
Not openly available; Baker (1987, pp. 131-132) describes the method.

Chalmers, R. P. (2018). Numerical approximation of the observed information matrix with Oakes'
identity. *British Journal of Mathematical and Statistical Psychology, 71*(3), 415-436.
https://doi.org/10.1111/bmsp.12127. Free to read at the publisher; pages cited are those of the
Early View version, 1-22.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via
the EM algorithm. *Journal of the Royal Statistical Society: Series B, 39*(1), 1-22.
https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. Not openly available; Woodruff and Hanson
(1996, pp. 3-8) describe the algorithm.

Du, Y., & Varadhan, R. (2020). SQUAREM: An R package for off-the-shelf acceleration of EM, MM and
other EM-like monotone algorithms. *Journal of Statistical Software, 92*(7), 1-41.
https://doi.org/10.18637/jss.v092.i07. Open access.

Engelen, R. J. H. (1987). *A review of different estimation procedures in the Rasch model* (Research
Report 87-6). University of Twente. Open copy: https://files.eric.ed.gov/fulltext/ED310126.pdf

Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *Journal of
the Royal Statistical Society: Series B, 44*(2), 226-233.
https://doi.org/10.1111/j.2517-6161.1982.tb01203.x. Not openly available; Monroe and Cai (2013,
p. 14) describe the method.

Meng, X.-L., & Rubin, D. B. (1991). Using EM to obtain asymptotic variance-covariance matrices: The
SEM algorithm. *Journal of the American Statistical Association, 86*(416), 899-909.
https://doi.org/10.1080/01621459.1991.10475130. Not openly available; Pritikin (2016, pp. 4-7)
describes the method.

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods for item response
models in an expectation-maximization framework. *Cogent Psychology, 4*(1), 1279435.
https://doi.org/10.1080/23311908.2017.1279435. Open access.

Thissen, D. (1982). Marginal maximum likelihood estimation for the one-parameter logistic model.
*Psychometrika, 47*(2), 175-186. https://doi.org/10.1007/BF02296273. Not openly available; Baker
(1987, pp. 133-134) describes the method.
