---
title: Calibrate by Markov chain Monte Carlo
kind: technique
status: blocked
priority: 4
effort: L
blocked_by: [estimation/mml-em-3pl]
raised_by: [moved from catsim's backlog]
---

# Calibrate by Markov chain Monte Carlo

## Problem

Markov chain Monte Carlo (MCMC) samples the joint posterior of the abilities and the item
parameters, so it reports the uncertainty of the item estimates as a whole distribution, not a
standard error at a mode. It is how the 4PL has been fitted (Loken & Rulison, 2010, p. 513;
Culpepper, 2016, Section 2). Its point estimates agree closely with those of BILOG (below), and the
run times reported are hours, so its value for pyitemcal is the posterior, and it comes last in the
build order.

## Evidence

The research pass (2026-11-27) checked the card carried over from catsim against Albert (1992),
Patz and Junker (1999a, 1999b), Kim and Bolt (2007), Gelman and Rubin (1992) and Cai (2010a):

- "MCMC is the most principled approach for 3PL/4PL estimation" has no source; Loken and Rulison
  (2010, p. 523) "do not mean to imply that fitting a 4PM necessarily requires a Bayesian approach";
- "MH-within-Gibbs is the standard": Patz and Junker (1999a, p. 149) present it as "easier to
  implement for novel models", not as the standard;
- "effective sample size" appears in none of the sources read; they use the potential scale
  reduction (Gelman & Rubin, 1992, p. 461), Geweke's and Raftery and Lewis's criteria (Kim & Bolt,
  2007, p. 42) and autocorrelations (Patz & Junker, 1999b, p. 353);
- "orders of magnitude slower than MML-EM" is not measured on the same data in any source; the run
  times they report are below.

## What pyitemcal and catsim have now

Nothing for sampling in either package; ADR-0005 governs how a sampler takes its random numbers.

## Proposal

A Metropolis-Hastings-within-Gibbs sampler for the logistic 1PL to 3PL (Patz & Junker, 1999a,
1999b), with the 3PL card's priors, several chains from one generator, and the potential scale
reduction as its convergence check. It uses NumPy only (ADR-0002).

## Tests

- **Agreement with Bayes modal estimation:** on a large simulated sample, the posterior modes agree
  with the 3PL card's Bayes modal estimates under the same priors, within a tolerance the test
  justifies from the chains' Monte Carlo error. Patz and Junker's (1999a, p. 163, Table 2) posterior
  means differed from BILOG's estimates by up to 0.01, and Kim and Bolt's (2007, p. 46) BILOG run used
  "the same priors"; the test departs from both by comparing modes.
- **Potential scale reduction:** for $m$ chains of length $2n$, of which the last $n$ are kept, with
  $B/n$ the variance between the chain means and $W$ the mean of the within-chain variances
  (Gelman & Rubin, 1992, p. 460), computed from hand-made chains against the closed forms
  $\hat\sigma^2 = \frac{n-1}{n}W + \frac{1}{n}B$ (p. 461, Equation 3), $\hat V = \hat\sigma^2 +
  B/(mn)$, $df = 2\hat V^2/\widehat{\operatorname{var}}(\hat V)$ with the variance of Equation 4, and
  $\sqrt{\hat R} = \sqrt{(\hat V/W)\,df/(df - 2)}$ (p. 461).
- **Detailed balance:** each Metropolis step's acceptance ratio, including the correction for its
  asymmetric proposal, checked against a direct computation of the posterior ratio.
- **Reproducibility** (ADR-0005): the same `rng` gives the same draws.

## Open Questions

1. **Whether to build it.** Its estimates matched BILOG's (Patz & Junker, 1999a, p. 163; Kim & Bolt,
   2007, p. 46); its advantage is the posterior itself: for highly discriminating items, Albert's
   (1992, p. 262) posterior standard deviations were 20% to 60% larger than the EM standard errors.
   Assumed: last in the build order, revisited after the 4PL card (YAGNI).
2. **The sampler.** Albert's (1992, pp. 258-259) data-augmentation Gibbs sampler and Culpepper's
   (2016, Section 2) need the normal-ogive model, outside ADR-0001's logistic models; Patz and
   Junker's (1999a, p. 159) random-walk Metropolis steps work for the logistic models, tuned to the
   acceptance rates Gelman and colleagues recommend, as Patz and Junker report them: "of about 50%"
   for single parameters, decreasing toward 25% for blocks. Assumed: Patz and Junker's.
3. **Convergence.** Gelman and Rubin (1992, p. 461) monitor until $\hat R$ is "near 1", with no fixed
   threshold. Assumed: several chains, with $\hat R$ reported for every parameter.
4. **The output.** Draws or summaries: decided with this card (ADR-0006).
5. **Missing responses.** Patz and Junker (1999b, pp. 351-352) impute them as an extra Gibbs step;
   the likelihood can also leave them out, as `pyitemcal.mml` does. Assumed: left out.

---

**Module:** the estimation layer of ADR-0006, named by the PR that implements it (planned)
**Reference:** Patz & Junker (1999a)
**Last validated:** 2026-11-27, against Albert (1992), Patz & Junker (1999a, 1999b), Kim & Bolt
(2007), Gelman & Rubin (1992), Loken & Rulison (2010), Culpepper (2016), Sheng (2008) and Cai (2010a)
on the paper shelf, and the mirt 1.48 reference manual.
**Implementation guidance:** not yet written; written against ADR-0006 to ADR-0008 when implemented.

## Motivation

A sampler draws from the joint posterior of the abilities and item parameters, so the uncertainty in
the items carries into what follows, and the posterior of each item curve is available, not only its
mode. Patz and Junker (1999a, p. 149) designed their sampler to be "easier to implement for novel
models" than the samplers before it.

## Definition

Each cycle draws every ability, then every item's parameters, each from its full conditional
distribution given the rest. For the logistic models the conditionals have no standard form, so
each draw is a random-walk Metropolis step: normal proposals for the abilities and difficulties, and
lognormal proposals for the discriminations (Patz & Junker, 1999a, p. 159); the 3PL adds a beta prior
and a proposal for the guessing parameter (Patz & Junker, 1999b, p. 353). Several chains from
dispersed starts are compared by the potential scale reduction, the ratio of the pooled to the
within-chain variance estimate (Gelman & Rubin, 1992, p. 461).

## Parameters

The number of chains, iterations and burn-in, the proposal scales, and the priors (the 3PL card).

## Behavior and Comparison

- **Run times reported:** 7,400 iterations took about 5.7 hours in S-Plus at 400 MHz, a time Patz
  and Junker attribute largely to their unoptimized S-Plus code (Patz & Junker, 1999a, p. 170). For a four-factor model, MCMC took 1 hour 20 minutes against 47 seconds for MH-RM
  (Cai, 2010a, p. 48).
- **Mixing:** 3PL parameters had significant autocorrelations at lags up to 350 (Patz & Junker,
  1999b, p. 353).
- **The posterior against the normal approximation:** Albert (1992, p. 262).

**The Metropolis-Hastings Robbins-Monro algorithm** (MH-RM) imputes the abilities by
Metropolis-Hastings sampling and updates the item parameters by stochastic approximation, for maximum
likelihood rather than posterior summaries (Cai, 2010a, p. 40); its computing time grows linearly
with the number of latent dimensions, against EM's exponential growth (p. 51). For one dimension, EM
was faster: 5 seconds against MH-RM's 10 (pp. 48-49), and EM "can be more efficient" at low dimension
(p. 46). The review therefore gives MH-RM no card for pyitemcal's unidimensional models; mirt's
manual likewise calls EM "generally effective with 1-3 factors" (mirt 1.48 manual, p. 126).

## Guarantees

1. The same `rng` and data give the same draws.
2. Every reported estimate comes with the chains' potential scale reduction.

## References

Albert, J. H. (1992). Bayesian estimation of normal ogive item response curves using Gibbs sampling.
*Journal of Educational Statistics, 17*(3), 251-269. https://doi.org/10.3102/10769986017003251. Not
openly available; Sheng (2008) describes the method.

Cai, L. (2010a). High-dimensional exploratory item factor analysis by a Metropolis-Hastings
Robbins-Monro algorithm. *Psychometrika, 75*(1), 33-57. https://doi.org/10.1007/s11336-009-9136-x.
Open access.

Chalmers, R. P. (2026). *mirt: Multidimensional item response theory* (R package version 1.48)
[Reference manual and source]. https://cran.r-project.org/package=mirt

Culpepper, S. A. (2016). Revisiting the 4-parameter item response model: Bayesian estimation and
application. *Psychometrika, 81*(4), 1142-1163. https://doi.org/10.1007/s11336-015-9477-6. Not
openly available; Zhang, Lu, Du and Zhang (2020) describe the method.

Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences.
*Statistical Science, 7*(4), 457-472. https://doi.org/10.1214/ss/1177011136. Open copy:
https://sites.stat.columbia.edu/gelman/research/published/itsim.pdf

Kim, J.-S., & Bolt, D. M. (2007). Estimating item response theory models using Markov chain Monte
Carlo methods. *Educational Measurement: Issues and Practice, 26*(4), 38-51.
https://doi.org/10.1111/j.1745-3992.2007.00107.x. Open copy:
https://www.ncme.org/wp-content/uploads/2025/10/Module-27-Markov-chain-Monte-Carlo-Estimation-Kim-1.pdf

Loken, E., & Rulison, K. L. (2010). Estimation of a four-parameter item response theory model.
*British Journal of Mathematical and Statistical Psychology, 63*(3), 509-525.
https://doi.org/10.1348/000711009X474502. Open copy (accepted manuscript):
https://web.archive.org/web/20240708235413/https://libres.uncg.edu/ir/uncg/f/K_Rulison_Estimation_2010.pdf
(archived copy of the UNCG repository's file)

Patz, R. J., & Junker, B. W. (1999a). A straightforward approach to Markov chain Monte Carlo methods
for item response models. *Journal of Educational and Behavioral Statistics, 24*(2), 146-178.
https://doi.org/10.3102/10769986024002146. Not openly available; Kim and Bolt (2007) describe the
method.

Patz, R. J., & Junker, B. W. (1999b). Applications and extensions of MCMC in IRT: Multiple item types,
missing data, and rated responses. *Journal of Educational and Behavioral Statistics, 24*(4),
342-366. https://doi.org/10.3102/10769986024004342. Not openly available; Kim and Bolt (2007, p. 42)
describe the Metropolis-within-Gibbs sampling it extends, citing this pair of papers with the a and b
labels swapped.

Sheng, Y. (2008). Markov chain Monte Carlo estimation of normal ogive IRT models in MATLAB. *Journal of
Statistical Software, 25*(8), 1-15. https://doi.org/10.18637/jss.v025.i08. Open access.

Zhang, J., Lu, J., Du, H., & Zhang, Z. (2020). Gibbs-slice sampling algorithm for estimating the
four-parameter logistic model. *Frontiers in Psychology, 11*, 2121.
https://doi.org/10.3389/fpsyg.2020.02121. Open access.

## Implementation Guidance

Follows ADR-0006 to ADR-0008, with the 3PL card's priors. The likelihood of an item given
abilities is the M-step's, so the sampler shares the per-item likelihood code with MML-EM.
