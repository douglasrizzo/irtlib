# Terms and Notation

The irtlib documentation uses one term for each concept. This page lists each term with the synonyms
you may know from the literature and the name to look for in the API, then the symbols that the
formulas use.

## Examinees and Items

examinee
: The person who answers a test's items. Also called *respondent*, *subject* or *person*. In the API
  an examinee is a row of the response matrix.

ability $\theta$
: The characteristic a test measures, on a continuous scale. Also called *proficiency*, *latent
  trait* or *person parameter*. In the API: `theta`.

ability estimate $\hat{\theta}$
: The estimate of an examinee's ability from its responses. In the API: the `theta` of an
  {class}`~irtlib.ability.AbilityEstimates`.

item
: One question of a test, scored correct or incorrect. In the API an item is a column of the
  response matrix and a row $[a, b, c, d]$ of an item matrix, `items`; in an item bank it also has a
  *name*, unique in its bank (`name`).

item bank
: The calibrated items of a test, with their names. Also called *item pool*. In the API:
  {class}`irtlib.ItemBank`, which holds $M$ items, `n_items`, and their `names`.

logistic models
: The item response models irtlib implements: the one-, two-, three- and four-parameter logistic
  models (1PL to 4PL). The 1PL with every discrimination equal to 1 is the *Rasch model*. In the API:
  {class}`irtlib.IrtModel`, and an item bank's `model`.

discrimination $a_i$
: How sharply item $i$ separates examinees of lower and higher ability: the slope of its
  characteristic curve. Also called the *slope parameter*. `ItemBank.discrimination`.

difficulty $b_i$
: The ability at which item $i$'s characteristic curve is steepest, on the same scale as $\theta$.
  Also called the *location* or *threshold*. `ItemBank.difficulty`.

guessing parameter $c_i$
: The lower asymptote of item $i$'s characteristic curve: the probability that an examinee of very
  low ability answers correctly. Also called the *pseudo-guessing parameter*.
  `ItemBank.pseudo_guessing`; 0 under the 1PL and 2PL.

upper asymptote $d_i$
: The upper asymptote of item $i$'s characteristic curve: the probability that an examinee of very
  high ability answers correctly. $1 - d_i$ is called the *slipping* or *carelessness* parameter.
  `ItemBank.upper_asymptote`; 1 under every model but the 4PL.

item characteristic curve $P_i(\theta)$
: The probability of a correct response to item $i$ at ability $\theta$. Also called the *item
  response function* (IRF). {func}`irtlib.irt.icc`.

item information $I_i(\theta)$
: The Fisher information of a response to item $i$ about the ability at $\theta$.
  {func}`irtlib.irt.information`.

test information $I(\theta)$
: The sum of the information of a set of items. {func}`irtlib.irt.total_information`.

standard error $SE(\hat{\theta})$
: The uncertainty of an ability estimate, on the scale of $\theta$. Also called the *standard error
  of estimation* (SEE) or *standard error of measurement* (SEM). Each estimator computes it by its
  own formula; for maximum likelihood it is $1/\sqrt{I(\hat{\theta})}$. In the API: the
  `standard_errors` of an {class}`~irtlib.ability.AbilityEstimates`, and {func}`irtlib.irt.see`.

## Responses

response $u_{ij}$
: Examinee $j$'s response to item $i$: 1 if correct, 0 if not.

response matrix
: The responses of a sample of examinees, with one row per examinee and one column per item, and NaN
  where an examinee did not answer an item. Also called the *data matrix*. In the API: `responses`.
  The responses are *complete* when no entry is missing.

number correct
: The number of items an examinee answered correctly. Also called the *raw score* or *number-correct
  score*.

rest score
: An examinee's number correct on the items other than the one in question.

proportion correct
: The share of the examinees who answered an item that answered it correctly. Also called the
  item's *p-value*. In the API: {func}`~irtlib.classical.proportion_correct`.

item-rest correlation
: The correlation of an item's responses with the examinees' rest scores, a point-biserial
  correlation. In the API: {func}`~irtlib.classical.item_rest_correlation`.

## Calibration

calibration
: Estimating the items' parameters from a response matrix. Also called *item parameter estimation*.
  These pages never say "estimation" alone where calibration and ability estimation could be
  confused.

calibrator
: An object that calibrates items by one method: {class}`~irtlib.mml.MMLCalibrator`,
  {class}`~irtlib.cml.CMLCalibrator` or {class}`~irtlib.jml.JMLCalibrator`. Its `calibrate` method
  returns a {class}`~irtlib.calibration.Calibration`.

marginal, conditional and joint likelihood
: The likelihoods the three calibration methods maximize: with the abilities integrated out over a
  population distribution (marginal), given each examinee's number correct (conditional), or given
  every examinee's ability, estimated with the items (joint). The methods are *marginal maximum
  likelihood* (MML), *conditional maximum likelihood* (CML) and *joint maximum likelihood* (JML).
  In the API: {class}`~irtlib.calibration.Likelihood`.

quadrature grid
: The abilities that calibration and ability estimation integrate over, the *nodes* $X_q$,
  $q = 1, \dots, Q$, each with a weight. Also called *quadrature points*. In the API:
  {class}`~irtlib.quadrature.QuadratureGrid`.

ability prior
: A density of ability: the population distribution that marginal maximum likelihood integrates
  over, or the prior of a Bayesian ability estimate. Its *mass* $A_q$ at node $X_q$ is proportional
  to the node's weight times the density there, and the masses sum to 1. In the API:
  {func}`~irtlib.quadrature.normal_log_prior`, {func}`~irtlib.quadrature.prior_masses`.

item priors
: Priors on the item parameters, under which the calibrated parameters are *Bayes modal* estimates:
  they maximize the likelihood times the priors. In the API: {class}`~irtlib.mml.Priors`.

cycle
: One round of a calibration's algorithm, such as an EM cycle. In the API: the `cycles` of a
  {class}`~irtlib.calibration.Calibration`; `converged` says whether its stopping rule was met.

## Ability Estimation

ability estimation
: Estimating each examinee's ability from its responses to items of known parameters.

estimator
: A method of ability estimation: maximum likelihood (ML), maximum a posteriori (MAP, also called
  *Bayes modal*), expected a posteriori (EAP) or Warm's weighted likelihood (WLE). In the API: the
  functions of {mod}`irtlib.ability`.

posterior
: An examinee's distribution of ability given its responses: proportional to the prior times the
  likelihood. In the API: {func}`~irtlib.quadrature.posterior`, over the nodes of a grid.

## Checking a Calibration

item fit
: How well the responses a calibration predicts for an item match those observed. In the API:
  {func}`~irtlib.fit.s_x2` and {func}`~irtlib.fit.mean_squares`.

person fit
: How well an examinee's responses match their expectations at the examinee's ability. In the API:
  {func}`~irtlib.fit.person_mean_squares`.

model comparison
: Whether a model with more parameters fits enough better to justify them. In the API: the
  functions of {mod}`irtlib.comparison`.

## Notation

| Symbol | Meaning | In the API |
|---|---|---|
| $\theta$ | ability | `theta` |
| $\hat{\theta}$ | ability estimate | `AbilityEstimates.theta` |
| $i$ | item index, a column of the response matrix | `item` |
| $j$ | examinee index, a row of the response matrix | |
| $M$ | number of items | `ItemBank.n_items` |
| $N$ | number of examinees | |
| $M_j$ | number of items examinee $j$ answered | |
| $a_i$, $b_i$, $c_i$, $d_i$ | discrimination, difficulty, guessing parameter and upper asymptote of item $i$ | `discrimination`, `difficulty`, `pseudo_guessing`, `upper_asymptote` |
| $P_i(\theta)$ | probability of a correct response to item $i$ | {func}`irtlib.irt.icc` |
| $Q_i(\theta)$ | $1 - P_i(\theta)$, probability of an incorrect response | |
| $I_i(\theta)$ | information of item $i$ | {func}`irtlib.irt.information` |
| $I(\theta)$ | test information, $\sum_i I_i(\theta)$ | {func}`irtlib.irt.total_information` |
| $SE(\hat{\theta})$ | standard error of the ability estimate | `standard_errors` |
| $u_{ij}$ | examinee $j$'s response to item $i$: 1 if correct, 0 if not | an entry of `responses` |
| $N_i$, $s_i$ | number of examinees who answered item $i$, and number who answered it correctly | |
| $p_i$ | proportion correct of item $i$, $s_i / N_i$ | {func}`~irtlib.classical.proportion_correct` |
| $R_{ij}$ | examinee $j$'s rest score for item $i$ | |
| $r_i$ | item-rest correlation of item $i$ | {func}`~irtlib.classical.item_rest_correlation` |
| $L_j(\theta)$ | likelihood of examinee $j$'s responses at ability $\theta$ | |
| $X_q$ | node $q$ of the quadrature grid | `QuadratureGrid.nodes` |
| $A_q$ | prior mass at node $X_q$ | {func}`~irtlib.quadrature.prior_masses` |
| $Q$ | number of nodes | |

$Q$ alone is the number of nodes, and $Q_i(\theta)$, with its subscript, the probability of an
incorrect response. $I$ always denotes information, never a set of items.
