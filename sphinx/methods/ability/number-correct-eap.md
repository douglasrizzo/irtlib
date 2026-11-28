# EAP by Number Correct

:API: {py:func}`irtlib.ability.number_correct_eap`; {py:class}`irtlib.ability.NumberCorrectEstimates`
:Sources: {cite:t}`Thissen1995`; {cite:t}`Bock1982`

## Motivation

A test scored by its number correct reports one ability estimate for each number correct: a
conversion table that every examinee with the same number correct shares, whichever items they
answered correctly. The EAP by number correct is the mean of the ability's posterior given the
number correct alone, with the posterior standard deviation as its standard error
{cite:p}`Thissen1995{p. 40}`. It gives up a little precision against the {doc}`eap` of each response
pattern: in Thissen et al.'s example, the posterior standard deviation given a number correct was
about 10% above the smallest given a response pattern with that number (p. 45).

The same computation gives the probability of each number correct, the distribution of number
correct that the items and the ability distribution predict. Set against the observed distribution,
it checks the fit of the model (p. 41).

## Definition

Let $S_k(\theta)$ be the probability of $k$ correct out of the $M$ items at ability $\theta$, which
{doc}`/methods/models/item-response-functions` defines, and $\phi(\theta)$ the prior density of
ability. The probability of a number correct of $k$ is

$$
P_k = \int S_k(\theta)\, \phi(\theta)\, d\theta
$$

{cite:p}`Thissen1995{p. 40, Equations 4-6}`, and its EAP estimate and posterior standard deviation are

$$
\mathrm{EAP}(\theta \mid k) = \frac{1}{P_k} \int \theta\, S_k(\theta)\, \phi(\theta)\, d\theta,
\qquad
\mathrm{SD}(\theta \mid k) = \left\{ \frac{1}{P_k} \int [\theta - \mathrm{EAP}(\theta \mid k)]^2\,
S_k(\theta)\, \phi(\theta)\, d\theta \right\}^{1/2}
$$

(p. 40, Equations 7 and 8). Thissen et al.'s likelihood of a number correct, $L_k(\theta)$, holds
the prior density: it is $S_k(\theta) \phi(\theta)$ (p. 43, Equation 17). These are the EAP estimate
and posterior standard deviation of {cite:t}`Bock1982{p. 433, Equations 4 and 5}`, with the
probability of the number correct in place of the likelihood of the response pattern
{cite:p}`Thissen1995{p. 40}`.

## Implementation

**The grid.** The integrals are sums over the nodes $X_q$ of a
{doc}`/methods/models/quadrature` grid, with the prior's masses $A_q$:
$P_k = \sum_q S_k(X_q) A_q$, and node $X_q$ has posterior mass $S_k(X_q) A_q / P_k$ given $k$.
Thissen et al. integrate by rectangular quadrature too (p. 43). The default grid and prior are those
of {doc}`eap`: 121 nodes on $[-6, 6]$, 0.1 apart, and the standard normal, with that page's
measurements of the grid's accuracy.

**The table.** Every number correct from 0 to $M$ has an estimate, both ends included: the prior
keeps them finite. The table depends on the items and the prior alone, so it takes no response
matrix ({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`): an examinee who answered every
item reads the entry of its number correct. An examinee who skipped items reads the table of the
items it answered.

:::{dropdown} Comparison with mirt

- mirt 1.48's `fscores(method = "EAPsum", full.scores = FALSE)` (manual p. 74) returns the same
  table: the estimate, its standard error, and the expected number of examinees with each number
  correct, $N P_k$ (`R/fscores.internal.R:777-1044`). Its default grid and prior are irtlib's, 121
  nodes on $[-6, 6]$ (`R/fscores.R:302, 318-319`) with the normal density at the nodes, normalized
  (`R/fscores.internal.R:842-847`).
- The parity tests first check that mirt's recorded estimates, standard errors and expected counts
  are the sums above, written out over every response pattern of a 2PL bank and a bank of 3PL and
  4PL items, then compare irtlib's with them within a relative $10^{-10}$
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 18 38 26

* - Parameter
  - Default
  - Description
  - Default from
* - `grid`
  - 121 nodes on $[-6, 6]$
  - The nodes the posterior is computed on.
  - mirt's (`R/fscores.R:302, 318-319`), as for {doc}`eap`
* - `log_prior`
  - the standard normal
  - The prior's log-density.
  - mirt's (`R/fscores.internal.R:846-847`); abilities $N(0, 1)$
    ({doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`)
```

## Guarantees

1. Every number correct from 0 to $M$ has an estimate within the grid's nodes, and the
   probabilities of the numbers correct sum to 1.
2. Among examinees whose abilities follow the prior and who answer every item, the estimate of a
   number correct is the mean ability of those with that number correct, and its probability their
   share, within the grid's accuracy.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.ability.number_correct_eap

.. autoclass:: irtlib.ability.NumberCorrectEstimates
```
