# Ability Estimation

```{py:module} irtlib.ability
```

Ability estimation gives each examinee's ability, with its standard error, from its responses to
items of known parameters. Each estimator is a function that takes a response matrix with one row
per examinee and one column per item, holding 1, 0 or NaN where the examinee did not answer the item,
and the items, one row `[a, b, c, d]` per column, with its settings as keyword arguments
({doc}`ADR-0010 </adr/0010-ability-estimation-contract>`). It returns an
{class}`~irtlib.ability.AbilityEstimates`: for each examinee an estimate, its standard error and the
number of evaluations of the estimator's objective or of its derivatives. The
{doc}`number-correct-eap` takes the items alone and returns one estimate per number correct, which
every examinee who answered every item reads by its number correct.

An examinee's estimate depends only on the items it answered, so its row gives the same estimate
alone as in any matrix: exactly where it answered every item, and otherwise to the rounding of a sum
over more terms, which can also move a search's last step and change its evaluation count by one,
since the maximum likelihood and weighted likelihood searches, and MAP's under a normal prior, search
every row of a matrix at once with the arithmetic of one row. On these pages, $L_j(\theta)$ is the
likelihood of examinee $j$'s responses $u_{ij}$, the product of $P_i^{u_{ij}} Q_i^{1 - u_{ij}}$ over
the items $i$ it answered, and $I(\theta)$ the test information of those items, both as
{doc}`/methods/models/item-response-functions` defines them.

| Estimator | Standard error | Responses all correct or all incorrect | No item answered |
|---|---|---|---|
| {doc}`ml` | $1/\sqrt{I(\hat\theta)}$ | $\pm\infty$, or the bound they point to | NaN |
| {doc}`map` | $1/\sqrt{I(\hat\theta) - (\log\pi)''(\hat\theta)}$ | finite, within the bounds | the prior's mode |
| {doc}`eap` | the posterior standard deviation | finite | the prior's mean |
| {doc}`wle` | $1/\sqrt{I(\hat\theta)}$ | finite, within the bounds | NaN |
| {doc}`number-correct-eap` | the posterior standard deviation given the number correct | finite | the prior's mean |

```{eval-rst}
.. autoclass:: irtlib.ability.AbilityEstimates
```

```{toctree}
:maxdepth: 1

ml
map
eap
wle
number-correct-eap
```
