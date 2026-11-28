# Mean Squares

:API: {py:func}`irtlib.fit.mean_squares`; {py:class}`irtlib.fit.MeanSquares`
:Sources: {cite:t}`Wright1982`

## Motivation

The weighted and unweighted mean squares compare each response with its expectation at the
examinee's estimated ability {cite:p}`Wright1982{pp. 97-99}`, summed item by item: the infit and the
outfit of the Rasch model. Wright and Masters define them for the models of their book, all built
from the Rasch model (p. 10; p. 100, Table 5.4a), whose model for dichotomous responses (p. 38,
Equation 3.1.1) is the 1PL, every item with the same slope $a$, with person parameters $a\theta_j$
and item parameters $ab_i$. For the other models, {doc}`s-x2` tests item fit.

## Definition

For examinee $j$ and item $i$, with $P_{ij}$ the probability of a correct response at the examinee's
ability and $Q_{ij} = 1 - P_{ij}$, the score residual $y_{ij} = u_{ij} - P_{ij}$
{cite:p}`Wright1982{p. 98, Equation 5.4.2}` has variance $W_{ij}$ and kurtosis $C_{ij}$
(p. 98, Equations 5.4.3 and 5.4.4), which for the two categories of a dichotomous item are

$$
W_{ij} = P_{ij} Q_{ij}, \quad C_{ij} = P_{ij} Q_{ij} (P_{ij}^3 + Q_{ij}^3),
$$

and the standardized residual is $z_{ij} = y_{ij} / W_{ij}^{1/2}$ (p. 99, Equation 5.4.5). Over the
$N$ examinees, the unweighted and the weighted mean squares, which Wright and Masters write $u_i$ and
$v_i$, are

$$
U_i = \frac{1}{N} \sum_j z_{ij}^2, \quad
V_i = \frac{\sum_j W_{ij} z_{ij}^2}{\sum_j W_{ij}} = \frac{\sum_j y_{ij}^2}{\sum_j W_{ij}}
$$

(p. 99, Equations 5.4.6 and 5.4.7). irtlib calls $U_i$ the outfit and $V_i$ the infit. $U_i$ is
sensitive to unexpected responses of examinees for whom the item is far too easy or far too hard, and
the weights $W_{ij}$, smallest for those examinees, lessen their influence on $V_i$ (pp. 99, 101).
When the data fit the model both have expectation 1, $U_i$ with variance
$\sum_j (C_{ij} / W_{ij}^2) / N^2 - 1 / N$ (p. 100, Table 5.4a) and $V_i$ with variance

$$
q_i^2 = \frac{\sum_j (C_{ij} - W_{ij}^2)}{\left(\sum_j W_{ij}\right)^2}
$$

(p. 101, Equation 5.4.8). The weighted mean square is standardized to

$$
t_i = (V_i^{1/3} - 1)(3 / q_i) + q_i / 3,
$$

which has a mean near 0 and a standard deviation near 1 when the data fit (p. 101; p. 100,
Table 5.4a). Wright and Masters estimate the person and item parameters from the same data as the
residuals, which lowers the expected mean square by the factor $(M - 1)(N - 1) / (MN)$ for $M$ items
(p. 101).

## Implementation

**The abilities** are the EAP estimates of {func}`irtlib.ability.expected_a_posteriori` on the
calibration's grid, abilities normal with mean 0 and variance 1
({doc}`ADR-0006 </adr/0006-calibrator-contract>`), so the mean squares take a calibration of the 1PL
by {doc}`/methods/calibration/mml`.

**Three choices no source on the shelf covers:**

- the unweighted mean square is standardized in the same way as the weighted one, with the standard
  deviation from Table 5.4a in place of $q_i$;
- a variance $W_{ij}$ whose square is below $10^{-5}$ is raised to $10^{-5/2}$, leaving $C_{ij}$ and
  $y_{ij}$ as they are, so that a mean square's variance can fall below 0, and its $t$ is then NaN;
- the standard deviation of either mean square is capped at 1.4142 before standardizing.

The [sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list all three.

**Complete responses.** Wright and Masters' sums run over every examinee, with no rule for a missing
response, so irtlib's mean squares need complete responses
({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

:::{dropdown} Comparison with mirt

- mirt 1.48 calls $U_i$ and $V_i$ outfit and infit too (`R/itemfit.R:584, 588`), and takes EAP
  estimates by default (`R/itemfit.R:268, 535-538`), on the grid of its `fscores`.
- irtlib's three choices above are mirt's: the standardized outfit (`R/itemfit.R:585, 587`), the
  floor on $W_{ij}$ (`R/personfit.R:189-200`) and the cap on the standard deviation
  (`R/itemfit.R:586, 591`).
- mirt computes the mean squares of any model from its probabilities (`R/personfit.R:181-194`), and
  with missing responses sums over the examinees who answered each item, with $N$ their number
  (`R/itemfit.R:577-584`).
- At mirt's estimates and abilities the parity tests find the mean squares and their standardized
  values within a relative $10^{-10}$; on irtlib's own calibration, the mean squares within a relative
  $10^{-4}$ and the standardized values within $10^{-3}$
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

{func}`~irtlib.fit.mean_squares` takes the calibration and the responses it was calibrated on, and
has no settings.

## Guarantees

1. Each item gets its outfit and infit, and their standardized values; a standardized value is NaN
   where its mean square's variance is not positive.
2. A missing response, a calibration whose likelihood is not marginal, or a calibration of a model
   other than the 1PL raises an error.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.fit.mean_squares

.. autoclass:: irtlib.fit.MeanSquares
```
