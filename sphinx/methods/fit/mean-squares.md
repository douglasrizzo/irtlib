# Mean Squares

:API: {py:func}`irtlib.fit.mean_squares`; {py:func}`irtlib.fit.person_mean_squares`; {py:class}`irtlib.fit.MeanSquares`
:Sources: {cite:t}`Wright1982`; {cite:t}`Wright1979`

## Motivation

The weighted and unweighted mean squares compare each response with its expectation at the
examinee's estimated ability {cite:p}`Wright1982{pp. 97-99}`, summed item by item: the infit and the
outfit of the Rasch model. Wright and Masters define them for the models of their book, all built
from the Rasch model (p. 10; p. 100, Table 5.4a), whose model for dichotomous responses (p. 38,
Equation 3.1.1) is the 1PL, every item with the same slope $a$, with person parameters $a\theta_j$
and item parameters $ab_i$. For the other models, {doc}`s-x2` tests item fit.

Summed examinee by examinee instead, over the items each examinee answered, the same residuals give
each examinee's mean squares: person fit, which flags the examinees whose response patterns the model
does not explain {cite:p}`Wright1982{p. 108}`. Item fit serves the construction and calibration of a
test, and person fit the validity of the measures a calibrated test makes (p. 109).

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

**Person fit.** The examinees' mean squares "parallel the corresponding item fit statistics exactly",
the squared residuals summed over the items for one examinee rather than over the examinees for one
item {cite:p}`Wright1982{p. 109}`. Over the $M_j$ items examinee $j$ answered, the weighted mean
square, its variance and its standardized value are

$$
V_j = \frac{\sum_i y_{ij}^2}{\sum_i W_{ij}}, \quad
q_j^2 = \frac{\sum_i (C_{ij} - W_{ij}^2)}{\left(\sum_i W_{ij}\right)^2}, \quad
t_j = (V_j^{1/3} - 1)(3 / q_j) + q_j / 3
$$

(pp. 108-109, Equations 5.7.6-5.7.8), and the unweighted mean square $U_j = \sum_i z_{ij}^2 / M_j$
has the variance $\sum_i (C_{ij} / W_{ij}^2) / M_j^2 - 1 / M_j$ of Table 5.4a (p. 100), with the items
in place of the examinees. {cite:t}`Wright1979{p. 77, Equations 4.4.1-4.4.2}` divide an examinee's sum
of squared standardized residuals by $M - 1$ instead, one less than the number of items, and
standardize it by a logarithm; irtlib follows Wright and Masters.

## Implementation

**The abilities.** The item mean squares take EAP estimates of
{func}`irtlib.ability.expected_a_posteriori` on the calibration's grid, abilities normal with mean 0
and variance 1 ({doc}`ADR-0006 </adr/0006-calibrator-contract>`), so they take a calibration of the
1PL by {doc}`/methods/calibration/mml`. The examinees' mean squares take the abilities from the caller,
estimated by any of the {doc}`/methods/ability/index`.

**The models.** The item mean squares take the 1PL, the model of Wright and Masters' book. The
examinees' mean squares take items of any model, 1PL to 4PL: $W_{ij}$ and $C_{ij}$ are written in terms of
the model's probabilities (p. 108, Equations 5.7.1-5.7.3), but no source on the shelf applies the mean
squares beyond the Rasch model.

**Missing responses.** Wright and Masters' sums run over every examinee and every item, with no rule
for a missing response. The item mean squares need complete responses
({doc}`ADR-0006 </adr/0006-calibrator-contract>`). The examinees' mean squares sum over the items each
examinee answered, $M_j$ of them, with no source on the shelf for that rule; an examinee who answered
no item, or whose ability is infinite or NaN, has none of the four values.

**Three choices no source on the shelf covers:**

- the unweighted mean square is standardized in the same way as the weighted one, with the standard
  deviation from Table 5.4a in place of $q_i$;
- a variance $W_{ij}$ whose square is below $10^{-5}$ is raised to $10^{-5/2}$, leaving $C_{ij}$ and
  $y_{ij}$ as they are, so that a mean square's variance can fall below 0, and its $t$ is then NaN;
- the standard deviation of either mean square is capped at 1.4142 before standardizing.

The [sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list these three, the examinees' mean squares of the 2PL to 4PL, and their sums over the answered items.

:::{dropdown} Comparison with mirt

- mirt 1.48 calls $U$ and $V$ outfit and infit too (`R/itemfit.R:584, 588`), and takes EAP
  estimates by default (`R/itemfit.R:268, 535-538`), on the grid of its `fscores`.
- irtlib's three choices above are mirt's: the standardized outfit (`R/itemfit.R:585, 587`), the
  floor on $W_{ij}$ (`R/personfit.R:189-200`) and the cap on the standard deviation
  (`R/itemfit.R:586, 591`; `R/personfit.R:205, 209`).
- mirt computes the mean squares of any model from its probabilities (`R/personfit.R:181-194`), and
  with missing responses sums over the examinees who answered each item, with $N$ their number
  (`R/itemfit.R:577-584`), and over the items each examinee answered, with $M_j$ their number
  (`R/personfit.R:195, 202-210`).
- Three departures, in the examinees' mean squares only. With missing responses, mirt divides the
  first term of the outfit's variance by the square of the number of all items, $M^2$, where irtlib
  divides it by $M_j^2$ (`R/personfit.R:204`). Where the floor on $W_{ij}$ leaves that variance
  negative, mirt takes its absolute value (`R/personfit.R:204`), and irtlib gives NaN. mirt replaces
  an infinite ability by the largest finite one plus 0.1, or the smallest less 0.1
  (`R/personfit.R:146-151`), and irtlib gives that examinee NaN.
- At mirt's estimates and abilities the parity tests find the item mean squares and their
  standardized values within a relative $10^{-10}$; on irtlib's own calibration, the mean squares
  within a relative $10^{-4}$ and the standardized values within $10^{-3}$
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`). At the same abilities, on 2PL, 3PL and 4PL
  items with responses complete and missing, they find mirt's mean squares of the examinees within a relative
  $10^{-10}$, and their standardized values too, except the standardized outfit with missing
  responses.
:::

## Parameters

{func}`~irtlib.fit.mean_squares` takes the calibration and the responses it was calibrated on, and
{func}`~irtlib.fit.person_mean_squares` the responses, the items and each examinee's ability. Neither
has settings.

## Guarantees

1. Each item, or each examinee, gets its outfit and infit, and their standardized values; a
   standardized value is NaN where its mean square's variance is not positive.
2. For the item mean squares, a missing response, a calibration whose likelihood is not marginal, or a
   calibration of a model other than the 1PL raises an error.
3. The mean squares of an examinee leave out the examinee's missing responses, and are NaN when
   the examinee answered no item or the ability is infinite or NaN, without a warning.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.fit.mean_squares

.. autofunction:: irtlib.fit.person_mean_squares

.. autoclass:: irtlib.fit.MeanSquares
```
