# Classical Item Statistics

:API: {py:mod}`irtlib.classical`
:Sources: {cite:t}`Wright1979`; {cite:t}`Baker1987`; {cite:t}`Haberman2004`; {cite:t}`Robitzsch2021`

## Motivation

Two statistics of classical test theory screen the responses before a calibration, item by item,
with no model fitted. The proportion correct, the item's "p-value", says how easy the item was for
the sample, and the correlation of the item with the examinees' test scores, the point-biserial
correlation, indexes how well it fits with the other items of the test. Wright and Stone call the
first the most familiar traditional item statistic, and the second the next most widely used
{cite:p}`Wright1979{pp. 25-26}`. Both depend on the sample: the proportion correct on the mean and
variance of its abilities (p. 25), and the point-biserial correlation on the spread and level of its
scores, falling toward 0 as the proportion correct nears 0 or 1; no size of it marks an item as
acceptable (p. 26).

irtlib correlates each item with the rest score, the examinee's number correct on the other items:
the point-biserial correlation of the item and the item-excluded test score
{cite:p}`Baker1987{p. 123}`.

## Definition

For item $i$, answered by $N_i$ examinees of whom $s_i$ answered it correctly, the proportion correct
is

$$
p_i = s_i / N_i
$$

{cite:p}`Wright1979{p. 25}`. Examinee $j$'s rest score for item $i$ is the sum of the examinee's
responses to the other items, $R_{ij} = \sum_{k \ne i} u_{kj}$, the score $S_p - X_{pi}$ of
{cite:t}`Robitzsch2021{p. 819}`. The item-rest correlation is the point-biserial correlation of the
responses $u_{ij}$ with the rest scores, by Haberman's standard estimate of the point-biserial
correlation {cite:p}`Haberman2004{p. 47}`, with the rest score in place of the test score:

$$
r_i = \frac{\sum_j u_{ij} R_{ij} - s_i \bar R_i}
{\left[ s_i (1 - s_i / N_i) \sum_j (R_{ij} - \bar R_i)^2 \right]^{1/2}},
$$

where the sums run over the $N_i$ examinees and $\bar R_i$ is their mean rest score. It is Pearson's
correlation of the two.

## Implementation

**Missing responses.** Wright and Stone's and Haberman's sums run over every examinee, with no rule
for a missing response. irtlib takes each item's statistics over the $N_i$ examinees who answered it,
and each examinee's rest score over the other items the examinee answered, with no source on the
shelf for that rule; the
[sources wanted](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/discovery/sources-wanted.md)
list it.

**Statistics with no value** are NaN: both statistics of an item nobody answered, and the
correlation of an item that takes one value among the examinees who answered it, or whose rest score
does, which includes a test of one item and a single examinee.

:::{dropdown} Comparison with mirt

- mirt 1.48's `itemstats` computes the same statistics, its columns `mean` and `cor_if_rm`, with the
  same rule for missing responses: each item's mean over the examinees who answered it, and the rest
  score as the sum of the other answered items, correlated over the examinees who answered the item
  (`R/itemstats.R:110-120, 148`). Its `empirical_plot` draws them, types `"difficulty"`,
  `"discrim"` and `"discrim_diff"` (manual p. 54).
- On LSAT6, LSAT7 and a simulated data set with a tenth of its responses missing, the parity tests
  find both statistics within a relative $10^{-10}$ of mirt's
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
:::

## Parameters

{func}`~irtlib.classical.proportion_correct` and {func}`~irtlib.classical.item_rest_correlation`
take the response matrix, and have no settings.

## Guarantees

1. Each item gets one value, from the examinees who answered it; missing responses raise no error.
2. A statistic with no value is NaN, without a warning, and leaves the other items' values as they
   are.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. automodule:: irtlib.classical
   :members:
```
