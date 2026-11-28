# S-X²

:API: {py:func}`irtlib.fit.s_x2`; {py:class}`irtlib.fit.ItemFit`
:Sources: {cite:t}`Orlando2000`

## Motivation

Orlando and Thissen's S-X² tests, item by item, whether a calibrated model predicts the responses
observed. It groups examinees by their number-correct score, which is observed, rather than by an
ability estimate, whose groups depend on the model {cite:p}`Orlando2000{p. 52}`, and it applies to
every model from the 1PL to the 4PL. For the Rasch model, {doc}`mean-squares` compare each response
with its expectation instead.

## Definition

For item $i$ of $M$ and score $k = 1, \dots, M - 1$, with $N_k$ examinees at score $k$ and $O_{ik}$
the observed proportion of them answering item $i$ correctly,

$$
S\text{-}X^2_i = \sum_{k=1}^{M-1} N_k \frac{(O_{ik} - E_{ik})^2}{E_{ik}(1 - E_{ik})}
$$

{cite:p}`Orlando2000{p. 54, Equation 13}`. The expected proportion is

$$
E_{ik} = \frac{\sum_q P_i(X_q)\, S^{*i}_{k-1}(X_q)\, A_q}{\sum_q S_k(X_q)\, A_q}
$$

(p. 54, Equation 12), where $S_k(X_q)$ is the likelihood of a score of $k$ at ability $X_q$ and
$S^{*i}_k$ the same without item $i$, both from Lord and Wingersky's recursion over items (p. 53,
Equations 6-11): adding an item with probability $T$ turns $S^*_k$ into
$T S^*_{k-1} + (1 - T) S^*_k$. The statistic sums $(O - E)^2 / E$ over the cells of the table of
scores by responses, and its degrees of freedom are the rows of the table less the number of
parameters estimated for the item (p. 56).

## Implementation

**The grid.** The nodes and their normal masses $A_q$ are the calibration's grid, so S-X² takes a
calibration by {doc}`/methods/calibration/mml` ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

**Collapsing the table.** Orlando and Thissen collapse the rows from both ends toward the middle until
every expected count is at least 1 {cite:p}`Orlando2000{p. 56}`. irtlib collapses the two cells of
each score row, incorrect and correct: rows no examinee reached are dropped, a row
with one examinee is merged into the next, the last into the one before, and while an expected count
is below `min_expected`, the first such row is merged into its neighbour, the first row into the
second, the last into the one before and any other into the neighbour with fewer examinees. The
degrees of freedom are floored at 0. An item whose collapsed table leaves no degrees of freedom,
including one collapsed to a single row, has none and no statistic.

**Complete responses.** S-X² "cannot handle incomplete data directly"
{cite:p}`Zhang2022{p. 705}`, so it needs complete responses
({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

:::{dropdown} Comparison with mirt

- The collapsing is mirt 1.48's (`R/utils.R:2410-2546`); the parity tests check the collapsed
  expected counts against mirt's within a relative $10^{-13}$.
- At mirt's estimates, the statistics and $p$-values agree with mirt's within a relative $10^{-10}$;
  on irtlib's own calibration, within a relative $10^{-4}$
  ({doc}`ADR-0003 </adr/0003-mirt-as-the-reference>`).
- mirt skips an item whose table collapses to a single row; irtlib reports it with no degrees of
  freedom and no statistic.
:::

## Parameters

```{list-table}
:header-rows: 1
:widths: 18 14 42 26

* - Parameter
  - Default
  - Description
  - Default from
* - `min_expected`
  - `1.0`
  - The smallest expected count a cell of the collapsed table may hold.
  - {cite:t}`Orlando2000{p. 56}`
```

## Guarantees

1. Every item of the calibration gets a statistic, the items its bank leaves out included, unless its
   collapsed table leaves no degrees of freedom.
2. A missing response, or a calibration whose likelihood is not marginal, raises an error.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autofunction:: irtlib.fit.s_x2

.. autoclass:: irtlib.fit.ItemFit
```
