# Item Fit

```{py:module} irtlib.fit
```

Item fit compares, item by item, the responses a calibration predicts with those observed. Both
statistics take a calibration by {doc}`/methods/calibration/mml`, whose grid and population they
use, and complete responses ({doc}`ADR-0006 </adr/0006-calibrator-contract>`).

| Statistic | Models | Groups the examinees by | Result |
|---|---|---|---|
| {doc}`s-x2` | 1PL to 4PL | number-correct score | {class}`~irtlib.fit.ItemFit`: a $\chi^2$ statistic, its degrees of freedom and $p$-value, and the observed and expected proportions correct at each number correct |
| {doc}`mean-squares` | 1PL | not grouped: each response against its expectation | {class}`~irtlib.fit.MeanSquares`: the infit and outfit, and their standardized values |

```{toctree}
:maxdepth: 1

s-x2
mean-squares
```
