# Item and Person Fit

```{py:module} irtlib.fit
```

Item fit compares, item by item, the responses a calibration predicts with those observed. Both
item statistics take a calibration by {doc}`/methods/calibration/mml`, whose grid and population
they use, and complete responses ({doc}`ADR-0006 </adr/0006-calibrator-contract>`). Person fit
compares, examinee by examinee, the responses with their expectations at the examinee's ability,
from the responses, the items and the abilities, with missing responses left out.

| Statistic | Models | Groups the examinees by | Result |
|---|---|---|---|
| {doc}`s-x2` | 1PL to 4PL | number-correct score | {class}`~irtlib.fit.ItemFit`: a $\chi^2$ statistic, its degrees of freedom and $p$-value |
| {doc}`mean-squares` of the items | 1PL | not grouped: each response against its expectation | {class}`~irtlib.fit.MeanSquares`: the infit and outfit, and their standardized values |
| {doc}`mean-squares` of the examinees | 1PL to 4PL | not grouped: each response against its expectation | {class}`~irtlib.fit.MeanSquares`, for each examinee |

```{toctree}
:maxdepth: 1

s-x2
mean-squares
```
