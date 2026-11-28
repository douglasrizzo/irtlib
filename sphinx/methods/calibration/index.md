# Calibration

Calibration estimates the items' parameters, with the covariance of each item's estimates and their
standard errors, from a response matrix.
irtlib has three methods, each a calibrator configured at construction whose `calibrate` method
returns a {class}`~irtlib.calibration.Calibration`
({doc}`ADR-0006 </adr/0006-calibrator-contract>`). They differ in what they do with the examinees'
abilities, which are unknown too:

| Method | Models | The abilities | Missing responses | Scale |
|---|---|---|---|---|
| {doc}`mml` | 1PL to 4PL, with optional priors | integrated out over a normal population | taken | abilities $N(0, 1)$ |
| {doc}`cml` | Rasch | conditioned out, on each number correct | refused | difficulties summing to 0 |
| {doc}`jml` | Rasch, 2PL | estimated with the items, and returned | refused | Rasch: difficulties with mean 0; 2PL: abilities with mean 0 and standard deviation 1 |

The scales follow {doc}`ADR-0008 </adr/0008-scale-priors-fixed-parameters>`, so the 1PL of marginal
maximum likelihood, with one estimated slope and abilities $N(0, 1)$, and the Rasch model of the
other two compare only after a linear change of the ability scale. Item fit and model comparison take
a calibration's result only when its likelihood is marginal.

```{toctree}
:maxdepth: 1

mml
cml
jml
```
