# API Reference

```{eval-rst}
.. automodule:: irtlib
```

Each object is documented at its public import path. The item bank, {class}`irtlib.ItemBank`, and
its model, {class}`irtlib.IrtModel`, come from {mod}`irtlib`; everything else comes from its
module:

- the item response functions and the quantities derived from them, from {mod}`irtlib.irt`;
- the quadrature grid, the ability priors and the posterior over the grid, from
  {mod}`irtlib.quadrature`;
- the calibrators, from {mod}`irtlib.mml`, {mod}`irtlib.cml` and {mod}`irtlib.jml`, with the result
  they share and the checks of a response matrix in {mod}`irtlib.calibration`;
- ability estimation, from {mod}`irtlib.ability`;
- item fit and model comparison, from {mod}`irtlib.fit` and {mod}`irtlib.comparison`;
- responses drawn from the model, from {mod}`irtlib.drawing`;
- the array types of the signatures, from {mod}`irtlib.typing`.

```{toctree}
:maxdepth: 2

irt
bank
quadrature
calibration
mml
cml
jml
ability
fit
comparison
drawing
typing
```
