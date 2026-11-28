# API Reference

```{eval-rst}
.. automodule:: irtlib
```

Each object is documented at its public import path. The item bank, {class}`irtlib.ItemBank`, and
its model, {class}`irtlib.IrtModel`, come from {mod}`irtlib`; everything else comes from its module.
The modules of the methods are documented on their method pages:

| Module | Page |
|---|---|
| {mod}`irtlib.irt` | {doc}`/methods/models/item-response-functions` |
| {mod}`irtlib.quadrature` | {doc}`/methods/models/quadrature` |
| {mod}`irtlib.mml`, {mod}`irtlib.cml`, {mod}`irtlib.jml` | {doc}`/methods/calibration/index` |
| {mod}`irtlib.ability` | {doc}`/methods/ability/index` |
| {mod}`irtlib.fit` | {doc}`/methods/fit/index` |
| {mod}`irtlib.comparison` | {doc}`/methods/comparison` |

The pages below document the rest: the item bank, the result and the response-matrix checks that the
calibrators share, drawing responses from the model, the plots, and the array types of the signatures.

```{toctree}
:maxdepth: 2

bank
calibration
drawing
plot
typing
```
