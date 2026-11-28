# Methods

Each method irtlib implements has a page: what it does and when to choose it, its definition with
the source of each formula, how irtlib computes it and where it departs from its sources, its
parameters with the origin of each default, what it guarantees, how it compares with the reference
implementations, and its API.

- {doc}`models/index`: the item response functions and the quantities derived from them, and the
  quadrature grid and ability priors that every method builds on.
- {doc}`calibration/index`: marginal, conditional and joint maximum likelihood.
- {doc}`ability/index`: maximum likelihood, maximum a posteriori, expected a posteriori and
  weighted likelihood.
- {doc}`fit/index`: S-X² and the mean squares of the items, and the mean squares of the examinees.
- {doc}`comparison`: the likelihood-ratio test, AIC, BIC and $G^2$.

```{toctree}
:hidden: true
:maxdepth: 2

models/index
calibration/index
ability/index
fit/index
comparison
```
