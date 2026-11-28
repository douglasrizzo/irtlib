# `irtlib` - Item Response Theory for Python

irtlib implements item response theory for the unidimensional dichotomous logistic models, 1PL to
4PL: the item response functions and the quantities derived from them, item calibration, ability
estimation, item fit and model comparison. It runs on NumPy and SciPy alone.

:::{admonition} irtlib has no release yet
:class: note

Install it from source, as the [README](https://github.com/douglasrizzo/irtlib#readme) shows. Its
interface may change before the first release.
:::

## Calibrate an item bank

Researchers estimate the items' parameters, with their standard errors, from the responses of a
sample of examinees: by marginal maximum likelihood for the 1PL to 4PL, which takes missing
responses ({mod}`irtlib.mml`), by
conditional maximum likelihood for the Rasch model ({mod}`irtlib.cml`), or by joint maximum
likelihood for the Rasch model and the 2PL ({mod}`irtlib.jml`). Item fit ({mod}`irtlib.fit`) and
model comparison ({mod}`irtlib.comparison`) then check the calibration.

## Score examinees

Researchers estimate each examinee's ability, with its standard error, from responses to items of
known parameters: by maximum likelihood, maximum a posteriori, expected a posteriori or weighted
likelihood ({mod}`irtlib.ability`).

## Build on irtlib

Package authors use irtlib's item bank ({class}`irtlib.ItemBank`), item response functions
({mod}`irtlib.irt`) and quadrature grid ({mod}`irtlib.quadrature`).
[catsim](https://douglasrizzo.com.br/catsim/), which computes and evaluates computerized adaptive
tests, is built on them.

## Find your way

- New to item response theory? The {doc}`introduction` explains the models and how calibration
  works, and {doc}`terms-and-notation` lists the terms and symbols.
- The {doc}`api/index` and the {doc}`architecture_decisions`.
- The [backlog](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/README.md): the methods
  planned next, and every other piece of planned work.

```{toctree}
:hidden: true
:maxdepth: 2

getting-started
API reference <api/index>
Architecture decisions <architecture_decisions>
contributing
references
```
