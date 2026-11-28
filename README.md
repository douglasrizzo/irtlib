<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/douglasrizzo/irtlib/dev/sphinx/logo/stacked-dark.svg" />
    <img src="https://raw.githubusercontent.com/douglasrizzo/irtlib/dev/sphinx/logo/stacked-light.svg" alt="irtlib" width="180" />
  </picture>
</p>

# irtlib

irtlib implements item response theory for the unidimensional dichotomous logistic models, 1PL to
4PL: the response functions and their quantities, item calibration, ability estimation and model
fit. Its item banks are the ones [catsim](https://github.com/douglasrizzo/catsim) runs computerized
adaptive tests on.

## Status

irtlib is in its design phase. Marginal maximum likelihood calibration of the 1PL to 4PL, with
priors, standard errors, item fit by S-X² and by the Rasch model's mean squares, and model
comparison, is implemented and checked against the R package mirt; conditional maximum likelihood
calibration of the Rasch model, and joint maximum likelihood calibration of the Rasch model and the
2PL, against published values and closed forms. The response functions, the quadrature grid and
priors, the item bank and ability estimation, taken from catsim, are implemented too, and irtlib
runs on NumPy and SciPy alone. The other calibration methods come next. Nothing is published on
PyPI.

## Install from source

irtlib needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/douglasrizzo/irtlib.git
cd irtlib
uv sync
```

## Where things are

- [The documentation](https://douglasrizzo.com.br/irtlib/): an introduction to the models and the
  calibration methods, the terms and notation, and the API reference.
- [The backlog](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/README.md): every piece of
  planned work, and the research rules.
- [The architecture decisions](https://github.com/douglasrizzo/irtlib/tree/dev/sphinx/adr): what
  irtlib covers and why it is built the way it is.
- [Contributing](https://github.com/douglasrizzo/irtlib/blob/dev/CONTRIBUTING.md): development
  setup and the quality gate.

## License

irtlib has no license yet. Until it has one, the code can be read but not reused.
