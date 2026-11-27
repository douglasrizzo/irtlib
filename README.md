# pyitemcal

pyitemcal estimates the parameters of item response theory models from test responses: it
calibrates items. It covers the unidimensional dichotomous logistic models, 1PL to 4PL, and gives
banks that [catsim](https://github.com/douglasrizzo/catsim) reads for computerized adaptive testing
simulations.

## Status

pyitemcal is in its research phase. No calibration method is implemented yet, and nothing is
published on PyPI. The first work is a review of the calibration literature from its primary
sources, then the design of the API, then the methods, each checked against the R package mirt.

## Install from source

pyitemcal needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/douglasrizzo/pyitemcal.git
cd pyitemcal
uv sync
```

## Where things are

- [The backlog](https://github.com/douglasrizzo/pyitemcal/blob/dev/backlog/README.md): every piece of
  planned work, and the research rules.
- [The architecture decisions](https://github.com/douglasrizzo/pyitemcal/tree/dev/sphinx/adr): what
  pyitemcal covers and why it is built the way it is.
- [Contributing](https://github.com/douglasrizzo/pyitemcal/blob/dev/CONTRIBUTING.md): development
  setup and the quality gate.

## License

pyitemcal has no license yet. Until it has one, the code can be read but not reused.
