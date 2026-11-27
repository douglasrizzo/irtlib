# ADR-0005: Every Draw Comes From a Generator the Caller Passes

## Status

Accepted 2026-11-26

## Context

Calibration draws random numbers in several places: Markov chain Monte Carlo samplers, stochastic
approximation, random starting values, and the responses that parameter-recovery studies draw from
known parameters. A study that repeats a calibration must be able to reproduce it, and a study that
compares two methods on the same simulated data must give both the same responses. Global random
state serves neither: any other code that draws from it changes the result, and a run cannot say
which numbers it used.

catsim takes every draw from a generator its caller passes (catsim's ADR-0013), so a study that
uses both packages can control them the same way.

## Decision

- Every function or method that draws takes an `rng` argument of type
  `int | numpy.random.Generator | None` and passes it through `numpy.random.default_rng`: `None`
  draws fresh entropy (NumPy's convention), an `int` fixes the state, and a `Generator` is used as
  is.
- irtlib never uses NumPy's global random state or the standard library's `random` module;
  Ruff's banned-API rule rejects `random`.
- A method that draws in more than one place derives its streams from the one generator it is
  given, so the same `rng` reproduces the whole call.

## Consequences

- The same seed, data and settings give the same estimates, on the same versions of irtlib, NumPy
  and SciPy. Across platforms results agree up to floating-point rounding.
- A recovery study can draw its responses and its estimator's random numbers from separate seeds,
  and compare methods on the same responses.
- A deterministic method, such as EM from fixed starting values, takes no `rng`.
- An unseeded call cannot be repeated, and a passed generator gives other results each time, since
  the call draws from it.

## Alternatives Considered

- A fixed default seed: unseeded replications in a loop are identical, against NumPy's convention.
- Global random state, seeded by the caller: any other code that draws from it changes the results.
- A generator stored on an object at construction: two calls on the same object would draw different
  numbers, so a call alone would not say what it drew.
