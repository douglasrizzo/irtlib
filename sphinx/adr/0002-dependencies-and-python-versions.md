# ADR-0002: NumPy and SciPy Are the Only Required Runtime Dependencies, on Python 3.12 and Later

## Status

Accepted 2026-11-26; catsim removed 2026-11-27; plot extra added 2026-11-28

## Context

Every package a library requires is one more version range that the projects installing it must
reconcile with their own, and one more range the tests must cover at both ends. Item response
theory is array arithmetic and numerical optimization, which NumPy and SciPy provide. The package
began by importing its item response functions, grid, priors and item bank from catsim, whose own
runtime dependencies are NumPy and SciPy; that code moved into irtlib, and catsim builds on irtlib
(ADR-0004).

catsim supports Python 3.11 and later. A measurement made when this package was set up, on
2026-11-26, compared the floors 3.11 to 3.14 for a representative EM iteration and for the
development lock file:

- the interpreter version changed neither step's speed beyond noise, since the work runs in NumPy,
  SciPy and BLAS;
- with a 3.11 floor the lock resolves NumPy and SciPy twice, older releases for 3.11 and newer ones
  from 3.12, because NumPy 2.5 and SciPy 1.18 require 3.12; from a 3.12 floor it resolves each once;
- `typing.override`, which every override carries, is in the standard library from 3.12; below it
  comes from typing-extensions;
- 3.12 is the Python version with the most NumPy downloads, and the Scientific Python ecosystem's
  support schedule (SPEC 0) drops it in late 2026.

## Decision

- irtlib's required runtime dependencies are NumPy and SciPy, with catsim's floors; the plot extra adds
  matplotlib (ADR-0011).
- catsim, at a fixed commit, is a development dependency of the benchmarks alone (ADR-0004), in a
  dependency group of its own that neither the tests nor CI install.
- It supports Python 3.12 and later. Its development lock and local tools use 3.13, as catsim's do.
- A new runtime dependency needs an ADR of its own.

## Consequences

- An environment that installs catsim gets nothing more when it installs irtlib.
- `import irtlib` works without catsim installed, which the CI job that tests the built wheel shows.
- Every supported Python resolves to the same NumPy and SciPy releases, so a failure on one Python
  is never a different library in disguise; the declared floors are tested in CI entries of their
  own.
- The code uses the Python 3.12 type-parameter syntax, and imports `override` from `typing`, with no
  compatibility module.
- A catsim user on Python 3.11 cannot install irtlib.
- Data-frame libraries and file formats other than the bank's CSV stay out of irtlib; its results are
  NumPy arrays that those libraries read.

## Alternatives Considered

- Python 3.11, catsim's floor: two NumPy and SciPy releases in the lock, and typing-extensions, for
  users of a Python version that reaches its end of life in October 2027.
- Python 3.13, SPEC 0's floor: little beyond 3.12 for this package's code, at the cost of the largest
  group of users.
- Python 3.14, for free threading: a thread-parallel M-step gains on free-threaded builds under any
  floor (`backlog/estimation/free-threaded-m-step.md`), so the floor need not exclude 3.12 and 3.13.
- An optional dependency for data frames: no feature needs one yet.
- catsim as an optional extra for the benchmarks: it would be part of the published package's
  metadata, offered to users, for code that only contributors run.
