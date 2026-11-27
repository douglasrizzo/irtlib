---
title: Drop catsim as a runtime dependency, and benchmark the moved code against it
kind: maintenance
status: blocked
priority: 2
effort: M
blocked_by: [irt/quadrature, irt/item-bank]
raised_by: [ADR-0004 as decided on 2026-11-27]
---

# Drop catsim as a runtime dependency, and benchmark the moved code against it

## Problem

ADR-0004 has irtlib import nothing from catsim at run time, and a released wheel cannot name
catsim's Git branch. Until the moves land, irtlib imports `catsim.irt.icc` and
`log_response_probabilities`, `catsim.estimation.QuadratureGrid` and `normal_log_prior`, and
`catsim.IrtModel` and `ItemBank` (`AGENTS.md`). It also keeps private versions of steps the moved
code provides: `mml._log_masses`, `_log_joint` and `_posterior` against the moved posterior, and the
grid's normal masses that `fit.s_x2` builds again (`src/irtlib/fit.py:104`).

catsim's code at a fixed commit is also the baseline that shows whether the moved code is as fast,
and still agrees, while catsim keeps its own copy (ADR-0004).

## Proposal

1. Import every moved name from irtlib, in one pull request, since an irtlib `ItemBank` never
   equals a catsim one: in `src/irtlib/mml.py:67-69`, `fit.py:40-41` and `calibration.py:10-11`, and
   in `tests/test_irt.py:17`, `test_mml.py:13-14`, `test_parity_mirt.py:27` and
   `test_comparison.py:13`. Replace the private versions with the moved functions; replace the
   inline response draws of the recovery tests with the drawing function; keep one constant for
   the four item columns, where `mml._N_PARAMETERS` repeats it.
2. A calibration reports its estimates as they come, and builds its bank from the valid items
   only: a miskeyed item, with a slope that is not positive, warns rather than raises (maintainer,
   2026-11-27). `calibration.validate_responses` builds on the response-matrix check,
   `irtlib._responses.response_matrix`, and adds its own rule, that every item is answered.
3. Remove catsim from `[project.dependencies]` and `[tool.uv.sources]`; edit ADR-0002 (runtime
   dependencies NumPy and SciPy) and `AGENTS.md` (the list of catsim names) in the same pull
   request.
4. Add catsim, pinned to commit `97b89f5023871391ba213e3e3c95f4ec8af8f20b` (catsim's `dev` on
   2026-11-27, which `uv.lock` pins today), to a development dependency group used only by the
   benchmarks.
5. Add benchmarks, run on demand and not in CI, where timings are noisy: each moved function timed
   against catsim's original with `timeit`, on the shapes catsim calls them with. Those are one
   ability against a bank of 500 items for the response functions and information, and one
   examinee's responses on catsim's 121-node default grid for the posterior. The estimators join
   them with `estimation/ability-estimation`, which this card does not wait for.
   They also check that the two agree within a relative `1e-12`. The pull request records every
   ratio; `CONTRIBUTING.md` says how to run them.

## Tests

- `import irtlib` with catsim uninstalled succeeds: a test in a CI job that installs the built
  wheel alone, which the "Test the built wheel" job already does.
- The benchmarks' agreement checks fail if a moved function's values drift from catsim's.

The batched posterior already takes 2.6 times catsim's time for one examinee (142 µs against 54
µs, 20 items on 121 nodes, a scratch script on 2026-11-27), so it gets a card for a faster version
as soon as the benchmarks confirm it.

## Open Questions

1. **When a slowdown needs a faster version.** A function slower than catsim's by more than 1.2× on
   catsim's shapes gets a card for a faster version in irtlib, as ADR-0004 says. The threshold is
   assumed until the maintainer sets another.
