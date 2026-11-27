# AGENTS.md

Repository guidance for coding agents working in `pyitemcal`.

## Scope

These instructions apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## Project Summary

`pyitemcal` calibrates items: it estimates the parameters of the unidimensional dichotomous 1PL to
4PL models, with their standard errors, from a response matrix with missing entries, and evaluates
item fit (ADR-0001). Its estimates use catsim's parameterization and bank format, and it builds on
catsim's public item response functions (ADR-0004).

The project is in its research phase: no calibration method is implemented yet. The order of work
is the research pass (`backlog/discovery/research-pass.md`), then the requirements
(`backlog/discovery/requirements.md`), then the mirt parity harness, then the methods.

## Repository Layout

- `src/pyitemcal/`: package source code
- `tests/`: automated test suite
- `backlog/`: every piece of planned work, as cards with frontmatter, and the citation rules
- `sphinx/adr/`: Architecture Decision Records
- `sphinx/references.bib`: the bibliography
- `papers/`: the local paper shelf, never committed (`backlog/README.md`, "The paper shelf")

## Core Architecture Rules

- Scope is ADR-0001: dichotomous, unidimensional, 1PL to 4PL. Work outside it is a parked card in
  `backlog/scope/` until the ADR changes.
- Runtime dependencies are NumPy, SciPy and catsim only (ADR-0002); a new one needs an ADR.
- Use catsim only through the names ADR-0004 lists, each imported from its public path. Using
  another catsim name changes ADR-0004 first. pyitemcal follows catsim's changes; catsim knows
  nothing of pyitemcal, so never propose a catsim change for pyitemcal's sake without the
  maintainer.
- Every method that mirt implements is checked against mirt by parity tests with recorded fixtures,
  and every method has parameter-recovery tests (ADR-0003).
- Every random draw comes from an `rng: int | numpy.random.Generator | None` argument, passed through
  `numpy.random.default_rng`, where None draws fresh entropy (ADR-0005). No global NumPy random
  state, no stdlib `random`.
- Performance is not a requirement. When a change makes pyitemcal slower, say so to the maintainer
  when it happens, with measurements.

## Research Rules

- Every formula, claim about a method and default rests on a source on the paper shelf, read at the
  location cited. Never cite a source, a page or an equation from memory. The citation convention
  and the revalidation checklist are in `backlog/README.md`.
- The method cards carried over from catsim are unverified until the research pass rewrites them.
  Do not implement from them, and do not copy their claims elsewhere.
- When a source is behind a paywall, list it for the maintainer to fetch; do not cite it through a
  summary.

## Source Code Guidelines

- Prefer small, explicit changes over broad implicit behavior.
- Keep type annotations accurate; this repository uses `ty` as part of its quality gates.
- Annotate NumPy arrays with `FloatArray` and `IntArray` from `catsim.typing` (ADR-0004). Ruff bans
  `numpy.typing.NDArray` elsewhere.
- Use the Python 3.12 type-parameter syntax for generics (ADR-0002).
- A method that overrides a base-class method carries `@override`, from `typing`; ty's
  `missing-override-decorator` enforces it.
- Closed option sets are `StrEnum`s, like catsim's `IrtModel`; code that branches on one uses `match`,
  with a final `case _`.
- Breaking an interface is fine when it gives the better design. A renamed or removed name is gone,
  with no backward-compatibility alias or deprecation shim; the same PR updates every caller, test,
  doc and card sketch.
- When changing architecture-level behavior, update or add an ADR in `sphinx/adr/`.
- Name things by catsim's one-name-per-concept table (`theta`, `items`, `responses`, `item`, `name`,
  `n_items`, `model`, `rng`), in catsim's ADR-0023, for every concept the two packages share.
- The PR that adds the package's second module adds an import-linter `layers` contract for its
  modules to `pyproject.toml` and the `import-linter` hook to `.pre-commit-config.yaml`, as catsim
  has; `lint-imports` cannot run without a contract.

## Documentation

Before writing or reviewing documentation of any kind, read `sphinx/AGENTS.md`, the documentation
charter. Each kind of document has one job and audience:

- `README.md`: what pyitemcal is, its status, install from source; newcomers on GitHub.
- Docstrings: the call contract; anyone calling the object.
- ADRs (`sphinx/adr/`): why the architecture is shaped this way; contributors.
- `backlog/`: every piece of planned work, as cards with frontmatter, whose index the README
  generates; contributors and agents. A PR that does a card's work deletes the card.

Hard rules:

- A docstring is the call contract: no method math, design rationale or history.
- History ("now", "no longer", "replaces", card IDs, PR numbers) goes only in the GitHub release
  notes and the Context section of an ADR. There is no CHANGELOG file.
- No internal names or design rationale in user pages: link the ADR.
- One home per topic; other documents link to it.

## Testing And Validation

- Use `uv` for project commands.
- Local commands run on Python 3.13 (`.python-version`), the version of CI's pre-commit job. CI's
  test matrix runs 3.12 to 3.15.
- Pre-commit runs ruff, ty, pylint, vulture, typos and the backlog index; run them all with
  `uv run pre-commit run --all-files`.
- Preferred validation commands:
  - `uv run pytest -q` (also runs the docstring examples in `src/pyitemcal`; warnings are errors)
  - `uv run ruff check .`
  - `uv run pylint src/pyitemcal`
  - `uv run ty check`
- Every test has a docstring naming what it tests for: the behaviour, or the bug that would make it
  fail. Test docstrings have no argument sections.
- Expected values come from a hand-worked case, a closed form, a published value, mirt's recorded
  output or a brute-force reference, never from the code under test.
- ty checks `tests/` too. A test double subclasses the base class it stands in for; a line that
  passes a deliberately wrong type carries `# ty: ignore[<rule>]`.
- The CI test and test-wheel jobs accept "no tests collected" (pytest's exit status 5) until the
  first module lands. The PR that adds the first tests removes that tolerance from both jobs and
  adds `--cov=pyitemcal` to the CI test command, which turns on the coverage gate (`fail_under` in
  `pyproject.toml`).

## Pre-commit And CI Expectations

- Pre-commit is part of the normal workflow and should remain aligned with the repository's real
  quality requirements.
- Do not add redundant checks to CI when they are already covered by pre-commit unless there is a
  clear defense-in-depth reason.
- Every action in `.github/workflows/` is pinned by commit SHA with a version comment, and every
  checkout sets `persist-credentials: false`.
- The branch rulesets require the CI checks by name; a PR that renames a job or a matrix entry
  updates the rulesets with the maintainer.

## Commit And Git Hygiene

- Pull requests into `dev` are squash-merged: each becomes one commit on `dev`, with the pull
  request's title and description as its message. The "Protect dev" ruleset allows no other merge
  method, and like "Protect main" it requires a pull request and the CI checks, and blocks deleting
  and force-pushing the branch, for administrators too.
- Commit once for the whole pull request, or at checkpoints that pass the gate (the pre-commit hook
  runs on every commit), with a message that says what the commit changes. Do not amend, rebase,
  reorder or squash commits to tidy a branch's history: the squash merge discards it.
- Keep each pull request to one concern.
- Merges from `dev` into `main` keep a merge commit, the only method the `main` ruleset allows.
- Never commit local-only directories: `.claude/`, `.pre-commit-cache/`, `papers/`, or other tool
  caches and scratch directories. A pre-commit hook rejects staged paths under `.claude/`,
  `.pre-commit-cache/` and `papers/`, even when added with `git add -f`.
- Preserve user changes in a dirty worktree. Do not revert unrelated modifications.

## Planned Work vs ADRs

- Planned work is a card in `backlog/` (`backlog/README.md`): methods, features, fixes, decisions
  for the maintainer and research. Follow-ups found during a task become cards too, never GitHub
  issues unless the maintainer asks.
- Use ADRs in `sphinx/adr/` for durable architectural decisions. A card's lasting rationale goes
  into an ADR when the card's work lands; the card itself is deleted.

## Practical Commands

- Run full tests: `uv run pytest -q`
- Run lint: `uv run ruff check .`
- Run every pre-commit hook: `uv run pre-commit run --all-files`
- Run type checks: `uv run ty check`
- Regenerate the backlog index: `uv run --no-project python backlog/_index.py`
- Move catsim's pinned `dev` commit forward: `uv lock --upgrade-package catsim`
