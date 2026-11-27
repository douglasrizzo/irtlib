# Contributing

pyitemcal is in its design phase: the planned work, and the order it comes in, is in the
[backlog](backlog/README.md). Pick a card, do its work in a pull request into `dev`, and delete the
card in that pull request.

## Development setup

Clone the repository, then install the development tools with [uv](https://docs.astral.sh/uv/) and
the pre-commit hooks:

```bash
uv sync --group dev
uv run pre-commit install
```

The first command installs pyitemcal with its runtime dependencies, declared in `pyproject.toml`, and
the tools for tests, linting, type checking and coverage. catsim comes from its `dev` branch on
GitHub, at the commit `uv.lock` pins. pre-commit then runs the repository's formatters and checks on
every commit.

## Quality gate

Changes that alter behavior come with new or updated tests in `tests/`. Before you send a pull
request, run the tests and every pre-commit hook, as CI does:

```bash
uv run pytest -q
uv run pre-commit run --all-files
```

pytest also runs the docstring examples in `src/pyitemcal`, and turns warnings into errors. The hooks
run Ruff, ty, pylint, vulture, typos and the check of the backlog's index.

## Parity tests with mirt

`tests/test_parity_mirt.py` compares pyitemcal with the R package mirt, its reference implementation
(ADR-0003), on outputs recorded in the JSON files of `tests/parity/mirt/`. The tests need no R.

The files are regenerated, never edited, when a case script beside them changes or the pinned mirt
version moves. That needs R and a C++ compiler, since mirt is built from source:

```bash
Rscript tests/parity/mirt/generate.R
```

The first run installs mirt, at the version `generate.R` pins, with its dependencies, into
`~/.cache/pyitemcal-parity/R-lib`, outside your own R library; set `PYITEMCAL_MIRT_LIB` to use
another folder. A rerun with the same mirt version writes the same bytes, so `git diff` shows only
what a change to the cases or the version moved.

## Sources

Every formula and every claim about a method cites a location in a source that its author has read.
The citation convention, the paper shelf and the revalidation checklist are in
[`backlog/README.md`](backlog/README.md).
