# Contributing

pyitemcal is in its research phase: the planned work, and the order it comes in, is in the
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

## Sources

Every formula and every claim about a method cites a location in a source that its author has read.
The citation convention, the paper shelf and the revalidation checklist are in
[`backlog/README.md`](backlog/README.md).
