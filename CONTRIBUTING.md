# Contributing

irtlib is in its design phase: the planned work, and the order it comes in, is in the
[backlog](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/README.md). Pick a card, do its
work in a pull request into `dev`, and delete the card in that pull request.

## Development setup

Clone the repository, then install the development tools with [uv](https://docs.astral.sh/uv/) and
the pre-commit hooks:

```bash
uv sync --group dev
uv run pre-commit install
```

The first command installs irtlib with its runtime dependencies, declared in `pyproject.toml`, and
the tools for tests, linting, type checking and coverage. pre-commit then runs the repository's
formatters and checks on every commit.

## Quality gate

Changes that alter behavior come with new or updated tests in `tests/`. Before you send a pull
request, run the tests and every pre-commit hook, as CI does:

```bash
uv run pytest -q
uv run pre-commit run --all-files
```

pytest also runs the docstring examples in `src/irtlib`, and turns warnings into errors. The hooks
run Ruff, ty, pylint, vulture, typos and the check of the backlog's index.

## Building the documentation

The documentation site's sources are in `sphinx/`: its pages, the ADRs and `references.bib`, with
the rules for each kind of page in `sphinx/AGENTS.md`. Each method has a page in `sphinx/methods/`,
written from `sphinx/methods/_template.md`. Build the HTML pages with the command below,
which also installs Sphinx and its extensions (the `docs` dependency group):

```bash
uv run --group docs sphinx-build -b html -W --keep-going sphinx sphinx/_build/html
```

`-W` turns warnings into errors, and the build also fails on a cross-reference that does not
resolve and on a public callable whose docstring leaves out a parameter. CI builds the site the same
way on every pull request, and publishes the site built from `dev` at
<https://douglasrizzo.com.br/irtlib/dev/>.

External links are checked once a week by the "Link check" workflow
(`.github/workflows/linkcheck.yml`), not in pull requests, and a broken link fails that run. To check
them yourself:

```bash
uv run --group docs sphinx-build -b linkcheck sphinx sphinx/_build/linkcheck
```

Links to the Python, NumPy and SciPy documentation come from those projects' intersphinx
inventories. The build downloads each inventory and, when a site cannot be reached, reads the copy
stored in `sphinx/_inventories/`. Refresh the stored copies with:

```bash
cd sphinx && for p in python=https://docs.python.org/3/ numpy=https://numpy.org/doc/stable/ scipy=https://docs.scipy.org/doc/scipy/; do curl -fsSL -o "_inventories/${p%%=*}.inv" "${p#*=}objects.inv"; done
```

## Parity tests with mirt

`tests/test_parity_mirt.py` compares irtlib with the R package mirt, its reference implementation
(ADR-0003), on outputs recorded in the JSON files of `tests/parity/mirt/`. The tests need no R.

The files are regenerated, never edited, when a case script beside them changes or the pinned mirt
version moves. That needs R and a C++ compiler, since mirt is built from source:

```bash
Rscript tests/parity/mirt/generate.R
```

The first run installs mirt, at the version `generate.R` pins, with its dependencies, into
`~/.cache/irtlib-parity/R-lib`, outside your own R library; set `IRTLIB_MIRT_LIB` to use
another folder. A rerun with the same mirt version writes the same bytes, so `git diff` shows only
what a change to the cases or the version moved.

## Parity tests with catR

`tests/test_parity_catr.py` compares irtlib's response functions and ability estimators with the R
package catR, their second reference implementation (ADR-0003), on outputs recorded in the JSON files
of `tests/parity/catr/`. These tests need no R either.

The files are regenerated, never edited, when a case script beside them changes or the pinned catR
version moves:

```bash
Rscript tests/parity/catr/generate.R
```

The first run installs catR, at the version `generate.R` pins, from CRAN's source archive into
`~/.cache/irtlib-parity/R-lib`, the library the mirt generator uses; set `IRTLIB_CATR_LIB` to use
another folder. Both generators write their files through `tests/parity/json.R`, and a rerun with
the same catR version writes the same bytes.

## Benchmarks against catsim

`benchmarks/catsim_baseline.py` times each function irtlib took from catsim, the ability estimators
included, against catsim's original, on the shapes catsim calls them with, and checks that the two
give the same values: within a relative 1e-12, or the estimators' search tolerance for the maximum a
posteriori and maximum likelihood estimates (ADR-0004). It runs on demand, not in CI, where timings
are noisy:

```bash
uv run --group benchmarks python benchmarks/catsim_baseline.py
```

The `benchmarks` dependency group installs catsim from GitHub, at the commit `pyproject.toml` pins.
The script prints each function's time in both packages and their ratio, flags a ratio above 1.2,
and exits with status 1 if a function's values disagree. A function more than 1.2 times slower than
catsim's gets a card in the backlog for a faster version, with the ratio and the shape it was
measured on.

## Sources

Every formula and every claim about a method cites a location in a source that its author has read.
The citation convention, the paper shelf and the revalidation checklist are in
[`backlog/README.md`](https://github.com/douglasrizzo/irtlib/blob/dev/backlog/README.md).
