# Backlog

Every piece of planned work on irtlib is a card in this folder:

- estimation methods, each with its full design;
- features, fixes and maintenance;
- decisions waiting for the maintainer;
- research.

Pick a card and do its work in a PR, which deletes the card. To add work, copy
[`_template.md`](_template.md) into the folder of its area. The index below is generated from the
cards' frontmatter by [`_index.py`](_index.py), which a pre-commit hook runs, so commit the
regenerated README with the card.

The folders:

- `discovery/`: the literature review and the parity harness, which come before any method, and
  open decisions;
- `irt/`: item response functions and their derivatives;
- `estimation/`: calibration methods;
- `fit/`: item fit;
- `maintenance/`: releases, documentation and tooling;
- `scope/`: work outside the scope ADR-0001 sets.

## The frontmatter

| Field | Values |
|---|---|
| `title` | What the card does, as a short imperative, or the question a decision card asks |
| `kind` | `technique` (a method, with its design), `feature`, `fix` (wrong behaviour or wrong docs), `docs`, `decision` (the maintainer chooses), `research` (a survey, a walkthrough or an audit), `maintenance` |
| `status` | `ready` (enough to start), `needs-research` (claims to verify, or a design to write, before work), `needs-decision` (the maintainer chooses first), `blocked` (by the cards in `blocked_by`, or by what the card says), `parked` (outside the scope an ADR sets; moving it into scope needs that ADR changed) |
| `priority` | `1` blocks the first calibration method · `2` needed before a release · `3` useful, though nothing is blocked · `4` a later phase or a scope change |
| `effort` | `XS`, `S`, `M`, `L`, `XL` |
| `blocked_by` | Paths of other cards, relative to this folder and without `.md`, such as `[estimation/cml-rasch]` |
| `raised_by` | What surfaced the card: a review, a measurement or a PR, as short text without commas |

Breaking an interface is fine when it gives the better design (`AGENTS.md`).

<!-- backlog-index:start -->

22 cards: 12 ready, 2 needs-research, 2 needs-decision, 2 blocked, 4 parked.

## Waiting for the maintainer

| Card | Area | Priority |
|---|---|---|
| [Choose irtlib's license](maintenance/license.md) | maintenance | 2 |
| [Should the item response plots move from catsim into irtlib?](discovery/irt-plots.md) | discovery | 4 |

## Priority 2

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [Choose irtlib's license](maintenance/license.md) | maintenance | decision | needs-decision | XS |  | maintainer during setup |

## Priority 3

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [Find sources for the claims and defaults that have none on the shelf](discovery/sources-wanted.md) | discovery | research | needs-research | S |  | maintainer on 2026-11-27 |
| [Bring the ability estimators within 1.2 times catsim's time for one examinee](estimation/faster-ability-estimation.md) | estimation | feature | ready | M |  | timing of the ability-estimation PR on 2026-11-27 |
| [Estimate Rasch difficulties by conditional maximum likelihood](estimation/cml-rasch.md) | estimation | technique | ready | S |  | moved from catsim's backlog |
| [Estimate items by joint maximum likelihood, for comparison studies](estimation/jml.md) | estimation | technique | ready | S |  | moved from catsim's backlog |
| [Measure EM acceleration on the 3PL and 4PL and add it if it pays](estimation/em-acceleration.md) | estimation | feature | ready | S |  | ADR-0007 |
| [Make the log-likelihood and the posterior of one examinee as fast as catsim's](irt/one-examinee-speed.md) | irt | fix | ready | S |  | benchmarks against catsim on 2026-11-27 |
| [Reject the out-of-range inputs that the code moved from catsim accepts](irt/inherited-input-checks.md) | irt | fix | ready | XS |  | reviews of the moves on 2026-11-27 |
| [Build the documentation site](maintenance/docs-site.md) | maintenance | docs | ready | M |  | setup of irtlib |
| [Publish releases to PyPI with trusted publishing](maintenance/pypi-publishing.md) | maintenance | maintenance | blocked | M | [license](maintenance/license.md), [docs-site](maintenance/docs-site.md) | setup of irtlib |
| [Share the complete-response check of item fit and G²](maintenance/complete-responses-check.md) | maintenance | maintenance | ready | XS |  | mean-squares PR on 2026-11-27 |
| [Write the tutorial notebooks](maintenance/notebooks.md) | maintenance | docs | ready | M |  | setup of irtlib |

## Priority 4

| Card | Area | Kind | Status | Effort | Blocked by | Raised by |
|---|---|---|---|---|---|---|
| [Find how EAP abilities lower the mean squares on data that fit the Rasch model](discovery/mean-squares-eap.md) | discovery | research | needs-research | S |  | mean-squares PR on 2026-11-27 |
| [Should the item response plots move from catsim into irtlib?](discovery/irt-plots.md) | discovery | decision | needs-decision | S |  | maintainer during the rescope of 2026-11-27 |
| [Calibrate by Markov chain Monte Carlo](estimation/bayesian-mcmc.md) | estimation | technique | ready | L |  | moved from catsim's backlog |
| [Run the M-step's item updates in threads on free-threaded Python](estimation/free-threaded-m-step.md) | estimation | feature | ready | S |  | Python-floor measurement of 2026-11-26 |
| [Make irtlib citable with CITATION.cff and Zenodo](maintenance/citation-zenodo.md) | maintenance | maintenance | blocked | XS | [pypi-publishing](maintenance/pypi-publishing.md) | setup of irtlib |
| [Report coverage to Codecov](maintenance/codecov.md) | maintenance | maintenance | ready | XS |  | setup of irtlib |

## Parked: outside the scope an ADR sets

| Card | Area | Effort |
|---|---|---|
| [Calibrate multidimensional item response models](scope/multidimensional-models.md) | scope | XL |
| [Calibrate new items seeded into adaptive tests](scope/online-calibration.md) | scope | L |
| [Calibrate polytomous item response models](scope/polytomous-models.md) | scope | XL |
| [Link separate calibrations onto one scale](scope/linking-equating.md) | scope | L |

## Background

- [Literature Review: Item Calibration](discovery/literature-review.md)

<!-- backlog-index:end -->

## Technique cards

A technique card carries the full design of a method irtlib does not implement yet, in the
sections of [`_template.md`](_template.md). The cards are MyST Markdown, ready for the documentation
site; GitHub does not render MyST roles and directives. mdformat is kept off this folder
(`.pre-commit-config.yaml`), because it rewrites that syntax and breaks the `$$` math.

Every technique card keeps only what its sources verify, and its "Last validated" line says against
which sources and when it was last checked.

### Workflow

1. **Research** (`needs-research`): verify the card's claims against the primary sources (and mirt's
   documentation or source, when the card cites mirt), check what irtlib and catsim already have,
   and write the technique sections from the template, following the citation convention below.
2. **Revalidate** the card with the checklist below before it is implemented.
3. **Implement** the method once its open questions are answered, with tests for its guarantees and
   for its "Tests" section, parity cases against mirt where mirt implements the method, and
   parameter-recovery tests (ADR-0003).
4. **Promote** the card into the documentation in the same PR, once the documentation site exists:
   its sections from Motivation to References become the method's page, its sources move into
   `sphinx/references.bib`, and the card is deleted. Until the site exists, the implementing PR
   moves the sources into `sphinx/references.bib`, keeps the card's design in the module's docstring
   and the ADRs it needs, and deletes the card.

### Citation convention

- Cite each formula and design choice where it appears, with its location in the source (page,
  equation, table or figure).
- **Cite only locations you have read in the source.** A source is cited once it is on the paper
  shelf (below) and has been read there; never from memory, a summary, or another card.
- Cite printed page numbers, which can differ from the PDF's; the shelf's index records each PDF's
  offset.
- When the source itself cannot be obtained, the claim waits, or it says whose description it relies
  on: "as described by X (Year, p. N)".
- A claim, formula or default with no source on the shelf goes into
  `discovery/sources-wanted.md`, with where it is used and what was searched, and the maintainer is
  asked for the source. When none can be found, the code may follow a reference implementation,
  cited at its version and its file and line, and the row stays until a published source is read.
- A claim about mirt cites the mirt version, and the page of its manual or the function in its
  source.
- Say where the card departs from its sources.
- Give each source its DOI. When the published version is not open, give an open copy ("Open
  copy: ...") or, if there is none, say so and name an open source that describes the method ("Not
  openly available; X (Year) describes the method.").

Cards write their citations by hand, e.g. "(Bock & Aitkin, 1981, p. 445, Equation 6)", and give each
source in full in "References", because GitHub does not render citation roles.

### The paper shelf

`papers/` at the repository root holds the PDFs of the sources. It is local and never committed
(`.gitignore`, and a pre-commit hook rejects it even with `git add -f`), because most sources may not
be redistributed. Its `papers/README.md` lists each file with its full citation and its page offset
(printed page = PDF page + offset). A source behind a paywall is listed for the maintainer to
fetch.

### Revalidation checklist

1. **Citations.** Authors, year, title, venue and DOI match the source. Every DOI resolves at
   doi.org: a publisher that refuses automated requests is fine, a 404 is not. Every entry has an
   open copy or the "Not openly available" note.
2. **Cited locations.** Every cited page, equation and table has been read in the source on the
   shelf; a location not read is not cited. Pages are the printed ones.
3. **Definitions and formulas** match the cited equations. Every departure from the source, or from
   mirt, is stated explicitly.
4. **mirt** is the reference implementation where it implements the method (ADR-0003). The card
   says where it agrees and where it departs, with the mirt version, and its tests name the parity
   cases the implementation adds. A card never says what mirt lacks.
5. **Implementation guidance** follows the calibrator contract and settings of ADR-0006 to
   ADR-0008, and imports nothing from catsim (ADR-0004).
6. **Status.** The top of the card says when it was last validated and against which sources.
