# Documentation charter

These rules govern every document in irtlib: the Sphinx pages under `sphinx/`, the docstrings they
render, `README.md`, `backlog/`, `CONTRIBUTING.md` and the GitHub release notes. They are
instructions for coding agents and contributors, not website content: `sphinx/conf.py` excludes this
file from the build. Read them before writing or reviewing documentation of any kind.

A kind of document the site does not have yet (guides, notebooks) gets its rules here in the pull
request that adds the first of them, adapted from catsim's charter.

## Audiences

In order of priority (ADR-0001):

1. **Researchers** who calibrate item banks, compare calibration methods or score examinees.
2. **Method authors** who add an estimation method.
3. **Package authors** who build on irtlib's item response theory, catsim's first.
4. **Contributors** changing irtlib itself.

Each kind of document below names its audience. Write for that reader only. Researchers know IRT
and may not know software design.

## Rules for every document

### Voice

- Write in the present tense. Describe what is, never what changed.
- Do not write "now", "no longer", "replaces", "instead of" or "previously" about an earlier
  version, "new" for something just added, card IDs, or PR and issue numbers. The exceptions are the
  Context section of an ADR and backlog cards.
- Put plain language first. Use a symbol only after the text has introduced it in words.
- Describe what users observe, not how irtlib computes it, unless the computation is the point, as
  on a method page.
- Use the terms and symbols of `terms-and-notation.md` (see "Terms and notation" below).
- In code, import each object from its shortest public path, in irtlib and in catsim:
  `from irtlib import ItemBank`, not `from irtlib.bank import ItemBank`.

### Sources

Every formula, every claim about a method's behaviour and every default rests on a source that has
been read: the citation convention and the revalidation checklist in `backlog/README.md` apply to
every document, not only to cards. Never cite a source or a location from memory.

On the Sphinx pages and in the docstrings, every source is an entry of `sphinx/references.bib`,
cited by its key: `` {cite:t}`key{p. N}` `` in a sentence and `` {cite:p}`key{p. N}` `` in
parentheses on a page, one role per source when it carries a location; the reST roles
`` :cite:t:`key` `` and `` :cite:p:`key` `` in a docstring. Add each source to `references.bib` with
its DOI and a note: `Open copy: \url{...}`, `Open access`, or `Not openly available; X (Year)
describes the method`. Escape a `%` in a URL as `\%`. Brace the title words whose case matters
(`{R}`, `{IRT}`, `{Bayesian}`, proper names): the style lowercases titles. A page that cites
sources ends with a References section holding a `{bibliography}` directive filtered to the page,
`:filter: docname in docnames`.

### mirt and other packages

irtlib's methods rest on their published sources, not on another package. mirt, the reference
implementation the parity tests compare irtlib with, and catR, the second reference for the response
functions and the ability estimators (ADR-0003), appear in three places only: each method page's
closed "Comparison with mirt" dropdown and its "Default from" column, the mirt ADR, and the parity
sections of `CONTRIBUTING.md`. An ADR may name another package where its practice was one of the forces behind
the decision. No document says what another package lacks, or presents an irtlib choice as an
improvement on another package's.

### History

The GitHub release notes are the only place for "what changed". There is no CHANGELOG file.

### Where a formula lives

- A method's formulas live with the method: on its backlog card until it is implemented, then on its
  method page, in the Definition section, with the source's page and equation.
- Its docstrings say in one sentence what it computes, in words, give the call contract and link the
  page: "The :doc:`/methods/calibration/mml` page gives the method, its sources and the origin of each
  default." A module's docstring is its summary and the conventions its callers share, with no
  formula.
- The item response functions and their quantities follow the same rule (ADR-0004): their formulas
  live on the Item Response Functions page, and the method pages refer to it rather than re-derive
  a quantity.
- The introduction shows $P_i(\theta)$ to teach the models. It is not a second reference.
- No formula appears twice on one rendered page: a method page renders its objects' docstrings under
  "API Reference", so a docstring that repeats the Definition shows the math twice.

### Implementation detail in user pages

Mention an implementation detail on the landing page and the concept pages only where it changes
what users observe, such as the methods that take missing responses; on a method page, the
implementation is the point. Describe it in the user's terms
and link the ADR for the reason. Leave out the rest: internal types and their fields, helpers,
validation rules, invariants.

### One home per topic

Each topic has one home. Other documents say what they need in a sentence and link the home; they
do not re-explain it.

| Topic | Home | Related documents |
|---|---|---|
| What irtlib is; install | README | landing page |
| What IRT is; the logistic models; how the calibration methods differ | Introduction | landing page |
| Terms and symbols | Terms and notation | every page |
| A method's definition, implementation, defaults, guarantees and departures | its method page | the introduction, in a sentence; its docstrings (one link) |
| A quantity's formula | the Item Response Functions page | the introduction shows $P_i(\theta)$; method pages |
| How the methods of a family compare | the family's index page | each method page |
| How irtlib agrees with mirt and catR, method by method | each method page's dropdown | `CONTRIBUTING.md` for running the parity tests |
| Why the architecture is shaped this way | the ADR | every page, by link |
| Planned work, and the research behind it | its card in `backlog/` | the backlog's generated index |
| How to cite sources; how a card is revalidated | `backlog/README.md` | this charter, for the bibliography's rules |
| How to contribute; the quality gate; building the docs | `CONTRIBUTING.md` | |
| What changed in a release | the GitHub release notes | |

When a home in this map does not exist yet, leave the material where it is and do not start a second
copy.

## Kinds of documents

Each kind has one job, audience and level. "Not here" names where the excluded material goes.

### README

- **Job:** tell a visitor on GitHub what irtlib is and where it stands: what it is (two
  sentences), its status, how to install it from source, where the documentation, the plans and the
  decisions are.
- **Audience and level:** every newcomer; introductory.
- **Not here:** developer setup (`CONTRIBUTING.md`), design notes (ADRs), method descriptions
  (docstrings and cards).
- **Rules:** every link is absolute, so that it works on PyPI too.

### Landing page (`sphinx/index.md`)

- **Job:** the website's front door, with an entry path for each audience the site serves:
  calibrating an item bank, scoring examinees and building on irtlib. It links the method pages, the
  concept pages, the API reference, the ADRs and the backlog. It has its own source, separate from the README.
  While irtlib has no release, a note says so and links the README's install from source.
- **Audience and level:** researchers and package authors; introductory.
- **Not here:** install steps and badges (README).
- **Length:** one screen, about 150-300 words.
- **Review:**
  - Can each audience reach its first page in one click?
  - Does it restate the README or the introduction rather than link it?

### Concept pages (introduction, terms and notation)

- **Job:** the introduction is the IRT primer: what a test measures, the response matrix, the
  logistic models, how the calibration methods differ, the ability estimators, how a calibration is
  checked, and a short map of irtlib's parts that links the API reference. "Terms and notation"
  gives one term per concept, its synonyms, its name in the API, and the symbols.
- **Audience and level:** researchers new to IRT or to calibration; introductory.
- **Belongs:** concepts, $P_i(\theta)$ as an illustration, each claim cited to a source read.
- **Not here:** a method's formulas (its method page), internal types and helpers, validation rules,
  design rationale (link the ADR).
- **Length:** the introduction reads in about 15 minutes (at most about 2,000 words); "Terms and
  notation" is as long as its lists need.
- **Review:**
  - Could a reader who has never seen irtlib's code follow every sentence?
  - Is every symbol introduced in words before it is used?
  - Do the terms and symbols match "Terms and notation"?

### Method pages (`sphinx/methods/`)

- **Job:** a method as a researcher needs it, centred on irtlib's implementation: what it does and
  when to choose it, its definition, how irtlib computes it, its parameters and where each default
  comes from, what it guarantees, how it agrees with mirt and catR, its sources and its API.
- **Audience and level:** researchers; intermediate to advanced in psychometrics.
- **Template:** `sphinx/methods/_template.md`: Motivation, Definition, Implementation (ending with a
  closed "Comparison with mirt" dropdown), Parameters, Guarantees, References, API Reference. Each
  family has an index page that compares its methods; a module that one page documents whole is
  rendered there with `automodule`, and a module split over several pages is declared on the family
  index with `{py:module}`.
- **Belongs:**
  - the formulas, each cited where it appears, with the source's page and equation;
  - in Implementation, the numerical choices a user can observe (the search, the grid, the starting
    values, the stopping rule, the acceleration, the standard errors, extreme and missing
    responses), each cited to its source or linked to the ADR where irtlib made it, and every
    departure from the sources, stated;
  - measurements behind a choice, with their date;
  - a Parameters table whose "Default from" column names each default's origin: a source's page,
    mirt's or catsim's default, an ADR, or no source yet, with a link to
    `backlog/discovery/sources-wanted.md`, where the default has its row;
  - Guarantees: properties a user relies on and can observe.
- **Not here:** software internals (private helpers, array layouts, caches); properties only an
  implementation sees, which go to the docstrings and the tests; a quantity's formula (link the Item
  Response Functions page); history.
- **Length:** about 600-1,500 words before the references; the calibrators and the item response
  functions run longer, with the measurements their choices rest on.
- **Review:**
  - Is every formula and choice cited to a location that someone has read, or linked to its ADR?
  - Does every default name its origin?
  - Do mirt and catR appear only in the dropdown and the "Default from" column?
  - Is every guarantee something a user can observe?
  - Does a docstring rendered under "API Reference" repeat a formula from the page?

### API reference (docstrings)

- **Job:** how to call an object correctly: the call contract, in numpy style: a one-line summary,
  the behaviour a caller relies on, Parameters (in the class docstring only, a dataclass's fields
  included), Returns, Raises, and a short doctest where it helps (pytest runs them). A method's
  module is rendered on its method page; the other modules each have a page in `sphinx/api/`. The
  objects that `irtlib` exports are documented at that path.
- **Audience and level:** anyone calling the object; precise and brief.
- **Belongs:** for a method's class or function, one link to its method page.
- **Not here:** a method's math (its method page), design rationale (link the ADR), history,
  comparisons with other packages (the method page's dropdown).
- **Length:** about 80 words of prose before the first numpydoc section.
- **Rules:** the docs build is strict: every cross-reference resolves, and numpydoc fails a public
  callable whose docstring leaves out a parameter (PR01). A module added to the package gets a method
  page or a page in `sphinx/api/`.
- **Review:** Could a caller use the object correctly from the docstring alone? Is any sentence
  rationale, history or method math?

### ADRs (`sphinx/adr/`)

- **Job:** why the architecture is shaped this way; contributors and method authors.
- **Template:** `sphinx/adr/_template.md`: Status, Context, Decision, Consequences, Alternatives
  Considered. The toctree of `sphinx/architecture_decisions.md` lists every ADR by its number.
- **Rules:**
  - Write in a timeless voice. Only Context may describe the code before the decision.
  - The Status line is dated: "Accepted 2026-11-26".
  - Show code only to state an interface, in a few lines.
  - When a decision changes, its ADR is edited in place, and an ADR whose decision no longer holds
    is deleted: git keeps the earlier text.
  - Cite by hand, "(Mislevy, 1986, p. 182)", so that the ADR reads on GitHub too.
- **Length:** 300-600 words.
- **Review:** Does it record a decision, or describe code? Would a reader who disagrees find the
  rejected options and the reasons?

### Backlog cards (`backlog/`)

- **Job:** one card per piece of planned work: a method with its full design, a feature, a fix, a
  decision for the maintainer, research. Its frontmatter feeds the index in `backlog/README.md`,
  which `backlog/_index.py` generates; the README defines every field.
- **Audience and level:** contributors and agents; advanced.
- **Template:** `backlog/_template.md`.
- **Rules:**
  - A card is working material, so it may cite its evidence: measurements, the review or PR that
    raised it, dates. The history rule above does not apply to cards.
  - A PR that does a card's work deletes the card; a lasting rationale goes into an ADR.
  - A card cites by hand, with a hand-written References list, following the citation convention in
    `backlog/README.md`, since GitHub does not render the citation roles.
- **Review:** Does the card say what is wrong or missing, with evidence, and what would show the work
  is done? Is every formula and claim cited to a location someone has read?

### CONTRIBUTING (`CONTRIBUTING.md`)

- **Job:** how to contribute: development setup, the quality gate, the parity tests, building the
  documentation and, once it exists, the release process. The site renders it as its Contributing
  page.
- **Not here:** architecture rationale (ADRs), instructions only agents need (`AGENTS.md`).
- **Rules:** a link to a file outside `sphinx/` is absolute, so that it resolves on the site too.
- **Review:** Does every command run as written? Does the gate match what pre-commit and CI run?

## Terms and notation

`sphinx/terms-and-notation.md` is the source of truth for terms and symbols. Use its preferred term
for each concept everywhere; its synonyms appear only there and in quotations. When a document needs
a term or symbol the page lacks, add it to the page first. The terms most often mixed up:

- **ability** $\theta$, not proficiency or latent trait;
- **guessing parameter** $c$, the lower asymptote (the API's `pseudo_guessing`), and **upper
  asymptote** $d$;
- **response matrix**, one row per examinee and one column per item, with $u_{ij}$ examinee $j$'s
  response to item $i$;
- **calibration** for estimating item parameters, and **ability estimation** for estimating
  $\theta$; never "estimation" alone where the two could be confused;
- **standard error**, not SEE or SEM;
- $N$ for the number of examinees, $M$ for the number of items, $j$ for the examinee index and $i$
  for the item index;
- **quadrature nodes** $X_q$, $q = 1, \dots, Q$, each with a weight, and **prior masses** $A_q$;
- $I$ is always information, never a set of items.

## Review checklist for every document

1. **Voice:** does anything tell history outside an ADR's Context or a card?
2. **Sources:** is every formula and claim cited to a location that has been read?
3. **Audience:** is there anything the reader does not need, such as an internal name or design
   rationale?
4. **One home:** does it re-explain a topic whose home is elsewhere, rather than link it?
5. **Terms:** does every term and symbol match "Terms and notation"?
6. **Code:** does every example run as written against the current API?
7. **Build:** does the docs build pass with no warnings?
