# Documentation charter

These rules govern every document in irtlib: the Sphinx pages under `sphinx/`, the docstrings they
render, `README.md`, `backlog/`, `CONTRIBUTING.md` and the GitHub release notes. They are
instructions for coding agents and contributors, not website content: `sphinx/conf.py` excludes this
file from the build. Read them before writing or reviewing documentation of any kind.

A kind of document the site does not have yet (method pages, guides, notebooks) gets its rules here
in the pull request that adds the first of them, adapted from catsim's charter.

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
  in a method's documentation.
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
implementation the parity tests compare irtlib with (ADR-0003), appears where a method's
documentation states its agreement and its departures, in the mirt ADR and in the parity section of
`CONTRIBUTING.md`. An ADR may name another package where its practice was one of the forces behind
the decision. No document says what another package lacks, or presents an irtlib choice as an
improvement on another package's.

### History

The GitHub release notes are the only place for "what changed". There is no CHANGELOG file.

### Where a formula lives

- A method's formulas live with the method: on its backlog card until it is implemented, then in its
  module's docstring, which the API reference renders, until the method has a page of its own. Its
  functions' docstrings say in one sentence what they compute, in words, and give the call contract.
- The item response functions and their quantities follow the same rule as a method (ADR-0004):
  their formulas, with their sources, live in their module's docstring, and code moved from catsim
  takes its formulas out of its function docstrings.
- The introduction shows $P_i(\theta)$ to teach the models. It is not a second reference.

### Implementation detail in user pages

Mention an implementation detail on the landing page and the concept pages only where it changes
what users observe, such as the methods that take missing responses. Describe it in the user's terms
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
| A method's definition, sources and departures | its module's docstring | the introduction, in a sentence |
| A quantity's formula | its `irtlib.irt` function's module docstring | the introduction shows $P_i(\theta)$ |
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
  calibrating an item bank, scoring examinees and building on irtlib. It links the concept pages,
  the API reference, the ADRs and the backlog. It has its own source, separate from the README.
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
- **Not here:** a method's formulas (its module's docstring), internal types and helpers, validation
  rules, design rationale (link the ADR).
- **Length:** the introduction reads in about 15 minutes (at most about 2,000 words); "Terms and
  notation" is as long as its lists need.
- **Review:**
  - Could a reader who has never seen irtlib's code follow every sentence?
  - Is every symbol introduced in words before it is used?
  - Do the terms and symbols match "Terms and notation"?

### API reference (docstrings)

- **Job:** how to call an object correctly: the call contract, in numpy style: a one-line summary,
  the behaviour a caller relies on, Parameters (in the class docstring only, a dataclass's fields
  included), Returns, Raises, and a short doctest where it helps (pytest runs them). Each module's
  page in `sphinx/api/` renders its docstrings with `automodule`; the objects that `irtlib` exports
  are documented at that path.
- **Audience and level:** anyone calling the object; precise and brief.
- **Not here:** design rationale (link the ADR), history, comparisons with other packages outside a
  method's statement of its departures. A method's math lives in its module's docstring, not in its
  functions'.
- **Length:** about 80 words of prose before the first numpydoc section.
- **Rules:** the docs build is strict: every cross-reference resolves, and numpydoc fails a public
  callable whose docstring leaves out a parameter (PR01). A module added to the package gets a page
  in `sphinx/api/` and an entry in its toctree.
- **Review:** Could a caller use the object correctly from the docstring alone? Is any sentence
  rationale or history?

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
