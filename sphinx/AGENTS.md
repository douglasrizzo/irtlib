# Documentation charter

These rules govern every document in irtlib: `README.md`, the docstrings, the ADRs under
`sphinx/adr/`, `backlog/`, `CONTRIBUTING.md` and the GitHub release notes. They are instructions
for coding agents and contributors. Read them before writing or reviewing documentation of any kind.

irtlib has no documentation website yet (`backlog/maintenance/docs-site.md`). The folder
`sphinx/` holds the ADRs and `references.bib` so that the site, when it comes, builds around them
without moving anything. The kinds of documents the site will add (concept pages, guides, method
pages, notebooks) get their rules here in the pull request that adds the first of them, adapted from
catsim's charter.

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
- In code, import each object from its shortest public path, in irtlib and in catsim.

### Sources

Every formula, every claim about a method's behaviour and every default rests on a source that has
been read: the citation convention and the revalidation checklist in `backlog/README.md` apply to
every document, not only to cards. Never cite a source or a location from memory.

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

- A method's formulas live with the method: on its backlog card until it is implemented, then on its
  documentation page; until the documentation site exists, its module's docstring is that page.
  Its functions' docstrings say in one sentence what they compute, in words, and give the call
  contract.
- The item response functions and their quantities follow the same rule as a method (ADR-0004):
  their formulas, with their sources, live in their module's docstring, and code moved from catsim
  takes its formulas out of its function docstrings.

### One home per topic

Each topic has one home. Other documents say what they need in a sentence and link the home.

| Topic | Home |
|---|---|
| What irtlib is; install | README |
| Why the architecture is shaped this way | the ADR |
| Planned work, and the research behind it | its card in `backlog/` |
| How to cite sources; how a card is revalidated | `backlog/README.md` |
| How to contribute; the quality gate | `CONTRIBUTING.md` |
| What changed in a release | the GitHub release notes |

## Kinds of documents

### README

- **Job:** tell a visitor on GitHub what irtlib is and where it stands: what it is (two
  sentences), its status, how to install it from source, where the plans and decisions are.
- **Audience and level:** every newcomer; introductory.
- **Not here:** developer setup (`CONTRIBUTING.md`), design notes (ADRs), method descriptions
  (cards).
- **Rules:** every link is absolute, so that it works on PyPI too.

### Docstrings

- **Job:** how to call an object correctly: the call contract, in numpy style: a one-line summary,
  the behaviour a caller relies on, Parameters (in the class docstring only), Returns, Raises, and a
  short doctest where it helps (pytest runs them).
- **Not here:** a method's math (its card or page), design rationale (the ADR), history,
  comparisons with other packages.
- **Length:** about 80 words of prose before the first numpydoc section.

### ADRs (`sphinx/adr/`)

- **Job:** why the architecture is shaped this way; contributors and method authors.
- **Template:** `sphinx/adr/_template.md`: Status, Context, Decision, Consequences, Alternatives
  Considered.
- **Rules:**
  - Write in a timeless voice. Only Context may describe the code before the decision.
  - The Status line is dated: "Accepted 2026-11-26".
  - Show code only to state an interface, in a few lines.
  - When a decision changes, its ADR is edited in place, and an ADR whose decision no longer holds
    is deleted: git keeps the earlier text.
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
    `backlog/README.md`.
- **Review:** Does the card say what is wrong or missing, with evidence, and what would show the work
  is done? Is every formula and claim cited to a location someone has read?

### CONTRIBUTING (`CONTRIBUTING.md`)

- **Job:** how to contribute: development setup, the quality gate and, once they exist, the parity
  tests and the release process.
- **Not here:** architecture rationale (ADRs), instructions only agents need (`AGENTS.md`).
- **Review:** Does every command run as written? Does the gate match what pre-commit and CI run?

## Terms and notation

Until irtlib has a terms page of its own (`backlog/maintenance/docs-site.md`), use the terms and
symbols of catsim's
[terms and notation](https://github.com/douglasrizzo/catsim/blob/dev/sphinx/terms-and-notation.md),
and these for calibration:

- **ability** $\theta$; **discrimination** $a$, **difficulty** $b$, **guessing parameter** $c$ and
  **upper asymptote** $d$;
- **responses** $u$; the **response matrix** has one row per examinee and one column per item, and
  $u_{ij}$ is examinee $j$'s response to item $i$;
- $N$ for the number of examinees, $M$ for the number of items, $j$ for the examinee index and $i$
  for the item index;
- **quadrature nodes** $X_q$ with weights $A_q$, $q = 1, \dots, Q$;
- **calibration** for estimating item parameters, and **ability estimation** for estimating
  $\theta$; never "estimation" alone where the two could be confused.

## Review checklist for every document

1. **Voice:** does anything tell history outside an ADR's Context or a card?
2. **Sources:** is every formula and claim cited to a location that has been read?
3. **Audience:** is there anything the reader does not need, such as an internal name or design
   rationale?
4. **One home:** does it re-explain a topic whose home is elsewhere, rather than link it?
5. **Terms:** does every term and symbol match the terms above?
6. **Code:** does every example run as written against the current API?
