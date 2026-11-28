---
title: Give each method a page of its own, with its formulas moved out of the module docstrings
kind: docs
status: ready
priority: 3
effort: L
blocked_by: []
raised_by: [maintainer on 2026-11-28]
---

# Give each method a page of its own, with its formulas moved out of the module docstrings

## Problem

A method's formulas, its sources, its numerical choices and its departures from mirt live in its
module's docstring (`sphinx/AGENTS.md`, "Where a formula lives"), which the API reference renders
above the call contracts. A researcher looking for what irtlib's MML-EM does reads it between
signatures, with nothing that says which defaults come from a paper, which from mirt and which
from an ADR. catsim gives each method a page, and its docstrings keep only the call contract.

The maintainer chose, on 2026-11-28, to move the math to method pages focused on the implementation
irtlib has, using the bibliography to show where each decision comes from.

## What irtlib and catsim have now

- Module docstrings with formulas and sources: `irt`, `quadrature`, `mml`, `cml`, `jml`, `ability`,
  `fit`, `comparison`; `bank` holds the choices behind `ItemBank.generate` and `drawing` its drawing
  rule.
- `sphinx/api/<module>.md` renders each module with `automodule`.
- catsim's `sphinx/api/techniques/` has one page per strategy, from
  `sphinx/api/techniques/_template.md`: Motivation, Definition (with an optional Computation
  subsection), Parameters, Behavior and Comparison (with a closed catR dropdown), Guarantees,
  References, API Reference.

## Proposal

**Pages.** A Methods section in the navbar, `sphinx/methods/`, of 13 pages in 5 families, each
family with an index page:

- Models: item response functions (`irt`: 1PL to 4PL, information, standard error, reliability, the
  log-likelihood), and quadrature and ability priors (`quadrature`). The method pages link these
  instead of re-deriving a quantity.
- Calibration: MML-EM, CML, JML.
- Ability estimation: ML, MAP, EAP, WLE.
- Item fit: S-X², mean squares.
- Model comparison: the likelihood-ratio test, AIC, BIC and G², on one page.

`bank`, `calibration`, `drawing` and `typing` keep their API reference pages, with no method page.

**Template** (`sphinx/methods/_template.md`), adapted from catsim's so that irtlib's implementation is
the core of the page:

1. Motivation: short; what the method does and when to choose it.
2. Definition: the formulas, each cited to its page and equation.
3. Implementation: required, and the longest section. The numerical choices a user can observe (EM
   steps, quadrature nodes, the convergence rule, acceleration, starting values, how the standard
   errors are computed, how extreme and missing responses are handled), each cited to its source or
   linked to the ADR where irtlib decided it.
4. Parameters: a table with a "Default from" column naming each default's origin: a paper's page,
   mirt's default, or an ADR.
5. Guarantees: properties a user relies on and can observe.
6. A closed "Comparison with mirt" dropdown, the page's only mention of mirt: the function and
   settings that match, what the parity tests find, and each departure.
7. References: a `{bibliography}` filtered to the page.
8. API Reference: the method's classes and functions.

Code internals stay out: private helpers, array layouts, caches.

**Docstrings.** Each module docstring keeps a sentence on what the module holds and links its page;
each function and class docstring says in one sentence what it computes, gives the call contract and
links the page, about 80 words before the first numpydoc section, catsim's budget. No formula
appears twice on one rendered page.

**Charter.** Add the method pages' rules to `sphinx/AGENTS.md`, adapted from catsim's "Technique
pages", with the template above; rewrite "Where a formula lives" and the one-home map; edit ADR-0004
in place if it still says that formulas live in module docstrings.

**Cards.** Align the technique sections of `backlog/_template.md` with the method-page template, and
the "Promote" step of the workflow in `backlog/README.md`.

## Tests

- The site builds with `-W`, every cross-reference resolved.
- Every source a page cites is an entry of `sphinx/references.bib` with its DOI and note, and every
  location cited is one read on the paper shelf (`backlog/README.md`, revalidation checklist).
- No rendered page shows a formula twice: a method page and the docstrings it renders under API
  Reference.
- Every default in a Parameters table names its origin.
