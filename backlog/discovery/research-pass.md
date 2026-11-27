---
title: Rebuild the research behind calibration from primary sources
kind: research
status: ready
priority: 1
effort: XL
blocked_by: []
raised_by: [setup review of the cards carried over from catsim]
---

# Rebuild the research behind calibration from primary sources

## Problem

Every card in `irt/`, `estimation/` and `fit/`, and the literature review, came over from catsim's
backlog, where they were written in April 2026 and never checked against their sources. Their
equations, ratings, defaults and references may be wrong or misleading, and the implementation work
would inherit any error. Nothing in pyitemcal can be designed or built on them as they are.

## Evidence

- The cards say so themselves: catsim parked them as outside its scope, and catsim's revalidation
  of its technique cards in November 2026 did not cover them.
- catsim's paper shelf has none of the calibration sources the cards cite: no Bock & Aitkin (1981),
  Baker & Kim (2004), Mislevy (1986), Patz & Junker (1999), Andersen (1972), Wright & Masters (1982)
  or Yen (1981).
- catsim's notes list Bock & Aitkin (1981) as having no open copy.

## Proposal

One pass, in PRs of reviewable size, that produces:

1. **The paper shelf.** `papers/`, local and gitignored (`backlog/README.md`, "The paper shelf"),
   with a PDF of every source used and a `papers/README.md` index of each file's full citation and
   page offset. Agents fetch the open copies; a source behind a paywall goes on a list for the
   maintainer to fetch.
2. **The bibliography.** An entry in `sphinx/references.bib` for every source used, with its DOI and
   its note (`Open copy: \url{...}`, `Open access`, or `Not openly available; ...`).
3. **A new literature review**, replacing `literature-review.md`: each method, what its primary
   sources establish, its known properties (consistency, bias, identification, failure modes), how
   mirt implements it and with which defaults, and what pyitemcal should build in what order. Every
   claim is cited to a location read on the shelf.
4. **The method cards, rewritten or deleted.** Each card in `irt/`, `estimation/` and `fit/` keeps
   only what the sources verify: wrong or misleading equations, claims, ratings and references are
   deleted, not corrected from memory. A card the review drops is deleted; a method the review adds
   gets a card. Each rewritten card has its "Last validated" line and passes the revalidation
   checklist.
5. **Questions for the requirements card**, listed in `discovery/requirements.md`, wherever the
   sources leave a design choice open.

### Starting list

The pass confirms or drops each item; the list was written from memory and is itself to be checked.

- **From the carried-over cards:** joint maximum likelihood; marginal maximum likelihood with the EM
  algorithm for the 1PL to 4PL (Bock & Aitkin, 1981); conditional maximum likelihood for the Rasch
  model (Andersen); Markov chain Monte Carlo (Albert, 1992; Patz & Junker, 1999); item fit.
- **Added:**
  - the Metropolis-Hastings Robbins-Monro algorithm (Cai, 2010);
  - standard errors of item estimates: Louis (1982), the supplemented EM algorithm (Meng & Rubin,
    1991), Oakes (1999), the cross-product approximation and the sandwich estimator;
  - Bayes modal estimation with priors on the item parameters, including the guessing parameter
    (Mislevy, 1986);
  - calibration of the 4PL (Loken & Rulison, 2010; Culpepper, 2016);
  - item-fit statistics: $S\text{-}X^2$ (Orlando & Thissen, 2000), Yen's $Q_1$ (Yen, 1981), infit and
    outfit;
  - starting values, convergence criteria, scale identification and EM acceleration;
  - model comparison: AIC, BIC and likelihood-ratio tests.
- **Software:** mirt's algorithms, defaults and outputs, as the reference (ADR-0003); ltm, TAM, girth
  and py-irt for context only.

## Tests

The pass is done when:

- every source cited anywhere in `backlog/` is on the shelf, indexed, and in `references.bib`;
- every card in `irt/`, `estimation/` and `fit/` has a "Last validated" line and no banner saying it
  is unverified, or has been deleted;
- a reviewer can pick any cited location at random and find the claim at that page of the source.

## Open Questions

1. How deep the software survey goes for packages other than mirt: assumed, their defaults only,
   where they differ from mirt's.
