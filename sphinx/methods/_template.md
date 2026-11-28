# \<Method Name>

:API: {py:class}`irtlib.<module>.<Name>`
:Sources: {cite:t}`<key>`; {cite:t}`<key>`

% METHOD PAGE TEMPLATE. Copy this file into sphinx/methods/<family>/, fill in each section, delete
% these comments and add the page to the toctree of the family's index.md. The rules for method
% pages are in sphinx/AGENTS.md ("Method pages"); the citation convention is in backlog/README.md.
% The header is a MyST field list: one :API: entry per class or function the page documents.
% Every source is an entry of sphinx/references.bib, cited by its key: {cite:t} in a sentence,
% {cite:p} in parentheses, one role per source when it carries a location, e.g.
% {cite:t}`Bock1981{p. 447}`. Link an ADR as {doc}`ADR-0007 </adr/0007-mml-em-settings>`.
% Budget: about 600-1,500 words before the references.

## Motivation

\<1-2 paragraphs for a researcher: what the method does, the problem it solves and when to choose
it over the alternatives. Plain language; introduce every symbol in words before using it.>

## Definition

\<The method's definition, with `$$` blocks for display equations. Cite the source of each formula
where it appears, with its page and equation, citing only locations you have read. Refer to the
Item Response Functions page for a quantity such as information or the log-likelihood rather than
re-derive it. Use the symbols of terms-and-notation.md.>

$$
<equation>
$$

## Implementation

\<How irtlib computes the definition: the numerical choices a user can observe, such as the search,
the grid, the starting values, the stopping rule, the acceleration, how the standard errors are
computed and how extreme or missing responses are handled. Cite each choice to its source, or link
the ADR where irtlib made it. State every departure from the sources.>

% A closed dropdown at the end of Implementation, the page's only mention of mirt: the mirt function
% and settings that match, with mirt's version, what the parity tests find, and each departure.
% Delete it when mirt has no counterpart: do not say that mirt lacks the method.

:::{dropdown} Comparison with mirt

\<The mirt function and settings that match, what the parity tests (ADR-0003) find, and each
departure.>
:::

## Parameters

% One row per parameter a caller sets. "Default from" names where the default comes from: a
% source's page ({cite:t}`key{p. N}`), mirt's default (with the manual's page), an ADR, or
% "irtlib; no source yet" with a link to backlog/discovery/sources-wanted.md, where the row is.

```{list-table}
:header-rows: 1
:widths: 18 14 42 26

* - Parameter
  - Default
  - Description
  - Default from
* - `<param>`
  - `<default>`
  - \<what it controls, its valid values and the range the literature recommends>
  - \<origin>
```

## Guarantees

% Properties a user relies on and can observe, in the user's terms. Properties only an
% implementation sees go to the docstring and the tests.

1. \<A property a user can observe, e.g. "Responses all correct or all incorrect raise no error.">

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: irtlib.<module>.<Name>
   :members:
```
