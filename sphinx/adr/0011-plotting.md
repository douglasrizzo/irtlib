# ADR-0011: Plotting Is an Optional Module With One Calling Convention

## Status

Accepted 2026-11-28

## Context

Item response theory is read from curves: an item's characteristic and information curves, a test's information, the
observed proportions correct against a model's. catsim, which builds on irtlib (ADR-0004), drew five plots of item
response quantities in `catsim.plot`, beside six plots of adaptive testing, so a researcher calibrating a fixed-form
test had to install a CAT package to see an item. ADR-0002 keeps plotting libraries out of irtlib's runtime
dependencies, and asks for an ADR of its own for a new one.

catsim settled how its plots are called (its ADR-0022): matplotlib as an optional extra, data first, options keyword
only, an `ax` to draw on, the caller's style. Two of its plots drew a second y scale on the right of one axes, where a
reader can take a crossing of the two curves for a fact about the item.

## Decision

- matplotlib is an optional dependency, installed with the `irtlib[plot]` extra. `irtlib.plot` is the only module
  that imports it; importing `irtlib`, or any other module, never loads it, and import-linter contracts in
  `pyproject.toml` check both rules. Importing `irtlib.plot` without matplotlib raises an `ImportError` that gives the
  install command.
- Every plot of an item response quantity that a fixed-form test has belongs in `irtlib.plot`; plots of adaptive
  sessions and simulation studies stay in catsim.
- A plot function takes its data first, as irtlib's objects and arrays, and its options as keyword-only arguments.
  It draws on the axes the caller passes as `ax`, or on those of a new figure, and returns the axes it drew on:

  ```python
  item_curve(items, item, *, kind="icc", ax=None, title=None, label=None) -> Axes
  ```

- Each function draws on one axes with one y scale. Two quantities on different scales are two calls on axes the
  caller lays out, sharing the ability axis.
- A plot of one series takes a `label`, so that several drawn on one axes compare.
- A plot follows the caller's matplotlib style, and sets no fonts, colours, title size or grid of its own. A colour it
  takes from the style's cycle is fixed when it is called.
- Items are labelled by their names in an item bank (ADR-0009); an item matrix names them by position.

## Consequences

- An installation without the extra fails at `import irtlib.plot`, with the fix in the message.
- catsim's item response plots move to irtlib, and catsim's documentation links to them.
- A figure with several panels is laid out with matplotlib's tools; the documentation's examples show how.
- A caller sets how plots look with matplotlib's styles alone. irtlib ships no style; the documentation's figures use
  `sphinx/_static/figures.mplstyle`.
- The tests and the documentation need matplotlib, which the development and documentation groups install.

## Alternatives Considered

- Leaving the plots in catsim: a fixed-form user would install a CAT package to draw an item.
- matplotlib required: every installation pays for figures it never draws.
- Importing matplotlib inside each function: the error would surface at the first call, far from the missing
  installation.
- A separate plotting package: a second distribution to version and release.
- A second y axis for two quantities: one axes, two scales, and crossings that mean nothing.
- A `figsize` argument: it does nothing when the caller passes `ax`, and `plt.subplots` provides it.
