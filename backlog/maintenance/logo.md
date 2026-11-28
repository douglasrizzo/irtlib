---
title: Draw irtlib's logo
kind: docs
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [maintainer on 2026-11-28]
---

# Draw irtlib's logo

## Problem

The site's navbar shows the project's name as text, the browser tab has no icon, and the README and
the repository's social preview have no image. catsim has a logo in each place, and irtlib is its
sister project.

## What irtlib and catsim have now

- catsim's `sphinx/logo/` (commit `bc277343` there): a mark drawn as a stylized test-progress plot,
  slate lines (`#1e293b` on light, `#e2e8f0` on dark) with orange nodes (`#ff9800`, Material Orange
  500), no outlines; the wordmark in Outfit 400 (SIL Open Font License) converted to outlines, with
  the dot of the "i" drawn as an orange node. Files: `icon-{light,dark}.svg`,
  `horizontal-{light,dark}.svg` (the navbar), `stacked-{light,dark}.svg` (the README),
  `favicon.svg` (a simpler mark on a slate tile, for light and dark tab bars from one file), and
  `social-preview.{svg,png}` (1280 × 640).
- irtlib's site colors are light blue (`sphinx/_static/irtlib.css`): `#0277bd` for links on the
  light theme and `#4fc3f7` on the dark one, with the logo's slate for a link under the pointer.

## Proposal

The maintainer chose on 2026-11-28:

- **Mark:** a slate item characteristic curve, an S-shaped logistic curve between dashed lower and
  upper asymptotes (the guessing parameter and the upper asymptote), with light-blue nodes for the
  observed proportions correct of groups of examinees, lying near the curve: calibration fitting the
  curve to the data. It reuses catsim's dashed lines and nodes.
- **Color:** the nodes in `#03a9f4` (Material Light Blue 500, the counterpart of catsim's Orange
  500); lines in catsim's slates.
- **Wordmark:** "irtlib" in Outfit 400, converted to outlines, with the dots of both i's drawn as
  light-blue nodes.
- **Files:** catsim's set, in `sphinx/logo/`. The navbar takes the horizontal logo per color mode
  (`html_theme_options["logo"]`), the favicon goes in `html_favicon`, the README shows the stacked
  logo by absolute URL, and the maintainer uploads the social preview in the repository's settings.

Draw a few candidates of the mark first, for the maintainer to choose from.

## Tests

- Each SVG is valid XML (the `check-xml` hook) and has a `<title>`.
- The nodes and lines are visible on both themes' backgrounds, and the favicon on light and dark tab
  bars.
- The site builds with `-W` with the logo set.
