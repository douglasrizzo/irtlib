---
title: Choose pyitemcal's license
kind: decision
status: needs-decision
priority: 2
effort: XS
blocked_by: []
raised_by: [maintainer during setup]
---

# Choose pyitemcal's license

## Problem

pyitemcal has no license: no `LICENSE` file and no `license` field in `pyproject.toml`. The
repository is public, so anyone can read the code, but without a license nobody may copy, modify or
redistribute it, and it cannot be published to PyPI as open source.

## What pyitemcal and catsim have now

catsim is BSD-3-Clause. NumPy's PyPI metadata gives `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND
CC0-1.0`, for its bundled code. mirt 1.48, the reference implementation pyitemcal's parity tests
compare against, is "GPL (>= 3)" in its CRAN DESCRIPTION; the parity fixtures will hold mirt's
outputs, not its code. (Checked 2026-11-26.)

## Proposal

Choose a license, then add the `LICENSE` file, the SPDX `license` field in `pyproject.toml` and the
README's license line in one PR. The options include BSD-3-Clause, as catsim and the scientific
Python stack use, a copyleft license, or another permissive one.

## Open Questions

1. Whether fixtures that record mirt's outputs need any notice. The research pass or the parity
   harness card can check mirt's license terms on that point.
