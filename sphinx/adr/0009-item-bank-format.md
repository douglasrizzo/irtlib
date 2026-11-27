# ADR-0009: The Item Bank Format

## Status

Accepted 2026-11-27

## Context

A calibration produces a bank of items and an adaptive test consumes one, so the bank's shape, its
item names and its file are a contract between packages. catsim defined them in its ADR-0003 to
ADR-0005; ADR-0001 and ADR-0004 make the format irtlib's, with catsim building on it. The format
below is catsim's, so a bank written by either package reads the same in the other.

## Decision

- **The matrix.** A bank holds an $n \times 4$ matrix of four-parameter logistic parameters, one row
  per item, with the columns $a$, $b$, $c$ and $d$. `ItemBank(items)` takes one to four columns,
  read as $[b]$, $[a, b]$, $[a, b, c]$ or $[a, b, c, d]$, and fills the missing parameters with
  $a = 1$, $c = 0$ and $d = 1$; a 1-D array is one item, and more than four columns are rejected.
- **Validity.** A bank always validates its items: every parameter finite, with $a > 0$ and
  $0 \le c < d \le 1$. A calibration's raw estimates may break these bounds; it reports them, and
  builds its bank from the items that keep them (ADR-0006).
- **Immutability.** A bank's arrays are read-only, so the values it caches from its parameters, such
  as each item's ability of maximum information, cannot go stale.
- **The model.** A bank reports the simplest model that fits every item as an `IrtModel`, a
  `StrEnum` whose members equal `"1PL"` to `"4PL"`.
- **Names and indices.** Every item has a name, a non-empty string unique within its bank, given or
  generated as `item_` and the zero-padded row index (`item_000` to `item_499` in a 500-item bank).
  Computations use row indices; outputs read by people, and error messages, carry names; other
  per-item inputs, such as a response matrix's columns, match items by position.
- **The file.** `ItemBank.from_csv` and `to_csv` read and write one CSV format: comma-separated with
  a dot as the decimal mark and a required header; parameter columns named `a` to `d` in any
  order, with `b` required and the defaults above for the others; an optional name column headed
  `name` or empty, as R's `write.csv` heads row names; any other column an error that names it.
  `to_csv` writes each number so that it reads back exactly.

## Consequences

- One shape serves every model: with $c = 0$ and $d = 1$ the four-parameter formulas are the
  two-parameter ones, so no function branches on the model.
- A bank read from a file or built from a matrix is valid, so nothing downstream checks its
  parameters again.
- A bank saved from R with `write.csv` loads with its names, and a bank written here reads back
  identical.
- Files in other layouts are read with pandas or NumPy and passed as `ItemBank(matrix, names=...)`;
  reading the format needs no dependency beyond the standard library.

## Alternatives Considered

- A matrix of only the estimated parameters per model: every function would branch on the model.
- Validating on request rather than always: every consumer would have to check, or trust, the bank.
- Identifying items by index only: an item's identity would be lost whenever a table is reordered.
- Other file formats, such as JSON or a binary array file: harder to edit and to exchange with R.
