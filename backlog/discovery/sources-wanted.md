---
title: Find sources for the claims and defaults that have none on the shelf
kind: research
status: needs-research
priority: 3
effort: S
blocked_by: []
raised_by: [maintainer on 2026-11-27]
---

# Find sources for the claims and defaults that have none on the shelf

## Problem

irtlib's research rules have every formula, claim about a method and default rest on a source read
on the paper shelf (`AGENTS.md`, `backlog/README.md`). Some of the code irtlib takes from catsim, and
some defaults, have no such source yet. This card lists each gap, so that it is not forgotten and an
agent can ask the maintainer to look for the source.

An agent that meets a claim, formula or default with no source on the shelf adds a row here, with
where irtlib uses it and what was searched, and asks the maintainer. When the maintainer cannot find
one, the code may follow a reference implementation, read at its file and line, and say so; the row
stays until a published source is read (maintainer, 2026-11-27; `backlog/README.md`, "Citation
convention").

## The gaps

| Claim or default | Where irtlib uses it | Source wanted | Searched | Stand-in |
|---|---|---|---|---|
| Reliability at an ability, catsim's $1 - 1/I(\theta)$ | `irtlib.irt.reliability` (`irt/response-functions`) | Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized adaptive testing: A primer* (2nd ed., pp. 159-184). Erlbaum. doi:10.4324/9781410605931-15 (catsim's `sphinx/references.bib`) | The maintainer could not find it (2026-11-27). Bock and Mislevy (1982, p. 437, Equation 7) define $1 - \mathrm{PSD}^2$ for the EAP estimate, a related quantity | mirt 1.48's reliability function, $I/(I + 1/\sigma^2)$ (`R/SingleGroup-methods.R:1474`) |
| Marginal reliability, should irtlib add it | not yet | Thissen, D., & Wainer, H. (2001). *Test scoring*. Erlbaum, which mirt 1.48's `marginal_rxx` cites (`R/marginal_rxx.R:4, 18`) | not searched | none |
| The closed-form integral of the item information | `irtlib.irt.info_integral` | any published closed form | Magis (2013) and Warm (1989) have none | a derivation from Magis (2013, p. 307, Equation 6), checked against mirt 1.48's `areainfo` (`R/areainfo.R:49-66`) and numerical integration |
| The 4PL information and the ability of maximum information for every $0 \le c < d \le 1$ | `irtlib.irt.information`, `max_info_theta` | a source without Magis's restriction | Magis (2013, p. 306) assumes $c \in (0, 0.5)$ and $d \in (0.5, 1)$ | the same formulas, tested by brute force over the whole range |
| Whether the second number of Barrada et al.'s $a \sim N(1.2, .25)$ and $c \sim N(.25, .02)$ is a standard deviation or a variance, and the truncation of $a$ and clipping of $c$ | `irtlib.ItemBank.generate` | a paper or code by the authors that states the notation or the bounds, or another source for these distributions that does | Barrada, Olea, Ponsoda and Abad (2010, p. 445), Barrada, Mazuela and Olea (2006, p. 157) and Barrada, Olea, Ponsoda and Abad (2008, p. 499) give the same three distributions; none defines the notation, truncates $a$ or bounds $c$ (all three read on the shelf, 2026-11-27). Indirect evidence leans to standard deviations without settling it: the ability of maximum information across the 2006 banks is $N(0.16, 1)$ (p. 159), which a simulation of the standard-deviation reading reproduces best, though a variance reading with $a$ bounded away from 0 comes close (the arithmetic of `papers/notes/barrada_mazuela_olea_2006.md`, not the paper's); and the operative bank of 2008 has a standard deviation of $c$ of .03 (p. 508, Table 5), the size of .02 read as one, which the paper does not link to its simulated banks | catsim's reading, as standard deviations, with $a$ truncated to $a > 0$ by redrawing and $c$ clipped at 0, labelled as irtlib's choices in `irtlib.bank`'s docstring |
| The upper asymptote of random 4PL banks, $d \sim U(0.94, 1)$ | `irtlib.ItemBank.generate` | any published distribution for $d$ | catsim calls it its own (`src/catsim/item_bank.py:399-402`); the simulated banks of Barrada et al. (2006, p. 157; 2008, p. 499; 2010, p. 445) draw only $a$, $b$ and $c$ | catsim's default, labelled as irtlib's choice in `irtlib.bank`'s docstring |
| The 201 search points of the MAP estimate and the 121 scan points of the WLE | ability estimation (`estimation/ability-estimation`) | a source, or the maintainer's acceptance as irtlib's choices | none | catsim's defaults |
| How catR treats constant patterns (`constantPatt`) | the catR parity cases of ability estimation | Magis, D., & Barrada, J. R. (2017). Computerized adaptive testing with R: Recent updates of the package catR. *Journal of Statistical Software, 76*(Code Snippet 1). doi:10.18637/jss.v076.c01 (catsim's `sphinx/references.bib`) | on catsim's shelf (`magis_barrada_2017.pdf`), not yet on irtlib's | none |

## Proposal

The maintainer looks for each source in the table. A source that arrives goes on the shelf, the claim
is cited at the page read, and its row leaves this card.
