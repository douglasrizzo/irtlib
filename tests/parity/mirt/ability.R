# Ability estimates and their standard errors (fscores) at fixed item parameters (irtlib.ability): EAP,
# MAP, ML and WLE, on a 2PL bank and on a bank of 3PL and 4PL items, for response patterns with
# missing responses, all correct or all incorrect, and none answered.
#
# Each pattern is scored alone, through response.pattern, which drops its missing items. EAP uses
# fscores' default grid, 121 nodes on [-6, 6], irtlib's; MAP the group's N(0, 1) prior. nlm's
# gradient tolerance is tightened to 1e-10 so that its estimates sit at the optimum. A missing value
# that fscores returns, NA, is written as NaN. Patterns are strings, one character per item: "1"
# correct, "0" incorrect and "." missing. fixed_items_model() comes from response_functions.R.
#
# The number-correct cases are fscores' EAP of each number correct (method = "EAPsum",
# full.scores = FALSE) on each bank, on the same grid: the estimate and its standard error for each
# number correct from 0 to 6, and the expected number of examinees with it, the model's probability
# of the number correct times the rows of the model's data, its 64 response patterns.

ABILITY_SETTINGS <- list(quadpts = 121L, theta_lim = c(-6, 6), gradtol = 1e-10)

# a, b, c, d
ABILITY_BANKS <- list(
  `2pl` = rbind(
    c(1.2, -1.5, 0, 1), c(0.8, -0.6, 0, 1), c(1.6, 0, 0, 1), c(1.0, 0.7, 0, 1), c(2.0, 1.3, 0, 1), c(1.4, -0.3, 0, 1)
  ),
  `4pl` = rbind(
    c(1.2, -1.5, 0.2, 1), c(0.8, -0.6, 0, 0.95), c(1.6, 0, 0.15, 1), c(1.0, 0.7, 0.1, 0.9), c(2.0, 1.3, 0.25, 1),
    c(1.4, -0.3, 0.1, 0.95)
  )
)

ABILITY_PATTERNS <- list(
  mixed = c("110100", "1.0.10", "011001", "10.01.", "111010", "0110.."),
  constant = c("111111", "000000", "1.11..", "..0.00"),
  unanswered = "......",
  # 3PL and 4PL patterns whose likelihood has its supremum at -inf, -inf and +inf
  supremum = c("000010", "0.0.1.", "111110")
)

# One entry per case: the method, the bank and the patterns it scores.
ABILITY_CASES <- list(
  list(name = "eap", method = "EAP", groups = c("mixed", "constant"), banks = c("2pl", "4pl")),
  list(name = "eap_unanswered", method = "EAP", groups = "unanswered", banks = "2pl"),
  list(name = "map", method = "MAP", groups = c("mixed", "constant"), banks = c("2pl", "4pl")),
  list(name = "map_unanswered", method = "MAP", groups = "unanswered", banks = "2pl"),
  list(name = "ml", method = "ML", groups = "mixed", banks = c("2pl", "4pl")),
  list(name = "ml_constant", method = "ML", groups = "constant", banks = c("2pl", "4pl")),
  list(name = "ml_supremum", method = "ML", groups = "supremum", banks = "4pl"),
  list(name = "ml_unanswered", method = "ML", groups = "unanswered", banks = "2pl"),
  list(name = "wle", method = "WLE", groups = c("mixed", "constant"), banks = c("2pl", "4pl")),
  list(name = "wle_unanswered", method = "WLE", groups = "unanswered", banks = "2pl")
)

pattern_row <- function(pattern) {
  characters <- strsplit(pattern, "")[[1]]
  as.numeric(ifelse(characters == ".", NA, characters))
}

score_pattern <- function(model, method, pattern) {
  rows <- rbind(pattern_row(pattern))
  # fscores stops on a single pattern with no response; in a matrix it drops such a row and returns NA
  # for it (R/fscores.internal.R:94-96, 185), so the pattern is scored beside an answered one
  if (all(is.na(rows))) rows <- rbind(rows, pattern_row(ABILITY_PATTERNS$mixed[1]))
  scores <- suppressWarnings(fscores(model,
    method = method, response.pattern = rows, full.scores.SE = TRUE,
    append_response.pattern = FALSE, quadpts = ABILITY_SETTINGS$quadpts, theta_lim = ABILITY_SETTINGS$theta_lim,
    gradtol = ABILITY_SETTINGS$gradtol, verbose = FALSE
  ))
  values <- c(scores[1, 1], scores[1, 2])
  values[is.na(values)] <- NaN
  values
}

ability_cases <- function() {
  models <- lapply(ABILITY_BANKS, fixed_items_model)
  cases <- list()
  for (case in ABILITY_CASES) {
    patterns <- unlist(ABILITY_PATTERNS[case$groups], use.names = FALSE)
    for (bank in case$banks) {
      scores <- vapply(patterns, function(pattern) score_pattern(models[[bank]], case$method, pattern), numeric(2))
      cases[[length(cases) + 1L]] <- list(
        id = paste0(case$name, "/", bank),
        method = case$method,
        bank = bank,
        patterns = as_array(patterns),
        counts = as_array(rep(1L, length(patterns))),
        theta = as_array(scores[1, ]),
        se = as_array(scores[2, ])
      )
    }
  }
  number_correct <- lapply(names(ABILITY_BANKS), function(bank) number_correct_case(bank, models[[bank]]))
  list(settings = ABILITY_SETTINGS, banks = ABILITY_BANKS, cases = cases, number_correct = number_correct)
}

number_correct_case <- function(bank, model) {
  table <- fscores(model,
    method = "EAPsum", full.scores = FALSE, quadpts = ABILITY_SETTINGS$quadpts,
    theta_lim = ABILITY_SETTINGS$theta_lim, verbose = FALSE
  )
  list(
    id = paste0("eap_sum/", bank),
    bank = bank,
    n_examinees = nrow(extract.mirt(model, "data")),
    theta = as_array(table$F1),
    se = as_array(table$SE_F1),
    expected = as_array(table$expected)
  )
}
