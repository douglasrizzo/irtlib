# S-X2 item fit (Orlando & Thissen, 2000) of 1PL and 2PL calibrations (pyitemcal.fit), with mirt's
# default collapsing (mincell = 1) on its default grid, 61 points on [-6, 6].
#
# Each case records mirt's estimates, so the Python test computes S-X2 at the same parameters, and
# mirt's S-X2, degrees of freedom and p-values. The fits use the settings of mml_em.R. S-X2 needs
# complete responses, so the simulated data set has none missing.
#
# The collapse cases run mirt's own collapsing, collapseCells (R/utils.R:2410-2546), on random tables
# of incorrect and correct counts by score, with empty rows, rows of one examinee, ties, small
# expected counts and expected counts of 0; an expected count of 0 is written as 0, where mirt marks
# it NA.

ITEM_FIT_SEED <- 20261128L
COLLAPSE_SEED <- 20261129L

item_fit_case <- function(name, x, model) {
  fitted <- fit(x, model, "Oakes")
  stopifnot(extract.mirt(fitted, "converged"))
  statistics <- itemfit(fitted, fit_stats = "S_X2", mincell = 1)
  list(
    id = paste0(name, "/", model),
    data = name,
    model = model,
    slope = item_values(fitted, "par", "a1"),
    intercept = item_values(fitted, "par", "d"),
    s_x2 = as_array(statistics$S_X2),
    df = as_array(as.integer(statistics$df.S_X2)),
    p_value = as_array(statistics$p.S_X2)
  )
}

item_fit_cases <- function() {
  data <- list(lsat6 = from_table(LSAT6), lsat7 = from_table(LSAT7), simulated = as_patterns(simulated_responses(ITEM_FIT_SEED, 0)))
  data$simulated$seed <- ITEM_FIT_SEED
  matrices <- lapply(data, from_patterns)
  cases <- list()
  for (name in names(data)) {
    for (model in c("2pl", "equal_slopes", "fixed_slopes")) {
      cases[[length(cases) + 1L]] <- item_fit_case(name, matrices[[name]], model)
    }
  }
  list(settings = MML_EM_SETTINGS, data = data, cases = cases, collapse = collapse_cases())
}

collapse_cases <- function() {
  set.seed(COLLAPSE_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  lapply(seq_len(200L), function(case) {
    repeat {
      totals <- sample(c(0, 1, 1, 2, 2, 3, 5, 8, 20), sample(3:8, 1L), replace = TRUE)
      if (sum(totals >= 2) >= 3L) break
    }
    correct <- rbinom(length(totals), totals, runif(length(totals)))
    proportion <- runif(length(totals), 0.02, 0.98)
    # every fourth table has a proportion of exactly 0 or 1, so an expected count of 0, which no rule merges
    if (case %% 4L == 0L) proportion[sample.int(length(proportion), 1L)] <- sample(c(0, 1), 1L)
    observed <- cbind(totals - correct, correct)
    expected <- cbind(totals * (1 - proportion), totals * proportion)
    collapsed <- mirt:::collapseCells(list(observed), list(expected), mincell = 1)
    collapsed_observed <- matrix(collapsed$O[[1L]], ncol = 2L)
    collapsed_expected <- matrix(collapsed$E[[1L]], ncol = 2L)
    collapsed_expected[is.na(collapsed_expected)] <- 0
    list(
      observed = observed,
      expected = expected,
      collapsed_observed = collapsed_observed,
      collapsed_expected = collapsed_expected
    )
  })
}
