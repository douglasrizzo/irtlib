# S-X2 item fit (Orlando & Thissen, 2000) of 1PL and 2PL calibrations (irtlib.fit), with mirt's
# default collapsing (mincell = 1) on its default grid, 61 points on [-6, 6].
#
# Each case records mirt's estimates, so the Python test computes S-X2 at the same parameters, and
# mirt's S-X2, degrees of freedom and p-values. The fits use the settings of mml_em.R. S-X2 needs
# complete responses, so the simulated data set has none missing. Each case also records the table
# S-X2 is computed from before collapsing, itemfit(return.tables = TRUE)'s O.org and E.org, which
# S_X2.plot draws: the number of examinees at each number correct from 1 to M - 1, and each item's
# correct responses and expected number correct there.
#
# The collapse cases run mirt's own collapsing, collapseCells (R/utils.R:2410-2546), on random tables
# of incorrect and correct counts by score, with empty rows, rows of one examinee, ties, small
# expected counts and expected counts of 0; an expected count of 0 is written as 0, where mirt marks
# it NA.
#
# The mean-square cases are mirt's itemfit(fit_stats = "infit") of Rasch fits of LSAT6 and LSAT7:
# slopes fixed at 1 and the ability variance estimated, irtlib's 1PL with its abilities multiplied
# by their standard deviation. The abilities are fscores' EAP estimates on irtlib's grid, the 61
# nodes on [-6, 6] multiplied by that standard deviation, where the posterior weights are irtlib's.
# The Rasch M-step runs nlminb, whose tolerance is rel.tol. Each case records the standard
# deviation, the intercepts, each pattern's ability and mirt's mean squares with their standardized
# values.
#
# The mean-square guard case runs itemfit(fit_stats = "infit") on every response pattern of three
# Rasch items of fixed parameters, with abilities passed in, five responses 7 or more logits from
# their item: mirt raises their variance (R/personfit.R:200), and caps the standard deviation of each
# unweighted mean square of the eight examinees (R/itemfit.R:586). fixed_items_model() comes from
# response_functions.R.

ITEM_FIT_SEED <- 20261128L
COLLAPSE_SEED <- 20261129L
# a, b, c, d
MEAN_SQUARE_GUARD_ITEMS <- rbind(c(1, -1, 0, 1), c(1, 0, 0, 1), c(1, 1.5, 0, 1))
MEAN_SQUARE_GUARD_THETA <- c(-8, -3, -1, 0, 0.5, 2, 4, 7)

item_fit_case <- function(name, x, model) {
  fitted <- fit(x, model, "Oakes")
  stopifnot(extract.mirt(fitted, "converged"))
  statistics <- itemfit(fitted, fit_stats = "S_X2", mincell = 1)
  tables <- itemfit(fitted, fit_stats = "S_X2", mincell = 1, return.tables = TRUE)
  list(
    id = paste0(name, "/", model),
    data = name,
    model = model,
    slope = item_values(fitted, "par", "a1"),
    intercept = item_values(fitted, "par", "d"),
    s_x2 = as_array(statistics$S_X2),
    df = as_array(as.integer(statistics$df.S_X2)),
    p_value = as_array(statistics$p.S_X2),
    # one row per item, one column per number correct from 1 to M - 1; column 2 of a table is the correct responses
    table_sizes = as_array(rowSums(tables$O.org[[1L]])),
    table_correct = t(sapply(tables$O.org, function(observed) observed[, 2L])),
    table_expected = t(sapply(tables$E.org, function(expected) expected[, 2L]))
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
  mean_squares <- lapply(c("lsat6", "lsat7"), function(name) mean_square_case(name, data[[name]], matrices[[name]]))
  list(
    settings = MML_EM_SETTINGS, data = data, cases = cases, collapse = collapse_cases(), mean_squares = mean_squares,
    mean_square_guards = mean_square_guard_case()
  )
}

mean_square_guard_case <- function() {
  model <- fixed_items_model(MEAN_SQUARE_GUARD_ITEMS)
  statistics <- itemfit(model, fit_stats = "infit", Theta = matrix(MEAN_SQUARE_GUARD_THETA))
  list(
    items = MEAN_SQUARE_GUARD_ITEMS,
    responses = unname(extract.mirt(model, "data")),
    theta = as_array(MEAN_SQUARE_GUARD_THETA),
    outfit = as_array(statistics$outfit),
    z_outfit = as_array(statistics$z.outfit),
    infit = as_array(statistics$infit),
    z_infit = as_array(statistics$z.infit)
  )
}

mean_square_case <- function(name, data, x) {
  fitted <- mirt(
    x, 1,
    itemtype = "Rasch", quadpts = MML_EM_SETTINGS$quadrature_points, TOL = MML_EM_SETTINGS$tol, verbose = FALSE,
    technical = list(NCYCLES = MML_EM_SETTINGS$max_cycles, theta_lim = MML_EM_SETTINGS$theta_limits),
    control = list(rel.tol = MML_EM_SETTINGS$reltol)
  )
  stopifnot(extract.mirt(fitted, "converged"))
  sd <- sqrt(coef(fitted, simplify = TRUE)$cov[1, 1])
  limits <- MML_EM_SETTINGS$theta_limits
  nodes <- seq(limits[1], limits[2], length.out = MML_EM_SETTINGS$quadrature_points)
  theta <- fscores(fitted, method = "EAP", custom_theta = matrix(sd * nodes), full.scores = TRUE, verbose = FALSE)
  statistics <- itemfit(fitted, fit_stats = "infit", Theta = theta)
  # the rows of a pattern are adjacent in the expanded matrix, so its first row gives its ability
  counts <- unlist(data$counts)
  first <- cumsum(counts) - counts + 1L
  list(
    id = paste0(name, "/rasch"),
    data = name,
    sd = sd,
    intercept = item_values(fitted, "par", "d"),
    theta = as_array(theta[first, 1]),
    outfit = as_array(statistics$outfit),
    z_outfit = as_array(statistics$z.outfit),
    infit = as_array(statistics$infit),
    z_infit = as_array(statistics$z.infit)
  )
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
