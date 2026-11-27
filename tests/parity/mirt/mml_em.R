# Marginal maximum likelihood with the EM algorithm for the 1PL and 2PL, with Oakes's standard errors
# and Louis's as a cross-check (pyitemcal.mml).
#
# Every fit runs to TOL = 1e-8 with an exact M-step, control = list(reltol = 1e-14), because mirt's
# defaults stop short of the maximum (ADR-0003, ADR-0007). The grid is mirt's default, 61 points on
# [-6, 6], passed explicitly.
#
# A data set's responses are stored as patterns with counts: one string per pattern, one character
# per item, "1" correct, "0" incorrect and "." missing. mirt fits the matrix the patterns expand to,
# the one the Python test reads.

MML_EM_SETTINGS <- list(tol = 1e-8, reltol = 1e-14, max_cycles = 10000L, quadrature_points = 61L, theta_limits = c(-6, 6))

# The simulated data set: 2PL responses of 1,000 examinees to 10 items, 10% missing completely at random.
SIMULATED_SEED <- 20261127L

as_patterns <- function(x) {
  rows <- apply(x, 1, function(r) paste(ifelse(is.na(r), ".", r), collapse = ""))
  # a radix sort orders the patterns bytewise in every locale, so the file's bytes do not depend on it
  counts <- table(factor(rows, levels = sort(unique(rows), method = "radix")))
  list(patterns = as_array(names(counts)), counts = as_array(as.integer(counts)))
}

from_patterns <- function(data) {
  rows <- rep(data$patterns, times = unlist(data$counts))
  x <- t(vapply(strsplit(unlist(rows), ""), function(chars) {
    suppressWarnings(as.numeric(chars)) # "." becomes NA
  }, numeric(nchar(data$patterns[[1]]))))
  colnames(x) <- sprintf("item%d", seq_len(ncol(x)))
  x
}

from_table <- function(table) {
  items <- as.matrix(table[, 1:5])
  list(
    patterns = as_array(apply(items, 1, paste, collapse = "")),
    counts = as_array(as.integer(table[, 6]))
  )
}

simulated_responses <- function() {
  set.seed(SIMULATED_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  n_examinees <- 1000L
  n_items <- 10L
  a <- rlnorm(n_items, 0, 0.3)
  b <- rnorm(n_items)
  theta <- rnorm(n_examinees)
  p <- plogis(sweep(outer(theta, b, "-"), 2, a, "*"))
  x <- matrix(as.integer(runif(n_examinees * n_items) < p), n_examinees, n_items)
  x[runif(n_examinees * n_items) < 0.1] <- NA
  x
}

fit <- function(x, model, se_type, pars = NULL) {
  n_items <- ncol(x)
  structure <- switch(model,
    `2pl` = list(spec = 1, itemtype = "2PL"),
    equal_slopes = list(spec = mirt.model(sprintf("F = 1-%d\nCONSTRAIN = (1-%d, a1)", n_items, n_items)), itemtype = "2PL"),
    fixed_slopes = list(spec = 1, itemtype = "1PL")
  )
  mirt(
    x, structure$spec,
    itemtype = structure$itemtype, pars = pars, quadpts = MML_EM_SETTINGS$quadrature_points,
    TOL = MML_EM_SETTINGS$tol, SE = is.null(pars), SE.type = se_type, verbose = FALSE,
    technical = list(NCYCLES = MML_EM_SETTINGS$max_cycles, theta_lim = MML_EM_SETTINGS$theta_limits),
    control = list(reltol = MML_EM_SETTINGS$reltol)
  )
}

item_values <- function(model, row, column, irt = FALSE) {
  items <- coef(model, printSE = TRUE, IRTpars = irt)
  items$GroupPars <- NULL
  as_array(vapply(items, function(item) item[row, column], numeric(1)))
}

# The log-likelihood at fixed 2PL parameters, which needs no optimizer.
fixed_log_likelihood <- function(x) {
  n_items <- ncol(x)
  slope <- seq(0.6, 1.6, length.out = n_items)
  intercept <- seq(1.5, -1.5, length.out = n_items)
  values <- fit(x, "2pl", "Oakes", pars = "values")
  values$value[values$name == "a1"] <- slope
  values$value[values$name == "d"] <- intercept
  values$est <- FALSE
  model <- fit(x, "2pl", "Oakes", pars = values)
  list(slope = as_array(slope), intercept = as_array(intercept), log_likelihood = extract.mirt(model, "logLik"))
}

mml_em_case <- function(name, x, model) {
  oakes <- fit(x, model, "Oakes")
  louis <- fit(x, model, "Louis")
  stopifnot(extract.mirt(oakes, "converged"), extract.mirt(louis, "converged"))
  case <- list(
    id = paste0(name, "/", model),
    data = name,
    model = model,
    slope = item_values(oakes, "par", "a1"),
    intercept = item_values(oakes, "par", "d"),
    intercept_se = item_values(oakes, "SE", "d"),
    intercept_se_louis = item_values(louis, "SE", "d"),
    b = item_values(oakes, "par", "b", irt = TRUE),
    b_se = item_values(oakes, "SE", "b", irt = TRUE),
    log_likelihood = extract.mirt(oakes, "logLik"),
    cycles = extract.mirt(oakes, "iterations")
  )
  if (model != "fixed_slopes") {
    case$slope_se <- item_values(oakes, "SE", "a1")
    case$slope_se_louis <- item_values(louis, "SE", "a1")
  }
  case
}

mml_em_cases <- function() {
  data <- list(lsat6 = from_table(LSAT6), lsat7 = from_table(LSAT7), simulated = as_patterns(simulated_responses()))
  data$simulated$seed <- SIMULATED_SEED
  matrices <- lapply(data, from_patterns)
  for (name in names(data)) {
    data[[name]]$fixed <- fixed_log_likelihood(matrices[[name]])
  }
  cases <- list()
  for (name in names(data)) {
    for (model in c("2pl", "equal_slopes", "fixed_slopes")) {
      cases[[length(cases) + 1L]] <- mml_em_case(name, matrices[[name]], model)
    }
  }
  list(settings = MML_EM_SETTINGS, data = data, cases = cases)
}
