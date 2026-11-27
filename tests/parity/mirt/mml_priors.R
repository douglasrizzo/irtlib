# Bayes modal MML-EM with priors on the item parameters (irtlib.mml): the 3PL with a normal prior
# on the logit of its guessing parameter, the 2PL with a lognormal prior on the slope and a normal
# prior on the intercept, the 1PL with a lognormal prior on its shared slope, which mirt counts once,
# and the 4PL with normal priors on the logits of c and d (ADR-0008). Settings as mml_em.R:
# TOL = 1e-8 with an exact M-step, 61 nodes on [-6, 6], Oakes standard errors.
#
# Each case records mirt's estimates in its parameters (slope, intercept, and the logits of the
# guessing parameter and upper asymptote where estimated), the standard errors of a, b, c and d from
# coef(IRTpars = TRUE), the log-likelihood, which leaves the priors out, and the log-prior.

PRIORS_SEED <- 20261130L
FOUR_PL_SEED <- 20261131L

# 3PL responses of 2,000 examinees to 10 items, 10% missing completely at random.
simulated_three_pl_responses <- function() {
  set.seed(PRIORS_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  n_examinees <- 2000L
  n_items <- 10L
  a <- rlnorm(n_items, 0, 0.3)
  b <- rnorm(n_items)
  c <- plogis(rnorm(n_items, qlogis(0.2), 0.3))
  theta <- rnorm(n_examinees)
  p <- sweep(sweep(plogis(sweep(outer(theta, b, "-"), 2, a, "*")), 2, 1 - c, "*"), 2, c, "+")
  x <- matrix(as.integer(runif(n_examinees * n_items) < p), n_examinees, n_items)
  x[runif(n_examinees * n_items) < 0.1] <- NA
  x
}

# 4PL responses of 5,000 examinees to 10 items, 10% missing completely at random.
simulated_four_pl_responses <- function() {
  set.seed(FOUR_PL_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  n_examinees <- 5000L
  n_items <- 10L
  a <- rlnorm(n_items, 0.3, 0.2)
  b <- rnorm(n_items)
  c <- plogis(rnorm(n_items, qlogis(0.2), 0.3))
  d <- plogis(rnorm(n_items, qlogis(0.9), 0.3))
  theta <- rnorm(n_examinees)
  logistic <- plogis(sweep(outer(theta, b, "-"), 2, a, "*"))
  p <- sweep(sweep(logistic, 2, d - c, "*"), 2, c, "+")
  x <- matrix(as.integer(runif(n_examinees * n_items) < p), n_examinees, n_items)
  x[runif(n_examinees * n_items) < 0.1] <- NA
  x
}

# The model "1PL" is mirt's 2PL with the slopes constrained equal, irtlib's 1PL.
fit_with_priors <- function(x, model, priors) {
  n_items <- ncol(x)
  tuples <- vapply(priors, function(prior) {
    sprintf("(1-%d, %s, %s, %.17g, %.17g)", n_items, prior$parameter, prior$family, prior$mean, prior$sd)
  }, "")
  constrain <- if (model == "1PL") sprintf("CONSTRAIN = (1-%d, a1)\n", n_items) else ""
  spec <- mirt.model(paste0(sprintf("F = 1-%d\n", n_items), constrain, "PRIOR = ", paste(tuples, collapse = ", ")))
  # mirt's M-step uses BFGS, whose tolerance is reltol, or nlminb when a prior bounds a parameter, whose tolerance
  # is rel.tol; each warns of the other's name ("unknown names in control", "unrecognized control elements"), and
  # only those warnings are muffled
  withCallingHandlers(mirt(
    x, spec,
    itemtype = if (model == "1PL") "2PL" else model, quadpts = MML_EM_SETTINGS$quadrature_points, TOL = MML_EM_SETTINGS$tol,
    SE = TRUE, SE.type = "Oakes", verbose = FALSE,
    technical = list(NCYCLES = MML_EM_SETTINGS$max_cycles, theta_lim = MML_EM_SETTINGS$theta_limits),
    control = list(reltol = MML_EM_SETTINGS$reltol, rel.tol = MML_EM_SETTINGS$reltol)
  ), warning = function(w) {
    if (grepl("unknown names in control|unrecognized control elements", conditionMessage(w))) {
      invokeRestart("muffleWarning")
    }
  })
}

priors_case <- function(name, x, itemtype, priors) {
  model <- fit_with_priors(x, itemtype, priors)
  stopifnot(extract.mirt(model, "converged"))
  case <- list(
    id = paste0(name, "/", tolower(itemtype), "_priors"),
    data = name,
    model = itemtype,
    priors = priors,
    slope = item_values(model, "par", "a1"),
    intercept = item_values(model, "par", "d"),
    a_se = item_values(model, "SE", "a", irt = TRUE),
    b_se = item_values(model, "SE", "b", irt = TRUE),
    log_likelihood = extract.mirt(model, "logLik"),
    log_prior = extract.mirt(model, "logPrior"),
    cycles = extract.mirt(model, "iterations")
  )
  if (itemtype %in% c("3PL", "4PL")) {
    case$guessing <- item_values(model, "par", "logit(g)")
    case$c_se <- item_values(model, "SE", "g", irt = TRUE)
  }
  if (itemtype == "4PL") {
    case$upper <- item_values(model, "par", "logit(u)")
    case$d_se <- item_values(model, "SE", "u", irt = TRUE)
  }
  case
}

mml_priors_cases <- function() {
  data <- list(
    lsat7 = from_table(LSAT7),
    simulated = as_patterns(simulated_responses(SIMULATED_SEED, 0.1)),
    simulated_3pl = as_patterns(simulated_three_pl_responses()),
    simulated_4pl = as_patterns(simulated_four_pl_responses())
  )
  data$simulated$seed <- SIMULATED_SEED
  data$simulated_3pl$seed <- PRIORS_SEED
  data$simulated_4pl$seed <- FOUR_PL_SEED
  matrices <- lapply(data, from_patterns)
  guessing <- list(list(parameter = "g", family = "norm", mean = qlogis(0.2), sd = 0.5))
  slope_and_intercept <- list(
    list(parameter = "a1", family = "lnorm", mean = 0, sd = 0.5),
    list(parameter = "d", family = "norm", mean = 0, sd = 2)
  )
  cases <- list(
    priors_case("lsat7", matrices$lsat7, "3PL", guessing),
    priors_case("simulated_3pl", matrices$simulated_3pl, "3PL", guessing),
    priors_case("simulated", matrices$simulated, "2PL", slope_and_intercept),
    priors_case("lsat7", matrices$lsat7, "1PL", list(list(parameter = "a1", family = "lnorm", mean = 0.5, sd = 0.3))),
    priors_case("simulated_4pl", matrices$simulated_4pl, "4PL", c(
      list(list(parameter = "g", family = "norm", mean = qlogis(0.2), sd = 0.2)),
      list(list(parameter = "u", family = "norm", mean = qlogis(0.9), sd = 0.2))
    ))
  )
  list(settings = MML_EM_SETTINGS, data = data, cases = cases)
}
