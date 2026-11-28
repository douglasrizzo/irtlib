# The mean squares of each examinee (irtlib.fit.person_mean_squares): mirt's personfit outfit and
# infit, with their standardized values, at abilities passed in, for items of fixed 2PL, 3PL and 4PL
# parameters. Nothing is estimated.
#
# The responses are drawn from the items at abilities drawn from N(0, 1), and four examinees, 7 logits
# or more from most items, answer against their ability: their variances fall under mirt's floor
# (R/personfit.R:200), and the standard deviation of their unweighted mean square above its cap
# (R/personfit.R:205). The case with missing responses leaves out a fifth of the responses at random,
# keeping at least one answered item per examinee.
#
# Each examinee's responses are stored as a pattern with a count of 1, in the examinees' order, one
# character per item: "1" correct, "0" incorrect and "." missing. fixed_items_model() comes from
# response_functions.R.

PERSON_FIT_SEED <- 20261201L
# a, b, c, d: two items of slope 1, two 2PL, two 3PL and two 4PL items
PERSON_FIT_ITEMS <- rbind(
  c(1, -1.2, 0, 1),
  c(1, 0.4, 0, 1),
  c(1.6, -0.5, 0, 1),
  c(0.7, 1.1, 0, 1),
  c(1.3, 0, 0.2, 1),
  c(2, 0.8, 0.15, 1),
  c(1.1, -0.8, 0.1, 0.92),
  c(0.9, 1.5, 0.05, 0.95)
)
# abilities far from most items, and responses against them
PERSON_FIT_EXTREMES <- list(theta = c(-8, -7, 7, 8), responses = rbind(
  c(0, 1, 1, 1, 0, 1, 0, 1),
  c(1, 0, 0, 1, 0, 0, 0, 1),
  c(0, 1, 0, 0, 1, 1, 1, 0),
  c(0, 0, 0, 1, 0, 0, 1, 1)
))

person_fit_responses <- function(theta, items) {
  a <- items[, 1]
  b <- items[, 2]
  c <- items[, 3]
  d <- items[, 4]
  p <- sweep(plogis(sweep(outer(theta, b, "-"), 2, a, "*")), 2, d - c, "*")
  p <- sweep(p, 2, c, "+")
  matrix(as.integer(runif(length(p)) < p), nrow(p), ncol(p))
}

person_fit_case <- function(id, x, theta) {
  model <- fixed_items_model(PERSON_FIT_ITEMS, x)
  statistics <- personfit(model, Theta = matrix(theta))
  list(
    id = id,
    data = list(
      patterns = as_array(apply(x, 1, function(r) paste(ifelse(is.na(r), ".", r), collapse = ""))),
      counts = as_array(rep(1L, nrow(x)))
    ),
    theta = as_array(theta),
    outfit = as_array(statistics$outfit),
    z_outfit = as_array(statistics$z.outfit),
    infit = as_array(statistics$infit),
    z_infit = as_array(statistics$z.infit)
  )
}

person_fit_cases <- function() {
  set.seed(PERSON_FIT_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  theta <- c(rnorm(200L), PERSON_FIT_EXTREMES$theta)
  x <- rbind(person_fit_responses(theta[1:200], PERSON_FIT_ITEMS), PERSON_FIT_EXTREMES$responses)
  missing <- x
  missing[runif(length(missing)) < 0.2] <- NA
  # an examinee left with no answer is given back the first item's response
  empty <- rowSums(!is.na(missing)) == 0
  missing[empty, 1] <- x[empty, 1]
  list(
    items = PERSON_FIT_ITEMS,
    cases = list(person_fit_case("complete", x, theta), person_fit_case("missing", missing, theta))
  )
}
