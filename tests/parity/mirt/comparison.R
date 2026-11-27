# Model comparison of MML-EM calibrations (irtlib.comparison): the log-likelihood, AIC, BIC and G2
# against the general multinomial of the 1PL (slopes constrained equal) and 2PL on LSAT6 and LSAT7,
# and the likelihood-ratio test of the 1PL against the 2PL from anova(). The fits use the settings of
# mml_em.R.

comparison_case <- function(name, x) {
  restricted <- fit(x, "equal_slopes", "Oakes")
  general <- fit(x, "2pl", "Oakes")
  stopifnot(extract.mirt(restricted, "converged"), extract.mirt(general, "converged"))
  summarize <- function(model) {
    list(
      log_likelihood = extract.mirt(model, "logLik"),
      n_parameters = extract.mirt(model, "nest"),
      aic = extract.mirt(model, "AIC"),
      bic = extract.mirt(model, "BIC"),
      g2 = extract.mirt(model, "G2"),
      df = as.integer(extract.mirt(model, "df")),
      p_value = extract.mirt(model, "p")
    )
  }
  test <- anova(restricted, general)
  list(
    data = name,
    n_examinees = nrow(x),
    restricted = summarize(restricted),
    general = summarize(general),
    likelihood_ratio = list(statistic = test$X2[2], df = as.integer(test$df[2]), p_value = test$p[2])
  )
}

comparison_cases <- function() {
  data <- list(lsat6 = from_table(LSAT6), lsat7 = from_table(LSAT7))
  matrices <- lapply(data, from_patterns)
  list(
    settings = MML_EM_SETTINGS,
    data = data,
    cases = lapply(names(data), function(name) comparison_case(name, matrices[[name]]))
  )
}
