# Classical item statistics (irtlib.classical): each item's proportion correct and its correlation
# with the rest score, mirt's itemstats() columns mean and cor_if_rm, on LSAT6, LSAT7 and the
# simulated data set of mml_em.R, 10% of whose responses are missing.
#
# With missing responses, itemstats takes each item's mean over the examinees who answered it, the
# rest score as the sum of the other items an examinee answered, and the correlation over the
# examinees who answered the item (R/itemstats.R:110-120, 148).

classical_cases <- function() {
  data <- list(
    lsat6 = from_table(LSAT6),
    lsat7 = from_table(LSAT7),
    simulated = as_patterns(simulated_responses(SIMULATED_SEED, 0.1))
  )
  data$simulated$seed <- SIMULATED_SEED
  cases <- lapply(names(data), function(name) {
    statistics <- itemstats(from_patterns(data[[name]]))$itemstats
    list(
      data = name,
      proportion_correct = as_array(statistics$mean),
      item_rest_correlation = as_array(statistics$cor_if_rm)
    )
  })
  list(data = data, cases = cases)
}
