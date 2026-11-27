# Quantities of the response functions at fixed item parameters (irtlib.irt): the reliability that
# plot(type = "rxx") draws, I(theta) / (I(theta) + 1 / var) with the abilities' variance fixed at 1,
# and the integral of each item's information over an interval, which areainfo() computes with R's
# integrate(), here to a relative 1e-12.
#
# mirt needs responses to build a model, so the model is set up on every response pattern of the
# items and its parameters are then fixed; nothing is estimated.

RESPONSE_FUNCTION_SETTINGS <- list(rxx_points = 13L, rel_tol = 1e-12)

# a, b, c, d: two 2PL, two 3PL and two 4PL items
RESPONSE_FUNCTION_ITEMS <- rbind(
  c(0.6, -1.4, 0, 1),
  c(1.8, -0.2, 0, 1),
  c(1.1, 0.3, 0.2, 1),
  c(2.4, 1.1, 0.1, 1),
  c(0.9, 2.2, 0.3, 0.92),
  c(1.5, -0.5, 0.15, 0.95)
)

fixed_items_model <- function(items) {
  x <- as.matrix(expand.grid(rep(list(0:1), nrow(items))))
  colnames(x) <- sprintf("item%d", seq_len(nrow(items)))
  values <- mirt(x, 1, itemtype = "4PL", pars = "values")
  values$value[values$name == "a1"] <- items[, 1]
  values$value[values$name == "d"] <- -items[, 1] * items[, 2]
  values$value[values$name == "g"] <- items[, 3]
  values$value[values$name == "u"] <- items[, 4]
  values$est <- FALSE
  mirt(x, 1, itemtype = "4PL", pars = values, verbose = FALSE)
}

response_function_cases <- function() {
  items <- RESPONSE_FUNCTION_ITEMS
  model <- fixed_items_model(items)
  subsets <- list(all = seq_len(nrow(items)), `2pl` = 1:2, `3pl` = 3:4, `4pl` = 5:6)
  reliability <- lapply(names(subsets), function(id) {
    which_items <- subsets[[id]]
    panel <- plot(model, type = "rxx", npts = RESPONSE_FUNCTION_SETTINGS$rxx_points, which.items = which_items)$panel.args[[1]]
    list(id = id, items = items[which_items, , drop = FALSE], thetas = panel$x, reliability = panel$y)
  })
  limits <- list(finite = c(-1.7, 2.4), from_minus_inf = c(-Inf, 0.5), to_inf = c(-0.5, Inf), whole = c(-Inf, Inf))
  info_integral <- lapply(names(limits), function(id) {
    limit <- limits[[id]]
    info <- vapply(seq_len(nrow(items)), function(i) {
      areainfo(model, limit, which.items = i, rel.tol = RESPONSE_FUNCTION_SETTINGS$rel_tol)$Info
    }, numeric(1))
    list(id = id, lower = limit[1], upper = limit[2], info = info)
  })
  list(
    settings = RESPONSE_FUNCTION_SETTINGS,
    items = items,
    reliability = reliability,
    info_integral = info_integral
  )
}
