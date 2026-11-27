# Writes the mirt outputs that tests/test_parity_mirt.py compares pyitemcal with (ADR-0003).
#
# Run from anywhere, with R installed:
#
#   Rscript tests/parity/mirt/generate.R
#
# mirt is loaded from the library named by the PYITEMCAL_MIRT_LIB environment variable, by default
# ~/.cache/pyitemcal-parity/R-lib, outside the repository and the user's R library. When the pinned
# mirt version is missing there, the script installs mirt from CRAN with its dependencies, then the
# pinned version from CRAN's source archive over it.
#
# Each case script builds the cases of one JSON file beside it: the response data, mirt's settings
# and mirt's outputs. A rerun with the same mirt version writes the same bytes.

MIRT_VERSION <- "1.48"
CRAN <- "https://cloud.r-project.org"

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

# Reads the installed DESCRIPTION without loading mirt, so a stale version is never loaded before
# the pinned one replaces it.
has_pinned <- function(lib) {
  nzchar(system.file(package = "mirt", lib.loc = lib)) &&
    as.character(packageVersion("mirt", lib.loc = lib)) == MIRT_VERSION
}

install_mirt <- function(lib) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  # mirt's dependencies come with the current release; the pinned version then replaces it.
  install.packages("mirt", lib = lib, repos = CRAN, quiet = TRUE)
  if (has_pinned(lib)) {
    return(invisible())
  }
  tarball <- sprintf("mirt_%s.tar.gz", MIRT_VERSION)
  # the current release sits in src/contrib, older ones in its Archive; success is the pinned
  # version being installed, since install.packages can warn without failing
  for (url in c(
    paste0(CRAN, "/src/contrib/", tarball),
    paste0(CRAN, "/src/contrib/Archive/mirt/", tarball)
  )) {
    try(install.packages(url, lib = lib, repos = NULL, type = "source", quiet = TRUE), silent = TRUE)
    if (has_pinned(lib)) {
      return(invisible())
    }
  }
  stop("could not install mirt ", MIRT_VERSION, " into ", lib)
}

lib <- Sys.getenv("PYITEMCAL_MIRT_LIB", file.path(path.expand("~"), ".cache", "pyitemcal-parity", "R-lib"))
if (!has_pinned(lib)) {
  install_mirt(lib)
}
stopifnot(has_pinned(lib))
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(mirt, lib.loc = lib))
stopifnot(as.character(getNamespaceVersion("mirt")) == MIRT_VERSION)

here <- script_dir()
for (helper in c("json.R", "mml_em.R")) {
  source(file.path(here, helper))
}

versions <- list(mirt_version = MIRT_VERSION, r_version = R.version.string)
outputs <- list(
  mml_em = mml_em_cases()
)
for (name in names(outputs)) {
  path <- file.path(here, paste0(name, ".json"))
  write_json(c(versions, outputs[[name]]), path)
  message("wrote ", path)
}
