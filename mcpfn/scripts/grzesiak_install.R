arguments <- commandArgs(trailingOnly = TRUE)
work <- normalizePath(arguments[[1]])
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 6)
lock <- jsonlite::read_json(file.path(work, "upstream/renv.lock"))$Packages
packages <- c("gdata", "mltools", "lme4", "rpart", "glmnet", "drf", "scoringRules", "pbapply", "Amelia", "bcv", "softImpute", "missMDA", "VIM")
for (package in packages) {
  version <- lock[[package]]$Version
  if (!requireNamespace(package, quietly = TRUE) || as.character(packageVersion(package)) != version) {
    tryCatch(remotes::install_version(package, version = version, upgrade = "never", dependencies = NA),
             error = function(error) message(package, ": ", conditionMessage(error)))
  }
}
remotes::install_github("KrystynaGrzesiak/miceDRF@d09d4f5c6b7d1ba1d1e6b1e78a9ca7bc5de46227", upgrade = "never", dependencies = NA)
installed <- installed.packages()[, c("Package", "Version")]
write.csv(installed, file.path(work, "installed_r_packages.csv"), row.names = FALSE)
