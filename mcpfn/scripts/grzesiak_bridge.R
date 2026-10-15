arguments <- commandArgs(trailingOnly = TRUE)
request <- jsonlite::read_json(arguments[[1]], simplifyVector = TRUE)
options(ranger.num.threads = 1, mc.cores = 1)
source(file.path(request$upstream, "R/mice.R"))
source(file.path(request$upstream, "R/missforest.R"))
source(file.path(request$upstream, "R/missmda.R"))
if (grepl("^areg", request$method)) {
  source(file.path(dirname(request$upstream), "dependency_sources/imputomics/utils.R"))
  bind_local_helpers <- function(expression) {
    if (!is.call(expression)) return(expression)
    if (identical(expression[[1]], as.name(":::")) && identical(expression[[2]], as.name("imputomics"))) return(expression[[3]])
    as.call(lapply(as.list(expression), bind_local_helpers))
  }
  for (expression in parse(file.path(request$upstream, "R/areg.R"))) eval(bind_local_helpers(expression))
}
set.seed(request$seed)
started <- proc.time()[["elapsed"]]
result <- tryCatch({
  missing <- if (grepl("\\.RDS$", request$input)) readRDS(request$input) else
    read.csv(request$input, check.names = FALSE, na.strings = "NA")
  if (request$method == "mice_drf") {
    suppressPackageStartupMessages(library(miceDRF))
    prediction <- miceDRF::impute_mice_drf(missing)
  } else if (request$method == "column_mean") {
    prediction <- missing
    for (column in seq_len(ncol(missing))) {
      prediction[is.na(missing[, column]), column] <- mean(missing[, column], na.rm = TRUE)
    }
  } else {
    prediction <- get(paste0("impute_", request$method))(missing)
  }
  if (!identical(dim(prediction), dim(missing))) stop("prediction shape mismatch")
  if (anyNA(prediction)) stop("unfilled missing values")
  numeric_prediction <- data.frame(lapply(prediction, as.numeric))
  numeric_missing <- data.frame(lapply(missing, as.numeric))
  if (any(!is.finite(as.matrix(numeric_prediction)))) stop("nonfinite prediction")
  observed <- !is.na(numeric_missing)
  if (!isTRUE(all.equal(as.matrix(numeric_prediction)[observed], as.matrix(numeric_missing)[observed], tolerance = 1.5e-5))) {
    stop("modified observed values")
  }
  for (column in seq_len(ncol(missing))) {
    if (is.factor(missing[[column]]) && any(!numeric_prediction[[column]] %in% seq_len(nlevels(missing[[column]])))) {
      stop("invalid category levels")
    }
  }
  write.csv(numeric_prediction, request$output, row.names = FALSE)
  list(status = "ok", seconds = proc.time()[["elapsed"]] - started)
}, error = function(error) list(status = "error", error = conditionMessage(error), seconds = proc.time()[["elapsed"]] - started))
jsonlite::write_json(result, request$result, auto_unbox = TRUE, pretty = TRUE)
