arguments <- commandArgs(trailingOnly = TRUE)
request <- jsonlite::read_json(arguments[[1]], simplifyVector = TRUE)
source(file.path(request$upstream, "R/mice.R"))
suppressPackageStartupMessages(library(miceDRF))
set.seed(request$seed)
calls <- 0L
imputation_function <- function(missing) {
  calls <<- calls + 1L
  if (request$method == "matrixformer") {
    handle <- curl::new_handle(postfields = jsonlite::toJSON(list(matrix = as.matrix(missing)),
                                                          matrix = "rowmajor", na = "null", digits = NA))
    curl::handle_setheaders(handle, "Content-Type" = "application/json", "Authorization" = request$token)
    response <- curl::curl_fetch_memory(request$url, handle = handle)
    if (response$status_code != 200) stop(rawToChar(response$content))
    prediction <- as.data.frame(jsonlite::fromJSON(rawToChar(response$content))$matrix)
    names(prediction) <- names(missing)
  } else if (request$method == "column_mean") {
    prediction <- missing
    for (column in seq_len(ncol(missing))) {
      prediction[is.na(missing[, column]), column] <- mean(missing[, column], na.rm = TRUE)
    }
  } else {
    prediction <- get(paste0("impute_", request$method))(missing)
  }
  if (!identical(dim(prediction), dim(missing)) || any(!is.finite(as.matrix(prediction)))) stop("invalid prediction")
  observed <- !is.na(missing)
  if (!isTRUE(all.equal(as.matrix(prediction)[observed], as.matrix(missing)[observed], tolerance = 1.5e-5))) stop("observed values changed")
  prediction
}
started <- proc.time()[["elapsed"]]
result <- tryCatch({
  missing <- as.data.frame(readRDS(request$input))
  stopifnot(all(vapply(missing, is.numeric, logical(1))))
  prediction <- imputation_function(missing)
  write.csv(prediction, request$output, row.names = FALSE)
  stochastic <- !request$method %in% c("matrixformer", "column_mean", "mice_norm_predict")
  score <- miceDRF::Iscore(X = missing, X_imp = prediction, N = 20, multiple = stochastic,
                          imputation_func = imputation_function, scale = FALSE)
  columns <- attr(score, "dat")
  write.csv(columns, request$columns, row.names = FALSE, na = "")
  if (!is.finite(score)) stop("nonfinite Energy-I-Score")
  list(status = "ok", energy_iscore = as.numeric(score), imputations_per_projection = if (stochastic) 20 else 1,
       scored_columns = sum(is.finite(columns$score)), requested_columns = nrow(columns),
       imputation_calls = calls, seconds = proc.time()[["elapsed"]] - started)
}, error = function(error) list(status = "error", error = conditionMessage(error),
                                imputation_calls = calls, seconds = proc.time()[["elapsed"]] - started))
jsonlite::write_json(result, request$result, auto_unbox = TRUE, pretty = TRUE, digits = NA)
