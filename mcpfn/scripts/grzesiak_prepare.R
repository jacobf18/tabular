arguments <- commandArgs(trailingOnly = TRUE)
work <- normalizePath(arguments[[1]])
library(dplyr)
library(jsonlite)
source(file.path(work, "upstream/R/amputation_mar.R"))
source(file.path(work, "upstream/R/amputation.R"))
manifest <- read.csv(file.path(work, "reference/dataset_manifest.csv"))
manifest <- unique(manifest[manifest$case %in% c("complete", "categorical"), ])
manifest <- manifest[order(manifest$case, manifest$set_id), ]
records <- list()
for (dataset_index in seq_len(nrow(manifest))) {
  dataset <- manifest$set_id[[dataset_index]]
  source_path <- file.path(work, "upstream/data/datasets/complete_backup",
                          if (manifest$case[[dataset_index]] == "complete") "numerical" else "categorical",
                          paste0(dataset, ".RDS"))
  original <- readRDS(source_path)
  if (anyNA(original)) stop(paste("Incomplete artificial-missingness input:", dataset))
  original <- as.data.frame(original)
  data_directory <- file.path(work, "fresh/data", dataset)
  dir.create(data_directory, recursive = TRUE, showWarnings = FALSE)
  write.csv(data.frame(lapply(original, as.numeric)), file.path(data_directory, "truth.csv"), row.names = FALSE)
  saveRDS(original, file.path(data_directory, "truth.RDS"))
  schema <- list(columns = names(original), rows = nrow(original), cols = ncol(original),
                 factors = lapply(original, function(column) if (is.factor(column)) levels(column) else NULL),
                 numeric_only = all(vapply(original, is.numeric, logical(1))))
  write_json(schema, file.path(data_directory, "schema.json"), auto_unbox = TRUE, pretty = TRUE)
  for (mechanism in c("mcar", "mar")) {
    for (ratio in c(0.1, 0.2, 0.3)) {
      for (replicate in seq_len(2)) {
        seed <- 56135L + length(records)
        case_id <- paste(dataset, toupper(mechanism), ratio, replicate, sep = "/")
        directory <- file.path(work, "fresh/cases", case_id)
        dir.create(directory, recursive = TRUE, showWarnings = FALSE)
        missing_path <- file.path(directory, "missing.RDS")
        if (file.exists(missing_path)) {
          missing <- readRDS(missing_path)
        } else {
          set.seed(seed)
          missing <- ampute_dataset(source_path, mechanism, ratio)
          saveRDS(missing, missing_path)
        }
        stopifnot(identical(dim(missing), dim(original)))
        write.csv(data.frame(lapply(missing, as.numeric)), file.path(directory, "missing.csv"), row.names = FALSE, na = "NA")
        records[[length(records) + 1]] <- data.frame(case_id = case_id, suite = "grzesiak",
          dataset = dataset, pattern = toupper(mechanism), missingness = ratio, rep = replicate,
          seed = seed, numeric_only = schema$numeric_only, rows = nrow(original), cols = ncol(original),
          source_directory = directory, truth_path = file.path(data_directory, "truth.csv"),
          truth_rds = file.path(data_directory, "truth.RDS"), missing_path = file.path(directory, "missing.csv"))
      }
    }
  }
  cat(dataset, nrow(original), ncol(original), "prepared\n")
}
write.csv(do.call(rbind, records), file.path(work, "fresh/cases.csv"), row.names = FALSE)
write_json(list(seed_base = 56135, scheme = "seed_base + deterministic case index",
                mask_status = "fresh independent masks, not historical target-specific RNG states",
                amputation = "unmodified upstream ampute_dataset / produce_NA", row_split = FALSE,
                r_version = R.version.string), file.path(work, "fresh/protocol.json"), auto_unbox = TRUE, pretty = TRUE)
