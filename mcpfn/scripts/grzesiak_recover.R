arguments <- commandArgs(trailingOnly = TRUE)
work <- normalizePath(arguments[[1]])
output <- file.path(work, "reference")
dir.create(output, recursive = TRUE, showWarnings = FALSE)
library(dplyr)
library(tidyr)
library(jsonlite)
read_sheet <- function(url, sheet) {
  metadata <- read.csv(file.path(work, "method_metadata.csv"), check.names = FALSE)
  metadata[, names(metadata) != "", drop = FALSE]
}
setwd(file.path(work, "upstream"))
source_lines <- readLines("results/filter_results_before_plotting.R")
source_lines <- source_lines[!grepl("^library\\(googlesheets4\\)", source_lines)]
source_lines <- sub("### Case 3 : Mixed", "audit_natural_numeric <- imputation_summary\n### Case 3 : Mixed", source_lines, fixed = TRUE)
source_lines <- sub("### Case 5 : ALL + Complete", "audit_natural_mixed <- imputation_summary\n### Case 5 : ALL + Complete", source_lines, fixed = TRUE)
eval(parse(text = source_lines))
write.csv(imputation_summary, file.path(output, "recovered_per_case.csv"), row.names = FALSE, na = "")
write.csv(imputation_summary_numerical, file.path(output, "numerical_per_case.csv"), row.names = FALSE, na = "")
write.csv(imputation_summary_categorical, file.path(output, "mixed_per_case.csv"), row.names = FALSE, na = "")
write.csv(imputation_summary_incomplete, file.path(output, "natural_per_case.csv"), row.names = FALSE, na = "")
write.csv(audit_natural_numeric, file.path(output, "natural_numerical_per_case.csv"), row.names = FALSE, na = "")
write.csv(audit_natural_mixed, file.path(output, "natural_mixed_per_case.csv"), row.names = FALSE, na = "")
datasets <- imputation_summary %>% distinct(set_id, case)
write.csv(datasets, file.path(output, "dataset_manifest.csv"), row.names = FALSE)
methods <- imputation_summary %>% distinct(method, imputation_fun)
write.csv(methods, file.path(output, "method_manifest.csv"), row.names = FALSE)
write_json(list(r_version = R.version.string,
                status = "author_artifacts_recovered_not_fresh_imputations",
                filter_script = "results/filter_results_before_plotting.R",
                packages = as.list(sapply(c("dplyr", "tidyr", "jsonlite"), function(package) as.character(packageVersion(package))))),
           file.path(output, "recovery.json"), auto_unbox = TRUE, pretty = TRUE)
print(table(datasets$case))
print(imputation_summary %>% group_by(case, measure) %>% summarise(rows = n(), methods = n_distinct(method), datasets = n_distinct(set_id), .groups = "drop"), n = 100)
