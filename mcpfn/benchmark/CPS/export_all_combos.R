# Batch exporter: run export_sdid_cps_draws.R logic for 4 paper combos.
# Output: sdid_cps_<outcome>_<assignment>_200/ per combo.

writable_r_lib <- Sys.getenv("R_LIBS_USER", unset = file.path(tempdir(), "Rlibs"))
dir.create(writable_r_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(writable_r_lib, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org", lib = writable_r_lib)
}
if (!requireNamespace("synthdid", quietly = TRUE)) {
  remotes::install_github("synth-inference/synthdid", ref = "sdid-paper",
                          lib = writable_r_lib, upgrade = "never")
}

library(synthdid)

R_draws      <- 200
rank_k       <- 4
N1_cap       <- 10
T1_post      <- 10
seed_value   <- 20260416

last.col <- function(X) X[, ncol(X)]

make_block_W <- function(N, T, N0, T0) {
  W <- matrix(0L, nrow = N, ncol = T)
  if (N0 < N && T0 < T) {
    W[(N0 + 1):N, (T0 + 1):T] <- 1L
  }
  W
}

script_file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[1]), mustWork = FALSE))
} else { getwd() }

data(CPS)

combos <- list(
  list(outcome = "hours", assignment = "open_carry"),
  list(outcome = "hours", assignment = "abort_ban"),
  list(outcome = "urate", assignment = "open_carry"),
  list(outcome = "urate", assignment = "abort_ban")
)

for (combo in combos) {
  outcome_name <- combo$outcome
  label_name   <- combo$assignment
  set.seed(seed_value)

  out_dir <- file.path(script_dir,
                       sprintf("sdid_cps_%s_%s_%03d", outcome_name, label_name, R_draws))
  if (dir.exists(out_dir)) {
    cat(sprintf("Skipping existing %s\n", out_dir)); next
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  Y_obs <- panel.matrices(CPS, treatment = "min_wage", outcome = outcome_name,
                          treated.last = FALSE)$Y
  w_obs <- last.col(
    panel.matrices(CPS, treatment = label_name, treated.last = FALSE)$W
  )

  cat(sprintf("\n== combo: outcome=%s, assignment=%s ==\n", outcome_name, label_name))
  params <- estimate_dgp(Y_obs, w_obs, rank = rank_k)

  saveRDS(params, file = file.path(out_dir, "estimated_dgp.rds"))
  saveRDS(list(outcome = outcome_name, assignment = label_name, R_draws = R_draws,
               rank_k = rank_k, N1_cap = N1_cap, T1_post = T1_post,
               seed_value = seed_value, Y_obs_dim = dim(Y_obs), w_obs_sum = sum(w_obs)),
          file = file.path(out_dir, "config.rds"))

  manifest_list <- vector("list", R_draws)
  for (r in seq_len(R_draws)) {
    sim <- simulate_dgp(params, N1 = N1_cap, T1 = T1_post)
    Y <- sim$Y; N0 <- sim$N0; T0 <- sim$T0
    N <- nrow(Y); TT <- ncol(Y)
    W <- make_block_W(N, TT, N0, T0)

    trs <- if (N0 < N) N0 + 1 else NA_integer_
    tre <- if (N0 < N) N else NA_integer_
    pcs <- if (T0 < TT) T0 + 1 else NA_integer_
    pce <- if (T0 < TT) TT else NA_integer_

    write.csv(Y, file = file.path(out_dir, sprintf("draw_%04d_Y.csv", r)), row.names = FALSE)
    write.csv(W, file = file.path(out_dir, sprintf("draw_%04d_W.csv", r)), row.names = FALSE)

    manifest_list[[r]] <- data.frame(
      draw_id = r, N = N, T = TT, N0 = N0, T0 = T0,
      N1_realized = N - N0, T1_realized = TT - T0,
      treated_row_start = trs, treated_row_end = tre,
      post_col_start = pcs, post_col_end = pce,
      outcome = outcome_name, assignment = label_name,
      stringsAsFactors = FALSE
    )
  }
  manifest <- do.call(rbind, manifest_list)
  write.csv(manifest, file = file.path(out_dir, "manifest.csv"), row.names = FALSE)
  cat(sprintf("  wrote %d draws -> %s\n", R_draws, out_dir))
}

cat("\nDone.\n")
