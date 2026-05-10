# export_sdid_cps_draws.R
#
# End-to-end exporter for the SDID CPS placebo benchmark.
# It estimates the CPS-based placebo DGP and saves synthetic draws to your Desktop.
#
# Paper-faithful CPS combinations used in the official vignette are:
#   1) outcome_name = "log_wage", label_name = "min_wage"
#   2) outcome_name = "log_wage", label_name = "open_carry"
#   3) outcome_name = "log_wage", label_name = "abort_ban"
#   4) outcome_name = "hours",    label_name = "min_wage"
#   5) outcome_name = "urate",    label_name = "min_wage"
#
# You can edit the CONFIG block below.

# =========================
# CONFIG
# =========================
outcome_name <- "log_wage"   # one of: "log_wage", "hours", "urate"
label_name   <- "min_wage"   # one of: "min_wage", "open_carry", "abort_ban"
R_draws      <- 200           # number of synthetic draws to save
rank_k       <- 4             # paper default
N1_cap       <- 10            # max treated units in each draw
T1_post      <- 10            # number of post-treatment periods
seed_value   <- 20260416      # reproducibility seed
save_csv     <- TRUE          # save Y/W as CSV in addition to RDS

# =========================
# PACKAGE SETUP
# =========================
writable_r_lib <- Sys.getenv("R_LIBS_USER", unset = file.path(tempdir(), "Rlibs"))
dir.create(writable_r_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(writable_r_lib, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org", lib = writable_r_lib)
}

if (!requireNamespace("synthdid", quietly = TRUE)) {
  remotes::install_github("synth-inference/synthdid", ref = "sdid-paper", lib = writable_r_lib, upgrade = "never")
}

library(synthdid)

# =========================
# HELPERS
# =========================
last.col <- function(X) X[, ncol(X)]

make_block_W <- function(N, T, N0, T0) {
  W <- matrix(0L, nrow = N, ncol = T)
  if (N0 < N && T0 < T) {
    W[(N0 + 1):N, (T0 + 1):T] <- 1L
  }
  W
}

# =========================
# REPRODUCIBILITY + OUTPUT
# =========================
set.seed(seed_value)

script_file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[1]), mustWork = FALSE))
} else {
  getwd()
}

out_dir <- file.path(
  script_dir,
  sprintf("sdid_cps_%s_%s_%03d", outcome_name, label_name, R_draws)
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# LOAD CPS AND BUILD SEED OBJECTS
# =========================
data(CPS)

# This mirrors the official vignette's CPS construction:
# the outcome matrix is taken from panel.matrices(..., treated.last = FALSE)
# and the assignment vector is taken from the last column of the chosen policy-label W matrix.
Y_obs <- panel.matrices(
  CPS,
  treatment = "min_wage",
  outcome = outcome_name,
  treated.last = FALSE
)$Y

w_obs <- last.col(
  panel.matrices(
    CPS,
    treatment = label_name,
    treated.last = FALSE
  )$W
)

# =========================
# ESTIMATE PLACEBO DGP
# =========================
params <- estimate_dgp(Y_obs, w_obs, rank = rank_k)

saveRDS(
  list(
    outcome = outcome_name,
    assignment = label_name,
    R_draws = R_draws,
    rank_k = rank_k,
    N1_cap = N1_cap,
    T1_post = T1_post,
    seed_value = seed_value,
    Y_obs_dim = dim(Y_obs),
    w_obs_sum = sum(w_obs)
  ),
  file = file.path(out_dir, "config.rds")
)

saveRDS(params, file = file.path(out_dir, "estimated_dgp.rds"))

# =========================
# GENERATE AND SAVE DRAWS
# =========================
manifest_list <- vector("list", R_draws)

for (r in seq_len(R_draws)) {
  sim <- simulate_dgp(params, N1 = N1_cap, T1 = T1_post)

  Y  <- sim$Y
  N0 <- sim$N0
  T0 <- sim$T0
  N  <- nrow(Y)
  TT <- ncol(Y)
  W  <- make_block_W(N, TT, N0, T0)

  treated_row_start <- if (N0 < N) N0 + 1 else NA_integer_
  treated_row_end   <- if (N0 < N) N else NA_integer_
  post_col_start    <- if (T0 < TT) T0 + 1 else NA_integer_
  post_col_end      <- if (T0 < TT) TT else NA_integer_
  N1_realized       <- N - N0
  T1_realized       <- TT - T0

  draw_obj <- list(
    draw_id = r,
    Y = Y,
    W = W,
    N0 = N0,
    T0 = T0,
    treated_row_start = treated_row_start,
    treated_row_end = treated_row_end,
    post_col_start = post_col_start,
    post_col_end = post_col_end,
    N1_realized = N1_realized,
    T1_realized = T1_realized,
    outcome = outcome_name,
    assignment = label_name
  )

  saveRDS(
    draw_obj,
    file = file.path(out_dir, sprintf("draw_%04d.rds", r))
  )

  if (isTRUE(save_csv)) {
    write.csv(
      Y,
      file = file.path(out_dir, sprintf("draw_%04d_Y.csv", r)),
      row.names = FALSE
    )
    write.csv(
      W,
      file = file.path(out_dir, sprintf("draw_%04d_W.csv", r)),
      row.names = FALSE
    )
  }

  manifest_list[[r]] <- data.frame(
    draw_id = r,
    N = N,
    T = TT,
    N0 = N0,
    T0 = T0,
    N1_realized = N1_realized,
    T1_realized = T1_realized,
    treated_row_start = treated_row_start,
    treated_row_end = treated_row_end,
    post_col_start = post_col_start,
    post_col_end = post_col_end,
    outcome = outcome_name,
    assignment = label_name,
    stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, manifest_list)
write.csv(manifest, file = file.path(out_dir, "manifest.csv"), row.names = FALSE)

# =========================
# PRINT SUMMARY
# =========================
cat("Done.\n")
cat("Saved", R_draws, "draws to:\n")
cat(normalizePath(out_dir, mustWork = FALSE), "\n\n")
cat("Main files:\n")
cat("  manifest.csv\n")
cat("  config.rds\n")
cat("  estimated_dgp.rds\n")
cat("  draw_0001.rds ... draw_", sprintf("%04d", R_draws), ".rds\n", sep = "")
if (isTRUE(save_csv)) {
  cat("  draw_0001_Y.csv ... draw_", sprintf("%04d", R_draws), "_Y.csv\n", sep = "")
  cat("  draw_0001_W.csv ... draw_", sprintf("%04d", R_draws), "_W.csv\n", sep = "")
}
cat("\nTreatment block for each draw is given by W, or equivalently by:\n")
cat("  treated rows    = treated_row_start : treated_row_end\n")
cat("  post columns    = post_col_start : post_col_end\n")
