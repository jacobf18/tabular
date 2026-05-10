#!/usr/bin/env python3
from __future__ import annotations

"""
Generate Kansas semi-synthetic causal-panel draws using R packages `augsynth` and `gsynth`.

This script is designed to mirror the *main* Kansas-calibrated simulation setup in
Ben-Michael, Feller, and Rothstein (2021):

    1. Load the built-in `kansas` panel from the `augsynth` R package.
    2. Fit a 3-factor interactive fixed-effects model to log GSP per capita using
       the `gsynth` package.
    3. Hold the estimated time effects and factors fixed, and simulate new panels by drawing
       new unit intercepts, factor loadings, and iid Gaussian noise.
    4. Sample exactly one treated row via the supplement's propensity-score recipe.
    5. Export:
         - manifest.csv
         - draw_XXXX_Y.csv   (the full latent untreated matrix)
         - draw_XXXX_W.csv   (the treated-post mask)

The output is meant to plug directly into downstream Python code that masks the treated-post block.

Notes
-----
* This script targets the paper's main factor-model DGP. Set `--noise-scale 4.0` to mimic the
  higher-noise factor DGP discussed in the supplement.
* The paper states that the factor model is fit with the Generalized Synthetic Control Method.
  Here I use `gsynth::interFE`, from the same R package, because it directly exposes the fitted
  factors, loadings, fixed effects, and residual variance needed for simulation.
* The row/column indexing written to manifest.csv is 0-based with end-exclusive intervals so it
  lines up naturally with NumPy/Python slicing conventions.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kansas semi-synthetic draws by calling R from Python."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where manifest.csv and draw_XXXX_[Y|W].csv files will be written.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=200,
        help="Number of synthetic draws to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Global RNG seed passed to R.",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default="lngdpcapita",
        help="Outcome column from augsynth::kansas to model. Default matches the paper.",
    )
    parser.add_argument(
        "--factor-r",
        type=int,
        default=3,
        help="Number of latent factors J in the fitted interactive fixed-effects model.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Selection-model scale θ used in the treatment-assignment logit.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier on the fitted residual SD. Use 4.0 to mimic the factor-4σ DGP.",
    )
    parser.add_argument(
        "--rscript",
        type=str,
        default=shutil.which("Rscript") or "Rscript",
        help="Path to Rscript.",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install missing R packages automatically (augsynth via GitHub, gsynth via CRAN).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    parser.add_argument(
        "--keep-r-script",
        action="store_true",
        help="Keep the temporary R script for debugging.",
    )
    return parser.parse_args()


R_SCRIPT = r'''
options(warn = 1)

output_dir <- Sys.getenv("OUTPUT_DIR")
n_draws <- as.integer(Sys.getenv("N_DRAWS"))
seed <- as.integer(Sys.getenv("SEED"))
outcome_var <- Sys.getenv("OUTCOME_VAR")
factor_r <- as.integer(Sys.getenv("FACTOR_R"))
theta <- as.numeric(Sys.getenv("THETA"))
noise_scale <- as.numeric(Sys.getenv("NOISE_SCALE"))
install_missing <- tolower(Sys.getenv("INSTALL_MISSING")) %in% c("1", "true", "yes")
overwrite <- tolower(Sys.getenv("OVERWRITE")) %in% c("1", "true", "yes")

if (is.na(n_draws) || n_draws <= 0) stop("N_DRAWS must be a positive integer.")
if (is.na(seed)) stop("SEED must be an integer.")
if (is.na(factor_r) || factor_r <= 0) stop("FACTOR_R must be a positive integer.")
if (is.na(theta)) stop("THETA must be numeric.")
if (is.na(noise_scale) || noise_scale <= 0) stop("NOISE_SCALE must be positive.")

ensure_pkg <- function(pkg, github_repo = NULL) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  if (!install_missing) {
    stop(sprintf(
      "Missing R package '%s'. Re-run with --install-missing or install it manually.",
      pkg
    ))
  }
  if (!is.null(github_repo)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    remotes::install_github(github_repo, upgrade = "never")
  } else {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Failed to install required package '%s'.", pkg))
  }
}

ensure_pkg("MASS")
ensure_pkg("gsynth")
ensure_pkg("augsynth", github_repo = "ebenmichael/augsynth")

suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(gsynth))
suppressPackageStartupMessages(library(augsynth))

if (!exists("interFE", where = asNamespace("gsynth"), inherits = FALSE)) {
  stop(
    "gsynth::interFE is not available in this R installation. ",
    "Try installing a gsynth version that still exports interFE."
  )
}

scale_safe <- function(x) {
  s <- stats::sd(x)
  if (is.na(s) || s <= 0) return(rep(0, length(x)))
  (x - mean(x)) / s
}

rescale_loadings <- function(phi) {
  if (is.null(dim(phi))) {
    return(matrix(scale_safe(phi), ncol = 1))
  }
  out <- apply(phi, 2, scale_safe)
  if (is.null(dim(out))) out <- matrix(out, ncol = 1)
  out
}

write_matrix_csv <- function(x, path) {
  utils::write.csv(
    as.data.frame(x, check.names = FALSE),
    file = path,
    row.names = FALSE,
    quote = FALSE
  )
}

if (dir.exists(output_dir)) {
  if (!overwrite) {
    stop("Output directory already exists. Pass --overwrite to replace it: ", output_dir)
  }
  unlink(output_dir, recursive = TRUE, force = TRUE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(seed)
data("kansas", package = "augsynth")
df <- kansas

required_cols <- c("fips", "state", "abb", "year_qtr", "treated", outcome_var)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("kansas dataset is missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Keep only what we need and rename the outcome to `y` for the model fit.
df <- as.data.frame(df[, required_cols])
names(df)[names(df) == outcome_var] <- "y"
df <- df[order(df$fips, df$year_qtr), ]

# gsynth::interFE has a bug with intercept-only formulas (y ~ 1) that crashes
# its Xname parsing. Add a trivial zero covariate so the formula has one X and
# the fitted coefficient on it is (numerically) zero.
df$zero_x <- stats::rnorm(nrow(df), mean = 0, sd = 1e-6)

if (any(is.na(df$y))) stop("Outcome column contains NA values.")

units <- sort(unique(df$fips))
times <- sort(unique(df$year_qtr))
N <- length(units)
TT <- length(times)

Y_obs <- xtabs(y ~ fips + year_qtr, data = df)
Y_obs <- Y_obs[as.character(units), as.character(times), drop = FALSE]

W_obs <- xtabs(treated ~ fips + year_qtr, data = df)
W_obs <- W_obs[as.character(units), as.character(times), drop = FALSE]
post_cols <- which(colSums(W_obs) > 0)
if (length(post_cols) == 0) stop("Could not find the post-treatment period in kansas::treated.")
T0 <- min(post_cols) - 1
if (T0 <= 0 || T0 >= TT) stop("Invalid T0 inferred from kansas::treated.")

state_map <- unique(df[, c("fips", "state", "abb")])
state_map <- state_map[match(units, state_map$fips), ]

# Fit the paper's main calibrated factor model.
fit <- gsynth::interFE(
  y ~ zero_x,
  data = df,
  index = c("fips", "year_qtr"),
  r = factor_r,
  force = "two-way",
  se = FALSE,
  tol = 1e-4
)

mu_hat <- as.matrix(fit$factor)
phi_hat <- as.matrix(fit$lambda)
alpha_hat <- as.numeric(fit$alpha)
xi_hat <- as.numeric(fit$xi)
sigma2_hat <- as.numeric(fit$sigma2)[1]

if (nrow(mu_hat) != TT) {
  stop(sprintf("Expected %d time rows in fit$factor but found %d.", TT, nrow(mu_hat)))
}
if (nrow(phi_hat) != N) {
  stop(sprintf("Expected %d unit rows in fit$lambda but found %d.", N, nrow(phi_hat)))
}
if (length(alpha_hat) != N) {
  stop(sprintf("Expected %d unit effects but found %d.", N, length(alpha_hat)))
}
if (length(xi_hat) != TT) {
  stop(sprintf("Expected %d time effects but found %d.", TT, length(xi_hat)))
}
if (is.na(sigma2_hat) || sigma2_hat < 0) stop("Invalid residual variance returned by interFE.")

sigma_eps <- sqrt(sigma2_hat) * noise_scale
alpha_mean <- mean(alpha_hat)
alpha_sd <- stats::sd(alpha_hat)
phi_cov <- stats::cov(phi_hat)
if (ncol(phi_hat) == 1) {
  phi_cov <- matrix(as.numeric(phi_cov), nrow = 1, ncol = 1)
}
# Numerical jitter in case phi_cov is nearly singular.
phi_cov <- phi_cov + diag(1e-8, nrow(phi_cov))

sample_draw <- function() {
  alpha_draw <- stats::rnorm(N, mean = alpha_mean, sd = alpha_sd)
  phi_draw <- MASS::mvrnorm(n = N, mu = rep(0, ncol(phi_hat)), Sigma = phi_cov)
  if (is.null(dim(phi_draw))) phi_draw <- matrix(phi_draw, ncol = 1)
  eps_draw <- matrix(stats::rnorm(N * TT, mean = 0, sd = sigma_eps), nrow = N, ncol = TT)

  alpha_mat <- matrix(alpha_draw, nrow = N, ncol = TT)
  xi_mat <- matrix(rep(xi_hat, each = N), nrow = N, ncol = TT)
  factor_part <- phi_draw %*% t(mu_hat)
  Y_true <- alpha_mat + xi_mat + factor_part + eps_draw

  alpha_z <- scale_safe(alpha_draw)
  phi_z <- rescale_loadings(phi_draw)
  score <- theta * (alpha_z + rowSums(phi_z))
  pi <- stats::plogis(score)
  pi <- pi / sum(pi)
  treated_row <- sample.int(N, size = 1, prob = pi)

  W <- matrix(0L, nrow = N, ncol = TT)
  W[treated_row, (T0 + 1):TT] <- 1L

  list(Y_true = Y_true, W = W, treated_row = treated_row)
}

manifest_rows <- vector("list", n_draws)
for (draw_id in seq_len(n_draws)) {
  draw <- sample_draw()
  draw_name <- sprintf("draw_%04d", draw_id - 1L)

  y_path <- file.path(output_dir, paste0(draw_name, "_Y.csv"))
  w_path <- file.path(output_dir, paste0(draw_name, "_W.csv"))
  write_matrix_csv(draw$Y_true, y_path)
  write_matrix_csv(draw$W, w_path)

  manifest_rows[[draw_id]] <- data.frame(
    draw_id = draw_id - 1L,
    outcome = outcome_var,
    assignment = sprintf("kansas_factor_r%d_theta%.3f_noise%.3f", factor_r, theta, noise_scale),
    N0 = N - 1L,
    T0 = T0,
    N1_realized = 1L,
    T1_realized = TT - T0,
    treated_row_start = draw$treated_row - 1L,
    treated_row_end = draw$treated_row,
    post_col_start = T0,
    post_col_end = TT,
    stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, manifest_rows)
utils::write.csv(manifest, file.path(output_dir, "manifest.csv"), row.names = FALSE, quote = FALSE)

utils::write.csv(
  data.frame(index0 = 0:(N - 1L), fips = units, state = state_map$state, abb = state_map$abb),
  file.path(output_dir, "row_index.csv"),
  row.names = FALSE,
  quote = FALSE
)

utils::write.csv(
  data.frame(index0 = 0:(TT - 1L), year_qtr = times),
  file.path(output_dir, "col_index.csv"),
  row.names = FALSE,
  quote = FALSE
)

metadata <- list(
  source_dataset = "augsynth::kansas",
  outcome = outcome_var,
  n_draws = n_draws,
  n_rows = N,
  n_cols = TT,
  T0 = T0,
  factor_r = factor_r,
  theta = theta,
  noise_scale = noise_scale,
  fitted_model = "gsynth::interFE(y ~ 1, force = 'two-way')",
  paper_target = "Kansas-calibrated factor-model simulation in Ben-Michael, Feller, Rothstein (2021)",
  note = paste(
    "Y files contain the full latent untreated outcome matrix; W files mark the single treated row",
    "for post-treatment columns. Apply the mask downstream to create observed panels."
  )
)
jsonlite_needed <- requireNamespace("jsonlite", quietly = TRUE)
if (jsonlite_needed) {
  writeLines(jsonlite::toJSON(metadata, auto_unbox = TRUE, pretty = TRUE), con = file.path(output_dir, "metadata.json"))
} else {
  dput(metadata, file = file.path(output_dir, "metadata.R"))
}

message(sprintf("Wrote %d Kansas semi-synthetic draws to %s", n_draws, output_dir))
'''


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_DIR": str(args.output_dir.resolve()),
            "N_DRAWS": str(args.draws),
            "SEED": str(args.seed),
            "OUTCOME_VAR": args.outcome,
            "FACTOR_R": str(args.factor_r),
            "THETA": str(args.theta),
            "NOISE_SCALE": str(args.noise_scale),
            "INSTALL_MISSING": "1" if args.install_missing else "0",
            "OVERWRITE": "1" if args.overwrite else "0",
        }
    )
    return env


def main() -> None:
    args = parse_args()

    if args.draws <= 0:
        raise SystemExit("--draws must be positive.")
    if args.factor_r <= 0:
        raise SystemExit("--factor-r must be positive.")
    if args.noise_scale <= 0:
        raise SystemExit("--noise-scale must be positive.")

    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}. Pass --overwrite to replace it."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="kansas_semisynth_"))
    r_script_path = temp_dir / "generate_kansas_semisynth.R"
    r_script_path.write_text(R_SCRIPT, encoding="utf-8")

    cmd = [args.rscript, str(r_script_path)]
    env = build_env(args)

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Could not find Rscript at '{args.rscript}'. Install R and/or pass --rscript PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"R generation step failed with exit code {exc.returncode}.") from exc
    finally:
        if args.keep_r_script:
            saved = args.output_dir.parent / f"{args.output_dir.name}_debug_generate_kansas_semisynth.R"
            try:
                saved.write_text(R_SCRIPT, encoding="utf-8")
                print(f"Saved debug R script to {saved}", file=sys.stderr)
            except OSError:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
