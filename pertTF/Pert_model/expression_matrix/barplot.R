#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(purrr)
})

# =========================
# CONFIG — EDIT IF NEEDED
# =========================
BASE_DIR <- "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/expression_matrix/"
OUT_DIR  <- file.path(BASE_DIR, "plots")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# We’ll plot these two columns if present
METRICS <- c("mean_cell_pearson", "mean_cell_cosine")

# File pattern: one CSV per fold, e.g., fold1_expression_metrics.csv
FILE_PATTERN <- "^fold[1-5]+_expression_metrics\\.csv$"

# -------------------------
# Helpers
# -------------------------
read_with_fold <- function(path) {
  # Extract fold number from filename like fold3_expression_metrics.csv
  fold <- str_match(basename(path), "^fold([0-9]+)_expression_metrics\\.csv$")[,2]
  fold <- suppressWarnings(as.integer(fold))
  df <- suppressWarnings(readr::read_csv(path, show_col_types = FALSE))
  df$fold <- fold
  df$source_file <- basename(path)
  df
}

harmonize_names <- function(df) {
  # In case column names drift
  rename_map <- c(
    "mean_pearson_cell" = "mean_cell_pearson",
    "mean_cosine_cell"  = "mean_cell_cosine"
  )
  for (old in names(rename_map)) {
    new <- rename_map[[old]]
    if (old %in% names(df) && !(new %in% names(df))) {
      names(df)[names(df) == old] <- new
    }
  }
  df
}

gather_metrics <- function(df, metrics = METRICS) {
  df <- harmonize_names(df)
  missing <- setdiff(metrics, colnames(df))
  if (length(missing) > 0) {
    warning(sprintf("Missing columns in %s: %s",
                    unique(df$source_file)[1],
                    paste(missing, collapse = ", ")))
  }
  present <- intersect(metrics, colnames(df))
  if (length(present) == 0) return(NULL)

  df %>%
    select(all_of(c("fold", "source_file", present))) %>%
    pivot_longer(cols = all_of(present),
                 names_to  = "metric",
                 values_to = "value") %>%
    mutate(metric = factor(metric, levels = metrics)) %>%
    filter(!is.na(value))
}

plot_mean_sd <- function(df, title = "Expression Metrics — mean ± SD",
                         out_base = "expression_bar_sd",
                         error_type = c("sd","sem")) {
  error_type <- match.arg(error_type)

  if (nrow(df) == 0) {
    message("[SKIP] No rows to plot.")
    return(invisible(NULL))
  }

  df <- df %>% mutate(fold = factor(fold, levels = sort(unique(fold))))

  # Summaries across folds
  summ <- df %>%
    group_by(metric) %>%
    summarise(
      mean = mean(value, na.rm = TRUE),
      sd   = sd(value, na.rm = TRUE),
      n    = sum(!is.na(value)),
      .groups = "drop"
    ) %>%
    mutate(
      sem  = sd / sqrt(pmax(n, 1)),
      ymin = ifelse(error_type == "sem", mean - sem, mean - sd),
      ymax = ifelse(error_type == "sem", mean + sem, mean + sd)
    )

  p <- ggplot() +
    geom_col(
      data = summ,
      aes(x = metric, y = mean),
      width = 0.6,
      fill  = "grey20"
    ) +
    geom_errorbar(
      data = summ,
      aes(x = metric,
          ymin = pmax(0, ymin),
          ymax = pmin(1, ymax)),
      width = 0.15,
      linewidth = 0.7,
      color = "black"
    ) +
    geom_point(
      data = df,
      aes(x = metric, y = value, color = fold),
      position = position_jitter(width = 0.07, height = 0),
      size = 2.1,
      alpha = 0.95
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = title,
      x = "Metric",
      y = "Score (0–1)",
      color = "Fold"
    ) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 15, hjust = 1)
    )

  pdf_path <- file.path(OUT_DIR, paste0(out_base, ".pdf"))
  png_path <- file.path(OUT_DIR, paste0(out_base, ".png"))
  ggsave(pdf_path, p, width = 6.8, height = 4.6, device = cairo_pdf)
  ggsave(png_path, p, width = 6.8, height = 4.6, dpi = 300)
  message(sprintf("[WRITE] %s", pdf_path))
}

# -------------------------
# Main
# -------------------------
files <- list.files(BASE_DIR, pattern = FILE_PATTERN, full.names = TRUE)
if (length(files) == 0) {
  stop(sprintf("No files matching %s in %s", FILE_PATTERN, BASE_DIR))
}

df_all <- files %>%
  map(read_with_fold) %>%
  bind_rows() %>%
  gather_metrics(METRICS)

if (is.null(df_all) || nrow(df_all) == 0) {
  stop("No usable metrics found in the expression metrics CSVs.")
}

plot_mean_sd(
  df_all,
  title    = "Expression Metrics (across folds) — mean ± SD",
  out_base = "expression_bar_sd",
  error_type = "sd" # switch to "sem" if you prefer
)

message("[DONE] Expression bar chart written.")
