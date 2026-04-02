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
BASE_DIR <- "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/metric_bovplot_validation/"
OUT_DIR  <- file.path(BASE_DIR, "plots")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Metrics to plot (must exist in the CSVs)
METRICS <- c(
  "acc",
  "precision_macro",
  "recall_macro",
  "f1_macro",
  "roc_auc_ovo_macro",
  "aupr_macro"
)

# File patterns per task
PATTERNS <- list(
  celltype     = "^validation_best_celltype_summary_.*_fold[0-9]+\\.csv$",
  genotype     = "^validation_best_genotype_summary_.*_fold[0-9]+\\.csv$",
  genotypeNEXT = "^validation_best_genotypeNEXT_summary_.*_fold[0-9]+\\.csv$"
)

# -------------------------
# Helpers
# -------------------------
read_with_fold <- function(path) {
  fold <- str_match(basename(path), "_fold([0-9]+)\\.csv$")[,2]
  fold <- suppressWarnings(as.integer(fold))
  df <- suppressWarnings(readr::read_csv(path, show_col_types = FALSE))
  df$fold <- fold
  df$source_file <- basename(path)
  df
}

# (Optional) gentle harmonization for occasional header drift
harmonize_names <- function(df) {
  rename_map <- c(
    "accuracy"        = "acc",
    "macro_precision" = "precision_macro",
    "macro_recall"    = "recall_macro",
    "macro_f1"        = "f1_macro",
    "roc_auc_macro"   = "roc_auc_ovo_macro",  # if someone wrote macro without OVO
    "aupr"            = "aupr_macro"
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
                 names_to = "metric",
                 values_to = "value") %>%
    mutate(metric = factor(metric, levels = METRICS)) %>%
    filter(!is.na(value))
}

plot_mean_sd <- function(df, task, out_dir = OUT_DIR, error_type = c("sd","sem")) {
  error_type <- match.arg(error_type)

  if (nrow(df) == 0) {
    message(sprintf("[SKIP] No rows for task %s", task))
    return(invisible(NULL))
  }

  df <- df %>% mutate(fold = factor(fold, levels = sort(unique(fold))))

  # Summaries by metric
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
    # Mean bars
    geom_col(
      data = summ,
      aes(x = metric, y = mean),
      width = 0.65,
      fill = "grey20"
    ) +
    # Error bars (clipped to [0,1])
    geom_errorbar(
      data = summ,
      aes(x = metric,
          ymin = pmax(0, ymin),
          ymax = pmin(1, ymax)),
      width = 0.15,
      linewidth = 0.7,
      color = "black"
    ) +
    # Fold-level dots
    geom_point(
      data = df,
      aes(x = metric, y = value, color = fold),
      position = position_jitter(width = 0.08, height = 0),
      size = 2.0,
      alpha = 0.9
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = paste0("Validation Metrics — ", task, " (mean ± SD)"),
      x = "Metric",
      y = "Score (0–1)",
      color = "Fold"
    ) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 20, hjust = 1)
    )

  pdf_path <- file.path(out_dir, paste0("validation_bar_sd_", task, ".pdf"))
  png_path <- file.path(out_dir, paste0("validation_bar_sd_", task, ".png"))
  ggsave(pdf_path, p, width = 8.8, height = 5.4, device = cairo_pdf)
  ggsave(png_path, p, width = 8.8, height = 5.4, dpi = 300)
  message(sprintf("[WRITE] %s", pdf_path))
}

# -------------------------
# Main
# -------------------------
all_plots_df <- list()

for (task in names(PATTERNS)) {
  pat <- PATTERNS[[task]]
  files <- list.files(BASE_DIR, pattern = pat, full.names = TRUE)
  if (length(files) == 0) {
    message(sprintf("[WARN] No files found for %s in %s", task, BASE_DIR))
    next
  }

  df_task <- files %>%
    map(read_with_fold) %>%
    bind_rows() %>%
    gather_metrics(METRICS) %>%
    mutate(task = task)

  if (!is.null(df_task) && nrow(df_task) > 0) {
    plot_mean_sd(df_task, task, error_type = "sd")  # change to "sem" if you prefer SEM
    all_plots_df[[task]] <- df_task
  } else {
    message(sprintf("[WARN] No usable metrics for %s", task))
  }
}

# Optional: combined faceted chart across tasks
if (length(all_plots_df) > 0) {
  df_all <- bind_rows(all_plots_df)

  # compute task-level summaries for error bars
  summ_all <- df_all %>%
    group_by(task, metric) %>%
    summarise(
      mean = mean(value, na.rm = TRUE),
      sd   = sd(value, na.rm = TRUE),
      n    = sum(!is.na(value)),
      .groups = "drop"
    ) %>%
    mutate(
      ymin = mean - sd,
      ymax = mean + sd
    )

  p_all <- ggplot() +
    geom_col(
      data = summ_all,
      aes(x = metric, y = mean),
      width = 0.65,
      fill = "grey20"
    ) +
    geom_errorbar(
      data = summ_all,
      aes(x = metric,
          ymin = pmax(0, ymin),
          ymax = pmin(1, ymax)),
      width = 0.15,
      linewidth = 0.7,
      color = "black"
    ) +
    geom_point(
      data = df_all,
      aes(x = metric, y = value, color = factor(fold)),
      position = position_jitter(width = 0.08, height = 0),
      size = 1.6,
      alpha = 0.9
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    facet_wrap(~ task, ncol = 1) +
    labs(
      title = "Validation Metrics — All Tasks (mean ± SD)",
      x = "Metric",
      y = "Score (0–1)",
      color = "Fold"
    ) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 20, hjust = 1)
    )

  pdf_path <- file.path(OUT_DIR, "validation_bar_sd_ALL.pdf")
  png_path <- file.path(OUT_DIR, "validation_bar_sd_ALL.png")
  ggsave(pdf_path, p_all, width = 9, height = 10, device = cairo_pdf)
  ggsave(png_path, p_all, width = 9, height = 10, dpi = 300)
  message(sprintf("[WRITE] %s", pdf_path))
}

message("[DONE] Bar charts (mean ± SD) written.")