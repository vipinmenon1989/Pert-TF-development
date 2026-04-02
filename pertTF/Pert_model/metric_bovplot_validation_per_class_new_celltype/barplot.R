#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(purrr)
  library(forcats)
})

# =========================
# CONFIG — EDIT IF NEEDED
# =========================
BASE_DIR <- "/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/metric_bovplot_validation_per_class_new_celltype/"
OUT_DIR  <- file.path(BASE_DIR, "plots_bar")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Metrics expected in each CSV
METRICS_PER_CLASS <- c("precision", "recall", "f1", "auc_roc", "ap")

# Only these filenames
PATTERN_CELLTYPE <- "^metrics_celltype_fold[1-5]\\.csv$"

# Error bars: "sd" or "sem"
ERROR_TYPE <- "sd"   # change to "sem" if you prefer

# =========================
# Helpers
# =========================
read_with_meta <- function(path) {
  m_fold <- str_match(basename(path), "fold([0-9]+)\\.csv$")[,2]
  fold   <- suppressWarnings(as.integer(m_fold))

  df <- suppressWarnings(readr::read_csv(path, show_col_types = FALSE))

  needed <- c("class", "support", METRICS_PER_CLASS)
  missing <- setdiff(needed, names(df))
  if (length(missing) > 0) {
    warning(sprintf("Missing cols in %s: %s", basename(path), paste(missing, collapse=", ")))
    for (nm in missing) df[[nm]] <- NA_real_
  }

  df$fold        <- fold
  df$source_file <- basename(path)
  df
}

make_per_class_barplots <- function(df, task = "celltype", out_dir = OUT_DIR, error_type = ERROR_TYPE) {
  if (nrow(df) == 0) {
    message(sprintf("[SKIP] No rows for %s", task)); return(invisible(NULL))
  }

  long <- df %>%
    select(class, support, fold, all_of(METRICS_PER_CLASS)) %>%
    pivot_longer(cols = all_of(METRICS_PER_CLASS),
                 names_to = "metric", values_to = "value") %>%
    filter(!is.na(value))

  # Order classes by mean support (desc)
  class_order <- long %>%
    group_by(class) %>%
    summarise(avg_support = mean(support, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(avg_support)) %>%
    pull(class)

  long <- long %>%
    mutate(
      class  = factor(class, levels = class_order),
      metric = factor(metric, levels = METRICS_PER_CLASS),
      fold   = factor(fold, levels = sort(unique(fold)))
    )

  # Summary stats per class & metric (INLINE — no tidy-eval weirdness)
  summ <- long %>%
    group_by(class, metric) %>%
    summarise(
      mean = mean(value, na.rm = TRUE),
      sd   = sd(value, na.rm = TRUE),
      n    = sum(!is.na(value)),
      .groups = "drop"
    ) %>%
    mutate(
      sem  = sd / sqrt(pmax(n, 1)),
      err  = ifelse(error_type == "sem", sem, sd),
      ymin = pmax(0, mean - err),
      ymax = pmin(1, mean + err)
    )

  n_classes <- length(unique(long$class))
  width_in  <- max(8.5, 0.28 * n_classes + 3.5)

  base_theme <- theme_bw(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 9),
      strip.background = element_rect(fill = "grey90", color = NA),
      strip.text = element_text(face = "bold")
    )

  # ---------- (A) Combined faceted figure across all metrics ----------
  p_all <- ggplot() +
    geom_col(
      data = summ,
      aes(x = class, y = mean),
      width = 0.7, fill = "grey20"
    ) +
    geom_errorbar(
      data = summ,
      aes(x = class, ymin = ymin, ymax = ymax),
      width = 0.25, linewidth = 0.6, color = "black"
    ) +
    geom_point(
      data = long,
      aes(x = class, y = value, color = fold),
      position = position_jitter(width = 0.15, height = 0),
      size = 1.6, alpha = 0.9
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    facet_wrap(~ metric, ncol = 2, scales = "fixed") +
    labs(
      title = paste0("Per-class Validation — ", task, " (mean ± ", toupper(error_type), " across folds)"),
      x = "Class",
      y = "Score (0–1)",
      color = "Fold"
    ) + base_theme

  out_base_all <- file.path(out_dir, paste0("per_class_bar_", error_type, "_", task, "_ALL"))
  ggsave(paste0(out_base_all, ".pdf"), p_all, width = width_in, height = 8.8, device = cairo_pdf)
  ggsave(paste0(out_base_all, ".png"), p_all, width = width_in, height = 8.8, dpi = 300)
  message(sprintf("[WRITE] %s.[pdf|png]", out_base_all))

  # ---------- (B) Individual figures per metric ----------
  for (m in levels(long$metric)) {
    summ_m <- filter(summ, metric == m)
    long_m <- filter(long, metric == m)

    p_m <- ggplot() +
      geom_col(
        data = summ_m,
        aes(x = class, y = mean),
        width = 0.7, fill = "grey20"
      ) +
      geom_errorbar(
        data = summ_m,
        aes(x = class, ymin = ymin, ymax = ymax),
        width = 0.25, linewidth = 0.6, color = "black"
      ) +
      geom_point(
        data = long_m,
        aes(x = class, y = value, color = fold),
        position = position_jitter(width = 0.15, height = 0),
        size = 1.8, alpha = 0.9
      ) +
      coord_cartesian(ylim = c(0, 1)) +
      labs(
        title = paste0("Per-class ", m, " — ", task, " (mean ± ", toupper(error_type), ")"),
        x = "Class",
        y = "Score (0–1)",
        color = "Fold"
      ) + base_theme

    out_base <- file.path(out_dir, paste0("per_class_bar_", error_type, "_", task, "_", m))
    ggsave(paste0(out_base, ".pdf"), p_m, width = width_in, height = 6.8, device = cairo_pdf)
    ggsave(paste0(out_base, ".png"), p_m, width = width_in, height = 6.8, dpi = 300)
    message(sprintf("[WRITE] %s.[pdf|png]", out_base))
  }
}

# =========================
# Main
# =========================
files <- list.files(BASE_DIR, pattern = PATTERN_CELLTYPE, full.names = TRUE)
if (length(files) == 0) {
  message(sprintf("[WARN] No files matching %s in %s", PATTERN_CELLTYPE, BASE_DIR))
} else {
  df_cell <- files %>% map(read_with_meta) %>% bind_rows()
  make_per_class_barplots(df_cell, task = "celltype")
}

message("[DONE] Per-class barplots written.]")
