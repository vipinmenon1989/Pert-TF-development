#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
  library(fs)
  library(purrr)
})

## ====================== USER KNOBS ======================
INPUT_DIR <- "."                  # folder with your CSVs
OUT_DIR   <- "./plots_fold_bars"  # output folder
FOLDS     <- 0:5
BASE_SIZE <- 13                   # base font size
WIDTH_BASE <- 12                  # base width (inches) before class-scaling
HEIGHT_IN  <- 8
DPI        <- 300

# Metrics to plot (normalized names)
METRICS <- c("AUC","AUPR","F1","Precision","Recall")

# Families to render (pattern pieces that distinguish file groups)
FAMILIES <- c("celltype_per_class", "genotype_per_class")
## =======================================================

dir_create(OUT_DIR)

# --- Filename patterns ---
LIST_RX  <- "^validation_fold[0-9]+_[A-Za-z0-9_]+_(celltype_per_class|genotype_per_class)_[A-Za-z0-9_]+\\.csv$"
PARSE_RX <- "^validation_fold(\\d+)_([A-Za-z0-9_]+)_(celltype_per_class|genotype_per_class)_([A-Za-z0-9_]+)\\.csv$"

epoch_number <- function(tag) {
  m <- stringr::str_match(tag, "e(?:poch)?(\\d+)$")
  if (is.na(m[1,2])) return(NA_real_)
  as.numeric(m[1,2])
}

pick_best_file <- function(df) {
  epnums <- pmax(epoch_number(df$ep1), epoch_number(df$ep2), na.rm = TRUE)
  if (all(is.na(epnums))) {
    df %>% arrange(desc(mtime)) %>% slice(1)
  } else {
    df %>%
      mutate(epnum_num = epnums) %>%
      arrange(desc(epnum_num), desc(mtime)) %>%
      slice(1) %>%
      select(-epnum_num)
  }
}

read_one_fold <- function(path, fold) {
  if (!file_exists(path)) {
    warning(sprintf("Missing file for fold %s: %s", fold, path))
    return(NULL)
  }
  df <- suppressMessages(readr::read_csv(path, show_col_types = FALSE))

  # Normalize column names
  cn  <- names(df)
  key <- tolower(gsub("[^a-z0-9]+", "", cn))
  alias <- c(
    "class" = "class",
    "label" = "class",
    "support" = "support",
    "precision" = "Precision",
    "recall" = "Recall",
    "f1" = "F1",
    "f1score" = "F1",
    "f1scores" = "F1",
    "auc" = "AUC",
    "aucroc" = "AUC",
    "rocauc" = "AUC",
    "ap" = "AUPR",
    "aupr" = "AUPR",
    "auprc" = "AUPR"
  )
  new_names <- cn
  for (i in seq_along(cn)) {
    if (key[i] %in% names(alias)) new_names[i] <- alias[[key[i]]]
  }
  names(df) <- new_names

  need_any <- intersect(c("class", METRICS, "support"), names(df))
  if (!"class" %in% need_any) {
    warning(sprintf("No 'class' column found in: %s", path))
    return(NULL)
  }
  present_metrics <- intersect(METRICS, need_any)
  if (length(present_metrics) == 0) {
    warning(sprintf("No expected metric columns found in: %s", path))
    return(NULL)
  }

  df %>%
    mutate(.fold = fold) %>%
    select(.fold, class, all_of(present_metrics), any_of("support"))
}

tidy_long <- function(df_list) {
  df <- bind_rows(df_list)
  present_metrics <- intersect(METRICS, names(df))
  df %>%
    pivot_longer(cols = all_of(present_metrics), names_to = "Metric", values_to = "Value") %>%
    mutate(Fold = factor(.fold, levels = sort(unique(.fold)))) %>%
    select(Fold, class, Metric, Value, any_of("support"))
}

collect_family_long <- function(input_dir, family_tag, folds) {
  all_files <- dir_ls(input_dir, type = "file", recurse = FALSE)
  all_files <- all_files[grepl(LIST_RX, path_file(all_files))]
  if (!length(all_files)) {
    warning(sprintf("No files matching pattern in: %s", input_dir))
    return(NULL)
  }

  m <- stringr::str_match(path_file(all_files), PARSE_RX)
  keep_idx <- which(!is.na(m[,1]))
  if (!length(keep_idx)) {
    warning("No files parsed successfully.")
    return(NULL)
  }

  meta <- tibble(
    path   = all_files[keep_idx],
    fold   = as.integer(m[keep_idx, 2]),
    ep1    = m[keep_idx, 3],
    family = m[keep_idx, 4],
    ep2    = m[keep_idx, 5],
    mtime  = file_info(all_files[keep_idx])$modification_time
  ) %>%
    filter(family == family_tag, fold %in% folds)

  if (nrow(meta) == 0) {
    warning(sprintf("No usable CSVs found for family '%s'", family_tag))
    return(NULL)
  }

  chosen <- meta %>%
    group_by(fold, family) %>%
    group_modify(~ pick_best_file(.x)) %>%
    ungroup() %>%
    arrange(fold) %>%
    mutate(epoch_tag = ifelse(ep1 == ep2, ep1, paste0(ep1, "/", ep2)))

  cat(sprintf("\n[INFO] Family: %s\n", family_tag))
  for (i in seq_len(nrow(chosen))) {
    cat(sprintf("  fold=%d  epoch_tag=%s  file=%s\n",
                chosen$fold[i], chosen$epoch_tag[i], chosen$path[i]))
  }

  dfs <- map2(chosen$path, chosen$fold, read_one_fold)
  dfs <- Filter(Negate(is.null), dfs)
  if (!length(dfs)) return(NULL)

  tidy_long(dfs)
}

# ============= Plotters (include Avg[1..5] bar in the SAME plot) =============

# single metric: grouped bars by Fold at each class + one "Avg" bar (mean across folds 1..5)
plot_one_metric_with_avg <- function(df_long, metric_name, family_label) {
  dsub <- df_long %>% filter(Metric == metric_name)

  # compute Avg across folds 1..5 ONLY
  davg <- dsub %>%
    filter(Fold %in% as.character(1:5)) %>%   # <-- exclude fold 0 from average
    group_by(class) %>%
    summarize(Value = mean(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(Fold = factor("Avg", levels = "Avg"))

  # bind: per-fold + Avg as one more "Fold"
  dplot <- bind_rows(
    dsub %>% select(Fold, class, Metric, Value),
    davg  %>% select(Fold, class, Value) %>% mutate(Metric = metric_name)
  )

  # order classes by Avg (descending) to make reading easy
  class_order <- davg %>% arrange(desc(Value)) %>% pull(class)
  if (length(class_order) == 0) {
    class_order <- dplot %>% distinct(class) %>% pull(class)
  }

  # Fold axis order: 0..5 then Avg (keep whatever folds exist)
  fold_levels_numeric <- sort(unique(as.integer(as.character(dsub$Fold))))
  fold_levels <- c(as.character(fold_levels_numeric), "Avg")

  dplot <- dplot %>% mutate(
    class = factor(class, levels = class_order),
    Fold  = factor(Fold, levels = fold_levels)
  )

  ggplot(dplot, aes(x = class, y = Value, fill = Fold)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.72) +
    labs(title = metric_name,
         subtitle = family_label,
         x = "Class", y = NULL) +
    coord_cartesian(ylim = c(0, max(1.0, max(dplot$Value, na.rm = TRUE) * 1.05))) +
    theme_classic(base_size = BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top",
      legend.title = element_text(size = BASE_SIZE - 1, face = "bold")
    )
}

# 5-panel grid (each panel already includes Avg[1..5] bar)
plot_family_grid_per_class_with_avg <- function(df_long, family_label) {
  plots <- lapply(METRICS, function(mn) plot_one_metric_with_avg(df_long, mn, family_label) + labs(subtitle = NULL))
  ((plots[[1]] | plots[[2]] | plots[[3]]) /
   (plots[[4]] | plots[[5]] | patchwork::plot_spacer())) +
    plot_annotation(
      title = paste0(family_label, " — Folds 0–5 + Avg(1–5)"),
      theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = BASE_SIZE + 2))
    )
}

save_plot <- function(p, file_prefix, n_classes, height_in = HEIGHT_IN) {
  width_in <- WIDTH_BASE + 0.25 * max(0, n_classes - 10)
  pdf_file <- path(OUT_DIR, paste0(file_prefix, ".pdf"))
  png_file <- path(OUT_DIR, paste0(file_prefix, ".png"))
  if (capabilities("cairo")) {
    ggsave(pdf_file, plot = p, width = width_in, height = height_in, device = cairo_pdf)
  } else {
    ggsave(pdf_file, plot = p, width = width_in, height = height_in)
  }
  ggsave(png_file, plot = p, width = width_in, height = height_in, dpi = DPI)
  message(sprintf("Wrote: %s", pdf_file))
  message(sprintf("Wrote: %s", png_file))
}

# ---- Main ----
for (fam in FAMILIES) {
  df_long <- collect_family_long(INPUT_DIR, fam, FOLDS)
  if (is.null(df_long) || nrow(df_long) == 0) next

  family_label <- if (str_detect(fam, "celltype")) {
    "Per-Class Metrics — Celltype"
  } else if (str_detect(fam, "genotype")) {
    "Per-Class Metrics — Genotype"
  } else {
    paste("Per-Class Metrics —", fam)
  }

  n_classes <- df_long %>% distinct(class) %>% nrow()

  # A) Combined 5-panel grid (folds 0..5 + Avg[1..5])
  p_grid <- plot_family_grid_per_class_with_avg(df_long, family_label)
  save_plot(p_grid, paste0("bars_perclass_", fam, "_folds_withavg_grid"), n_classes)

  # B) Five separate plots (each metric: folds 0..5 + Avg[1..5])
  for (mn in METRICS) {
    p_one <- plot_one_metric_with_avg(df_long, mn, family_label)
    out_prefix <- paste0("bars_perclass_", fam, "_", tolower(mn), "_folds_withavg")
    save_plot(p_one, out_prefix, n_classes)
  }
}

cat("\nDone.\n")