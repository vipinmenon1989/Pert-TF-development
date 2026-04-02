suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
  library(purrr);   library(stringr)
})

#=====================
# CONFIG
#=====================
base_dir <- "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Pert_model/Metric_fold/"
pattern  <- "^curve_metrics_epoch_fold[1-5]+\\.csv$"
out_dir  <- file.path(base_dir, "per_metric_timecourses_lineplots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

EPOCH_MAX   <- 100   # X-axis upper limit
WIDTH <- 7; HEIGHT <- 4
SHOW_POINTS <- FALSE # set TRUE if you want markers on the lines

# Order metrics: common first, then the rest alphabetically
metric_priority <- c("cls","mse","mre","dab")

#=====================
# HELPERS
#=====================
extract_fold_from_name <- function(fname)
  as.integer(sub("^.*fold([1-5]+)\\.csv$", "\\1", basename(fname)))

normalize_epoch_col <- function(df){
  if (!"epoch" %in% names(df)) names(df)[1] <- "epoch"
  df$epoch <- suppressWarnings(as.numeric(df$epoch))
  df
}

#=====================
# LOAD
#=====================
files <- list.files(base_dir, pattern = pattern, full.names = TRUE)
stopifnot(length(files) > 0)

raw_list <- lapply(files, function(f){
  dat <- read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  dat <- normalize_epoch_col(dat)
  dat$fold <- extract_fold_from_name(f)
  dat$file_source <- basename(f)
  dat
})

all_df <- bind_rows(raw_list)

metric_cols <- setdiff(colnames(all_df), c("epoch","fold","file_source"))
# Make metric cols numeric where possible
all_df[metric_cols] <- lapply(all_df[metric_cols], function(x){
  if (is.numeric(x)) x else suppressWarnings(as.numeric(x))
})

plot_df <- all_df %>% filter(!is.na(epoch))

#=====================
# LONG & TAG SPLIT
#=====================
long_df <- plot_df %>%
  pivot_longer(cols = all_of(metric_cols), names_to = "metric", values_to = "value") %>%
  mutate(
    split = case_when(
      startsWith(metric, "train_") ~ "train",
      startsWith(metric, "valid_") ~ "valid",
      TRUE ~ NA_character_
    ),
    base_metric = sub("^(train_|valid_)", "", metric)
  ) %>%
  filter(!is.na(split), !is.na(value)) %>%
  arrange(epoch)

# Metric ordering
bases <- unique(long_df$base_metric)
ord_idx <- match(bases, metric_priority)
ord_idx[is.na(ord_idx)] <- max(length(metric_priority), 0) + rank(bases[is.na(ord_idx)], ties.method = "first")
bases <- bases[order(ord_idx)]

#=====================
# PLOTTING
#=====================
plot_idx <- 1
save_plot <- function(p, split, bm){
  fname <- sprintf("%02d_%s_%s_epoch_vs_value_by_fold.pdf", plot_idx, split, bm)
  ggsave(filename = file.path(out_dir, fname), plot = p, width = WIDTH, height = HEIGHT)
}

make_line_plot <- function(df, split, bm){
  g <- ggplot(df, aes(x = epoch, y = value, color = factor(fold), group = fold)) +
    geom_line(linewidth = 0.9, na.rm = TRUE)
  if (SHOW_POINTS) {
    g <- g + geom_point(size = 1.2, alpha = 0.8, na.rm = TRUE)
  }
  g +
    scale_x_continuous(limits = c(0, EPOCH_MAX), breaks = seq(0, EPOCH_MAX, by = 10)) +
    labs(
      title = sprintf("(%d) %s %s vs Epoch (by fold)", plot_idx,
                      tools::toTitleCase(split), bm),
      x = "Epoch", y = bm, color = "Fold"
    ) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "right",
      plot.title = element_text(face = "bold")
    )
}

# 1) TRAIN plots first
for (bm in bases) {
  df_bm <- long_df %>% filter(base_metric == bm, split == "train")
  if (nrow(df_bm) == 0) next
  p <- make_line_plot(df_bm, "train", bm)
  save_plot(p, "train", bm); plot_idx <- plot_idx + 1
}

# 2) VALID plots continue numbering
for (bm in bases) {
  df_bm <- long_df %>% filter(base_metric == bm, split == "valid")
  if (nrow(df_bm) == 0) next
  p <- make_line_plot(df_bm, "valid", bm)
  save_plot(p, "valid", bm); plot_idx <- plot_idx + 1
}

message("Done. Line plots saved to: ", out_dir)
