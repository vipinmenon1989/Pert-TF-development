library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# --- 1. List CSVs ---
files <- list.files(
  path = "/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/Metrics_all_folds",
  pattern = "^curve_metrics_epoch[0-9]+\\.csv$",
  full.names = TRUE
)

# --- 2. Extract epoch number from filename ---
extract_epoch_from_name <- function(fname) {
  as.integer(sub("^.*epoch([0-9]+)\\.csv$", "\\1", basename(fname)))
}

# --- 3. Read each file & extract matching epoch row ---
df_list <- lapply(files, function(f) {
  target_epoch <- extract_epoch_from_name(f)
  dat <- read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"epoch" %in% names(dat)) names(dat)[1] <- "epoch"
  dat$epoch <- as.numeric(dat$epoch)
  subdat <- dplyr::filter(dat, epoch == target_epoch)
  subdat$file_source <- basename(f)
  subdat
})

# --- 4. Combine all rows ---
all_df <- bind_rows(df_list)
if ("epoch" %in% colnames(all_df)) all_df <- all_df %>% select(-epoch)

# --- 5. Long format ---
long_df <- all_df %>%
  pivot_longer(
    cols = -file_source,
    names_to = "metric",
    values_to = "value"
  )

# --- 6. Split metric into 'split' (train/valid) and 'base_metric' (mse, mre, cls...) ---
long_df <- long_df %>%
  mutate(
    split = ifelse(startsWith(metric, "train_"), "train",
             ifelse(startsWith(metric, "valid_"), "valid", NA)),
    base_metric = sub("^(train_|valid_)", "", metric)
  )

# --- 7. Log-transform and then min–max scale to [0,1] within each base_metric ---
long_df <- long_df %>%
  mutate(log_value = log10(value + 1e-6)) %>%
  group_by(base_metric) %>%
  mutate(
    min_log = min(log_value, na.rm = TRUE),
    max_log = max(log_value, na.rm = TRUE),
    scaled_value = ifelse(
      max_log > min_log,
      (log_value - min_log) / (max_log - min_log),
      0.5
    )
  ) %>%
  ungroup()

# --- 8. Make one combined plot (train vs valid) per base_metric ---
unique_bases <- unique(long_df$base_metric)
dir.create("/local/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/Identity_model/Metrics_all_folds/per_metric_plots_scaled", showWarnings = FALSE)

plot_base_metric <- function(bm) {
  df_bm <- long_df %>% filter(base_metric == bm, !is.na(split))
  
  p <- ggplot(df_bm, aes(x = split, y = scaled_value, fill = split)) +
    geom_boxplot(outlier.shape = 16, outlier.size = 1.5, alpha = 0.6) +
    geom_jitter(width = 0.15, alpha = 0.6, color = "black") +
    scale_y_continuous(limits = c(0,1)) +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("Normalized log10 Distribution of ", bm, " (train vs valid)"),
      x = "Split",
      y = "Scaled log10(value)"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")
  
  ggsave(file.path("per_metric_plots_scaled",
                   paste0("boxplot_", bm, "_train_valid_scaled.pdf")),
         p, width = 5, height = 5)
}

purrr::walk(unique_bases, plot_base_metric)
