suppressPackageStartupMessages({
  library(data.table)
  library(stringr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(RColorBrewer)  # for Set1 palette in manual scale
})

# ---------- Config ----------
input_dir  <- "."   # change if needed
pattern    <- "^curve_metrics_epoch.*_f[0-5]\\.(csv|tsv)$"
out_dir    <- "plots_metrics_by_fold"
dpi_png    <- 300
width_in   <- 7.0
height_in  <- 4.5

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------- Load & combine ----------
files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
if (length(files) == 0) stop("No files matched. Check input_dir/pattern.")

read_one <- function(fp){
  sep <- if (grepl("\\.tsv$", fp, ignore.case = TRUE)) "\t" else ","
  dt  <- fread(fp, sep = sep)
  if (!"epoch" %in% names(dt)) stop(paste("Missing 'epoch' in", fp))
  fold_num <- stringr::str_match(basename(fp), "_f([0-5])")[,2]
  if (is.na(fold_num)) stop(paste("Couldn't parse fold from filename:", fp))
  dt[, fold := as.integer(fold_num)]
  dt
}

df <- rbindlist(lapply(files, read_one), use.names = TRUE, fill = TRUE)

# ---------- Identify metrics (everything except epoch & fold) ----------
metric_cols <- setdiff(names(df), c("epoch","fold"))
if (length(metric_cols) == 0) stop("No metric columns found besides 'epoch' and 'fold'.")

# ---------- Long format ----------
df_long <- df |>
  pivot_longer(cols = all_of(metric_cols), names_to = "metric", values_to = "value") |>
  arrange(metric, fold, epoch) |>
  mutate(Fold = as.character(fold))  # char for clean legend/factoring

# ---------- Build Avg(1..5) and combined plotting frame ----------
df_avg <- df_long |>
  filter(Fold %in% as.character(1:5)) |>
  group_by(metric, epoch) |>
  summarize(value = mean(value, na.rm = TRUE), .groups = "drop") |>
  mutate(Fold = "Avg")

df_withavg <- bind_rows(
  df_long %>% select(Fold, epoch, metric, value),
  df_avg
)

# ---------- Palette helper (folds use Set1, Avg is black) ----------
fold_color_map <- function(fold_levels) {
  base_levels <- setdiff(fold_levels, "Avg")
  n <- length(base_levels)
  # Set1 supports up to 9 distinct colors
  if (n > 0) {
    pal <- RColorBrewer::brewer.pal(max(3, min(9, n)), "Set1")[seq_len(n)]
  } else {
    pal <- character(0)
  }
  vals <- c(pal, "Avg" = "#000000")
  names(vals) <- c(base_levels, "Avg")
  vals[fold_levels]  # return in the same order as levels
}

# ---------- Plotting helpers ----------
plot_metric <- function(metric_name) {
  d <- df_long |> filter(metric == metric_name)
  ggplot(d, aes(x = epoch, y = value, color = Fold, group = Fold)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 1.8, alpha = 0.9) +
    scale_color_brewer(palette = "Set1", name = "Fold") +
    labs(
      title = paste0("Epoch vs ", metric_name),
      x = "Epoch",
      y = metric_name
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
}
plot_metric_withavg <- function(metric_name) {
  d <- df_withavg |> filter(metric == metric_name)
  folds_present <- sort(unique(as.integer(d$Fold[d$Fold != "Avg"])))
  fold_levels <- c(as.character(folds_present), "Avg")
  d <- d |> mutate(Fold = factor(Fold, levels = fold_levels))

  # Light palette for folds
  base_cols <- RColorBrewer::brewer.pal(max(3, length(folds_present)), "Set1")[seq_along(folds_present)]
  light_cols <- scales::alpha(base_cols, 0.7)  # folds moderately light

  # Avg in silver with higher transparency
  colors <- c(
    setNames(light_cols, as.character(folds_present)),
    "Avg" = scales::alpha("grey60", 0.5)
  )

  ggplot(d, aes(x = epoch, y = value, color = Fold, group = Fold)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.6) +
    scale_color_manual(
      values = colors,
      labels = c(paste0("Fold ", folds_present), "Avg (1–5)")
    ) +
    labs(
      title = paste0("Epoch vs ", metric_name),
      subtitle = "Folds 0–5 in light colors, Avg(1–5) in semi-transparent silver",
      x = "Epoch",
      y = metric_name
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
}

# ---------- Save per-metric PNG + PDF (original) ----------
for (m in metric_cols) {
  p <- plot_metric(m)
  png_path <- file.path(out_dir, paste0("epoch_vs_", m, ".png"))
  pdf_path <- file.path(out_dir, paste0("epoch_vs_", m, ".pdf"))
  ggsave(png_path, p, width = width_in, height = height_in, dpi = dpi_png)
  ggsave(pdf_path, p, width = width_in, height = height_in, dpi = dpi_png)
  message("Saved: ", basename(png_path), " and ", basename(pdf_path))
}

# ---------- Save per-metric PNG + PDF (WITH Avg(1..5)) ----------
for (m in metric_cols) {
  p_avg <- plot_metric_withavg(m)
  png_path <- file.path(out_dir, paste0("epoch_vs_", m, "_withavg.png"))
  pdf_path <- file.path(out_dir, paste0("epoch_vs_", m, "_withavg.pdf"))
  ggsave(png_path, p_avg, width = width_in, height = height_in, dpi = dpi_png)
  ggsave(pdf_path, p_avg, width = width_in, height = height_in, dpi = dpi_png)
  message("Saved: ", basename(png_path), " and ", basename(pdf_path))
}

# ---------- Optional: one multi-page PDF for all metrics (original) ----------
all_pdf <- file.path(out_dir, "ALL_metrics_multipage.pdf")
pdf(all_pdf, width = width_in, height = height_in, onefile = TRUE)
for (m in metric_cols) print(plot_metric(m))
dev.off()
message("Also saved multi-page PDF: ", basename(all_pdf))

# ---------- Optional: one multi-page PDF for all metrics (WITH Avg) ----------
all_pdf_avg <- file.path(out_dir, "ALL_metrics_multipage_withavg.pdf")
pdf(all_pdf_avg, width = width_in, height = height_in, onefile = TRUE)
for (m in metric_cols) print(plot_metric_withavg(m))
dev.off()
message("Also saved multi-page PDF: ", basename(all_pdf_avg))