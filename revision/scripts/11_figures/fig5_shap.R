# Figure 5 — SHAP mean |value| bars for the primary 517-feature CatBoost
# Reads official shap_mean_abs.csv (native TreeSHAP, all 630 discs).
# Layout matches the original 3 x 2 figure: grades I–V + global, top 8 each.
# Does not recompute SHAP and does not use the superseded shapviz RDS.

rm(list = ls())

root <- "revision"
shap_path <- file.path(root, "results/09_interpretability/shap_mean_abs.csv")
out_dir <- file.path(root, "figures")

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

shap <- read.csv(shap_path, stringsAsFactors = FALSE)
stopifnot("mean_abs_shap_all" %in% names(shap))
stopifnot(all(paste0("mean_abs_shap_class_", 1:5) %in% names(shap)))

# Imc2 must be global rank 1 (official 0.094).
stopifnot(shap$feature[1] == "log-sigma-3-0-mm-3D_glcm_Imc2")
stopifnot(abs(shap$mean_abs_shap_all[1] - 0.09446554) < 1e-6)

shorten <- function(x) {
  x <- gsub("log-sigma-([0-9]+)-0-mm-3D_", "LoG-\\1mm_", x)
  x <- gsub("lbp-3D-", "LBP-", x)
  x <- gsub("original_shape_", "shape_", x)
  x <- gsub("wavelet-", "wav-", x)
  x <- gsub("exponential_", "exp_", x)
  x <- gsub("squareroot_", "sqrt_", x)
  x <- gsub("square_", "sq_", x)
  x <- gsub("logarithm_", "log_", x)
  x <- gsub("gradient_", "grad_", x)
  x
}

importance_color <- "#3182bd"
max_features <- 8L

make_panel <- function(values, features, title) {
  ord <- order(values, decreasing = TRUE)
  keep <- ord[seq_len(min(max_features, length(ord)))]
  imp_df <- data.frame(
    Feature = factor(shorten(features[keep]),
                     levels = rev(shorten(features[keep]))),
    Importance = values[keep],
    stringsAsFactors = FALSE
  )
  ggplot(imp_df, aes(x = Feature, y = Importance)) +
    geom_col(fill = importance_color, color = "black",
             linewidth = 0.2, width = 0.7) +
    coord_flip() +
    labs(title = title, x = NULL, y = "Mean |SHAP Value|") +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.text.y = element_text(size = 8, face = "bold"),
      axis.text.x = element_text(size = 9),
      axis.title.x = element_text(size = 10, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(5, 8, 5, 5)
    )
}

roman <- c("I", "II", "III", "IV", "V")
labels <- c("(a)", "(b)", "(c)", "(d)", "(e)")
plots <- vector("list", 6)

for (k in 1:5) {
  col <- paste0("mean_abs_shap_class_", k)
  plots[[k]] <- make_panel(
    shap[[col]],
    shap$feature,
    sprintf("%s Pfirrmann Grade %s", labels[k], roman[k])
  )
}

plots[[6]] <- make_panel(
  shap$mean_abs_shap_all,
  shap$feature,
  "(f) Global"
)

combined <- wrap_plots(plots, nrow = 2, ncol = 3) +
  plot_annotation(
    title = "SHAP Feature Importance Across Lumbar Intervertebral Disc Degeneration Grades",
    subtitle = "Top features ranked by mean |SHAP value| for each Pfirrmann grade",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40")
    )
  )

ggsave(file.path(out_dir, "Figure5_Importance_Combined.pdf"),
       combined, width = 16, height = 10)
ggsave(file.path(out_dir, "Figure5_Importance_Combined.png"),
       combined, width = 16, height = 10, dpi = 600)
ggsave(file.path(out_dir, "Figure5_Importance_Combined.tiff"),
       combined, width = 16, height = 10, dpi = 600, compression = "lzw")

top8 <- shap$feature[seq_len(8)]
cat("Figure 5 written.\n")
cat("Global top 8:\n")
cat(paste(sprintf("  %d  %s  %.3f", seq_along(top8), top8,
                  shap$mean_abs_shap_all[seq_along(top8)]),
          collapse = "\n"), "\n")
