# Figure 4 — primary CatBoost OVR ROC + confusion matrix
# Reads official patient-level predictions and class-wise AUCs.
# ROC coordinates come from the saved probabilities; AUC/CI annotations
# are taken from class_auc.csv so the figure matches Table 4.

rm(list = ls())

root <- "revision"
pred_path <- file.path(root, "results/02_primary/test_predictions_3d.csv")
auc_path <- file.path(root, "results/02_primary/class_auc.csv")
out_dir <- file.path(root, "figures")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(pROC)
  library(patchwork)
})

pred <- read.csv(pred_path, stringsAsFactors = FALSE)
auc_tab <- read.csv(auc_path, stringsAsFactors = FALSE)
auc_tab <- auc_tab[auc_tab$config == "3D_primary", ]
stopifnot(nrow(pred) == 126L)
stopifnot(nrow(auc_tab) == 5L)

grades <- 1:5
roman <- c("I", "II", "III", "IV", "V")
grade_titles <- paste("Pfirrmann Grade", roman)
roc_color <- "#3182bd"
cm_low_color <- "#f7fbff"
cm_high_color <- "#08519c"

pred$y_true <- factor(pred$y_true, levels = grades, labels = roman)
pred$y_pred <- factor(pred$y_pred, levels = grades, labels = roman)

fmt_ci <- function(lo, hi) {
  sprintf("%.3f-%.3f", lo, hi)
}

create_roc_plot <- function(i) {
  g <- grades[i]
  y_bin <- ifelse(as.integer(pred$y_true) == i, 1, 0)
  score <- pred[[paste0("prob_", g)]]
  roc_obj <- roc(response = y_bin, predictor = score, quiet = TRUE, direction = "<")
  roc_df <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities
  )
  row <- auc_tab[auc_tab$class == g, ]
  auc_label <- sprintf(
    "AUC: %.3f\nCI: %s",
    row$AUC,
    fmt_ci(row$AUC_ci_low, row$AUC_ci_high)
  )
  ggplot(roc_df, aes(x = fpr, y = tpr)) +
    annotate("segment", x = 0, y = 0, xend = 1, yend = 1,
             linetype = "dashed", color = "gray50", linewidth = 0.7) +
    geom_step(color = roc_color, linewidth = 1, direction = "hv") +
    annotate("text", x = 1, y = 0.08, label = auc_label,
             size = 3.8, hjust = 1) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = c(0.01, 0.01)) +
    labs(
      title = grade_titles[i],
      x = "1-Specificity (FPR)",
      y = "Sensitivity (TPR)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.title = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9, color = "black"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 0.8),
      aspect.ratio = 1,
      plot.margin = margin(5, 5, 5, 5)
    )
}

roc_plots <- lapply(grades, create_roc_plot)

cm <- as.data.frame(table(Predicted = pred$y_pred, Actual = pred$y_true))
colnames(cm) <- c("Predicted", "Actual", "Count")
cm <- cm %>%
  group_by(Actual) %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  ungroup()
cm$Label <- sprintf("%d\n(%.1f%%)", cm$Count, cm$Percentage)
cm$Predicted <- factor(cm$Predicted, levels = roman)
cm$Actual <- factor(cm$Actual, levels = roman)

sens <- cm$Percentage[as.character(cm$Predicted) == as.character(cm$Actual)]
sens <- sens[match(roman, as.character(cm$Actual[as.character(cm$Predicted) == as.character(cm$Actual)]))]
# Diagonal percentages must match Table 4 sensitivity.
stopifnot(abs(sens[1] - 16.7) < 0.05)
stopifnot(abs(sens[2] - 85.7) < 0.05)
stopifnot(abs(sens[3] - 70.6) < 0.05)
stopifnot(abs(sens[4] - 80.0) < 0.05)
stopifnot(abs(sens[5] - 66.7) < 0.05)

cm_plot <- ggplot(cm, aes(x = Predicted, y = Actual, fill = Percentage)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = Label), size = 3.5,
            color = ifelse(cm$Percentage > 50, "white", "black")) +
  scale_fill_gradient(low = cm_low_color, high = cm_high_color,
                      limits = c(0, 100), name = "%") +
  labs(
    title = "Confusion Matrix",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
    axis.title = element_text(size = 10, face = "bold"),
    axis.text = element_text(size = 9, color = "black"),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 9),
    aspect.ratio = 1,
    plot.margin = margin(5, 5, 5, 5)
  )

combined <- (roc_plots[[1]] | roc_plots[[2]]) /
  (roc_plots[[3]] | roc_plots[[4]]) /
  (roc_plots[[5]] | cm_plot) +
  plot_annotation(
    title = "CatBoost Classification Performance",
    subtitle = "ROC Curves for Lumbar Intervertebral Disc Degeneration Grades",
    tag_levels = "a",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray30")
    )
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave(file.path(out_dir, "Figure4_CatBoost_ROC_CM.pdf"),
       combined, width = 10, height = 12)
ggsave(file.path(out_dir, "Figure4_CatBoost_ROC_CM.png"),
       combined, width = 10, height = 12, dpi = 600)
ggsave(file.path(out_dir, "Figure4_CatBoost_ROC_CM.tiff"),
       combined, width = 10, height = 12, dpi = 600, compression = "lzw")

cat("Figure 4 written.\n")
print(as.data.frame.matrix(table(Actual = pred$y_true, Predicted = pred$y_pred)))
cat(sprintf("Grade AUCs: %s\n",
            paste(sprintf("%s=%.3f", roman, auc_tab$AUC[match(grades, auc_tab$class)]),
                  collapse = "; ")))
