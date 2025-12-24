# ================================================================
# Model Performance Comparison Visualization
# ================================================================
# Author: [Your Name]
# Description: Generate publication-ready performance comparison plots
# License: MIT
# ================================================================

# Clear environment
rm(list = ls())

# ================================================================
# CONFIGURATION
# ================================================================

config <- list(
  # Input files
  train_performance_file = "../output/1_All_Models_Train_Performance.csv",
  test_performance_file = "../output/1_All_Models_Test_Performance.csv",
  performance_with_ci_file = "../output/Test_Performance.csv",
  models_dir = "../output/models",
  
  # Output settings
  output_dir = "../output/figures",
  
  # Plot settings
  figure_width = 14,
  figure_height = 10,
  dpi = 1200
)

# ================================================================
# LOAD REQUIRED PACKAGES
# ================================================================

required_packages <- c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "patchwork"
)

suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})

# Create output directory if not exists
if (!dir.exists(config$output_dir)) {
  dir.create(config$output_dir, recursive = TRUE)
}

# ================================================================
# COLOR SCHEME (Publication-ready)
# ================================================================

color_mapping <- c(
  CatBoost = "#3182bd",            # Deep blue
  DecisionTree = "#B09C85",        # Beige
  KNN = "#DC91A7",                 # Pink
  Lasso = "#F39B7F",               # Coral
  MultinomialLogistic = "#91D1C2", # Light green
  NaiveBayes = "#4DBBD5",          # Cyan
  NeuralNetwork = "#7E6148",       # Brown
  RandomForest = "#E64B35",        # Red
  Ridge = "#00A087",               # Green
  XGBoost = "#8491B4"              # Gray-blue
)

# ================================================================
# LOAD DATA
# ================================================================

message("\n===== Loading Performance Data =====\n")

# Load training set performance
if (file.exists(config$train_performance_file)) {
  train_results <- read.csv(config$train_performance_file)
  message("Training set performance loaded: ", nrow(train_results), " models")
} else {
  message("Warning: Training performance file not found")
  train_results <- NULL
}

# Load test set performance
if (file.exists(config$test_performance_file)) {
  test_mean <- read.csv(config$test_performance_file)
  message("Test set performance loaded: ", nrow(test_mean), " models")
} else {
  stop("Test performance file not found: ", config$test_performance_file)
}

# Load performance with confidence intervals (if available)
if (file.exists(config$performance_with_ci_file)) {
  performance_df <- read.csv(config$performance_with_ci_file)
  message("Performance with CI loaded")
  has_ci <- TRUE
} else {
  message("Warning: Performance with CI file not found, using point estimates only")
  has_ci <- FALSE
}

# ================================================================
# PREPARE DATA FOR PLOTTING
# ================================================================

# Define metrics to plot
main_metrics <- c("AUC", "Accuracy", "F1", "Sensitivity", "Specificity", "Precision")

# Create metric labels with subplot identifiers
metric_labels <- c(
  "AUC" = "(a) AUC",
  "Accuracy" = "(b) Accuracy",
  "F1" = "(c) F1",
  "Sensitivity" = "(d) Sensitivity",
  "Specificity" = "(e) Specificity",
  "Precision" = "(f) Precision"
)

if (has_ci) {
  plot_data <- performance_df %>%
    filter(Metric %in% main_metrics) %>%
    mutate(
      Metric_Label = factor(metric_labels[Metric], levels = metric_labels),
      Model = factor(Model, levels = test_mean$Model[order(test_mean$AUC, decreasing = TRUE)])
    )
} else {
  # Reshape test_mean for plotting without CI
  plot_data <- test_mean %>%
    tidyr::pivot_longer(
      cols = all_of(main_metrics),
      names_to = "Metric",
      values_to = "Mean"
    ) %>%
    mutate(
      Metric_Label = factor(metric_labels[Metric], levels = metric_labels),
      Model = factor(Model, levels = test_mean$Model[order(test_mean$AUC, decreasing = TRUE)]),
      Lower = Mean,
      Upper = Mean
    )
}

# ================================================================
# FIGURE: Multi-metric Performance Comparison
# ================================================================

message("\n===== Generating Performance Comparison Plot =====\n")

multi_metric_plot <- ggplot(plot_data, aes(x = Model, y = Mean, fill = Model)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", linewidth = 0.3) +
  {if (has_ci) geom_errorbar(aes(ymin = Lower, ymax = Upper), 
                              width = 0.25, 
                              linewidth = 0.3,
                              color = "black")} +
  geom_text(aes(label = sprintf("%.3f", Mean)),
            vjust = -1,
            size = 3,
            color = "black") +
  scale_fill_manual(values = color_mapping, name = "Model") +
  facet_wrap(~ Metric_Label, scales = "free_y", ncol = 3) +
  labs(
    title = "Model Performance Comparison",
    subtitle = "Multi-class Classification Results",
    x = NULL,
    y = "Performance Value",
    caption = "Error bars represent 95% confidence intervals"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 11, face = "bold"),
    strip.text = element_text(size = 11, face = "bold", hjust = 0.5),
    strip.background = element_rect(fill = "gray90", color = "black"),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 10, face = "bold"),
    panel.spacing = unit(1, "lines"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  guides(fill = guide_legend(nrow = 2))

# Display plot
print(multi_metric_plot)

# ================================================================
# SAVE FIGURES
# ================================================================

message("\n===== Saving Figures =====\n")

# Save in multiple formats
ggsave(file.path(config$output_dir, "Model_Performance_Comparison.pdf"), 
       multi_metric_plot, 
       width = config$figure_width, 
       height = config$figure_height, 
       dpi = config$dpi)

ggsave(file.path(config$output_dir, "Model_Performance_Comparison.png"), 
       multi_metric_plot, 
       width = config$figure_width, 
       height = config$figure_height, 
       dpi = config$dpi)

ggsave(file.path(config$output_dir, "Model_Performance_Comparison.tiff"), 
       multi_metric_plot, 
       width = config$figure_width, 
       height = config$figure_height, 
       dpi = config$dpi, 
       compression = "lzw")

message("Figures saved to: ", config$output_dir)
message("  - Model_Performance_Comparison.pdf")
message("  - Model_Performance_Comparison.png")
message("  - Model_Performance_Comparison.tiff")

# ================================================================
# SUMMARY TABLE
# ================================================================

message("\n===== Performance Summary =====\n")

summary_table <- test_mean %>%
  arrange(desc(AUC)) %>%
  mutate(Rank = row_number())

print(summary_table)

message("\n===== Visualization Complete =====\n")
