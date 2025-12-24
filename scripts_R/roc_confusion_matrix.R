# ================================================================
# ROC Curves and Confusion Matrix Visualization
# ================================================================
# Author: [Your Name]
# Description: Generate publication-ready ROC curves and confusion
#              matrix heatmaps for multi-class classification
# License: MIT
# ================================================================

# Clear environment
rm(list = ls())

# ================================================================
# CONFIGURATION
# ================================================================

config <- list(
  # Data settings
  data_file = "../data/extracted_data.xlsx",
  response_var = "quality",
  class_levels = c("I", "II", "III", "IV", "V"),
  
  # Model settings
  model_file = "../output/models/catboost_model.cbm",  # or .rds for caret models
  model_name = "CatBoost",
  model_type = "catboost",  # "catboost" or "caret"
  
  # Data split (must match training)
  train_ratio = 0.8,
  random_seed = 4321,
  
  # Output settings
  output_dir = "../output/figures",
  subplots_dir = "../output/figures/subplots",
  
  # Plot settings
  figure_dpi = 900,
  
  # Color scheme
  roc_color = "#3182bd",
  cm_low_color = "#f7fbff",
  cm_high_color = "#08519c"
)

# ================================================================
# LOAD REQUIRED PACKAGES
# ================================================================

required_packages <- c(
  "caret",
  "pROC",
  "ggplot2",
  "dplyr",
  "tidyr",
  "patchwork",
  "readxl"
)

# Load catboost if using catboost model
if (config$model_type == "catboost") {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    message("Note: catboost package not installed. Install with:")
    message('  install.packages("catboost", repos = "https://catboost.ai/r-package")')
  } else {
    required_packages <- c(required_packages, "catboost")
  }
}

suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})

# Create output directories
for (dir in c(config$output_dir, config$subplots_dir)) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
}

# ================================================================
# LOAD DATA
# ================================================================

message("\n===== Loading Data =====\n")

if (!file.exists(config$data_file)) {
  stop("Data file not found: ", config$data_file)
}

# Load and preprocess data
data <- read_excel(config$data_file)

# Convert response variable
if (is.numeric(data[[config$response_var]])) {
  data[[config$response_var]] <- as.integer(data[[config$response_var]])
  data[[config$response_var]] <- as.character(as.roman(data[[config$response_var]]))
}
data[[config$response_var]] <- factor(data[[config$response_var]], 
                                       levels = config$class_levels)

# Split data (same as training)
set.seed(config$random_seed)
train_index <- createDataPartition(data[[config$response_var]], 
                                   p = config$train_ratio, 
                                   list = FALSE)
traindata <- data[train_index, ]
testdata <- data[-train_index, ]

testdata[[config$response_var]] <- factor(testdata[[config$response_var]], 
                                           levels = config$class_levels)

message("Test set: ", nrow(testdata), " samples")

# ================================================================
# LOAD MODEL AND GET PREDICTIONS
# ================================================================

message("\n===== Loading Model and Making Predictions =====\n")

class_names <- levels(testdata[[config$response_var]])
n_classes <- length(class_names)

if (config$model_type == "catboost") {
  # Load CatBoost model
  model <- catboost.load_model(config$model_file)
  
  test_pool <- catboost.load_pool(
    data = testdata[, -which(names(testdata) == config$response_var)],
    label = as.numeric(testdata[[config$response_var]]) - 1
  )
  
  pred_probs <- catboost.predict(model, test_pool, prediction_type = "Probability")
  colnames(pred_probs) <- class_names
  
} else {
  # Load caret model
  model <- readRDS(config$model_file)
  pred_probs <- predict(model, testdata, type = "prob")
}

pred_class <- factor(class_names[apply(pred_probs, 1, which.max)], 
                     levels = class_names)
truth <- testdata[[config$response_var]]

message("Predictions generated successfully")

# ================================================================
# HELPER FUNCTIONS
# ================================================================

# Function: Create ROC plot for combination figure
create_roc_plot <- function(roc_obj, title_text, roc_color) {
  
  roc_df <- data.frame(
    sens = roc_obj$sensitivities,
    spec = roc_obj$specificities
  )
  
  auc_val <- round(as.numeric(roc_obj$auc), 3)
  ci_val <- ci(roc_obj, conf.level = 0.95)
  ci_lower <- round(ci_val[1], 3)
  ci_upper <- round(ci_val[3], 3)
  
  auc_label <- sprintf("AUC: %.3f\nCI: %.3f-%.3f", auc_val, ci_lower, ci_upper)
  
  p <- ggplot(roc_df, aes(x = 1 - spec, y = sens)) +
    geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
                 linetype = "dashed", color = "gray50", linewidth = 0.7) +
    geom_step(color = roc_color, linewidth = 1, direction = "hv") +
    annotate("text", x = 1, y = 0.08, label = auc_label,
             size = 3.8, hjust = 1, fontface = "plain") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                       expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                       expand = c(0.01, 0.01)) +
    labs(
      title = title_text,
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
  
  return(p)
}

# Function: Create ROC plot for standalone figure
create_roc_plot_standalone <- function(roc_obj, title_text, model_name, roc_color) {
  
  roc_df <- data.frame(
    sens = roc_obj$sensitivities,
    spec = roc_obj$specificities
  )
  
  auc_val <- round(as.numeric(roc_obj$auc), 3)
  ci_val <- ci(roc_obj, conf.level = 0.95)
  ci_lower <- round(ci_val[1], 3)
  ci_upper <- round(ci_val[3], 3)
  
  auc_label <- sprintf("%s\nAUC: %.3f\nCI: %.3f-%.3f", 
                       model_name, auc_val, ci_lower, ci_upper)
  
  p <- ggplot(roc_df, aes(x = 1 - spec, y = sens)) +
    geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1),
                 linetype = "dashed", color = "gray50", linewidth = 0.7) +
    geom_step(color = roc_color, linewidth = 1, direction = "hv") +
    annotate("text", x = 1, y = 0.08, label = auc_label,
             size = 3.8, hjust = 1, fontface = "plain") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                       expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                       expand = c(0.01, 0.01)) +
    labs(
      title = title_text,
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
  
  return(p)
}

# Function: Create confusion matrix heatmap
create_confusion_matrix_plot <- function(truth, prediction, cm_low_color, cm_high_color) {
  
  cm <- table(Predicted = prediction, Actual = truth)
  cm_df <- as.data.frame(cm)
  colnames(cm_df) <- c("Predicted", "Actual", "Count")
  
  cm_df <- cm_df %>%
    group_by(Actual) %>%
    mutate(Percentage = Count / sum(Count) * 100) %>%
    ungroup()
  
  cm_df$Label <- sprintf("%d\n(%.1f%%)", cm_df$Count, cm_df$Percentage)
  
  p <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Percentage)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = Label), size = 3.5,
              color = ifelse(cm_df$Percentage > 50, "white", "black")) +
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
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 9),
      aspect.ratio = 1,
      plot.margin = margin(5, 5, 5, 5)
    )
  
  return(p)
}

# Function: Create standalone confusion matrix
create_confusion_matrix_standalone <- function(truth, prediction, model_name, 
                                                cm_low_color, cm_high_color) {
  
  cm_obj <- confusionMatrix(prediction, truth)
  cm <- table(Predicted = prediction, Actual = truth)
  cm_df <- as.data.frame(cm)
  colnames(cm_df) <- c("Predicted", "Actual", "Count")
  
  cm_df <- cm_df %>%
    group_by(Actual) %>%
    mutate(Percentage = Count / sum(Count) * 100) %>%
    ungroup()
  
  cm_df$Label <- sprintf("%d\n(%.1f%%)", cm_df$Count, cm_df$Percentage)
  overall_acc <- round(cm_obj$overall["Accuracy"] * 100, 1)
  
  p <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Percentage)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = Label), size = 4,
              color = ifelse(cm_df$Percentage > 50, "white", "black")) +
    scale_fill_gradient(low = cm_low_color, high = cm_high_color,
                        limits = c(0, 100), name = "Percentage\n(%)") +
    labs(
      title = "Confusion Matrix for Multi-class Classification",
      subtitle = paste(model_name, "Model"),
      x = "Predicted Class",
      y = "Actual Class",
      caption = sprintf("Overall Accuracy: %.1f%% | n = %d", overall_acc, length(truth))
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray40"),
      plot.caption = element_text(hjust = 0.5, size = 10, color = "gray50", face = "italic"),
      axis.title = element_text(size = 11, face = "bold"),
      axis.text = element_text(size = 10, color = "black"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 10, face = "bold"),
      aspect.ratio = 1,
      plot.margin = margin(10, 10, 10, 10)
    )
  
  return(p)
}

# ================================================================
# GENERATE ROC CURVES
# ================================================================

message("\n===== Generating ROC Curves =====\n")

roc_plots <- list()
roc_plots_standalone <- list()
roc_objects <- list()

# Define grade labels (customize for your application)
grade_labels <- c(
  "I" = "Grade I",
  "II" = "Grade II",
  "III" = "Grade III",
  "IV" = "Grade IV",
  "V" = "Grade V"
)

for (i in seq_len(min(5, n_classes))) {
  class <- class_names[i]
  
  # Create binary classification for one-vs-rest
  binary_truth <- factor(
    ifelse(truth == class, class, "Other"),
    levels = c(class, "Other")
  )
  
  class_prob <- pred_probs[, class]
  
  # Calculate ROC
  roc_obj <- roc(response = binary_truth, predictor = class_prob,
                 levels = c("Other", class), quiet = TRUE)
  roc_objects[[i]] <- roc_obj
  
  # Get plot title
  plot_title <- ifelse(class %in% names(grade_labels), 
                       grade_labels[class], 
                       paste0("Class ", class))
  
  # Create plots
  roc_plots[[i]] <- create_roc_plot(roc_obj, plot_title, config$roc_color)
  roc_plots_standalone[[i]] <- create_roc_plot_standalone(
    roc_obj, plot_title, config$model_name, config$roc_color
  )
  
  message("  ROC curve for ", class, " - AUC: ", round(as.numeric(roc_obj$auc), 3))
}

# Fill with empty plots if fewer than 5 classes
while (length(roc_plots) < 5) {
  empty_plot <- ggplot() + 
    theme_void() + 
    theme(plot.background = element_rect(fill = "white", color = NA))
  roc_plots[[length(roc_plots) + 1]] <- empty_plot
  roc_plots_standalone[[length(roc_plots_standalone) + 1]] <- empty_plot
}

# ================================================================
# GENERATE CONFUSION MATRIX
# ================================================================

message("\n===== Generating Confusion Matrix =====\n")

cm_plot <- create_confusion_matrix_plot(truth, pred_class, 
                                         config$cm_low_color, config$cm_high_color)

cm_plot_standalone <- create_confusion_matrix_standalone(
  truth, pred_class, config$model_name, 
  config$cm_low_color, config$cm_high_color
)

# ================================================================
# CREATE COMBINED FIGURE
# ================================================================

message("\n===== Creating Combined Figure =====\n")

combined_plot <- (roc_plots[[1]] | roc_plots[[2]]) /
  (roc_plots[[3]] | roc_plots[[4]]) /
  (roc_plots[[5]] | cm_plot) +
  plot_annotation(
    title = paste(config$model_name, "Classification Performance"),
    subtitle = "ROC Curves and Confusion Matrix",
    tag_levels = 'a',
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray30")
    )
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

# ================================================================
# SAVE FIGURES
# ================================================================

message("\n===== Saving Figures =====\n")

# Save combined figure
ggsave(file.path(config$output_dir, "ROC_Confusion_Matrix_Combined.pdf"), 
       combined_plot, width = 10, height = 12, dpi = config$figure_dpi)
ggsave(file.path(config$output_dir, "ROC_Confusion_Matrix_Combined.png"), 
       combined_plot, width = 10, height = 12, dpi = config$figure_dpi)
ggsave(file.path(config$output_dir, "ROC_Confusion_Matrix_Combined.tiff"), 
       combined_plot, width = 10, height = 12, dpi = config$figure_dpi, 
       compression = "lzw")

message("Combined figure saved")

# Save individual ROC curves
message("Saving individual ROC curves...")
for (i in seq_len(min(5, n_classes))) {
  file_base <- file.path(config$subplots_dir, paste0("ROC_Class_", class_names[i]))
  
  ggsave(paste0(file_base, ".pdf"), roc_plots_standalone[[i]], 
         width = 5, height = 5, dpi = config$figure_dpi)
  ggsave(paste0(file_base, ".png"), roc_plots_standalone[[i]], 
         width = 5, height = 5, dpi = config$figure_dpi)
}

# Save confusion matrix
ggsave(file.path(config$subplots_dir, "Confusion_Matrix.pdf"), 
       cm_plot_standalone, width = 6, height = 5, dpi = config$figure_dpi)
ggsave(file.path(config$subplots_dir, "Confusion_Matrix.png"), 
       cm_plot_standalone, width = 6, height = 5, dpi = config$figure_dpi)

# ================================================================
# EXPORT AUC SUMMARY
# ================================================================

message("\n===== Exporting AUC Summary =====\n")

auc_summary <- data.frame()

for (i in seq_len(min(5, n_classes))) {
  class <- class_names[i]
  roc_obj <- roc_objects[[i]]
  ci_val <- ci(roc_obj)
  
  auc_summary <- rbind(auc_summary, data.frame(
    Class = class,
    AUC = round(as.numeric(roc_obj$auc), 3),
    CI_Lower = round(ci_val[1], 3),
    CI_Upper = round(ci_val[3], 3)
  ))
}

write.csv(auc_summary, file.path(config$output_dir, "AUC_Summary.csv"), 
          row.names = FALSE)

print(auc_summary)

# Display combined plot
print(combined_plot)

message("\n===== Visualization Complete =====\n")
message("Output directory: ", config$output_dir)
message("Subplots directory: ", config$subplots_dir)
