#!/usr/bin/env Rscript
# ============================================================================
# CatBoost SHAP Analysis for Lumbar Intervertebral Disc Degeneration
# ============================================================================
#
# Description:
#   This script performs SHAP (SHapley Additive exPlanations) analysis on a
#   trained CatBoost model for multi-class Pfirrmann grading of lumbar disc
#   degeneration. It generates comprehensive visualizations including feature
#   importance plots, beeswarm plots, waterfall plots, and combined figures.
#
# Author: [GUO Dongqi]
# Date: 2025-10-23
# License: MIT
#
# Requirements:
#   R >= 4.0.0
#   Packages: catboost, shapviz, ggplot2, dplyr, tidyr, readxl, caret, 
#             patchwork, viridis
#
# Usage:
#   1. Set the configuration parameters in the "USER CONFIGURATION" section
#   2. Run: Rscript catboost_shap_analysis.R
#   Or source in RStudio: source("catboost_shap_analysis.R")
#
# ============================================================================

# Clear environment
rm(list = ls())

# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Please modify the following parameters according to your data

config <- list(
  # ----- File Paths -----
  # Working directory (set to NULL to use current directory)
  working_dir = NULL,  # e.g., "path/to/your/project"
  
  # Model file path (CatBoost .cbm format)
  model_path = "../output/models/catboost_model.cbm",
  
  # Data file path (Excel format)
  data_path = "../data/extracted_data.xlsx",
  
  # Feature list file (optional, set to NULL if not available)
  feature_list_path = "../output/results/final_selected_features.csv",
  
  # ----- Data Configuration -----
  # Response variable name in the dataset
  response_var = "quality",
  
  # Class labels (Pfirrmann grades)
  class_labels = c("I", "II", "III", "IV", "V"),
  
  # Random seed for reproducibility
  seed = 4321,
  
  # Train-test split ratio
  train_ratio = 0.8,
  
  # ----- SHAP Analysis Configuration -----
  # Dataset for SHAP analysis: "test" (recommended) or "train"
  # "test"  - Calculate SHAP on test set (reflects generalization)
  # "train" - Calculate SHAP on training set (larger sample size)
  shap_dataset = "test",
  
  # ----- Visualization Configuration -----
  # Output directory
  output_dir = "../output/SHAP_Analysis_Results",
  
  # Plot resolution (DPI)
  plot_dpi = 300,
  
  # Color for importance plots
  importance_color = "#3182bd",
  
  # Maximum number of features to display
  max_features_display = 9,
  
  # Smart filtering: filter features with very low SHAP values
  smart_filter = FALSE,
  filter_threshold = 0.01,  # 1% of max value
  
  # ----- Output Formats -----
  save_pdf = TRUE,
  save_png = TRUE,
  save_tiff = TRUE,
  
  # ----- GPU Configuration -----
  use_gpu = FALSE
)

# ============================================================================
# 1. SETUP AND PACKAGE LOADING
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  CatBoost SHAP Analysis for Disc Degeneration Grading\n")
cat("============================================================\n\n")

# Set working directory if specified
if (!is.null(config$working_dir)) {
  setwd(config$working_dir)
  cat(sprintf("Working directory: %s\n", config$working_dir))
}

# Load required packages
cat("Loading required packages...\n")

required_packages <- c(
  "catboost",    # CatBoost model

"shapviz",     # SHAP visualization
  "ggplot2",     # Plotting
  "dplyr",       # Data manipulation
  "tidyr",       # Data tidying
  "readxl",      # Excel reading
  "caret",       # Data splitting
  "patchwork",   # Plot composition
  "viridis"      # Color palettes
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  cat(sprintf("  [OK] %s\n", pkg))
}

cat("\n")

# ============================================================================
# 2. LOAD MODEL AND DATA
# ============================================================================

cat("Loading model and data...\n")

# Check model file exists
if (!file.exists(config$model_path)) {
  stop(sprintf("Model file not found: %s", config$model_path))
}

# Load CatBoost model
catboost_model <- catboost.load_model(config$model_path)
cat(sprintf("  [OK] Model loaded from: %s\n", config$model_path))

# Load data
if (!file.exists(config$data_path)) {
  stop(sprintf("Data file not found: %s", config$data_path))
}

data <- read_excel(config$data_path)
cat(sprintf("  [OK] Data loaded: %d samples x %d features\n", nrow(data), ncol(data)))

# Convert response variable to factor with Roman numeral levels
data[[config$response_var]] <- as.integer(data[[config$response_var]])
data[[config$response_var]] <- as.character(as.roman(data[[config$response_var]]))
data[[config$response_var]] <- factor(data[[config$response_var]], 
                                       levels = config$class_labels)

cat("\nClass distribution:\n")
print(table(data[[config$response_var]]))

# ============================================================================
# 3. DATA SPLITTING
# ============================================================================

cat("\nSplitting data...\n")

set.seed(config$seed)
train_index <- createDataPartition(data[[config$response_var]], 
                                   p = config$train_ratio, 
                                   list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat(sprintf("  Training set: %d samples\n", nrow(train_data)))
cat(sprintf("  Test set: %d samples\n", nrow(test_data)))

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================

cat("\nPreparing features...\n")

# Load feature list if available
if (!is.null(config$feature_list_path) && file.exists(config$feature_list_path)) {
  final_features <- read.csv(config$feature_list_path)$Feature
  cat(sprintf("  [OK] Loaded %d selected features from file\n", length(final_features)))
  
  # Subset data to selected features
  train_data <- train_data[, c(final_features, config$response_var)]
  test_data <- test_data[, c(final_features, config$response_var)]
} else {
  cat("  [INFO] No feature list file found, using all features\n")
  final_features <- setdiff(names(train_data), config$response_var)
}

# Ensure factor levels are consistent
train_data[[config$response_var]] <- factor(train_data[[config$response_var]], 
                                             levels = config$class_labels)
test_data[[config$response_var]] <- factor(test_data[[config$response_var]], 
                                            levels = config$class_labels)

cat(sprintf("  Final feature count: %d\n", length(final_features)))

# ============================================================================
# 5. COMPUTE SHAP VALUES
# ============================================================================

cat("\nComputing SHAP values...\n")

# Prepare feature matrices
X_train <- train_data[, final_features]
X_test <- test_data[, final_features]
y_train <- train_data[[config$response_var]]
y_test <- test_data[[config$response_var]]

# Create CatBoost pools
train_pool <- catboost.load_pool(data = X_train)
test_pool <- catboost.load_pool(data = X_test)

# Select dataset for SHAP analysis
if (config$shap_dataset == "test") {
  X_shap <- X_test
  y_shap <- y_test
  shap_pool <- test_pool
  cat("  Using test set for SHAP analysis (recommended)\n")
} else {
  X_shap <- X_train
  y_shap <- y_train
  shap_pool <- train_pool
  cat("  Using training set for SHAP analysis\n")
}

cat(sprintf("  SHAP analysis samples: %d\n", nrow(X_shap)))

# Compute SHAP values
shap_values_matrix <- catboost.get_feature_importance(
  model = catboost_model,
  pool = shap_pool,
  type = "ShapValues"
)

n_classes <- nlevels(y_shap)
n_features <- ncol(X_shap)
n_samples <- nrow(X_shap)

cat(sprintf("  SHAP matrix dimensions: %d samples x %d classes x %d features\n", 
            n_samples, n_classes, n_features))

# ============================================================================
# 6. CREATE SHAPVIZ OBJECTS
# ============================================================================

cat("\nCreating shapviz objects for each class...\n")

sv_list <- list()

for (class_idx in 1:n_classes) {
  class_name <- levels(y_shap)[class_idx]
  
  # Initialize SHAP matrix for current class
  class_shap <- matrix(0, nrow = n_samples, ncol = n_features)
  
  # Extract SHAP values for each sample
  for (i in 1:n_samples) {
    sample_shap <- shap_values_matrix[i, , ]
    class_shap[i, ] <- sample_shap[class_idx, 1:n_features]
  }
  
  colnames(class_shap) <- colnames(X_shap)
  sv_list[[class_name]] <- shapviz(class_shap, X = as.data.frame(X_shap))
  
  cat(sprintf("  [OK] Class %s\n", class_name))
}

# ============================================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================================

if (!dir.exists(config$output_dir)) {
  dir.create(config$output_dir, recursive = TRUE)
}
cat(sprintf("\nOutput directory: %s\n", config$output_dir))

# ============================================================================
# 8. HELPER FUNCTIONS
# ============================================================================

# Function to save plots in multiple formats
save_plot <- function(plot, filename, width = 10, height = 8) {
  base_path <- file.path(config$output_dir, filename)
  
  if (config$save_pdf) {
    ggsave(paste0(base_path, ".pdf"), plot, width = width, height = height, 
           dpi = config$plot_dpi)
  }
  if (config$save_png) {
    ggsave(paste0(base_path, ".png"), plot, width = width, height = height, 
           dpi = config$plot_dpi)
  }
  if (config$save_tiff) {
    ggsave(paste0(base_path, ".tiff"), plot, width = width, height = height, 
           dpi = config$plot_dpi, compression = "lzw")
  }
}

# Function to create importance plot
create_importance_plot <- function(sv_obj = NULL, global_data = NULL, 
                                   title, is_global = FALSE) {
  
  if (is_global) {
    imp_df <- global_data[1:config$max_features_display, c("Feature", "Overall_Importance")]
    names(imp_df) <- c("Feature", "Importance")
  } else {
    mean_abs_shap <- colMeans(abs(sv_obj$S))
    imp_df <- data.frame(
      Feature = names(mean_abs_shap),
      Importance = as.numeric(mean_abs_shap)
    )
    imp_df <- imp_df[order(-imp_df$Importance), ]
    imp_df <- imp_df[1:min(config$max_features_display, nrow(imp_df)), ]
  }
  
  imp_df$Feature <- factor(imp_df$Feature, levels = rev(imp_df$Feature))
  
  ggplot(imp_df, aes(x = Feature, y = Importance)) +
    geom_bar(stat = "identity", fill = config$importance_color, 
             color = "black", linewidth = 0.2, width = 0.7) +
    coord_flip() +
    labs(title = title, x = NULL, y = "Mean |SHAP Value|") +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.text.y = element_text(size = 9),
      axis.text.x = element_text(size = 9),
      axis.title.x = element_text(size = 10, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(5, 8, 5, 5)
    )
}

# ============================================================================
# 9. GENERATE VISUALIZATIONS
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Generating SHAP Visualizations\n")
cat("============================================================\n\n")

# ----- 9.1 Feature Importance Plots for Each Class -----
cat("Generating class-specific importance plots...\n")

for (class_name in names(sv_list)) {
  sv_obj <- sv_list[[class_name]]
  
  # Calculate valid features
  mean_abs_shap <- colMeans(abs(sv_obj$S))
  max_importance <- max(mean_abs_shap)
  
  if (config$smart_filter) {
    valid_features <- names(mean_abs_shap)[mean_abs_shap >= max_importance * config$filter_threshold]
    n_display <- min(config$max_features_display, length(valid_features))
  } else {
    n_display <- config$max_features_display
  }
  
  # Importance plot
  p_imp <- sv_importance(sv_obj, kind = "bar", max_display = n_display) +
    labs(title = sprintf("SHAP Feature Importance - Grade %s", class_name),
         x = "Mean |SHAP Value|") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 10)
    )
  
  save_plot(p_imp, sprintf("importance_grade_%s", class_name))
  cat(sprintf("  [OK] Grade %s importance plot\n", class_name))
}

# ----- 9.2 Beeswarm Plots for Each Class -----
cat("\nGenerating beeswarm plots...\n")

for (class_name in names(sv_list)) {
  sv_obj <- sv_list[[class_name]]
  
  mean_abs_shap <- colMeans(abs(sv_obj$S))
  max_importance <- max(mean_abs_shap)
  
  if (config$smart_filter) {
    valid_features <- names(mean_abs_shap)[mean_abs_shap >= max_importance * config$filter_threshold]
    n_display <- min(config$max_features_display, length(valid_features))
  } else {
    n_display <- config$max_features_display
  }
  
  p_bee <- sv_importance(sv_obj, kind = "beeswarm", max_display = n_display) +
    labs(title = sprintf("SHAP Beeswarm Plot - Grade %s", class_name)) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 10)
    )
  
  save_plot(p_bee, sprintf("beeswarm_grade_%s", class_name))
  cat(sprintf("  [OK] Grade %s beeswarm plot\n", class_name))
}

# ----- 9.3 Global Feature Importance -----
cat("\nCalculating global feature importance...\n")

global_importance <- data.frame(Feature = colnames(X_shap))

for (class_name in names(sv_list)) {
  sv_obj <- sv_list[[class_name]]
  mean_abs_shap <- colMeans(abs(sv_obj$S))
  global_importance[[paste0("Class_", class_name)]] <- mean_abs_shap
}

global_importance$Overall_Importance <- rowMeans(global_importance[, -1])
global_importance <- global_importance[order(-global_importance$Overall_Importance), ]

# Save global importance data
write.csv(global_importance, 
          file.path(config$output_dir, "global_feature_importance.csv"), 
          row.names = FALSE)
cat("  [OK] Global importance saved to CSV\n")

# Global importance plot
top_n <- min(config$max_features_display, nrow(global_importance))
plot_data_global <- global_importance[1:top_n, ]
plot_data_global$Feature <- factor(plot_data_global$Feature, 
                                    levels = rev(plot_data_global$Feature))

p_global <- ggplot(plot_data_global, aes(x = Feature, y = Overall_Importance)) +
  geom_bar(stat = "identity", fill = config$importance_color, 
           color = "black", linewidth = 0.3) +
  coord_flip() +
  labs(
    title = "Global SHAP Feature Importance",
    subtitle = sprintf("Mean |SHAP Value| across all classes (Top %d)", top_n),
    x = NULL,
    y = "Mean |SHAP Value|"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 11, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

save_plot(p_global, "global_importance")
cat("  [OK] Global importance plot\n")

# ----- 9.4 Combined Figure (3x2 Layout) -----
cat("\nGenerating combined figure...\n")

plots <- list()
subplot_labels <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")

# Create subplots for each grade
for (i in seq_along(config$class_labels)) {
  class_name <- config$class_labels[i]
  sv_obj <- sv_list[[class_name]]
  
  plots[[i]] <- create_importance_plot(
    sv_obj = sv_obj,
    title = sprintf("%s Pfirrmann Grade %s", subplot_labels[i], class_name),
    is_global = FALSE
  )
}

# Add global plot
plots[[6]] <- create_importance_plot(
  global_data = global_importance,
  title = "(f) Global",
  is_global = TRUE
)

# Combine plots
combined_plot <- wrap_plots(plots, nrow = 2, ncol = 3) +
  plot_annotation(
    title = "SHAP Feature Importance Across Lumbar Intervertebral Disc Degeneration Grades",
    subtitle = "Top features ranked by mean |SHAP value| for each Pfirrmann grade",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40")
    )
  )

save_plot(combined_plot, "figure_combined_importance", width = 16, height = 10)
cat("  [OK] Combined figure (3x2 layout)\n")

# ----- 9.5 Combined Beeswarm Plot -----
cat("\nGenerating combined beeswarm plot...\n")

beeswarm_plots <- list()

for (i in seq_along(names(sv_list))) {
  class_name <- names(sv_list)[i]
  sv_obj <- sv_list[[class_name]]
  
  mean_abs_shap <- colMeans(abs(sv_obj$S))
  max_importance <- max(mean_abs_shap)
  valid_features <- names(mean_abs_shap)[mean_abs_shap >= max_importance * config$filter_threshold]
  n_valid <- min(config$max_features_display, length(valid_features))
  
  beeswarm_plots[[i]] <- sv_importance(sv_obj, kind = "beeswarm", max_display = n_valid) +
    labs(title = sprintf("Grade %s", class_name)) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
      axis.text.y = element_text(size = 7),
      axis.text.x = element_text(size = 8),
      legend.position = "none"
    )
}

combined_beeswarm <- wrap_plots(beeswarm_plots, ncol = min(3, length(beeswarm_plots))) +
  plot_annotation(
    title = "SHAP Beeswarm Plots by Pfirrmann Grade",
    subtitle = "Feature impact on each class prediction",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray40")
    )
  )

save_plot(combined_beeswarm, "figure_combined_beeswarm", width = 16, height = 12)
cat("  [OK] Combined beeswarm plot\n")

# ============================================================================
# 10. GENERATE DIAGNOSTIC REPORT
# ============================================================================

cat("\nGenerating diagnostic report...\n")

shap_diagnosis <- data.frame(
  Class = character(),
  N_Samples = integer(),
  Max_SHAP = numeric(),
  Mean_SHAP = numeric(),
  Valid_Features = integer(),
  Status = character(),
  stringsAsFactors = FALSE
)

for (class_name in names(sv_list)) {
  sv_obj <- sv_list[[class_name]]
  mean_abs_shap <- colMeans(abs(sv_obj$S))
  max_shap <- max(mean_abs_shap)
  
  valid_count <- sum(mean_abs_shap >= max_shap * config$filter_threshold)
  
  status <- if (max_shap < 0.001) {
    "Warning: Very low SHAP values"
  } else if (valid_count < 3) {
    "Warning: Few informative features"
  } else {
    "OK"
  }
  
  shap_diagnosis <- rbind(shap_diagnosis, data.frame(
    Class = class_name,
    N_Samples = nrow(sv_obj$X),
    Max_SHAP = max_shap,
    Mean_SHAP = mean(mean_abs_shap),
    Valid_Features = valid_count,
    Status = status
  ))
}

write.csv(shap_diagnosis, 
          file.path(config$output_dir, "shap_diagnostic_report.csv"), 
          row.names = FALSE)

# ============================================================================
# 11. SAVE DATA FOR FUTURE USE
# ============================================================================

cat("\nSaving analysis objects...\n")

saveRDS(sv_list, file.path(config$output_dir, "shapviz_objects.rds"))
saveRDS(shap_values_matrix, file.path(config$output_dir, "shap_values_matrix.rds"))
saveRDS(config, file.path(config$output_dir, "analysis_config.rds"))

cat("  [OK] All objects saved\n")

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Analysis Complete!\n")
cat("============================================================\n\n")

cat("Configuration Summary:\n")
cat(sprintf("  - SHAP dataset: %s (%d samples)\n", config$shap_dataset, n_samples))
cat(sprintf("  - Features: %d\n", n_features))
cat(sprintf("  - Classes: %d\n", n_classes))
cat(sprintf("  - Plot DPI: %d\n", config$plot_dpi))
cat(sprintf("  - Max features displayed: %d\n", config$max_features_display))

cat("\nDiagnostic Summary:\n")
for (i in 1:nrow(shap_diagnosis)) {
  cat(sprintf("  Grade %s: %s (Valid features: %d, Max SHAP: %.4f)\n",
              shap_diagnosis$Class[i],
              shap_diagnosis$Status[i],
              shap_diagnosis$Valid_Features[i],
              shap_diagnosis$Max_SHAP[i]))
}

cat("\nTop 10 Global Features:\n")
print(head(global_importance[, c("Feature", "Overall_Importance")], 10))

cat("\nOutput Files:\n")
cat(sprintf("  Directory: %s/\n", config$output_dir))
cat("  - importance_grade_*.pdf/png/tiff  : Class-specific importance plots\n")
cat("  - beeswarm_grade_*.pdf/png/tiff    : Class-specific beeswarm plots\n")
cat("  - global_importance.pdf/png/tiff   : Global feature importance\n")
cat("  - figure_combined_importance.*     : Combined 3x2 importance figure\n")
cat("  - figure_combined_beeswarm.*       : Combined beeswarm figure\n")
cat("  - global_feature_importance.csv    : Feature importance data\n")
cat("  - shap_diagnostic_report.csv       : Diagnostic report\n")
cat("  - shapviz_objects.rds              : SHAP visualization objects\n")
cat("  - shap_values_matrix.rds           : Raw SHAP values\n")
cat("  - analysis_config.rds              : Analysis configuration\n")

cat("\n============================================================\n")
cat("  For questions or issues, please open an issue on GitHub\n")
cat("============================================================\n")
