# ================================================================
# Multi-class Machine Learning Models with SHAP Explanations
# ================================================================
# Author: [Your Name]
# Description: Train 10 machine learning models for multi-class 
#              classification with SHAP-based feature importance
# License: MIT
# ================================================================

# Clear environment
rm(list = ls())

# ================================================================
# CONFIGURATION - Modify these parameters for your project
# ================================================================

config <- list(
  # Data settings
  data_file = "../data/extracted_data.xlsx",
  response_var = "quality",
  class_levels = c("I", "II", "III", "IV", "V"),
  
  # Train/test split
  train_ratio = 0.8,
  random_seed = 4321,
  
  # Feature selection
  bonferroni_alpha = 0.05,
  correlation_threshold = 0.9,
  
  # SHAP settings
  shap_background_samples = 20,
  
  # Output directories
  output_dir = "../output",
  models_dir = "../output/models",
  figures_dir = "../output/figures"
)

# ================================================================
# LOAD REQUIRED PACKAGES
# ================================================================

required_packages <- c(
  "caret",        # Machine learning framework
  "pROC",         # ROC curve analysis
  "ggplot2",      # Visualization
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "rpart.plot",   # Decision tree visualization
  "yardstick",    # Multi-class metrics
  "glmnet",       # Lasso/Ridge regression
  "kernlab",      # SVM
  "nnet",         # Neural network
  "xgboost",      # XGBoost
  "patchwork",    # Combine plots
  "shapviz",      # SHAP visualization
  "kernelshap",   # SHAP values calculation
  "readxl",       # Read Excel files
  "boot"          # Bootstrap confidence intervals
)

# Install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages) > 0) {
    message("Installing missing packages: ", paste(new_packages, collapse = ", "))
    install.packages(new_packages, dependencies = TRUE)
  }
}

install_if_missing(required_packages)

# Load packages
suppressPackageStartupMessages({
  lapply(required_packages, library, character.only = TRUE)
})

# ================================================================
# CREATE OUTPUT DIRECTORIES
# ================================================================

create_dirs <- function(dirs) {
  for (dir in dirs) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
      message("Created directory: ", dir)
    }
  }
}

create_dirs(c(config$output_dir, config$models_dir, config$figures_dir))

# ================================================================
# DATA LOADING AND PREPROCESSING
# ================================================================

message("\n===== Loading Data =====\n")

# Check if data file exists
if (!file.exists(config$data_file)) {
  stop("Data file not found: ", config$data_file, 
       "\nPlease place your data file in the 'data/' directory.")
}

# Load data
data <- read_excel(config$data_file)
message("Data loaded: ", nrow(data), " samples, ", ncol(data), " features")

# Check response variable exists
if (!config$response_var %in% names(data)) {
  stop("Response variable '", config$response_var, "' not found in data.")
}

# Convert response variable to factor
if (is.numeric(data[[config$response_var]])) {
  data[[config$response_var]] <- as.integer(data[[config$response_var]])
  data[[config$response_var]] <- as.character(as.roman(data[[config$response_var]]))
}
data[[config$response_var]] <- factor(data[[config$response_var]], 
                                       levels = config$class_levels)

message("Class distribution:")
print(table(data[[config$response_var]]))

# ================================================================
# TRAIN/TEST SPLIT
# ================================================================

message("\n===== Splitting Data =====\n")

set.seed(config$random_seed)
train_index <- createDataPartition(data[[config$response_var]], 
                                   p = config$train_ratio, 
                                   list = FALSE)
traindata <- data[train_index, ]
testdata <- data[-train_index, ]

# Ensure factor levels are consistent
traindata[[config$response_var]] <- factor(traindata[[config$response_var]], 
                                            levels = config$class_levels)
testdata[[config$response_var]] <- factor(testdata[[config$response_var]], 
                                           levels = config$class_levels)

message("Training set: ", nrow(traindata), " samples")
message("Test set: ", nrow(testdata), " samples")

# ================================================================
# FEATURE SELECTION: Kruskal-Wallis Test with Bonferroni Correction
# ================================================================

message("\n===== Feature Selection: Kruskal-Wallis Test =====\n")

feature_cols <- setdiff(names(traindata), config$response_var)
X_train <- traindata[, feature_cols]
y_train <- traindata[[config$response_var]]

message("Number of features to evaluate: ", length(feature_cols))

# Perform Kruskal-Wallis test for each feature
kw_results <- data.frame(
  Feature = feature_cols,
  Statistic = NA_real_,
  P_value = NA_real_,
  stringsAsFactors = FALSE
)

for (i in seq_along(feature_cols)) {
  feat <- feature_cols[i]
  tryCatch({
    kw_test <- kruskal.test(X_train[[feat]] ~ y_train)
    kw_results$Statistic[i] <- kw_test$statistic
    kw_results$P_value[i] <- kw_test$p.value
  }, error = function(e) {
    kw_results$Statistic[i] <- NA
    kw_results$P_value[i] <- 1
  })
}

# Apply Bonferroni correction
kw_results$P_adjusted <- p.adjust(kw_results$P_value, method = "bonferroni")
kw_results$Significant <- ifelse(kw_results$P_adjusted < config$bonferroni_alpha, 
                                  "Yes", "No")
kw_results <- kw_results[order(kw_results$P_adjusted), ]

# Export results
write.csv(kw_results, 
          file.path(config$output_dir, "Kruskal_Wallis_results.csv"), 
          row.names = FALSE)

# Select significant features
significant_features <- kw_results$Feature[kw_results$P_adjusted < config$bonferroni_alpha]
message("Original features: ", length(feature_cols))
message("Significant features (Bonferroni p < ", config$bonferroni_alpha, "): ", 
        length(significant_features))

# Update datasets
traindata <- traindata[, c(significant_features, config$response_var)]
testdata <- testdata[, c(significant_features, config$response_var)]

# ================================================================
# FEATURE SELECTION: Remove Near-Zero Variance Features
# ================================================================

message("\n===== Removing Near-Zero Variance Features =====\n")

feature_cols <- setdiff(names(traindata), config$response_var)
X_train <- traindata[, feature_cols]
X_test <- testdata[, feature_cols]

nzv_result <- nearZeroVar(X_train, saveMetrics = TRUE)
nzv_idx <- which(nzv_result$nzv == TRUE)

if (length(nzv_idx) > 0) {
  removed_features <- names(X_train)[nzv_idx]
  X_train <- X_train[, -nzv_idx]
  X_test <- X_test[, names(X_train)]
  message("Removed ", length(nzv_idx), " near-zero variance features")
  message("Remaining features: ", ncol(X_train))
} else {
  message("No near-zero variance features found")
}

# ================================================================
# FEATURE SELECTION: Correlation-based Redundancy Removal
# ================================================================

message("\n===== Removing Highly Correlated Features =====\n")

# Calculate feature importance (Kruskal-Wallis H statistic)
feature_importance <- sapply(names(X_train), function(feat) {
  tryCatch({
    kruskal.test(X_train[[feat]] ~ y_train)$statistic
  }, error = function(e) 0)
})

# Calculate Spearman correlation matrix
cor_matrix <- cor(X_train, method = "spearman", use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0
cor_matrix_abs <- abs(cor_matrix)
diag(cor_matrix_abs) <- 0

# Remove redundant features (keep more important one in correlated pairs)
feature_names <- names(X_train)
removed_features <- c()
importance_order <- order(feature_importance)

for (idx in importance_order) {
  feat_name <- feature_names[idx]
  if (feat_name %in% removed_features) next
  
  high_cor_idx <- which(cor_matrix_abs[idx, ] > config$correlation_threshold)
  if (length(high_cor_idx) > 0) {
    high_cor_names <- feature_names[high_cor_idx]
    high_cor_names <- high_cor_names[!high_cor_names %in% removed_features]
    
    for (cor_feat in high_cor_names) {
      if (feature_importance[feat_name] <= feature_importance[cor_feat] && 
          !(feat_name %in% removed_features)) {
        removed_features <- c(removed_features, feat_name)
        break
      }
    }
  }
}

removed_features <- unique(removed_features)
if (length(removed_features) > 0) {
  X_train <- X_train[, !names(X_train) %in% removed_features]
  X_test <- X_test[, names(X_train)]
  message("Removed ", length(removed_features), " highly correlated features")
}

message("Final number of features: ", ncol(X_train))

# Update datasets
traindata <- cbind(X_train, traindata[config$response_var])
testdata <- cbind(X_test, testdata[config$response_var])

# ================================================================
# MODEL TRAINING
# ================================================================

message("\n===== Training Machine Learning Models =====\n")

# Define cross-validation settings
cv_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

# Define algorithms
algorithms <- list(
  list(name = "DecisionTree", method = "rpart"),
  list(name = "RandomForest", method = "rf"),
  list(name = "XGBoost", method = "xgbTree"),
  list(name = "Lasso", method = "glmnet", tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, 0, length = 20))),
  list(name = "Ridge", method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4, 0, length = 20))),
  list(name = "NeuralNetwork", method = "nnet"),
  list(name = "MultinomialLogistic", method = "multinom"),
  list(name = "KNN", method = "knn"),
  list(name = "NaiveBayes", method = "naive_bayes")
)

# Train models
all_models <- list()
train_results <- data.frame()
test_results <- data.frame()

for (algo in algorithms) {
  message("Training ", algo$name, "...")
  
  tryCatch({
    set.seed(config$random_seed)
    
    train_args <- list(
      form = as.formula(paste(config$response_var, "~ .")),
      data = traindata,
      method = algo$method,
      trControl = cv_control,
      metric = "AUC"
    )
    
    if (!is.null(algo$tuneGrid)) {
      train_args$tuneGrid <- algo$tuneGrid
    }
    
    model <- do.call(train, train_args)
    all_models[[algo$name]] <- model
    
    # Save model
    saveRDS(model, file.path(config$models_dir, paste0(algo$name, "_model.rds")))
    
    # Evaluate on training set
    train_pred <- predict(model, traindata, type = "prob")
    
    # Evaluate on test set
    test_pred <- predict(model, testdata, type = "prob")
    test_class <- predict(model, testdata)
    
    message("  ", algo$name, " training completed")
    
  }, error = function(e) {
    message("  Error training ", algo$name, ": ", e$message)
  })
}

message("\n", length(all_models), " models trained successfully")

# ================================================================
# MODEL EVALUATION
# ================================================================

message("\n===== Evaluating Models =====\n")

evaluate_model <- function(model, newdata, response_var, class_levels) {
  pred_prob <- predict(model, newdata, type = "prob")
  pred_class <- predict(model, newdata)
  truth <- newdata[[response_var]]
  
  # Calculate multi-class AUC
  roc_list <- list()
  for (class in class_levels) {
    binary_truth <- factor(ifelse(truth == class, class, "Other"), 
                           levels = c(class, "Other"))
    if (class %in% colnames(pred_prob)) {
      roc_obj <- pROC::roc(binary_truth, pred_prob[, class], 
                           levels = c("Other", class), quiet = TRUE)
      roc_list[[class]] <- as.numeric(roc_obj$auc)
    }
  }
  
  avg_auc <- mean(unlist(roc_list), na.rm = TRUE)
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, truth)
  
  return(list(
    AUC = avg_auc,
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Sensitivity = mean(cm$byClass[, "Sensitivity"], na.rm = TRUE),
    Specificity = mean(cm$byClass[, "Specificity"], na.rm = TRUE),
    Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
    F1 = mean(cm$byClass[, "F1"], na.rm = TRUE)
  ))
}

# Evaluate all models
performance_df <- data.frame()

for (model_name in names(all_models)) {
  model <- all_models[[model_name]]
  
  train_metrics <- evaluate_model(model, traindata, config$response_var, config$class_levels)
  test_metrics <- evaluate_model(model, testdata, config$response_var, config$class_levels)
  
  performance_df <- rbind(performance_df, data.frame(
    Model = model_name,
    Dataset = "Train",
    AUC = train_metrics$AUC,
    Accuracy = train_metrics$Accuracy,
    Sensitivity = train_metrics$Sensitivity,
    Specificity = train_metrics$Specificity,
    Precision = train_metrics$Precision,
    F1 = train_metrics$F1
  ))
  
  performance_df <- rbind(performance_df, data.frame(
    Model = model_name,
    Dataset = "Test",
    AUC = test_metrics$AUC,
    Accuracy = test_metrics$Accuracy,
    Sensitivity = test_metrics$Sensitivity,
    Specificity = test_metrics$Specificity,
    Precision = test_metrics$Precision,
    F1 = test_metrics$F1
  ))
}

# Save performance results
write.csv(performance_df, 
          file.path(config$output_dir, "Model_Performance.csv"), 
          row.names = FALSE)

# Find best model
test_performance <- performance_df[performance_df$Dataset == "Test", ]
best_model_name <- test_performance$Model[which.max(test_performance$AUC)]
best_model <- all_models[[best_model_name]]

message("\nBest model: ", best_model_name, " (AUC: ", 
        round(max(test_performance$AUC), 3), ")")

# ================================================================
# SHAP ANALYSIS
# ================================================================

message("\n===== Computing SHAP Values =====\n")

X <- traindata[, -which(names(traindata) == config$response_var)]

# Define prediction wrapper
pred_wrapper <- function(model, newdata) {
  probs <- predict(model, newdata = newdata, type = "prob")
  return(as.matrix(probs))
}

# Compute SHAP values
message("Computing SHAP values for ", best_model_name, "...")
message("Using ", config$shap_background_samples, " background samples")

shap_values <- kernelshap::kernelshap(
  best_model,
  X = X,
  bg_X = X[1:min(config$shap_background_samples, nrow(X)), ],
  pred_fun = function(m, x) pred_wrapper(m, x)
)

sv <- shapviz(shap_values, X = X)

# Generate SHAP plots
message("Generating SHAP plots...")

# Feature importance plot
p_importance <- sv_importance(sv) +
  labs(title = paste(best_model_name, "- SHAP Feature Importance")) +
  theme_minimal()
ggsave(file.path(config$figures_dir, "SHAP_Feature_Importance.pdf"), 
       p_importance, width = 10, height = 8)

# Beeswarm plot
p_beeswarm <- sv_importance(sv, kind = "beeswarm", show_numbers = TRUE) +
  labs(title = paste(best_model_name, "- SHAP Value Distribution")) +
  theme_minimal()
ggsave(file.path(config$figures_dir, "SHAP_Beeswarm.pdf"), 
       p_beeswarm, width = 12, height = 8)

# Waterfall plot for first sample
p_waterfall <- sv_waterfall(sv, row_id = 1)
ggsave(file.path(config$figures_dir, "SHAP_Waterfall.pdf"), 
       p_waterfall, width = 10, height = 6)

# Force plot
p_force <- sv_force(sv, row_id = 1)
ggsave(file.path(config$figures_dir, "SHAP_Force.pdf"), 
       p_force, width = 12, height = 7)

message("\n===== Analysis Complete =====\n")
message("Results saved to: ", config$output_dir)
message("Models saved to: ", config$models_dir)
message("Figures saved to: ", config$figures_dir)
