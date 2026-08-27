# Figure 3 — ten-model test performance (patient-level, 517 features)
# Reads official point estimates / CIs from 02_primary/.
# F1 and precision are not in that table; they are computed from
# ten_models_test_predictions.csv with the same unstratified percentile
# bootstrap (n = 1000, seed = 4321) used for the official CIs.
# CatBoost is drawn first because it is the pre-specified primary model.
# Random Forest (macro AUC 0.941) is drawn at its true height.

rm(list = ls())

root <- "revision"
perf_path <- file.path(root, "results/02_primary/ten_models_test_performance.csv")
pred_path <- file.path(root, "results/02_primary/ten_models_test_predictions.csv")
out_csv <- file.path(root, "results/11_figures/fig3_metrics_long.csv")
out_dir <- file.path(root, "figures")

dir.create(file.path(root, "results/11_figures"), showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

SEED <- 4321L
N_BOOT <- 1000L
CLASSES <- 1:5

display_name <- c(
  CatBoost = "CatBoost",
  RandomForest = "Random Forest",
  NaiveBayes = "Naive Bayes",
  XGBoost = "XGBoost",
  NeuralNetwork = "Neural Network",
  Ridge = "Ridge",
  KNN = "KNN",
  MultinomialLogistic = "Logistic Regression",
  Lasso = "Lasso",
  DecisionTree = "Decision Tree"
)

# Primary first, then remaining models by official macro AUC (descending).
model_levels <- c(
  "CatBoost",
  "Random Forest",
  "Naive Bayes",
  "XGBoost",
  "Neural Network",
  "Ridge",
  "KNN",
  "Logistic Regression",
  "Lasso",
  "Decision Tree"
)

color_mapping <- c(
  "CatBoost" = "#3182bd",
  "Naive Bayes" = "#4DBBD5",
  "Ridge" = "#00A087",
  "Random Forest" = "#E64B35",
  "Lasso" = "#F39B7F",
  "XGBoost" = "#8491B4",
  "KNN" = "#DC91A7",
  "Logistic Regression" = "#91D1C2",
  "Neural Network" = "#7E6148",
  "Decision Tree" = "#B09C85"
)

metric_levels <- c(
  "(a) AUC",
  "(b) Accuracy",
  "(c) F1",
  "(d) Sensitivity",
  "(e) Specificity",
  "(f) Precision"
)

perf <- read.csv(perf_path, stringsAsFactors = FALSE)
pred <- read.csv(pred_path, stringsAsFactors = FALSE)
stopifnot(all(c("Model", "macro_AUC", "accuracy") %in% names(perf)))
stopifnot(all(c("Model", "y_true", "y_pred") %in% names(pred)))

official <- bind_rows(
  transmute(perf,
            Model = unname(display_name[Model]),
            Metric = "(a) AUC",
            Mean = macro_AUC,
            Lower = macro_AUC_ci_low,
            Upper = macro_AUC_ci_high,
            Source = "ten_models_test_performance.csv"),
  transmute(perf,
            Model = unname(display_name[Model]),
            Metric = "(b) Accuracy",
            Mean = accuracy,
            Lower = accuracy_ci_low,
            Upper = accuracy_ci_high,
            Source = "ten_models_test_performance.csv"),
  transmute(perf,
            Model = unname(display_name[Model]),
            Metric = "(d) Sensitivity",
            Mean = macro_sensitivity,
            Lower = macro_sensitivity_ci_low,
            Upper = macro_sensitivity_ci_high,
            Source = "ten_models_test_performance.csv"),
  transmute(perf,
            Model = unname(display_name[Model]),
            Metric = "(e) Specificity",
            Mean = macro_specificity,
            Lower = macro_specificity_ci_low,
            Upper = macro_specificity_ci_high,
            Source = "ten_models_test_performance.csv")
)

macro_prf <- function(y_true, y_pred) {
  precs <- f1s <- numeric(length(CLASSES))
  for (i in seq_along(CLASSES)) {
    cls <- CLASSES[i]
    tp <- sum(y_true == cls & y_pred == cls)
    fp <- sum(y_true != cls & y_pred == cls)
    fn <- sum(y_true == cls & y_pred != cls)
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
    precs[i] <- prec
    f1s[i] <- f1
  }
  c(precision = mean(precs), f1 = mean(f1s))
}

set.seed(SEED)
models_raw <- unique(pred$Model)
prf_rows <- vector("list", length(models_raw))

for (i in seq_along(models_raw)) {
  m <- models_raw[i]
  d <- pred[pred$Model == m, , drop = FALSE]
  y <- d$y_true
  p <- d$y_pred
  n <- length(y)
  pt <- macro_prf(y, p)
  boot_prec <- numeric(N_BOOT)
  boot_f1 <- numeric(N_BOOT)
  for (b in seq_len(N_BOOT)) {
    idx <- sample.int(n, n, replace = TRUE)
    br <- macro_prf(y[idx], p[idx])
    boot_prec[b] <- br[["precision"]]
    boot_f1[b] <- br[["f1"]]
  }
  prf_rows[[i]] <- bind_rows(
    data.frame(
      Model = unname(display_name[m]),
      Metric = "(c) F1",
      Mean = unname(pt[["f1"]]),
      Lower = unname(quantile(boot_f1, 0.025, names = FALSE, type = 7)),
      Upper = unname(quantile(boot_f1, 0.975, names = FALSE, type = 7)),
      Source = "ten_models_test_predictions.csv",
      stringsAsFactors = FALSE
    ),
    data.frame(
      Model = unname(display_name[m]),
      Metric = "(f) Precision",
      Mean = unname(pt[["precision"]]),
      Lower = unname(quantile(boot_prec, 0.025, names = FALSE, type = 7)),
      Upper = unname(quantile(boot_prec, 0.975, names = FALSE, type = 7)),
      Source = "ten_models_test_predictions.csv",
      stringsAsFactors = FALSE
    )
  )
}

plot_data <- bind_rows(official, bind_rows(prf_rows)) %>%
  mutate(
    Model = factor(Model, levels = model_levels),
    Metric = factor(Metric, levels = metric_levels),
    label_y = pmin(pmax(Mean, Upper) + 0.035, 0.975)
  )

stopifnot(nrow(plot_data) == 60L)
stopifnot(!anyNA(plot_data$Model))
write.csv(plot_data[, c("Model", "Metric", "Mean", "Lower", "Upper", "Source")],
          out_csv, row.names = FALSE)

p <- ggplot(plot_data, aes(x = Model, y = Mean, fill = Model)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.3) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper),
                width = 0.25, linewidth = 0.3, color = "black") +
  geom_text(aes(y = label_y, label = sprintf("%.3f", Mean)),
            size = 2.8, color = "black") +
  scale_fill_manual(values = color_mapping, name = "Model", drop = FALSE) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0, 0.02))
  ) +
  facet_wrap(~ Metric, ncol = 3, scales = "fixed") +
  labs(
    title = "Model Performance in Lumbar Intervertebral Disc Degeneration Grades",
    x = NULL,
    y = "Performance Value"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
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

ggsave(file.path(out_dir, "Figure3_Model_Performance_by_Metric.pdf"),
       p, width = 14, height = 10)
ggsave(file.path(out_dir, "Figure3_Model_Performance_by_Metric.png"),
       p, width = 14, height = 10, dpi = 600)
ggsave(file.path(out_dir, "Figure3_Model_Performance_by_Metric.tiff"),
       p, width = 14, height = 10, dpi = 600, compression = "lzw")

cat_auc <- plot_data$Mean[plot_data$Model == "CatBoost" & plot_data$Metric == "(a) AUC"]
rf_auc <- plot_data$Mean[plot_data$Model == "Random Forest" & plot_data$Metric == "(a) AUC"]
cat(sprintf("Wrote %s\n", out_csv))
cat(sprintf("CatBoost AUC = %.3f; Random Forest AUC = %.3f\n", cat_auc, rf_auc))
cat("Figure 3 written to revision/figures/Figure3_Model_Performance_by_Metric.{pdf,png,tiff}\n")
