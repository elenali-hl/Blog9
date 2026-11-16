
# ============================================================
# BLOG POST 9: STOCK RETURN PREDICTION
# 02 - ORIGINAL XGBOOST MODEL
# ============================================================

# 1. LIBRARIES -----------------------------------------------

packages <- c(
  "tidyverse",
  "tidymodels",
  "xgboost"
)

missing_packages <- packages[
  !vapply(packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(tidyverse)
library(tidymodels)
library(xgboost)

set.seed(123)

# 2. LOAD PREPARED DATA --------------------------------------

# Run from the Blog9 project root.

train <- read_csv(
  "data/train.csv",
  show_col_types = FALSE
)

validation <- read_csv(
  "data/validation.csv",
  show_col_types = FALSE
)

test <- read_csv(
  "data/test.csv",
  show_col_types = FALSE
)

# Ensure dates have the correct type
train <- train %>%
  mutate(
    date = as.Date(date),
    target_end_date = as.Date(target_end_date)
  )

validation <- validation %>%
  mutate(
    date = as.Date(date),
    target_end_date = as.Date(target_end_date)
  )

test <- test %>%
  mutate(
    date = as.Date(date),
    target_end_date = as.Date(target_end_date)
  )

cat("Training observations:", nrow(train), "\n")
cat("Validation observations:", nrow(validation), "\n")
cat("Test observations:", nrow(test), "\n")

# 3. DEFINE ORIGINAL PREDICTORS ------------------------------

# Same six predictors as the classroom experiment.

predictors <- c(
  "mom_1m",
  "mom_3m",
  "mom_6m",
  "vol_1m",
  "ret_5d",
  "excess_ret"
)

stopifnot(
  all(c(predictors, "target") %in% names(train)),
  all(c(predictors, "target") %in% names(validation)),
  all(c(predictors, "target") %in% names(test))
)

# 4. PREPROCESSING -------------------------------------------

# Date, symbol, and target_end_date are identifiers.
# They must not enter the model as predictors.

rec <- recipe(target ~ ., data = train) %>%
  update_role(
    date,
    symbol,
    target_end_date,
    new_role = "id"
  )

# 5. ORIGINAL XGBOOST SPECIFICATION --------------------------

# Preserve the professor's original hyperparameters.

original_spec <- boost_tree(
  trees = 300,
  tree_depth = 5,
  learn_rate = 0.05,
  loss_reduction = 0,
  min_n = 10
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# 6. CREATE WORKFLOW -----------------------------------------

original_workflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(original_spec)

# 7. TRAIN ORIGINAL MODEL ------------------------------------

cat("\nTraining original XGBoost model...\n")

original_fit <- fit(
  original_workflow,
  data = train
)

cat("Original model training completed!\n")

# 8. GENERATE PREDICTIONS ------------------------------------

validation_preds <- predict(
  original_fit,
  new_data = validation
) %>%
  bind_cols(validation) %>%
  rename(predicted_return = .pred) %>%
  mutate(model = "Original XGBoost")

test_preds <- predict(
  original_fit,
  new_data = test
) %>%
  bind_cols(test) %>%
  rename(predicted_return = .pred) %>%
  mutate(model = "Original XGBoost")

# 9. EVALUATION FUNCTION ------------------------------------

evaluate_model <- function(data, dataset_name) {
  
  rmse_value <- sqrt(
    mean(
      (data$target - data$predicted_return)^2
    )
  )
  
  mae_value <- mean(
    abs(data$target - data$predicted_return)
  )
  
  r2_value <- 1 -
    sum(
      (data$target - data$predicted_return)^2
    ) /
    sum(
      (data$target - mean(data$target))^2
    )
  
  # Cross-sectional rank correlation for each date.
  # Only dates with sufficient variation are used.
  
  daily_ic <- data %>%
    group_by(date) %>%
    summarise(
      rank_ic = if (
        n() >= 3 &&
        sd(predicted_return) > 0 &&
        sd(target) > 0
      ) {
        cor(
          predicted_return,
          target,
          method = "spearman"
        )
      } else {
        NA_real_
      },
      .groups = "drop"
    )
  
  tibble(
    model = "Original XGBoost",
    dataset = dataset_name,
    RMSE = rmse_value,
    MAE = mae_value,
    R2 = r2_value,
    Rank_IC = mean(daily_ic$rank_ic, na.rm = TRUE)
  )
}

# 10. VALIDATION AND TEST METRICS ----------------------------

validation_metrics <- evaluate_model(
  validation_preds,
  "Validation"
)

test_metrics <- evaluate_model(
  test_preds,
  "Test"
)

original_metrics <- bind_rows(
  validation_metrics,
  test_metrics
)

cat("\n====================================\n")
cat("ORIGINAL XGBOOST MODEL RESULTS\n")
cat("====================================\n")

print(original_metrics)

# 11. SAVE MODEL AND PREDICTIONS -----------------------------

dir.create(
  "results/models",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "results/predictions",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "results/tables",
  recursive = TRUE,
  showWarnings = FALSE
)

saveRDS(
  original_fit,
  "results/models/original_xgboost.rds"
)

write_csv(
  validation_preds,
  "results/predictions/original_validation.csv"
)

write_csv(
  test_preds,
  "results/predictions/original_test.csv"
)

write_csv(
  original_metrics,
  "results/tables/original_model_metrics.csv"
)

# 12. MODEL FEATURE IMPORTANCE -------------------------------

# Extract the fitted XGBoost engine.

xgb_engine <- extract_fit_engine(original_fit)

importance <- xgboost::xgb.importance(
  model = xgb_engine
)

print(importance)

write_csv(
  as_tibble(importance),
  "results/tables/original_feature_importance.csv"
)

cat("\nOriginal XGBoost analysis completed!\n")
