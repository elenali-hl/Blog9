
# ============================================================
# BLOG POST 9: STOCK RETURN PREDICTION
# 03 - IMPROVED XGBOOST MODEL
# ============================================================

# 1. LIBRARIES -----------------------------------------------

library(tidyverse)
library(tidymodels)
library(xgboost)
library(slider)

set.seed(123)

# 2. LOAD DATA -----------------------------------------------

train <- read_csv("data/train.csv",
                  show_col_types = FALSE)

validation <- read_csv("data/validation.csv",
                       show_col_types = FALSE)

prices <- read_csv("data-raw/stock_prices.csv",
                   show_col_types = FALSE)

spy <- read_csv("data/spy_benchmark.csv",
                show_col_types = FALSE)

# 3. ADD NEW FEATURES ----------------------------------------

# All new signals use historical information only.
# We preserve the original six variables.

extra_features <- prices %>%
  mutate(date = as.Date(date)) %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    daily_ret = adjusted / lag(adjusted) - 1,
    
    # Correctly oriented historical momentum
    momentum_1m_new =
      lag(adjusted, 1) / lag(adjusted, 22) - 1,
    
    momentum_3m_new =
      lag(adjusted, 1) / lag(adjusted, 64) - 1,
    
    # Short-term past return
    past_return_10d =
      lag(adjusted, 1) / lag(adjusted, 11) - 1,
    
    # Short-run volatility
    volatility_10d = slide_dbl(
      lag(daily_ret),
      sd,
      .before = 9,
      .complete = TRUE
    ),
    
    # Longer-run volatility
    volatility_3m = slide_dbl(
      lag(daily_ret),
      sd,
      .before = 62,
      .complete = TRUE
    )
  ) %>%
  ungroup() %>%
  select(
    date, symbol,
    momentum_1m_new,
    momentum_3m_new,
    past_return_10d,
    volatility_10d,
    volatility_3m
  )

# Add historical market momentum.

spy_features <- spy %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  mutate(
    market_momentum_1m =
      lag(spy_price, 1) / lag(spy_price, 22) - 1
  ) %>%
  select(date, market_momentum_1m)

# 4. MERGE FEATURES ------------------------------------------

add_features <- function(data) {
  data %>%
    mutate(
      date = as.Date(date),
      target_end_date = as.Date(target_end_date)
    ) %>%
    left_join(
      extra_features,
      by = c("date", "symbol")
    ) %>%
    left_join(
      spy_features,
      by = "date"
    ) %>%
    drop_na()
}

train_improved <- add_features(train)
validation_improved <- add_features(validation)

cat("Improved training rows:",
    nrow(train_improved), "\n")

cat("Improved validation rows:",
    nrow(validation_improved), "\n")

stopifnot(
  nrow(train_improved) > 0,
  nrow(validation_improved) > 0
)

# 5. DEFINE PREDICTORS ---------------------------------------

original_features <- c(
  "mom_1m",
  "mom_3m",
  "mom_6m",
  "vol_1m",
  "ret_5d",
  "excess_ret"
)

new_features <- c(
  "momentum_1m_new",
  "momentum_3m_new",
  "past_return_10d",
  "volatility_10d",
  "volatility_3m",
  "market_momentum_1m"
)

all_features <- c(
  original_features,
  new_features
)

cat("Number of predictors:",
    length(all_features), "\n")

# 6. MODEL RECIPE --------------------------------------------

# Explicitly select the predictors.
# Date and symbol cannot accidentally enter the model.

model_formula <- reformulate(
  all_features,
  response = "target"
)

rec_improved <- recipe(
  model_formula,
  data = train_improved
)

# 7. HYPERPARAMETER CANDIDATES -------------------------------

# A compact, transparent tuning experiment.

candidate_grid <- tibble(
  candidate = c(
    "Model_A",
    "Model_B",
    "Model_C",
    "Model_D"
  ),
  trees = c(200, 300, 500, 500),
  tree_depth = c(2, 3, 3, 5),
  learn_rate = c(0.03, 0.03, 0.05, 0.03),
  min_n = c(20, 20, 30, 30)
)

# 8. TRAIN AND VALIDATE CANDIDATES ----------------------------

results_list <- list()
model_list <- list()

for (i in seq_len(nrow(candidate_grid))) {
  
  config <- candidate_grid[i, ]
  
  cat("\nTraining:", config$candidate, "\n")
  
  spec <- boost_tree(
    trees = config$trees,
    tree_depth = config$tree_depth,
    learn_rate = config$learn_rate,
    min_n = config$min_n,
    loss_reduction = 0
  ) %>%
    set_engine("xgboost") %>%
    set_mode("regression")
  
  wf <- workflow() %>%
    add_recipe(rec_improved) %>%
    add_model(spec)
  
  fit_model <- fit(
    wf,
    data = train_improved
  )
  
  pred <- predict(
    fit_model,
    new_data = validation_improved
  )$.pred
  
  actual <- validation_improved$target
  
  rmse_value <- sqrt(
    mean((actual - pred)^2)
  )
  
  mae_value <- mean(
    abs(actual - pred)
  )
  
  results_list[[i]] <- config %>%
    mutate(
      Validation_RMSE = rmse_value,
      Validation_MAE = mae_value
    )
  
  model_list[[config$candidate]] <- fit_model
}

# 9. COMPARE CANDIDATES --------------------------------------

tuning_results <- bind_rows(
  results_list
) %>%
  arrange(Validation_RMSE)

cat("\n====================================\n")
cat("IMPROVED MODEL TUNING RESULTS\n")
cat("====================================\n")

print(tuning_results)

# 10. SELECT BEST MODEL --------------------------------------

best_candidate <- tuning_results$candidate[1]

best_model <- model_list[[best_candidate]]

cat("\nSelected model:",
    best_candidate, "\n")

# 11. SAVE VALIDATION PREDICTIONS ----------------------------

validation_preds <- predict(
  best_model,
  new_data = validation_improved
) %>%
  bind_cols(validation_improved) %>%
  rename(predicted_return = .pred) %>%
  mutate(model = "Improved XGBoost")

# 12. VALIDATION RANK IC -------------------------------------

daily_rank_ic <- validation_preds %>%
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

cat(
  "Validation Rank IC:",
  mean(daily_rank_ic$rank_ic, na.rm = TRUE),
  "\n"
)

# 13. FEATURE IMPORTANCE -------------------------------------

xgb_engine <- extract_fit_engine(best_model)

importance <- xgboost::xgb.importance(
  model = xgb_engine
)

print(importance)

# 14. SAVE RESULTS -------------------------------------------

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
  best_model,
  "results/models/improved_xgboost.rds"
)

write_csv(
  tuning_results,
  "results/tables/improved_tuning_results.csv"
)

write_csv(
  validation_preds,
  "results/predictions/improved_validation.csv"
)

write_csv(
  as_tibble(importance),
  "results/tables/improved_feature_importance.csv"
)

write_csv(
  train_improved,
  "data/train_improved.csv"
)

write_csv(
  validation_improved,
  "data/validation_improved.csv"
)

cat("\nImproved model training completed!\n")
