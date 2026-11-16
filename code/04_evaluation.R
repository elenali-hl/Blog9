
# ============================================================
# BLOG POST 9
# 04 - OUT-OF-SAMPLE EVALUATION
# ============================================================

# 1. LIBRARIES -----------------------------------------------

library(tidyverse)
library(tidymodels)
library(xgboost)
library(slider)

set.seed(123)

# 2. LOAD DATA AND MODELS ------------------------------------

train <- read_csv(
  "data/train.csv",
  show_col_types = FALSE
)

test <- read_csv(
  "data/test.csv",
  show_col_types = FALSE
) %>%
  mutate(
    date = as.Date(date),
    target_end_date = as.Date(target_end_date)
  )

prices <- read_csv(
  "data-raw/stock_prices.csv",
  show_col_types = FALSE
)

spy <- read_csv(
  "data/spy_benchmark.csv",
  show_col_types = FALSE
) %>%
  mutate(date = as.Date(date))

original_model <- readRDS(
  "results/models/original_xgboost.rds"
)

improved_model <- readRDS(
  "results/models/improved_xgboost.rds"
)

# 3. RECREATE IMPROVED FEATURES ------------------------------

# Use the same feature engineering as Step 3.

extra_features <- prices %>%
  mutate(date = as.Date(date)) %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    daily_ret = adjusted / lag(adjusted) - 1,
    
    momentum_1m_new =
      lag(adjusted, 1) / lag(adjusted, 22) - 1,
    
    momentum_3m_new =
      lag(adjusted, 1) / lag(adjusted, 64) - 1,
    
    past_return_10d =
      lag(adjusted, 1) / lag(adjusted, 11) - 1,
    
    volatility_10d = slide_dbl(
      lag(daily_ret),
      sd,
      .before = 9,
      .complete = TRUE
    ),
    
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

spy_features <- spy %>%
  arrange(date) %>%
  mutate(
    market_momentum_1m =
      lag(spy_price, 1) / lag(spy_price, 22) - 1
  ) %>%
  select(date, market_momentum_1m)

test_improved <- test %>%
  left_join(
    extra_features,
    by = c("date", "symbol")
  ) %>%
  left_join(
    spy_features,
    by = "date"
  )

stopifnot(
  nrow(test_improved) == nrow(test),
  !anyNA(test_improved)
)

# 4. GENERATE TEST PREDICTIONS -------------------------------

original_preds <- predict(
  original_model,
  new_data = test
) %>%
  bind_cols(test) %>%
  rename(predicted_return = .pred) %>%
  mutate(model = "Original XGBoost")

improved_preds <- predict(
  improved_model,
  new_data = test_improved
) %>%
  bind_cols(test_improved) %>%
  rename(predicted_return = .pred) %>%
  mutate(model = "Improved XGBoost")

# Historical mean benchmark:
# No information from validation or test is used.

historical_mean <- mean(train$target)

benchmark_preds <- test %>%
  mutate(
    predicted_return = historical_mean,
    model = "Historical Mean"
  )

# 5. COMBINE PREDICTIONS -------------------------------------

all_preds <- bind_rows(
  original_preds %>%
    select(date, symbol, target,
           target_end_date, predicted_return, model),
  
  improved_preds %>%
    select(date, symbol, target,
           target_end_date, predicted_return, model),
  
  benchmark_preds %>%
    select(date, symbol, target,
           target_end_date, predicted_return, model)
)

# 6. PREDICTIVE PERFORMANCE ----------------------------------

evaluate_predictions <- function(data) {
  
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
    RMSE = sqrt(mean(
      (data$target - data$predicted_return)^2
    )),
    MAE = mean(
      abs(data$target - data$predicted_return)
    ),
    R2 = 1 -
      sum(
        (data$target - data$predicted_return)^2
      ) /
      sum(
        (data$target - mean(data$target))^2
      ),
    Rank_IC = if (all(is.na(daily_ic$rank_ic))) {
      NA_real_
    } else {
      mean(daily_ic$rank_ic, na.rm = TRUE)
    }
  )
}

predictive_results <- all_preds %>%
  group_by(model) %>%
  group_modify(
    ~ evaluate_predictions(.x)
  ) %>%
  ungroup()

cat("\nPREDICTIVE PERFORMANCE:\n")
print(predictive_results)

# 7. CREATE NON-OVERLAPPING REBALANCE DATES ------------------

# Each target is measured from t+1 to t+20.
# Select dates at least 20 trading sessions apart.

trading_dates <- sort(unique(spy$date))

eligible_dates <- sort(unique(test$date))

rebalance_dates <- as.Date(character())
last_end_index <- 0

for (d in eligible_dates) {
  
  current_index <- match(d, trading_dates)
  
  if (is.na(current_index)) next
  
  # Require the full future 20-session horizon.
  if (current_index + 20 > length(trading_dates)) next
  
  # Avoid overlapping return measurement windows.
  if (current_index >= last_end_index) {
    
    rebalance_dates <- c(rebalance_dates, d)
    
    last_end_index <- current_index + 20
  }
}

cat(
  "\nNumber of rebalance periods:",
  length(rebalance_dates), "\n"
)

# 8. CONSTRUCT TOP-10 PORTFOLIOS -----------------------------

# Equal-weighted portfolios avoid additional
# assumptions about prediction-based weights.

make_portfolio <- function(data) {
  
  data %>%
    filter(date %in% rebalance_dates) %>%
    group_by(date) %>%
    arrange(desc(predicted_return), symbol,
            .by_group = TRUE) %>%
    slice_head(n = 10) %>%
    mutate(weight = 1 / n()) %>%
    ungroup()
}

original_portfolio <- make_portfolio(original_preds)

improved_portfolio <- make_portfolio(improved_preds)

# 9. PORTFOLIO RETURNS ---------------------------------------

portfolio_returns <- bind_rows(
  original_portfolio,
  improved_portfolio
) %>%
  group_by(model, date) %>%
  summarise(
    gross_return = sum(weight * target),
    .groups = "drop"
  )

# 10. SPY MATCHED-HORIZON RETURNS ----------------------------

# Match the same t+1 to t+20 return horizon.
# Avoid comparing a 19-session strategy return
# with a full calendar-month SPY return.

spy_horizon <- spy %>%
  arrange(date) %>%
  mutate(
    spy_future_return =
      lead(spy_price, 20) /
      lead(spy_price, 1) - 1
  ) %>%
  filter(date %in% rebalance_dates) %>%
  select(date, spy_future_return)

portfolio_returns <- portfolio_returns %>%
  left_join(spy_horizon, by = "date")

stopifnot(
  !anyNA(portfolio_returns$spy_future_return)
)

# 11. TRANSACTION COSTS --------------------------------------

# Illustrative round-trip cost: 20 basis points.
# Applied to each portfolio rebalance period.
# This is a simplified assumption, not actual
# trade-by-trade turnover accounting.

transaction_cost <- 0.002

portfolio_returns <- portfolio_returns %>%
  mutate(
    net_return = gross_return - transaction_cost
  )

# 12. PERFORMANCE METRICS ------------------------------------

# Approximate annualization:
# 252 trading sessions / 20-session spacing.

periods_per_year <- 252 / 20

performance_summary <- function(data) {
  
  r <- data$net_return
  market <- data$spy_future_return
  
  wealth <- prod(1 + r)
  market_wealth <- prod(1 + market)
  
  annual_return <- wealth^(
    periods_per_year / length(r)
  ) - 1
  
  annual_volatility <- sd(r) *
    sqrt(periods_per_year)
  
  # Zero risk-free rate assumption.
  sharpe <- if (sd(r) > 0) {
    mean(r) / sd(r) *
      sqrt(periods_per_year)
  } else {
    NA_real_
  }
  
  capm <- lm(r ~ market)
  
  alpha <- unname(coef(capm)[1]) *
    periods_per_year
  
  beta <- unname(coef(capm)[2])
  
  tibble(
    Periods = length(r),
    Total_Return = wealth - 1,
    Annual_Return = annual_return,
    Annual_Volatility = annual_volatility,
    Sharpe = sharpe,
    CAPM_Alpha = alpha,
    CAPM_Beta = beta,
    SPY_Total_Return = market_wealth - 1
  )
}

investment_results <- portfolio_returns %>%
  group_by(model) %>%
  group_modify(
    ~ performance_summary(.x)
  ) %>%
  ungroup()

cat("\nINVESTMENT PERFORMANCE:\n")
print(investment_results, width = Inf)

# 13. CUMULATIVE WEALTH --------------------------------------

cumulative_returns <- portfolio_returns %>%
  arrange(model, date) %>%
  group_by(model) %>%
  mutate(
    cumulative_gross = cumprod(1 + gross_return),
    cumulative_net = cumprod(1 + net_return),
    cumulative_spy = cumprod(1 + spy_future_return)
  ) %>%
  ungroup()

# 14. SAVE ALL RESULTS ---------------------------------------

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

write_csv(
  improved_preds,
  "results/predictions/improved_test.csv"
)

write_csv(
  benchmark_preds,
  "results/predictions/benchmark_test.csv"
)

write_csv(
  predictive_results,
  "results/tables/predictive_comparison.csv"
)

write_csv(
  investment_results,
  "results/tables/investment_comparison.csv"
)

write_csv(
  portfolio_returns,
  "results/tables/portfolio_period_returns.csv"
)

write_csv(
  cumulative_returns,
  "results/tables/cumulative_returns.csv"
)

write_csv(
  bind_rows(original_portfolio, improved_portfolio),
  "results/tables/portfolio_holdings.csv"
)

cat("\nStep 4 completed successfully!\n")
