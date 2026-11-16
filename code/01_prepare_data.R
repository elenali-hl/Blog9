
# ============================================================
# BLOG POST 9: STOCK RETURN PREDICTION
# 01 - DATA PREPARATION
# ============================================================

# 1. LIBRARIES -----------------------------------------------

packages <- c(
  "tidyverse",
  "tidyquant",
  "lubridate",
  "slider"
)

missing_packages <- packages[
  !vapply(packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(tidyverse)
library(tidyquant)
library(lubridate)
library(slider)

set.seed(123)

# 2. PROJECT DIRECTORIES ------------------------------------

# Run this script from the Blog9 project root.

dir.create("data-raw", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)
dir.create("results/tables",
           recursive = TRUE,
           showWarnings = FALSE)

# 3. STOCK UNIVERSE -----------------------------------------

# Same stock universe as the original classroom code.

tickers <- c(
  # Technology
  "AAPL","MSFT","NVDA","GOOG","META","AMZN",
  "AVGO","ORCL","ADBE","CRM","CSCO","AMD",
  "INTC","QCOM","TXN","IBM","NOW","INTU",
  "AMAT","MU",
  
  # Financials
  "JPM","BAC","WFC","GS","MS","BLK","C",
  "SCHW","SPGI","AXP","USB","PNC","TFC","COF",
  
  # Healthcare
  "UNH","JNJ","LLY","PFE","MRK","ABBV",
  "TMO","DHR","ABT","BMY","AMGN","GILD",
  "CVS","ISRG","MDT",
  
  # Consumer Discretionary
  "TSLA","HD","MCD","NKE","SBUX","LOW",
  "TJX","BKNG","CMG","ORLY",
  
  # Consumer Staples
  "WMT","COST","PG","KO","PEP","PM",
  "MO","CL","EL",
  
  # Industrials
  "CAT","GE","HON","UPS","BA","RTX",
  "DE","LMT","MMM","ETN",
  
  # Energy
  "XOM","CVX","COP","SLB","EOG",
  
  # Communication
  "NFLX","DIS","CMCSA","TMUS","VZ",
  
  # Real Estate, Utilities, Materials
  "AMT","PLD","NEE","DUK","SO","LIN","APD"
)

cat("Number of requested stocks:",
    length(tickers), "\n")

# 4. DOWNLOAD HISTORICAL PRICES ------------------------------

# Use a fixed end date for reproducibility.
# Yahoo Finance may revise historical data.

start_date <- "2022-01-01"
end_date <- "2026-11-01"

stock_file <- "data-raw/stock_prices.csv"
spy_file <- "data-raw/spy_prices.csv"

if (file.exists(stock_file) &&
    file.exists(spy_file)) {
  
  cat("Loading previously saved price data...\n")
  
  prices <- read_csv(stock_file,
                     show_col_types = FALSE)
  
  spy_raw <- read_csv(spy_file,
                      show_col_types = FALSE)
  
} else {
  
  cat("Downloading historical stock prices...\n")
  
  prices <- tq_get(
    tickers,
    from = start_date,
    to = end_date,
    get = "stock.prices"
  )
  
  cat("Downloading SPY benchmark...\n")
  
  spy_raw <- tq_get(
    "SPY",
    from = start_date,
    to = end_date,
    get = "stock.prices"
  )
  
  write_csv(prices, stock_file)
  write_csv(spy_raw, spy_file)
}

# 5. DATA QUALITY CHECKS ------------------------------------

prices <- prices %>%
  mutate(date = as.Date(date)) %>%
  arrange(symbol, date) %>%
  distinct(symbol, date, .keep_all = TRUE)

spy_raw <- spy_raw %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  distinct(date, .keep_all = TRUE)

stopifnot(
  nrow(prices) > 0,
  nrow(spy_raw) > 0,
  all(c("symbol", "date", "adjusted") %in%
        names(prices)),
  all(c("date", "adjusted") %in%
        names(spy_raw))
)

missing_tickers <- setdiff(
  tickers, unique(prices$symbol)
)

cat("Stocks successfully downloaded:",
    n_distinct(prices$symbol), "\n")

if (length(missing_tickers) > 0) {
  cat("Missing tickers:",
      paste(missing_tickers, collapse = ", "),
      "\n")
}

# 6. PREPARE SPY BENCHMARK ----------------------------------

spy <- spy_raw %>%
  transmute(
    date,
    spy_price = adjusted
  ) %>%
  arrange(date) %>%
  mutate(
    spy_ret = spy_price / lag(spy_price) - 1
  )

# 7. FEATURE ENGINEERING ------------------------------------

# Preserve the original classroom feature definitions.
# All rolling calculations are performed within each stock.

df <- prices %>%
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    return = adjusted / lag(adjusted) - 1
  ) %>%
  ungroup() %>%
  left_join(
    spy %>% select(date, spy_ret),
    by = "date"
  ) %>%
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    
    # Momentum: original classroom definitions
    mom_1m = lag(adjusted, 21) /
      lag(adjusted, 1) - 1,
    
    mom_3m = lag(adjusted, 63) /
      lag(adjusted, 1) - 1,
    
    mom_6m = lag(adjusted, 126) /
      lag(adjusted, 1) - 1,
    
    # Historical return volatility
    vol_1m = slide_dbl(
      lag(return),
      sd,
      .before = 21,
      .complete = TRUE
    ),
    
    # Short-term reversal signal
    ret_5d = lag(adjusted, 5) /
      lag(adjusted, 1) - 1,
    
    # Return relative to SPY
    excess_ret = return - spy_ret,
    
    # Future return target: original definition
    target = lead(adjusted, 20) /
      lead(adjusted, 1) - 1,
    
    # Last date needed to observe the target
    target_end_date = lead(date, 20)
  ) %>%
  ungroup()

# 8. SELECT MODEL VARIABLES ---------------------------------

# Keep only the six classroom predictors.
# Date and symbol remain identifiers.

model_data <- df %>%
  select(
    date,
    symbol,
    mom_1m,
    mom_3m,
    mom_6m,
    vol_1m,
    ret_5d,
    excess_ret,
    target,
    target_end_date
  ) %>%
  drop_na() %>%
  arrange(date, symbol)

# 9. TIME-BASED DATA SPLIT ----------------------------------

# Training: 2022-2023
# Validation: 2024
# Test: 2025 onward
#
# Exclude observations whose future target period
# crosses the next split boundary.

train <- model_data %>%
  filter(
    date < as.Date("2024-01-01"),
    target_end_date < as.Date("2024-01-01")
  )

validation <- model_data %>%
  filter(
    date >= as.Date("2024-01-01"),
    date < as.Date("2025-01-01"),
    target_end_date < as.Date("2025-01-01")
  )

test <- model_data %>%
  filter(
    date >= as.Date("2025-01-01")
  )

stopifnot(
  nrow(train) > 0,
  nrow(validation) > 0,
  nrow(test) > 0
)

# 10. SAVE CLEAN DATA ---------------------------------------

write_csv(model_data, "data/stock_model_data.csv")
write_csv(train, "data/train.csv")
write_csv(validation, "data/validation.csv")
write_csv(test, "data/test.csv")

write_csv(spy, "data/spy_benchmark.csv")

# 11. DATA SUMMARY ------------------------------------------

data_summary <- bind_rows(
  train %>% mutate(dataset = "Training"),
  validation %>% mutate(dataset = "Validation"),
  test %>% mutate(dataset = "Test")
) %>%
  group_by(dataset) %>%
  summarise(
    observations = n(),
    stocks = n_distinct(symbol),
    start_date = min(date),
    end_date = max(date),
    mean_target = mean(target),
    sd_target = sd(target),
    .groups = "drop"
  )

print(data_summary)

write_csv(
  data_summary,
  "results/tables/data_summary.csv"
)

cat("\nData preparation completed successfully!\n")
