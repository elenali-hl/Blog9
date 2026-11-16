
# ============================================================
# BLOG POST 9
# 05 - VISUALIZATION
# ============================================================

# 1. LIBRARIES -----------------------------------------------

library(tidyverse)
library(scales)

# 2. OUTPUT DIRECTORIES --------------------------------------

dir.create(
  "results/figures",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "results/tables",
  recursive = TRUE,
  showWarnings = FALSE
)

# 3. LOAD SAVED RESULTS --------------------------------------

tuning <- read_csv(
  "results/tables/improved_tuning_results.csv",
  show_col_types = FALSE
)

predictive <- read_csv(
  "results/tables/predictive_comparison.csv",
  show_col_types = FALSE
)

investment <- read_csv(
  "results/tables/investment_comparison.csv",
  show_col_types = FALSE
)

importance <- read_csv(
  "results/tables/improved_feature_importance.csv",
  show_col_types = FALSE
)

cumulative <- read_csv(
  "results/tables/cumulative_returns.csv",
  show_col_types = FALSE
) %>%
  mutate(date = as.Date(date))

# 4. FIGURE 1: HYPERPARAMETER TUNING -------------------------

fig1 <- tuning %>%
  mutate(
    candidate = factor(
      candidate,
      levels = rev(candidate[order(Validation_RMSE)])
    )
  ) %>%
  ggplot(aes(
    x = candidate,
    y = Validation_RMSE
  )) +
  geom_col(
    fill = "#457B9D",
    width = 0.65
  ) +
  geom_text(
    aes(label = sprintf("%.4f", Validation_RMSE)),
    hjust = -0.15,
    size = 4
  ) +
  coord_flip() +
  scale_y_continuous(
    limits = c(0, max(tuning$Validation_RMSE) * 1.2),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Hyperparameter Tuning: Validation RMSE",
    subtitle = "Lower RMSE indicates better predictive accuracy",
    x = "Candidate Model",
    y = "Validation RMSE"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "results/figures/figure1_tuning.png",
  fig1,
  width = 8,
  height = 5,
  dpi = 300
)

# 5. FIGURE 2: TEST PREDICTIVE PERFORMANCE -------------------

fig2 <- predictive %>%
  mutate(
    model = factor(
      model,
      levels = c(
        "Historical Mean",
        "Improved XGBoost",
        "Original XGBoost"
      )
    )
  ) %>%
  ggplot(aes(
    x = model,
    y = RMSE,
    fill = model
  )) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = sprintf("%.4f", RMSE)),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "Historical Mean" = "#8D99AE",
      "Improved XGBoost" = "#2A9D8F",
      "Original XGBoost" = "#457B9D"
    )
  ) +
  scale_y_continuous(
    limits = c(0, max(predictive$RMSE) * 1.15),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Out-of-Sample Predictive Performance",
    subtitle = "Test period: 2025-2026",
    x = NULL,
    y = "RMSE (Lower Is Better)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    )
  )

ggsave(
  "results/figures/figure2_test_rmse.png",
  fig2,
  width = 8,
  height = 5,
  dpi = 300
)

# 6. FIGURE 3: FEATURE IMPORTANCE ----------------------------

fig3 <- importance %>%
  slice_max(Gain, n = 10) %>%
  mutate(
    Feature = reorder(Feature, Gain)
  ) %>%
  ggplot(aes(
    x = Feature,
    y = Gain
  )) +
  geom_col(
    fill = "#2A9D8F",
    width = 0.7
  ) +
  coord_flip() +
  scale_y_continuous(
    labels = percent_format(accuracy = 1)
  ) +
  labs(
    title = "Improved XGBoost: Feature Importance",
    subtitle = "Relative importance measured by Gain",
    x = NULL,
    y = "Relative Gain"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "results/figures/figure3_feature_importance.png",
  fig3,
  width = 8,
  height = 6,
  dpi = 300
)

# 7. FIGURE 4: CUMULATIVE INVESTMENT PERFORMANCE -------------

# The SPY series is repeated for each model in the
# cumulative file. Keep only one SPY series.

strategy_wealth <- cumulative %>%
  transmute(
    date,
    strategy = model,
    wealth = cumulative_net
  )

spy_wealth <- cumulative %>%
  filter(model == "Original XGBoost") %>%
  transmute(
    date,
    strategy = "SPY Benchmark",
    wealth = cumulative_spy
  )

# Add initial wealth of $1 for each strategy

initial_date <- min(cumulative$date) - 1

initial_wealth <- tibble(
  date = rep(initial_date, 3),
  strategy = c(
    "Improved XGBoost",
    "Original XGBoost",
    "SPY Benchmark"
  ),
  wealth = c(1, 1, 1)
)

wealth_data <- bind_rows(
  initial_wealth,
  strategy_wealth,
  spy_wealth
) %>%
  arrange(strategy, date)

fig4 <- ggplot(
  wealth_data,
  aes(
    x = date,
    y = wealth,
    color = strategy
  )
) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(
    values = c(
      "Improved XGBoost" = "#2A9D8F",
      "Original XGBoost" = "#457B9D",
      "SPY Benchmark" = "#E76F51"
    )
  ) +
  scale_y_continuous(
    labels = dollar_format(prefix = "$")
  ) +
  labs(
    title = "Simulated Portfolio Performance vs. SPY",
    subtitle = paste(
      "Matched-horizon backtest;",
      "20 bps assumed cost per strategy period"
    ),
    x = "Rebalance Date",
    y = "Compounded Wealth from $1",
    color = NULL,
    caption = paste(
      "Hypothetical non-continuous holding-period returns;",
      "not an executable continuous NAV."
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom"
  )

ggsave(
  "results/figures/figure4_cumulative_returns.png",
  fig4,
  width = 9,
  height = 5.5,
  dpi = 300
)

# 8. SAVE FORMATTED COMPARISON TABLES ------------------------

predictive_table <- predictive %>%
  mutate(
    across(
      c(RMSE, MAE, R2, Rank_IC),
      ~ round(.x, 4)
    )
  )

investment_table <- investment %>%
  mutate(
    across(
      c(
        Total_Return,
        Annual_Return,
        Annual_Volatility,
        CAPM_Alpha,
        SPY_Total_Return
      ),
      ~ round(.x * 100, 2)
    ),
    across(
      c(Sharpe, CAPM_Beta),
      ~ round(.x, 3)
    )
  )

write_csv(
  predictive_table,
  "results/tables/predictive_comparison_formatted.csv"
)

write_csv(
  investment_table,
  "results/tables/investment_comparison_formatted.csv"
)

# 9. PRINT SUMMARY -------------------------------------------

cat("\n====================================\n")
cat("FIGURES GENERATED SUCCESSFULLY\n")
cat("====================================\n")

print(list.files("results/figures"))

cat("\nPredictive comparison:\n")
print(predictive_table)

cat("\nInvestment comparison:\n")
print(investment_table, width = Inf)

cat("\nStep 5 completed successfully!\n")


