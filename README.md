# Blog Post 9: Predicting Stock Returns with Gradient Boosting

## Overview

This project revisits an in-class stock-return prediction experiment using Extreme Gradient Boosting (XGBoost).

The objective is to examine whether additional financial features and hyperparameter tuning improve out-of-sample stock-return predictions and whether those improvements translate into economically meaningful investment performance.

The analysis compares:

1. Original XGBoost model
2. Improved XGBoost model
3. Historical-mean prediction benchmark

It also evaluates hypothetical top-10 stock portfolios against the SPY market benchmark.

## Data

Historical adjusted stock prices were retrieved from Yahoo Finance using the `tidyquant` R package.

- Stock universe: 95 large U.S. stocks
- Historical price period: January 2022 to October 2026
- Training sample: 2022–2023
- Validation sample: 2024
- Test sample: 2025–2026
- Prediction target: Forward adjusted-price return from trading day t+1 to t+20

The raw data are cached in `data-raw/` to support reproducibility.

## Project Structure

```text
Blog9/
├── Blog9.Rproj
├── index.qmd
├── README.md
├── code/
│   ├── 01_prepare_data.R
│   ├── 02_original_model.R
│   ├── 03_improved_model.R
│   ├── 04_evaluation.R
│   └── 05_visualization.R
├── data-raw/
│   ├── stock_prices.csv
│   └── spy_prices.csv
├── data/
│   ├── stock_model_data.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── results/
    ├── figures/
    ├── tables/
    ├── models/
    └── predictions/
```

Additional feature-engineered datasets are saved in `data/`.

## Methods

### Original XGBoost

The baseline uses six engineered financial predictors and the classroom model's original hyperparameters:

- 300 trees
- Maximum tree depth of 5
- Learning rate of 0.05
- Minimum node size of 10

This is a simplified reproduction of the classroom experiment rather than an exact replication of every original input column.

### Improved XGBoost

The improved model adds six financial features, including corrected historical momentum, short- and long-horizon volatility, recent returns, and market momentum.

Four hyperparameter configurations are compared on the validation sample.

The best-performing configuration uses:

- 200 trees
- Maximum tree depth of 2
- Learning rate of 0.03
- Minimum node size of 20

### Evaluation

Predictive performance is evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared
- Cross-sectional Spearman Rank Information Coefficient (Rank IC)

Economic performance is evaluated using hypothetical top-10 stock portfolios and a matched-horizon SPY benchmark.

## Main Results

### Out-of-Sample Prediction

| Model | RMSE | MAE | R-squared | Rank IC |
|---|---:|---:|---:|---:|
| Historical Mean | 0.0948 | 0.0682 | -0.0000 | N/A |
| Improved XGBoost | 0.0960 | 0.0694 | -0.0245 | 0.0921 |
| Original XGBoost | 0.1020 | 0.0729 | -0.1470 | 0.0183 |

The improved model reduces RMSE relative to the original XGBoost specification but does not outperform the historical-mean benchmark.

### Hypothetical Investment Performance

| Strategy | Total Simulated Return |
|---|---:|
| Improved XGBoost | 79.4% |
| Original XGBoost | 34.8% |
| SPY Benchmark | 31.4% |

These results are based on a simplified matched-horizon backtest with an assumed transaction cost of 20 basis points per strategy period.

They should not be interpreted as returns from a fully executable continuous trading strategy.

## Reproducibility

### Required R Packages

```r
install.packages(c(
  "tidyverse",
  "tidyquant",
  "tidymodels",
  "xgboost",
  "lubridate",
  "slider",
  "scales"
))
```

### Running the Analysis

Open `Blog9.Rproj` in RStudio.

Run the following scripts in order:

```r
source("code/01_prepare_data.R")
source("code/02_original_model.R")
source("code/03_improved_model.R")
source("code/04_evaluation.R")
source("code/05_visualization.R")
```

The scripts generate the processed data, fitted models, prediction files, summary tables, and figures.

If cached raw data are present, the preparation script loads those files rather than downloading historical prices again.

### Rendering the Blog

From the Blog9 project root:

```bash
quarto render index.qmd
```

The rendered HTML uses figures saved under `results/figures/`.

## Limitations

- Fixed stock universe and possible survivorship bias
- Limited historical sample
- One validation period for hyperparameter selection
- Simplified transaction-cost assumptions
- Non-continuous matched-horizon portfolio simulation
- No formal statistical test of abnormal investment returns

## Conclusion

Additional features and lower-complexity XGBoost tuning improved performance relative to the original model.

However, the historical-mean benchmark still achieved the lowest out-of-sample prediction error.

The findings highlight the distinction between improved machine learning performance, cross-sectional stock ranking, and economically robust investment predictability.
```
