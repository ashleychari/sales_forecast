"""Score two simple baseline forecasts on each evaluation window.

Naive:          next week's sales = this week's sales     (sales_lag1)
Seasonal naive: next week's sales = same week last year   (sales_lag52)

These set the bar the LightGBM model has to beat.
"""
import pandas as pd


def WAPE(predicted, actual):
    """Weighted absolute percentage error: total absolute error / total actual sales.
    Stays stable when individual sales are zero, and weights big series more."""
    return (actual - predicted).abs().sum() / actual.sum()


def MAE(predicted, actual):
    """Mean absolute error: the average miss, in units sold per row."""
    return (actual - predicted).abs().mean()


if __name__ == "__main__":
    # Only runs when this file is run directly, not when train_lgbm.py imports WAPE and MAE
    features = pd.read_csv("data/features.csv", parse_dates=["week"])

    # Each baseline's prediction is already a column in the features table
    models = {"Naive": "sales_lag1", "Seasonal Naive": "sales_lag52"}

    # Evaluation windows: 3 backtest windows + the final held-out test (8 weeks each)
    dates = [("2016-09-05", "2016-10-24"), ("2016-12-05", "2017-01-23"),
             ("2017-03-06", "2017-04-24"), ("2017-06-19", "2017-08-07")]
    windows = ["September", "December", "March", "Final Test"]

    rows = []

    for name, (start, end) in zip(windows, dates):
        # Keep only the rows (every store and family) in this window's 8 weeks
        window = features[features["week"].between(start, end)]
        print(name, "weeks:", window["week"].nunique())   # sanity check: should be 8

        # Score each baseline against actual sales in this window
        for model_key, col in models.items():
            rows.append({
                "window": name,
                "model": model_key,
                "WAPE": WAPE(window[col], window["sales"]),
                "MAE": MAE(window[col], window["sales"]),
            })

    # One row per window per model (4 x 2 = 8 rows)
    results_table = pd.DataFrame(rows)
    print(results_table)
    results_table.to_csv("results/baseline_results.csv", index=False)