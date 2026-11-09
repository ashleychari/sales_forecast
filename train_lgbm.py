"""Train LightGBM on each evaluation window, score it, and analyze the final test.

For each window: train on every week before the cutoff, predict the next 8 weeks,
and score with WAPE, MAE and bias. Then break the final-test error down by product
family, and plot forecast vs actual for two example series.
"""
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from baselines import WAPE, MAE


def get_train_test_splits(train_data, dates, windows):
    """For each window, train = every week before its first week; test = its 8 weeks.
    Splitting by time (not randomly) means the model never sees the weeks it's tested on."""
    train_test_data = {window: {} for window in windows}

    for (start, end), window in zip(dates, windows):
        train = train_data[train_data['week'] < start]
        test = train_data[(train_data['week'] >= start) & (train_data['week'] <= end)]
        train_test_data[window]['train'] = train
        train_test_data[window]['test'] = test

    return train_test_data


def train_test_lgbm(train_test_dict, eq, save_filename):
    """Train one LightGBM model per window, score it, and return the results
    plus each window's test rows with a 'pred' column attached."""
    # Every feature is known before the week starts: lagged/shifted sales, the calendar,
    # planned promotions and fixed store attributes. Oil price is left out (not known in advance).
    feature_cols = ['store_nbr', 'family', 'onpromotion', 'city', 'state',
                    'type', 'cluster', 'holiday_week', 'sales_lag1',
                    'sales_lag2', 'sales_lag52', 'roll_mean_4', 'roll_mean_12',
                    'month', 'week_of_year', 'payday_week']
    if eq:
        feature_cols = feature_cols + ["earthquake"]

    # CSV loses the category type, so store attributes are converted back here.
    # LightGBM handles pandas category columns automatically.
    cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']

    results_dicts = []
    test_with_preds = {}

    for window, data_dict in train_test_dict.items():
        # Train on log1p(sales): errors count in relative terms, so the largest series
        # don't dominate training. The +1 handles zero-sales weeks.
        X_train = data_dict['train'][feature_cols].copy()
        y_train = np.log1p(data_dict['train']['sales'])
        X_test = data_dict['test'][feature_cols].copy()
        for c in cat_cols:
            X_train[c] = X_train[c].astype('category')
            X_test[c] = X_test[c].astype('category')

        model = lgb.LGBMRegressor(random_state=42, verbose=-1)   # default settings, no tuning
        model.fit(X_train, y_train)

        # Convert predictions back from log units; sales can't be negative
        pred = np.expm1(model.predict(X_test)).clip(0)
        actual = data_dict['test']['sales']

        # Keep the test rows with predictions attached, for the family breakdown and charts
        test = data_dict['test'].copy()
        test['pred'] = pred
        test_with_preds[window] = test

        # Score against actual sales. Bias = total predicted / total actual
        # (below 1 = predicting too low, above 1 = too high).
        wape = WAPE(pred, actual)
        mae = MAE(pred, actual)
        bias = pred.sum() / actual.sum()
        results_dicts.append({'Window': window, 'WAPE': wape, 'MAE': mae, 'bias': bias})
        print(f"Window: {window}, WAPE: {wape:.2%}, MAE: {mae:.1f}, bias: {bias:.3f}")

    results_df = pd.DataFrame(results_dicts)
    print(results_df)
    results_df.to_csv(f"results/{save_filename}", index=False)
    return results_df, test_with_preds


def family_breakdown(test, save_filename):
    """WAPE for each product family on one window, LightGBM vs naive."""
    test = test.copy()
    test['err_lgbm'] = (test['sales'] - test['pred']).abs()
    test['err_naive'] = (test['sales'] - test['sales_lag1']).abs()

    # Sum errors and sales per family first, then divide (that's WAPE).
    # Averaging each row's percentage error would give a different, wrong number.
    by_fam = test.groupby('family')[['err_lgbm', 'err_naive', 'sales']].sum()
    by_fam['wape_lgbm'] = by_fam['err_lgbm'] / by_fam['sales']
    by_fam['wape_naive'] = by_fam['err_naive'] / by_fam['sales']
    by_fam['sales_share'] = by_fam['sales'] / by_fam['sales'].sum()   # how much each family affects overall WAPE
    by_fam['lgbm_beats_naive'] = by_fam['wape_lgbm'] < by_fam['wape_naive']

    # Biggest families first, since they decide the overall result
    by_fam = (by_fam[['sales_share', 'wape_lgbm', 'wape_naive', 'lgbm_beats_naive']]
              .sort_values('sales_share', ascending=False)
              .reset_index())

    print(by_fam.to_string(index=False))
    by_fam.to_csv(f"results/{save_filename}", index=False)
    return by_fam


def plot_forecast_v_actual(test_w_preds, store, family, filename):
    """Plot actual sales, LightGBM and naive for one store-family series on the final test."""
    final = test_w_preds['Final Test']
    one = final[(final['store_nbr'] == store) & (final['family'] == family)]

    plt.figure(figsize=(12, 5))
    plt.plot(one['week'], one['sales'], label='Actual')
    plt.plot(one['week'], one['pred'], label='LightGBM')
    plt.plot(one['week'], one['sales_lag1'], label='Naive', linestyle='--')   # naive = last week's sales
    plt.legend()
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.title(f'Actual vs predicted sales: store {store}, {family}')
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=120)   # save before closing, or the image can come out blank
    plt.close()


if __name__ == "__main__":
    features = pd.read_csv("data/features.csv", parse_dates=["week"])

    # 3 backtest windows + the final held-out test, 8 weeks each (first week = cutoff)
    dates = [('2016-09-05', '2016-10-24'), ('2016-12-05', '2017-01-23'),
             ('2017-03-06', '2017-04-24'), ('2017-06-19', '2017-08-07')]
    windows = ['September', 'December', 'March', 'Final Test']

    # --- Main model (no earthquake flag) ---
    train_test_dict = get_train_test_splits(features, dates, windows)
    results_df, test_with_preds = train_test_lgbm(train_test_dict, False, "lgbm_backtest_no_eq_results.csv")

    # Final-test error by product family
    family_breakdown(test_with_preds['Final Test'], "final_test_by_family.csv")

    # Forecast vs actual: a large series where the model does well, and a failure case
    plot_forecast_v_actual(test_with_preds, 44, "GROCERY I", "forecast_vs_actual_store44_grocery.png")
    plot_forecast_v_actual(test_with_preds, 44, "SCHOOL AND OFFICE SUPPLIES", "forecast_vs_actual_store44_school.png")

    # Check: did last year's values (seasonal naive) predict school supplies better? (They didn't.)
    school = test_with_preds['Final Test']
    school = school[school["family"] == "SCHOOL AND OFFICE SUPPLIES"]
    print("School supplies WAPE  LightGBM:", f"{WAPE(school['pred'], school['sales']):.1%}",
          " Naive:", f"{WAPE(school['sales_lag1'], school['sales']):.1%}",
          " Seasonal naive:", f"{WAPE(school['sales_lag52'], school['sales']):.1%}")

    # --- Comparison: same model plus an earthquake flag (tested, not kept) ---
    # Flags the weeks of Apr 11 - May 9, 2016, after the April 16 earthquake.
    eq_features = features.copy()
    eq_features["earthquake"] = eq_features["week"].between("2016-04-11", "2016-05-09").astype(int)
    eq_train_test_dict = get_train_test_splits(eq_features, dates, windows)
    train_test_lgbm(eq_train_test_dict, True, "lgbm_backtest_eq_results.csv")