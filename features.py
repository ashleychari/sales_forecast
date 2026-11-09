import pandas as pd

# Read the weekly table and parse the week column as dates
data = pd.read_csv("data/weekly_table.csv", parse_dates=["week"])

# Sort so each store-family series is in time order (needed for shift)
data = data.sort_values(["store_nbr", "family", "week"]).reset_index(drop=True)

# --- Lags: sales 1, 2, and 52 weeks ago, within each store-family series ---
g = data.groupby(["store_nbr", "family"])["sales"]
for lag in [1, 2, 52]:
    data[f"sales_lag{lag}"] = g.shift(lag)

# --- Rolling averages of the previous 4 and 12 weeks ---
# shift(1) first so the current week's sales are never included (no leakage)
data["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
data["roll_mean_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())

# --- Calendar features ---
data["month"] = data["week"].dt.month
data["week_of_year"] = data["week"].dt.isocalendar().week.astype(int)

# Payday: does the week (Monday to Sunday) contain the 15th or a month end?
weeks = pd.Series(data["week"].unique())
days_in_week = [weeks + pd.Timedelta(days=k) for k in range(7)]
has_payday = pd.concat(
    [(d.dt.day == 15) | d.dt.is_month_end for d in days_in_week], axis=1
).any(axis=1)
data["payday_week"] = data["week"].map(dict(zip(weeks, has_payday.astype(int))))

# holiday_week was already built in build_weekly_table.py, so nothing to add here.
# onpromotion is already in the table too (known in advance, so it's allowed).

# --- Store info as categorical features ---
for c in ["store_nbr", "family", "type", "city"]:
    data[c] = data[c].astype("category")

# --- Drop 2013: it was only kept so 2014 rows could have a 52-week lag ---
data = data[data["week"] >= "2014-01-01"].reset_index(drop=True)

print(data.shape)
print("Missing lag52 values:", data["sales_lag52"].isna().sum())

data.to_csv("data/features.csv", index=False)