"""Build the weekly modeling table from Kaggle's daily files.

Output: data/weekly_table.csv, one row per store, product family and week (weeks start Monday),
with weekly sales, promotions, store info, average oil price and a national holiday flag.
"""
import pandas as pd

# --- Daily sales -> weekly sales ---
train_data = pd.read_csv("data/train.csv", parse_dates=["date"])

# Label each day with the Monday that starts its week (weeks run Monday to Sunday)
train_data["week"] = train_data["date"].dt.to_period("W-SUN").dt.start_time

# Sum daily sales and items on promotion into weekly totals for each store-family series
weekly = train_data.groupby(["store_nbr", "family", "week"])[["sales", "onpromotion"]].sum()

# Drop series that never sold anything (e.g. a store that doesn't carry a category).
# transform("sum") writes each series' total on every one of its rows, so we can filter rows directly.
# Zero-sales weeks inside real series are kept: they're real outcomes the model should see.
weekly["series_total"] = weekly.groupby(["store_nbr", "family"])["sales"].transform("sum")
weekly = weekly[weekly["series_total"] > 0].drop(columns="series_total")

# groupby put store_nbr/family/week into the index; this turns them back into columns
weekly = weekly.reset_index()

# --- Drop partial weeks ---
# The data starts on a Tuesday (Jan 1, 2013) and ends on a Tuesday (Aug 15, 2017),
# so the first and last weeks are incomplete and their totals would look falsely low.
# Christmas weeks also have only 6 days (stores are closed on Dec 25), but they're kept:
# those 6 days are the whole selling week, and dropping them would leave gaps in the 52-week lag.
edge_weeks = [train_data["week"].min(), train_data["week"].max()]
full = weekly[~weekly["week"].isin(edge_weeks)]

# --- Store info ---
# Adds city, state, store type and cluster. Every store_nbr appears in stores.csv,
# so the inner merge keeps every row.
stores = pd.read_csv("data/stores.csv")
full_stores = full.merge(stores, on="store_nbr", how="inner")

# --- Weekly average oil price ---
# Kept in the table for exploration, but not used as a model feature: the current week's
# price isn't known at forecast time, and its slow effect is already captured by lagged sales.
oil_data = pd.read_csv("data/oil.csv", parse_dates=["date"])
oil_data["dcoilwtico"] = oil_data["dcoilwtico"].ffill()   # fill missing days with the last known price
oil_data["week"] = oil_data["date"].dt.to_period("W-SUN").dt.start_time

oil_data = oil_data.groupby("week")["dcoilwtico"].mean().reset_index()
oil_data.columns = ["week", "oil_price"]

# Left merge keeps every sales row even if a week had no oil data
full_store_oil = full_stores.merge(oil_data, on="week", how="left")

# --- National holiday flag ---
holidays = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])
holidays = holidays[holidays["transferred"] != True]   # transferred holidays were moved to another day
holidays["week"] = holidays["date"].dt.to_period("W-SUN").dt.start_time

# Keep national holidays only; "Work Day" rows are make-up workdays, not days off
nat = holidays[(holidays["locale"] == "National") & (holidays["type"] != "Work Day")]

# One row per holiday week. Some weeks have several holidays, and duplicates would
# copy sales rows when merging.
holiday_weeks = nat[["week"]].drop_duplicates()
holiday_weeks["holiday_week"] = 1

# Left merge: weeks with no national holiday get NaN, which we fill with 0
weekly_table = full_store_oil.merge(holiday_weeks, on="week", how="left")
weekly_table["holiday_week"] = weekly_table["holiday_week"].fillna(0).astype(int)

weekly_table.to_csv("data/weekly_table.csv", index=False)