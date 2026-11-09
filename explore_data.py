import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

train = pd.read_csv("data/train.csv", parse_dates=["date"])
holidays = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])

print(train.head())
print("Date range:", train["date"].min(), "to", train["date"].max())
print("Stores:", train["store_nbr"].nunique(), "| Families:", train["family"].nunique())

# --- 1. Total sales over time ---
daily = train.groupby("date")["sales"].sum()

fig, ax = plt.subplots(figsize=(12, 4))
daily.plot(ax=ax, linewidth=0.5, label="Daily")
daily.rolling(28).mean().plot(ax=ax, label="28-day average")
ax.axvline(pd.Timestamp("2016-04-16"), color="red", linestyle="--", label="Earthquake")
ax.set_title("Total daily sales")
ax.legend()
fig.savefig("results/total_sales.png", dpi=120, bbox_inches="tight")

# --- 2. A few individual store-family series ---
examples = [(1, "GROCERY I"), (44, "BEVERAGES"), (10, "PRODUCE"), (25, "SCHOOL AND OFFICE SUPPLIES")]

fig, axes = plt.subplots(len(examples), 1, figsize=(12, 10), sharex=True)
for ax, (store, family) in zip(axes, examples):
    s = train[(train["store_nbr"] == store) & (train["family"] == family)]
    ax.plot(s["date"], s["sales"], linewidth=0.5)
    ax.set_title(f"Store {store} - {family}")
fig.tight_layout()
fig.savefig("results/example_series.png", dpi=120)

# --- 3. Always-zero series and zero-sales share ---
series_total = train.groupby(["store_nbr", "family"])["sales"].sum()
always_zero = series_total[series_total == 0]
print(f"\nAlways-zero series: {len(always_zero)} of {len(series_total)}")
print(always_zero.reset_index()["family"].value_counts().head(10))

print(f"Share of rows with zero sales: {(train['sales'] == 0).mean():.1%}")

# --- 4. Promotion patterns ---
promo_by_year = train.groupby(train["date"].dt.year)["onpromotion"].mean()
print("\nAverage items on promotion per row, by year:")
print(promo_by_year)

# --- 5. Earthquake period (two months either side) ---
quake = daily["2016-02-15":"2016-06-15"]
fig, ax = plt.subplots(figsize=(10, 4))
quake.plot(ax=ax)
ax.axvline(pd.Timestamp("2016-04-16"), color="red", linestyle="--")
ax.set_title("Sales around the April 16, 2016 earthquake")
fig.savefig("results/earthquake.png", dpi=120, bbox_inches="tight")

# --- 6. Holidays: average sales on national holidays vs other days ---
national = holidays[(holidays["locale"] == "National") & (holidays["transferred"] == False)]
is_holiday = daily.index.isin(national["date"])
print(f"\nAvg daily sales on national holidays: {daily[is_holiday].mean():,.0f}")
print(f"Avg daily sales on other days:        {daily[~is_holiday].mean():,.0f}")

print("\nPlots saved to results/")