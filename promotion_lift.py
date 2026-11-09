"""Estimate how much promotions are associated with higher sales.

Regression of log(1 + sales) on log(1 + promotions) at the national level, with
product-family and week fixed effects and standard errors clustered by family.
The result is an association, not a proven causal effect (see README).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


if __name__ == "__main__":
    weekly = pd.read_csv("data/weekly_table.csv", parse_dates=["week"])

    # --- National table: one row per family per week ---
    # Sum sales and promotions across all stores. onpromotion becomes item-store-days:
    # how widespread promotions were that week (not how deep the discounts were).
    nat = weekly.groupby(["family", "week"])[["sales", "onpromotion"]].sum().reset_index()

    # --- Keep only weeks after promotions start being recorded ---
    # onpromotion is 0 everywhere early in the data because promotions weren't tracked yet.
    # Treating "no data" as "no promotions" would bias the estimate.
    promo_by_week = nat.groupby("week")["onpromotion"].sum()
    first_promo_week = promo_by_week[promo_by_week > 0].index.min()
    nat = nat[nat["week"] >= first_promo_week].copy()
    print("First week with promotions:", first_promo_week.date())

    # --- Log both sides so the coefficient reads as an elasticity ---
    # log(1 + x) handles weeks with zero sales or zero promotions.
    nat["log_sales"] = np.log1p(nat["sales"])
    nat["log_promo"] = np.log1p(nat["onpromotion"])
    nat = nat.dropna(subset=["log_sales", "log_promo"]).reset_index(drop=True)

    # --- Regression with fixed effects ---
    # C(family): each family gets its own baseline, so big and small categories aren't compared.
    # C(week):   each week gets its own baseline, removing holidays, paydays and other events
    #            that affect all families at once.
    # Clustered standard errors: a family's weeks aren't independent, so the confidence
    # interval is computed with each family as one group (33 groups).
    groups = pd.factorize(nat["family"])[0]   # family names -> numbers for the clustering
    model = smf.ols("log_sales ~ log_promo + C(family) + C(week)", data=nat).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    # --- Results ---
    coef = model.params["log_promo"]
    lo, hi = model.conf_int().loc["log_promo"]

    # Convert the elasticity into "a 10% increase in promotions -> X% increase in sales"
    lift_10 = 1.1 ** coef - 1
    lift_lo, lift_hi = 1.1 ** lo - 1, 1.1 ** hi - 1

    summary = (
        f"Elasticity (log_promo coefficient): {coef:.4f}\n"
        f"95% CI: {lo:.4f} to {hi:.4f}\n"
        f"A 10% increase in promotion activity is associated with a {lift_10:.1%} increase in sales "
        f"(95% CI: {lift_lo:.1%} to {lift_hi:.1%})\n"
        f"Rows used: {len(nat)}, families: {nat['family'].nunique()}, weeks: {nat['week'].nunique()}\n"
    )
    print(summary)

    with open("results/ols_model_results.txt", "w") as f:
        f.write(summary)