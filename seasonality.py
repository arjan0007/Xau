import yfinance as yf
import pandas as pd
import numpy as np


MONTH_NAMES = ["Jan","Feb","Mar","Apr","Maj","Qer","Kor","Gus","Sht","Tet","Nën","Dhj"]

def fetch_seasonality(years: int = 10) -> dict:
    """
    Analyze gold's historical seasonal patterns.
    Returns monthly avg returns, best/worst months, day-of-week patterns.
    """
    df = yf.Ticker("GC=F").history(period=f"{years}y", interval="1d")
    if df.empty:
        return {}

    df.index = pd.to_datetime(df.index)
    df["return"] = df["Close"].pct_change()
    df["month"] = df.index.month
    df["month_name"] = df.index.month.map(lambda m: MONTH_NAMES[m-1])
    df["year"] = df.index.year
    df["dow"] = df.index.dayofweek  # 0=Mon
    df["dow_name"] = df.index.dayofweek.map({0:"E Hënë",1:"E Martë",2:"E Mërkurë",3:"E Enjte",4:"E Premte"})
    df["week"] = df.index.isocalendar().week.astype(int)

    # Monthly average returns
    monthly = df.groupby("month")["return"].agg(["mean","std","count"]).reset_index()
    monthly.columns = ["month","avg_return","std","count"]
    monthly["avg_return_pct"] = (monthly["avg_return"] * 100).round(3)
    monthly["month_name"] = monthly["month"].map(lambda m: MONTH_NAMES[m-1])

    # Monthly win rate (% of days positive)
    df["positive"] = df["return"] > 0
    monthly_wr = df.groupby("month")["positive"].mean().reset_index()
    monthly_wr.columns = ["month","win_rate"]
    monthly = monthly.merge(monthly_wr, on="month")

    # Monthly total return per year (more reliable)
    monthly_perf = df.groupby(["year","month"]).apply(
        lambda g: (g["Close"].iloc[-1] / g["Close"].iloc[0] - 1) * 100
        if len(g) > 0 else 0
    ).reset_index(name="monthly_pct")

    avg_by_month = monthly_perf.groupby("month")["monthly_pct"].agg(["mean","median","std"]).reset_index()
    avg_by_month.columns = ["month","avg_pct","median_pct","std_pct"]
    avg_by_month["month_name"] = avg_by_month["month"].map(lambda m: MONTH_NAMES[m-1])

    # Best / worst months
    best_month  = avg_by_month.loc[avg_by_month["avg_pct"].idxmax()]
    worst_month = avg_by_month.loc[avg_by_month["avg_pct"].idxmin()]
    current_month = pd.Timestamp.now().month
    current_stats = avg_by_month[avg_by_month["month"] == current_month]

    # Day of week
    dow_perf = df[df["dow"] < 5].groupby("dow_name")["return"].agg(["mean","count"])
    dow_perf["avg_pct"] = (dow_perf["mean"] * 100).round(3)
    dow_perf = dow_perf.reset_index()

    # Q1/Q2/Q3/Q4
    df["quarter"] = df.index.quarter
    quarterly = df.groupby("quarter")["return"].mean() * 100

    # Heatmap data (month x year)
    heatmap = monthly_perf.pivot(index="year", columns="month", values="monthly_pct").round(2)

    return {
        "monthly_avg": avg_by_month.sort_values("month"),
        "best_month": best_month,
        "worst_month": worst_month,
        "current_month": current_month,
        "current_month_name": MONTH_NAMES[current_month-1],
        "current_stats": current_stats.iloc[0] if not current_stats.empty else None,
        "dow_perf": dow_perf,
        "quarterly": quarterly,
        "heatmap": heatmap,
        "years_analyzed": years,
    }
