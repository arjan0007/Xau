"""
COT (Commitment of Traders) Report for Gold Futures.
Data source: CFTC public data (free, no API key).
Gold futures COT data is in the "Disaggregated" report under "GOLD - COMMODITY EXCHANGE INC."
"""
import requests
import pandas as pd
import io
from datetime import datetime


CFTC_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"
CFTC_CURRENT = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2024.zip"

# Simpler: use the weekly CSV from CFTC
COT_CSV_URL = "https://www.cftc.gov/sites/default/files/files/dea/history/fut_fin_xls_2024.zip"

GOLD_IDENTIFIER = "GOLD"


def fetch_cot_data() -> pd.DataFrame | None:
    """
    Fetch the latest COT data for gold from CFTC.
    Falls back to simulated recent data if fetch fails.
    """
    try:
        # Try CFTC legacy format CSV
        url = "https://www.cftc.gov/dea/newcot/fut_disagg_txt.zip"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            import zipfile, io as _io
            with zipfile.ZipFile(_io.BytesIO(r.content)) as z:
                fname = [f for f in z.namelist() if f.endswith(".txt") or f.endswith(".csv")]
                if fname:
                    with z.open(fname[0]) as f:
                        df = pd.read_csv(f, low_memory=False)
                        gold_df = df[df["Market_and_Exchange_Names"].str.contains("GOLD", na=False)]
                        if not gold_df.empty:
                            return _parse_cot(gold_df)
    except Exception:
        pass

    # Return simulated/cached data structure for demo
    return _simulated_cot()


def _parse_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Parse relevant columns from CFTC disaggregated data."""
    cols_map = {
        "Report_Date_as_MM_DD_YYYY": "date",
        "Prod_Merc_Positions_Long_All": "commercial_long",
        "Prod_Merc_Positions_Short_All": "commercial_short",
        "Money_Manager_Positions_Long_All": "noncommercial_long",
        "Money_Manager_Positions_Short_All": "noncommercial_short",
        "Other_Rept_Positions_Long_All": "other_long",
        "Other_Rept_Positions_Short_All": "other_short",
    }
    available = {k: v for k, v in cols_map.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date")

    if "noncommercial_long" in result.columns and "noncommercial_short" in result.columns:
        result["net_noncommercial"] = result["noncommercial_long"] - result["noncommercial_short"]
    if "commercial_long" in result.columns and "commercial_short" in result.columns:
        result["net_commercial"] = result["commercial_long"] - result["commercial_short"]

    return result.tail(52)  # last year


def _simulated_cot() -> pd.DataFrame:
    """Return simulated COT structure for display when CFTC fetch fails."""
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=52, freq="W")
    np.random.seed(42)
    base_nc_long  = 180000
    base_nc_short = 70000

    nc_long  = base_nc_long  + np.cumsum(np.random.randn(52) * 3000)
    nc_short = base_nc_short + np.cumsum(np.random.randn(52) * 2000)
    c_long   = 100000 + np.cumsum(np.random.randn(52) * 1500)
    c_short  = 180000 + np.cumsum(np.random.randn(52) * 1500)

    df = pd.DataFrame({
        "date": dates,
        "noncommercial_long":  nc_long.astype(int),
        "noncommercial_short": nc_short.astype(int),
        "commercial_long":     c_long.astype(int),
        "commercial_short":    c_short.astype(int),
    })
    df["net_noncommercial"] = df["noncommercial_long"] - df["noncommercial_short"]
    df["net_commercial"]    = df["commercial_long"] - df["commercial_short"]
    return df


def cot_summary(df: pd.DataFrame) -> dict:
    """Summarize current COT positioning."""
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    nc_net = int(latest.get("net_noncommercial", 0))
    nc_prev = int(prev.get("net_noncommercial", 0))
    c_net  = int(latest.get("net_commercial", 0))

    nc_change = nc_net - nc_prev
    nc_long   = int(latest.get("noncommercial_long", 0))
    nc_short  = int(latest.get("noncommercial_short", 0))

    # Sentiment from COT
    if nc_net > 50000:
        sentiment = "🟢 Bullish — Spekulatorët janë NET LONG"
    elif nc_net < -10000:
        sentiment = "🔴 Bearish — Spekulatorët janë NET SHORT"
    else:
        sentiment = "⚪ Neutral"

    # Commercials (hedgers) — inverse indicator
    if c_net < -100000:
        commercial_view = "⚠ Hedgers janë shumë SHORT — kujdes me BUY"
    else:
        commercial_view = "➡ Hedgers normal"

    return {
        "date": str(latest.get("date", ""))[:10],
        "nc_long": nc_long,
        "nc_short": nc_short,
        "nc_net": nc_net,
        "nc_change_wk": nc_change,
        "commercial_net": c_net,
        "sentiment": sentiment,
        "commercial_view": commercial_view,
        "is_simulated": True,
    }
