import pandas as pd
import yfinance as yf


def fetch_data(interval: str, period: str) -> pd.DataFrame:
    """Fetch XAUUSD/Gold OHLC data for model training and inference."""
    needs_4h_resample = interval == "4h"
    fetch_interval = "60m" if needs_4h_resample else interval
    fetch_period = "120d" if needs_4h_resample else period

    df = pd.DataFrame()
    for symbol in ("GC=F", "XAUUSD=X"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=fetch_period, interval=fetch_interval)
            if not df.empty and len(df) >= 50:
                break
        except Exception:
            continue

    if df.empty:
        return df

    df.index = pd.to_datetime(df.index)

    if needs_4h_resample:
        df = df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = (df["High"] - df["Low"]).rolling(3, min_periods=1).mean() * 1000

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
