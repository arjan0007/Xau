import yfinance as yf
import pandas as pd
import numpy as np


ASSETS = {
    "Gold (XAU)":    "GC=F",
    "DXY (Dollar)":  "DX-Y.NYB",
    "S&P 500":       "^GSPC",
    "10Y Yield":     "^TNX",
    "Silver":        "SI=F",
    "Oil (WTI)":     "CL=F",
    "VIX":           "^VIX",
    "EUR/USD":       "EURUSD=X",
}

DESCRIPTIONS = {
    "DXY (Dollar)":  "Korrelacion negativ — dollar i fortë → gold bie",
    "S&P 500":       "Kur S&P bie (fear) → gold ngrihet (safe haven)",
    "10Y Yield":     "Yield lart → gold bie (kostoja e mbajtjes rritet)",
    "Silver":        "Korrelacion pozitiv — lëvizin bashkë zakonisht",
    "Oil (WTI)":     "Korrelacion pozitiv — të dy janë commodity",
    "VIX":           "Fear index — VIX lart → gold ngrihet",
    "EUR/USD":       "EUR/USD lart → DXY bie → gold ngrihet",
}


def fetch_correlations(period: str = "1y") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch daily price data for all assets and compute correlation matrix.
    Returns: (returns_df, correlation_matrix)
    """
    closes = {}
    for name, ticker in ASSETS.items():
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
            if not df.empty:
                closes[name] = df["Close"]
        except Exception:
            continue

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    price_df = pd.DataFrame(closes).dropna()
    returns_df = price_df.pct_change().dropna()
    corr_matrix = returns_df.corr().round(3)

    return returns_df, corr_matrix


def gold_correlation_summary(corr_matrix: pd.DataFrame) -> list[dict]:
    """Get gold's correlation with each asset."""
    if corr_matrix.empty or "Gold (XAU)" not in corr_matrix.columns:
        return []

    results = []
    for asset in corr_matrix.columns:
        if asset == "Gold (XAU)":
            continue
        corr = corr_matrix.loc["Gold (XAU)", asset]
        strength = _corr_strength(corr)
        results.append({
            "asset": asset,
            "correlation": round(corr, 3),
            "strength": strength,
            "description": DESCRIPTIONS.get(asset, ""),
            "signal": _corr_signal(asset, corr),
        })

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def rolling_correlation(returns_df: pd.DataFrame,
                        window: int = 30) -> pd.DataFrame:
    """Compute rolling 30-day correlation of gold with other assets."""
    if "Gold (XAU)" not in returns_df.columns:
        return pd.DataFrame()

    gold = returns_df["Gold (XAU)"]
    rolling_corrs = {}
    for col in returns_df.columns:
        if col != "Gold (XAU)":
            rolling_corrs[col] = gold.rolling(window).corr(returns_df[col])

    return pd.DataFrame(rolling_corrs)


def _corr_strength(c: float) -> str:
    a = abs(c)
    if a >= 0.7:  return "Shumë i fortë"
    if a >= 0.5:  return "I fortë"
    if a >= 0.3:  return "Mesatar"
    return "I dobët"


def _corr_signal(asset: str, corr: float) -> str:
    """What does this correlation mean right now for gold."""
    if asset == "DXY (Dollar)":
        return "🔴 Dollar lart → Gold bie" if corr < -0.3 else "➡ Korrelacion i dobët"
    if asset == "VIX":
        return "🟢 Fear lart → Gold ngrihet" if corr > 0.3 else "➡ Korrelacion i dobët"
    if asset == "S&P 500":
        return "🔴 Risk-on → Gold bie" if corr < -0.3 else "🟢 Risk-off → Gold ngrihet" if corr > 0.3 else "➡ Neutral"
    if asset == "10Y Yield":
        return "🔴 Yield lart → Gold bie" if corr < -0.3 else "➡ Korrelacion i dobët"
    return f"{'🟢' if corr > 0 else '🔴'} Korrelacion {'+' if corr > 0 else ''}{corr:.2f}"
