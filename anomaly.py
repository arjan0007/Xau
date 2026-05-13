import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    Detect anomalous price movements using Isolation Forest.
    Returns df with 'anomaly' (bool) and 'anomaly_score' columns.
    """
    df = df.copy()

    # Features for anomaly detection
    df["ret"] = df["Close"].pct_change()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
    df["ret_z"] = (df["ret"] - df["ret"].rolling(20).mean()) / (df["ret"].rolling(20).std() + 1e-9)
    df["range_z"] = (df["range_pct"] - df["range_pct"].rolling(20).mean()) / (df["range_pct"].rolling(20).std() + 1e-9)

    feat_cols = ["ret_z", "range_z", "vol_z"]
    clean = df[feat_cols].dropna()

    if len(clean) < 30:
        df["anomaly"] = False
        df["anomaly_score"] = 0.0
        return df

    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)
    scores = iso.score_samples(X)

    df.loc[clean.index, "anomaly"] = preds == -1
    df.loc[clean.index, "anomaly_score"] = -scores  # higher = more anomalous
    df["anomaly"] = df["anomaly"].fillna(False)
    df["anomaly_score"] = df["anomaly_score"].fillna(0.0)

    return df


def zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Simple Z-score based anomaly detection on returns."""
    df = df.copy()
    ret = df["Close"].pct_change()
    mu = ret.rolling(20).mean()
    sigma = ret.rolling(20).std()
    df["zscore"] = (ret - mu) / (sigma + 1e-9)
    df["zscore_anomaly"] = df["zscore"].abs() > threshold
    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Get summary of recent anomalies."""
    if "anomaly" not in df.columns:
        df = detect_anomalies(df)
    if "zscore" not in df.columns:
        df = zscore_anomalies(df)

    recent = df.tail(50)
    anomalies = recent[recent["anomaly"] == True]
    zscore_anom = recent[recent.get("zscore_anomaly", pd.Series([False]*len(recent), index=recent.index))]

    latest_zscore = float(df["zscore"].iloc[-1]) if "zscore" in df.columns else 0.0
    latest_score = float(df["anomaly_score"].iloc[-1]) if "anomaly_score" in df.columns else 0.0

    is_anomaly_now = bool(df["anomaly"].iloc[-1]) if "anomaly" in df.columns else False
    is_zscore_now = abs(latest_zscore) > 3.0

    return {
        "is_anomaly_now": is_anomaly_now or is_zscore_now,
        "anomaly_score": round(latest_score, 3),
        "zscore": round(latest_zscore, 2),
        "anomalies_last_50": len(anomalies),
        "recent_anomalies": anomalies.tail(5)[["Close","ret","range_pct"]].round(4) if not anomalies.empty else pd.DataFrame(),
        "severity": _severity(latest_score, abs(latest_zscore)),
    }


def _severity(iso_score: float, zscore: float) -> str:
    if iso_score > 0.6 or zscore > 4.0:
        return "🔴 E Lartë"
    if iso_score > 0.4 or zscore > 3.0:
        return "🟡 Mesatare"
    return "🟢 Normale"
