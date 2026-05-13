import pandas as pd
import numpy as np


# ── Support & Resistance ───────────────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, window: int = 20, n_levels: int = 5):
    """
    Detect S/R levels using local minima/maxima + simple clustering.
    Returns two lists: support_levels, resistance_levels
    """
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    pivots = []
    for i in range(window, len(df) - window):
        # Local high
        if highs[i] == max(highs[i - window: i + window]):
            pivots.append(("R", highs[i]))
        # Local low
        if lows[i] == min(lows[i - window: i + window]):
            pivots.append(("S", lows[i]))

    if not pivots:
        return [], []

    # Cluster nearby levels (within 0.3% of each other)
    def cluster(levels, tol=0.003):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for v in levels[1:]:
            if abs(v - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [np.mean(c) for c in clusters]

    current = closes[-1]
    support_raw = [v for t, v in pivots if t == "S"]
    resist_raw = [v for t, v in pivots if t == "R"]

    supports = sorted(cluster(support_raw), reverse=True)[:n_levels]
    resistances = sorted(cluster(resist_raw))[:n_levels]

    return supports, resistances


# ── Fibonacci Retracement ──────────────────────────────────────────────────────

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_COLORS = ["#f0c040", "#40c0f0", "#00c853", "#ffffff", "#f040f0", "#ff6b00", "#f0c040"]


def fibonacci_levels(df: pd.DataFrame, lookback: int = 100):
    """
    Calculate Fibonacci retracement levels over the last `lookback` candles.
    Returns a dict {level_pct: price}
    """
    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    diff = high - low

    # Determine trend direction
    first_close = recent["Close"].iloc[0]
    last_close = recent["Close"].iloc[-1]
    uptrend = last_close > first_close

    levels = {}
    for fib in FIB_LEVELS:
        if uptrend:
            price = high - diff * fib
        else:
            price = low + diff * fib
        levels[fib] = round(price, 2)

    return levels, uptrend


# ── Chart Patterns ─────────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame, window: int = 50):
    """
    Detect basic chart patterns in the last `window` candles.
    Returns list of detected pattern names.
    """
    recent = df.tail(window).copy()
    closes = recent["Close"].values
    patterns = []

    if len(closes) < 20:
        return patterns

    # Double Top: two peaks close in value with a valley in between
    peaks = _find_peaks(closes, 10)
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if abs(closes[p1] - closes[p2]) / closes[p1] < 0.015:
            valley_min = min(closes[p1:p2])
            if valley_min < min(closes[p1], closes[p2]) * 0.99:
                patterns.append("Double Top 🔴")

    # Double Bottom: two troughs close in value
    troughs = _find_troughs(closes, 10)
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if abs(closes[t1] - closes[t2]) / closes[t1] < 0.015:
            peak_max = max(closes[t1:t2])
            if peak_max > max(closes[t1], closes[t2]) * 1.01:
                patterns.append("Double Bottom 🟢")

    # Uptrend: price consistently above 20-period MA
    ma20 = pd.Series(closes).rolling(20).mean().dropna().values
    if len(ma20) > 5:
        if all(closes[-(len(ma20)):] > ma20 * 0.998):
            patterns.append("Uptrend Channel 📈")
        elif all(closes[-(len(ma20)):] < ma20 * 1.002):
            patterns.append("Downtrend Channel 📉")

    # Bullish/Bearish engulfing (last 2 candles)
    opens = recent["Open"].values
    if len(opens) >= 2:
        c1_bear = opens[-2] > closes[-2]  # prev candle bearish
        c2_bull = closes[-1] > opens[-1]  # curr candle bullish
        if c1_bear and c2_bull and closes[-1] > opens[-2] and opens[-1] < closes[-2]:
            patterns.append("Bullish Engulfing 🟢")

        c1_bull = closes[-2] > opens[-2]
        c2_bear = opens[-1] > closes[-1]
        if c1_bull and c2_bear and closes[-1] < opens[-2] and opens[-1] > closes[-2]:
            patterns.append("Bearish Engulfing 🔴")

    return patterns if patterns else ["Nuk u gjet pattern i qartë ⚪"]


def _find_peaks(arr, min_dist=5):
    peaks = []
    for i in range(min_dist, len(arr) - min_dist):
        if arr[i] == max(arr[i - min_dist: i + min_dist + 1]):
            peaks.append(i)
    return peaks


def _find_troughs(arr, min_dist=5):
    troughs = []
    for i in range(min_dist, len(arr) - min_dist):
        if arr[i] == min(arr[i - min_dist: i + min_dist + 1]):
            troughs.append(i)
    return troughs
