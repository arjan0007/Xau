"""
Auto-Learning Bot Brain — tracks every prediction, evaluates outcomes,
and feeds back into the model with weighted retraining.

Reinforcement-style loop:
1. log_prediction()  →  store every BUY/SELL signal with entry, SL, TP, features
2. evaluate_pending() →  for each pending prediction, check if SL/TP hit using
                         subsequent candles. Mark WIN/LOSS/UNDECIDED.
3. get_performance() →  real-time win-rate, profit-factor, recent streak.
4. auto_tune()       →  adjust ML/confluence thresholds up/down based on
                         recent precision. If precision drops below target,
                         tighten gates; if too few signals, loosen them.
"""
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

DB_PATH = "C:/Users/User/Desktop/xauusd/bot_brain.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mode TEXT,
        interval TEXT,
        signal TEXT,
        confidence REAL,
        ml_proba REAL,
        confluence REAL,
        entry REAL,
        sl REAL,
        tp1 REAL,
        tp2 REAL,
        atr REAL,
        features TEXT,
        outcome TEXT DEFAULT 'PENDING',
        exit_price REAL,
        exit_ts TEXT,
        pnl_pips REAL,
        bars_to_exit INTEGER
    );
    CREATE TABLE IF NOT EXISTS thresholds_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mode TEXT,
        ml_threshold REAL,
        confluence_threshold REAL,
        recent_precision REAL,
        recent_count INTEGER,
        action TEXT
    );
    """)
    c.commit()
    c.close()


def log_prediction(mode: str, interval: str, signal: str, confidence: float,
                   ml_proba: float, confluence: float,
                   entry: float, sl: float, tp1: float, tp2: float,
                   atr: float, features: dict = None) -> int:
    """Record a new BUY/SELL prediction. Returns prediction ID."""
    if signal not in ("BUY", "SELL"):
        return -1
    init_db()
    c = _conn()
    cur = c.execute(
        """INSERT INTO predictions
           (ts, mode, interval, signal, confidence, ml_proba, confluence,
            entry, sl, tp1, tp2, atr, features)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(timespec="seconds"),
         mode, interval, signal, confidence, ml_proba, confluence,
         entry, sl, tp1, tp2, atr, json.dumps(features or {})),
    )
    pid = cur.lastrowid
    c.commit()
    c.close()
    return pid


def has_recent(signal: str, within_minutes: int = 30) -> bool:
    """Skip logging dupes when sidebar re-runs fire same signal twice."""
    init_db()
    c = _conn()
    row = c.execute(
        """SELECT ts FROM predictions
           WHERE signal=? AND outcome='PENDING'
           ORDER BY id DESC LIMIT 1""",
        (signal,),
    ).fetchone()
    c.close()
    if not row:
        return False
    last = datetime.fromisoformat(row[0])
    return (datetime.utcnow() - last).total_seconds() / 60 < within_minutes


def evaluate_pending(price_df: pd.DataFrame, max_bars_lookahead: int = 24) -> dict:
    """
    Walk through pending predictions and check if SL or TP was hit using
    the recent OHLC data. Updates each row with outcome.

    Returns: {"evaluated": N, "wins": W, "losses": L, "still_pending": P}
    """
    init_db()
    if price_df is None or price_df.empty:
        return {"evaluated": 0, "wins": 0, "losses": 0, "still_pending": 0}

    df = price_df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    c = _conn()
    rows = c.execute(
        "SELECT id, ts, signal, entry, sl, tp1 FROM predictions WHERE outcome='PENDING'"
    ).fetchall()

    wins = losses = still = 0
    for pid, ts_str, sig, entry, sl, tp1 in rows:
        ts = pd.to_datetime(ts_str)
        # candles strictly after the prediction timestamp
        future = df[df.index > ts].head(max_bars_lookahead)
        if future.empty:
            still += 1
            continue

        outcome = exit_p = exit_ts = bars = None
        for i, (idx, row) in enumerate(future.iterrows(), start=1):
            hi, lo = row["High"], row["Low"]
            if sig == "BUY":
                if lo <= sl:
                    outcome, exit_p, exit_ts, bars = "LOSS", sl, idx, i; break
                if hi >= tp1:
                    outcome, exit_p, exit_ts, bars = "WIN", tp1, idx, i; break
            else:  # SELL
                if hi >= sl:
                    outcome, exit_p, exit_ts, bars = "LOSS", sl, idx, i; break
                if lo <= tp1:
                    outcome, exit_p, exit_ts, bars = "WIN", tp1, idx, i; break

        if outcome is None:
            # Ran out of bars; if max_bars exceeded, mark as expired/timeout LOSS
            if len(future) >= max_bars_lookahead:
                outcome = "TIMEOUT"
                exit_p = float(future["Close"].iloc[-1])
                exit_ts = future.index[-1]
                bars = len(future)
            else:
                still += 1
                continue

        pnl_pips = (exit_p - entry) / 0.01 if sig == "BUY" else (entry - exit_p) / 0.01
        c.execute(
            """UPDATE predictions
               SET outcome=?, exit_price=?, exit_ts=?, pnl_pips=?, bars_to_exit=?
               WHERE id=?""",
            (outcome, float(exit_p), str(exit_ts), float(pnl_pips), int(bars), pid),
        )
        if outcome == "WIN":
            wins += 1
        else:
            losses += 1

    c.commit()
    c.close()
    return {"evaluated": wins + losses, "wins": wins, "losses": losses, "still_pending": still}


def get_performance(window: int = 50) -> dict:
    """Recent performance over last `window` evaluated predictions."""
    init_db()
    c = _conn()
    rows = c.execute(
        """SELECT outcome, pnl_pips, signal, confidence, bars_to_exit
           FROM predictions
           WHERE outcome IN ('WIN','LOSS','TIMEOUT')
           ORDER BY id DESC LIMIT ?""",
        (window,),
    ).fetchall()
    total_count = c.execute(
        "SELECT COUNT(*) FROM predictions WHERE outcome!='PENDING'"
    ).fetchone()[0]
    pending = c.execute(
        "SELECT COUNT(*) FROM predictions WHERE outcome='PENDING'"
    ).fetchone()[0]
    c.close()

    if not rows:
        return {"total_evaluated": 0, "pending": pending, "win_rate": 0,
                "wins": 0, "losses": 0, "avg_bars": 0, "profit_factor": 0,
                "lifetime_evaluated": total_count}

    wins = sum(1 for r in rows if r[0] == "WIN")
    losses = sum(1 for r in rows if r[0] in ("LOSS", "TIMEOUT"))
    n = wins + losses
    win_rate = round(wins / n * 100, 1) if n else 0
    gross_win  = sum(r[1] for r in rows if r[1] and r[1] > 0)
    gross_loss = abs(sum(r[1] for r in rows if r[1] and r[1] < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
    avg_bars = round(np.mean([r[4] for r in rows if r[4]]), 1) if any(r[4] for r in rows) else 0
    streak = _current_streak(rows)
    return {
        "total_evaluated": n, "pending": pending,
        "win_rate": win_rate, "wins": wins, "losses": losses,
        "avg_bars": avg_bars, "profit_factor": pf,
        "current_streak": streak, "lifetime_evaluated": total_count,
    }


def _current_streak(rows):
    if not rows:
        return 0
    first = "W" if rows[0][0] == "WIN" else "L"
    s = 0
    for r in rows:
        kind = "W" if r[0] == "WIN" else "L"
        if kind == first:
            s += 1
        else:
            break
    return s if first == "W" else -s


def auto_tune(target_precision: float = 0.70,
              min_signals_per_day: float = 1.0,
              window: int = 30) -> Optional[dict]:
    """
    Adjust thresholds based on recent performance:
    - If precision << target → raise ML threshold (be pickier)
    - If precision >> target & few signals → lower ML threshold (be more active)
    Returns adjustment dict or None if no change.
    """
    perf = get_performance(window=window)
    if perf["total_evaluated"] < 10:
        return None  # too few samples

    import model as ml_mod
    current_ml = ml_mod.HIGH_CONF_THRESHOLD
    current_conf = ml_mod.CONFLUENCE_THRESHOLD
    new_ml, new_conf, action = current_ml, current_conf, "hold"

    precision = perf["win_rate"] / 100

    # Estimate signals/day from evaluated window
    if precision < target_precision - 0.05:
        # Tighten — too many losses
        new_ml = min(0.85, current_ml + 0.03)
        new_conf = min(6, current_conf + 1)
        action = "tighten"
    elif precision > target_precision + 0.10:
        # Loosen — being too picky, missing trades
        new_ml = max(0.40, current_ml - 0.03)
        new_conf = max(1, current_conf - 1)
        action = "loosen"

    if action == "hold":
        return None

    ml_mod.HIGH_CONF_THRESHOLD = new_ml
    ml_mod.CONFLUENCE_THRESHOLD = new_conf

    init_db()
    c = _conn()
    c.execute(
        """INSERT INTO thresholds_history
           (ts, mode, ml_threshold, confluence_threshold,
            recent_precision, recent_count, action)
           VALUES (?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(timespec="seconds"),
         ml_mod.ACTIVE_MODE, new_ml, new_conf, precision,
         perf["total_evaluated"], action),
    )
    c.commit()
    c.close()
    return {
        "action": action,
        "old_ml": current_ml, "new_ml": new_ml,
        "old_conf": current_conf, "new_conf": new_conf,
        "precision": precision, "samples": perf["total_evaluated"],
    }


def get_recent_predictions(limit: int = 20) -> pd.DataFrame:
    init_db()
    c = _conn()
    df = pd.read_sql_query(
        """SELECT id, ts, mode, interval, signal, confidence, entry, sl, tp1,
                  outcome, exit_price, pnl_pips, bars_to_exit
           FROM predictions ORDER BY id DESC LIMIT ?""",
        c, params=[limit],
    )
    c.close()
    return df


def get_training_feedback(min_samples: int = 20) -> Optional[pd.DataFrame]:
    """
    Return WIN/LOSS rows with stored features so the next training pass
    can up-weight winners and down-weight losers (or feed them as extra
    labeled samples). Returns None when not enough data yet.
    """
    init_db()
    c = _conn()
    df = pd.read_sql_query(
        """SELECT signal, features, outcome, pnl_pips
           FROM predictions
           WHERE outcome IN ('WIN','LOSS','TIMEOUT') AND features IS NOT NULL""",
        c,
    )
    c.close()
    if len(df) < min_samples:
        return None
    return df
