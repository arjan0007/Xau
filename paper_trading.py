"""
Paper Trading Engine — virtual account that auto-executes every AI signal
so the user can measure real performance without risking money.

Each new actionable signal (BUY/SELL with high confidence) opens a virtual
position with the same SL/TP and position sizing the user would use live.
Subsequent candles close the position on SL/TP hit. Stats are tracked in
a SQLite DB so the user sees an honest, audited track record.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

DB_PATH = "C:/Users/User/Desktop/xauusd/paper_trading.db"
INITIAL_BALANCE = 10000.0


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        balance REAL NOT NULL,
        equity REAL NOT NULL,
        starting_balance REAL NOT NULL,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        opened_at TEXT NOT NULL,
        signal TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp1 REAL NOT NULL,
        tp2 REAL,
        lot REAL NOT NULL,
        risk_usd REAL,
        confidence REAL,
        mode TEXT,
        session TEXT,
        status TEXT DEFAULT 'OPEN',
        closed_at TEXT,
        exit_price REAL,
        exit_reason TEXT,
        pnl_usd REAL,
        pnl_pips REAL,
        bars_open INTEGER
    );
    """)
    c.execute("INSERT OR IGNORE INTO paper_account (id, balance, equity, starting_balance, last_updated) VALUES (1, ?, ?, ?, ?)",
              (INITIAL_BALANCE, INITIAL_BALANCE, INITIAL_BALANCE,
               datetime.utcnow().isoformat(timespec="seconds")))
    c.commit()
    c.close()


def get_account() -> dict:
    init_db()
    c = _conn()
    row = c.execute("SELECT balance, equity, starting_balance, last_updated FROM paper_account WHERE id=1").fetchone()
    c.close()
    if not row:
        return {"balance": INITIAL_BALANCE, "equity": INITIAL_BALANCE,
                "starting": INITIAL_BALANCE, "last_updated": ""}
    return {"balance": float(row[0]), "equity": float(row[1]),
            "starting": float(row[2]), "last_updated": row[3] or ""}


def reset_account(starting: float = INITIAL_BALANCE):
    init_db()
    c = _conn()
    c.execute("UPDATE paper_account SET balance=?, equity=?, starting_balance=?, last_updated=? WHERE id=1",
              (starting, starting, starting, datetime.utcnow().isoformat(timespec="seconds")))
    c.execute("DELETE FROM paper_trades")
    c.commit()
    c.close()


def open_paper_trade(signal: str, entry: float, sl: float, tp1: float,
                     tp2: float, lot: float, risk_usd: float,
                     confidence: float, mode: str, session: str) -> int:
    if signal not in ("BUY", "SELL"):
        return -1
    init_db()
    c = _conn()
    cur = c.execute("""INSERT INTO paper_trades
        (opened_at, signal, entry, sl, tp1, tp2, lot, risk_usd, confidence, mode, session)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(timespec="seconds"),
         signal, entry, sl, tp1, tp2, lot, risk_usd, confidence, mode, session))
    pid = cur.lastrowid
    c.commit()
    c.close()
    return pid


def has_recent_paper_trade(signal: str, within_minutes: int = 30) -> bool:
    """Prevent duplicate paper trades when streamlit reruns."""
    init_db()
    c = _conn()
    row = c.execute(
        """SELECT opened_at FROM paper_trades
           WHERE signal=? AND status='OPEN' ORDER BY id DESC LIMIT 1""",
        (signal,)).fetchone()
    c.close()
    if not row:
        return False
    last = datetime.fromisoformat(row[0])
    return (datetime.utcnow() - last).total_seconds() / 60 < within_minutes


def evaluate_open_trades(df: pd.DataFrame, max_bars: int = 48) -> dict:
    """Walk every open paper trade through recent candles and close it
    when SL or TP hits. Returns summary of evaluations."""
    init_db()
    if df is None or df.empty:
        return {"closed": 0, "wins": 0, "losses": 0}
    dfi = df.copy()
    dfi.index = pd.to_datetime(dfi.index)
    if dfi.index.tz is not None:
        dfi.index = dfi.index.tz_convert("UTC").tz_localize(None)

    c = _conn()
    rows = c.execute("""SELECT id, opened_at, signal, entry, sl, tp1, tp2, lot
                        FROM paper_trades WHERE status='OPEN'""").fetchall()
    closed = wins = losses = 0
    pnl_total = 0.0
    for pid, ts_str, sig, entry, sl, tp1, tp2, lot in rows:
        ts = pd.to_datetime(ts_str)
        fut = dfi[dfi.index > ts].head(max_bars)
        if fut.empty:
            continue
        exit_price = exit_reason = exit_ts = bars = None
        for i, (idx, row) in enumerate(fut.iterrows(), start=1):
            hi, lo = row["High"], row["Low"]
            if sig == "BUY":
                if lo <= sl:
                    exit_price, exit_reason, exit_ts, bars = sl, "SL_HIT", idx, i; break
                if hi >= tp1:
                    exit_price, exit_reason, exit_ts, bars = tp1, "TP1_HIT", idx, i; break
            else:
                if hi >= sl:
                    exit_price, exit_reason, exit_ts, bars = sl, "SL_HIT", idx, i; break
                if lo <= tp1:
                    exit_price, exit_reason, exit_ts, bars = tp1, "TP1_HIT", idx, i; break
        if exit_price is None:
            if len(fut) >= max_bars:
                exit_price = float(fut["Close"].iloc[-1])
                exit_reason = "TIMEOUT"
                exit_ts = fut.index[-1]
                bars = len(fut)
            else:
                continue
        price_diff = (exit_price - entry) if sig == "BUY" else (entry - exit_price)
        pnl_usd  = price_diff * lot * 100   # gold: 1 lot = 100 oz, $1 move = $100/lot
        pnl_pips = price_diff / 0.01
        c.execute("""UPDATE paper_trades SET status='CLOSED', closed_at=?,
                     exit_price=?, exit_reason=?, pnl_usd=?, pnl_pips=?, bars_open=?
                     WHERE id=?""",
                  (str(exit_ts), float(exit_price), exit_reason,
                   round(pnl_usd, 2), round(pnl_pips, 1), int(bars), pid))
        pnl_total += pnl_usd
        closed += 1
        if pnl_usd > 0: wins += 1
        else: losses += 1

    if closed > 0:
        cur_bal = c.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()[0]
        new_bal = cur_bal + pnl_total
        c.execute("UPDATE paper_account SET balance=?, equity=?, last_updated=? WHERE id=1",
                  (new_bal, new_bal, datetime.utcnow().isoformat(timespec="seconds")))
    c.commit()
    c.close()
    return {"closed": closed, "wins": wins, "losses": losses, "pnl_total": round(pnl_total, 2)}


def get_open_trades() -> pd.DataFrame:
    init_db()
    c = _conn()
    df = pd.read_sql_query(
        "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC", c)
    c.close()
    return df


def get_closed_trades(limit: int = 100) -> pd.DataFrame:
    init_db()
    c = _conn()
    df = pd.read_sql_query(
        "SELECT * FROM paper_trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ?",
        c, params=[limit])
    c.close()
    return df


def get_stats() -> dict:
    init_db()
    closed = get_closed_trades(limit=10000)
    acc = get_account()
    if closed.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
                "max_dd_pct": 0, "best_trade": 0, "worst_trade": 0,
                "sharpe": 0, "return_pct": 0,
                "balance": acc["balance"], "starting": acc["starting"]}
    wins = closed[closed["pnl_usd"] > 0]
    losses = closed[closed["pnl_usd"] <= 0]
    total = len(closed)
    gp = wins["pnl_usd"].sum() if not wins.empty else 0
    gl = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
    pf = round(gp / gl, 2) if gl > 0 else float("inf")
    # Drawdown
    closed_sorted = closed.sort_values("closed_at")
    bal = acc["starting"]
    peak = bal; max_dd = 0
    eq_curve = []
    for p in closed_sorted["pnl_usd"]:
        bal += p
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100 if peak else 0
        if dd > max_dd: max_dd = dd
        eq_curve.append(bal)
    rets = closed_sorted["pnl_usd"] / acc["starting"]
    sharpe = round(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252), 2) if len(rets) > 1 else 0
    return {
        "trades": total, "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / total * 100, 1) if total else 0,
        "total_pnl": round(closed["pnl_usd"].sum(), 2),
        "profit_factor": pf,
        "avg_win": round(wins["pnl_usd"].mean(), 2) if not wins.empty else 0,
        "avg_loss": round(losses["pnl_usd"].mean(), 2) if not losses.empty else 0,
        "max_dd_pct": round(max_dd, 2),
        "best_trade": round(closed["pnl_usd"].max(), 2),
        "worst_trade": round(closed["pnl_usd"].min(), 2),
        "sharpe": sharpe,
        "return_pct": round((acc["balance"] - acc["starting"]) / acc["starting"] * 100, 2),
        "balance": acc["balance"], "starting": acc["starting"],
        "equity_curve": eq_curve,
    }


def stats_by_session(closed_df: pd.DataFrame = None) -> pd.DataFrame:
    if closed_df is None:
        closed_df = get_closed_trades(limit=10000)
    if closed_df.empty:
        return pd.DataFrame()
    g = closed_df.groupby("session").agg(
        trades=("id", "count"),
        wins=("pnl_usd", lambda s: (s > 0).sum()),
        pnl=("pnl_usd", "sum"),
    ).reset_index()
    g["win_rate"] = (g["wins"] / g["trades"] * 100).round(1)
    return g.sort_values("pnl", ascending=False)


def stats_by_day_of_week(closed_df: pd.DataFrame = None) -> pd.DataFrame:
    if closed_df is None:
        closed_df = get_closed_trades(limit=10000)
    if closed_df.empty:
        return pd.DataFrame()
    closed_df = closed_df.copy()
    closed_df["dow"] = pd.to_datetime(closed_df["opened_at"]).dt.day_name()
    g = closed_df.groupby("dow").agg(
        trades=("id", "count"),
        wins=("pnl_usd", lambda s: (s > 0).sum()),
        pnl=("pnl_usd", "sum"),
    ).reset_index()
    g["win_rate"] = (g["wins"] / g["trades"] * 100).round(1)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g["_order"] = g["dow"].apply(lambda d: order.index(d) if d in order else 99)
    return g.sort_values("_order").drop(columns=["_order"])
