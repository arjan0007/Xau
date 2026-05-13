"""
Trading Journal with SQLite backend.
Stores manual trades, calculates P&L, statistics.
"""
import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "C:/Users/User/Desktop/xauusd/trading_journal.db"


def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_open   TEXT NOT NULL,
            date_close  TEXT,
            pair        TEXT DEFAULT 'XAUUSD',
            direction   TEXT NOT NULL,
            lot_size    REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss   REAL,
            take_profit REAL,
            exit_price  REAL,
            pnl_usd     REAL,
            pips        REAL,
            status      TEXT DEFAULT 'OPEN',
            setup       TEXT,
            notes       TEXT,
            screenshot  TEXT,
            emotion     TEXT,
            rating      INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS journal_notes (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            date    TEXT NOT NULL,
            note    TEXT NOT NULL,
            mood    TEXT,
            market  TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_trade(direction: str, lot_size: float, entry_price: float,
              stop_loss: float = None, take_profit: float = None,
              setup: str = "", notes: str = "", emotion: str = "Neutral",
              rating: int = 3) -> int:
    """Add a new open trade. Returns trade ID."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (date_open, direction, lot_size, entry_price,
                            stop_loss, take_profit, setup, notes, emotion, rating, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"),
          direction, lot_size, entry_price,
          stop_loss, take_profit, setup, notes, emotion, rating))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def close_trade(trade_id: int, exit_price: float, notes: str = "") -> dict:
    """Close an open trade and calculate P&L."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM trades WHERE id=?", (trade_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": "Trade nuk u gjet"}

    cols = [d[0] for d in c.description]
    trade = dict(zip(cols, row))

    entry = trade["entry_price"]
    lot   = trade["lot_size"]
    direction = trade["direction"]

    pips = (exit_price - entry) / 0.01 if direction == "BUY" else (entry - exit_price) / 0.01
    pnl_usd = pips * lot  # simplified pip value

    existing_notes = trade.get("notes", "") or ""
    combined_notes = f"{existing_notes}\nClose: {notes}".strip() if notes else existing_notes

    c.execute("""
        UPDATE trades SET date_close=?, exit_price=?, pnl_usd=?, pips=?,
                          status='CLOSED', notes=?
        WHERE id=?
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"),
          exit_price, round(pnl_usd, 2), round(pips, 1),
          combined_notes, trade_id))
    conn.commit()
    conn.close()

    return {"trade_id": trade_id, "pnl_usd": round(pnl_usd, 2),
            "pips": round(pips, 1), "result": "WIN" if pnl_usd > 0 else "LOSS"}


def get_all_trades(status: str = None) -> pd.DataFrame:
    """Get all trades, optionally filtered by status."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM trades"
    params = []
    if status:
        query += " WHERE status=?"
        params.append(status)
    query += " ORDER BY date_open DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_statistics() -> dict:
    """Calculate overall trading statistics."""
    df = get_all_trades("CLOSED")
    if df.empty:
        return {"total": 0}

    wins   = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] <= 0]
    total  = len(df)

    win_rate   = round(len(wins) / total * 100, 1) if total > 0 else 0
    total_pnl  = round(df["pnl_usd"].sum(), 2)
    avg_win    = round(wins["pnl_usd"].mean(), 2) if not wins.empty else 0
    avg_loss   = round(losses["pnl_usd"].mean(), 2) if not losses.empty else 0
    best_trade = round(df["pnl_usd"].max(), 2)
    worst_trade= round(df["pnl_usd"].min(), 2)
    avg_pips   = round(df["pips"].mean(), 1) if "pips" in df.columns else 0

    # Monthly P&L
    df["month"] = pd.to_datetime(df["date_open"], errors="coerce").dt.strftime("%Y-%m")
    monthly = df.groupby("month")["pnl_usd"].sum().reset_index()

    # Streak
    results = df.sort_values("date_close")["pnl_usd"].apply(lambda x: 1 if x > 0 else -1).tolist()
    cur_streak = max_streak = 0
    if results:
        cur = results[-1]
        cur_streak = 0
        for r in reversed(results):
            if r == cur:
                cur_streak += 1
            else:
                break
        max_streak = max(
            len(list(g)) for _, g in
            __import__("itertools").groupby(results)
        )

    gross_profit = wins["pnl_usd"].sum() if not wins.empty else 0
    gross_loss   = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "total": total, "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "total_pnl": total_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "best_trade": best_trade, "worst_trade": worst_trade,
        "avg_pips": avg_pips, "profit_factor": pf,
        "current_streak": cur_streak, "max_streak": max_streak,
        "monthly": monthly,
    }


def add_note(note: str, mood: str = "Neutral", market: str = ""):
    """Add a journal note."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO journal_notes (date, note, mood, market) VALUES (?,?,?,?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M"), note, mood, market))
    conn.commit()
    conn.close()


def get_notes(limit: int = 20) -> pd.DataFrame:
    """Get recent journal notes."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM journal_notes ORDER BY date DESC LIMIT ?",
        conn, params=[limit],
    )
    conn.close()
    return df


def delete_trade(trade_id: int):
    """Delete a trade by ID."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM trades WHERE id=?", (trade_id,))
    conn.commit()
    conn.close()
