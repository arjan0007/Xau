"""
Position Monitor & Early Exit Detector

Tracks active trades and continuously scans for "invalidation signals" —
patterns that suggest the trade is about to fail — so the user can exit
manually BEFORE the SL is hit.

The detector runs the same indicators the model uses, but interprets them
from the perspective of an OPEN POSITION: did the setup that justified
entry now reverse? If yes → DANGER → suggest early exit.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

DB_PATH = "C:/Users/User/Desktop/xauusd/active_positions.db"


# ════════════════════════════════════════════════════════════════════════════
# DB: store user's open positions
# ════════════════════════════════════════════════════════════════════════════
def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        opened_at TEXT NOT NULL,
        signal TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp1 REAL NOT NULL,
        tp2 REAL,
        lot REAL DEFAULT 0.1,
        atr_at_entry REAL,
        status TEXT DEFAULT 'OPEN',
        closed_at TEXT,
        exit_price REAL,
        exit_reason TEXT,
        pnl_usd REAL,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS alerts_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER,
        ts TEXT NOT NULL,
        severity TEXT,
        signal TEXT,
        score INTEGER,
        reasons TEXT,
        action_recommended TEXT,
        acknowledged INTEGER DEFAULT 0
    );
    """)
    c.commit()
    c.close()


def open_position(signal: str, entry: float, sl: float, tp1: float,
                  tp2: float = None, lot: float = 0.1,
                  atr_at_entry: float = 5.0, notes: str = "") -> int:
    init_db()
    c = _conn()
    cur = c.execute(
        """INSERT INTO positions
           (opened_at, signal, entry, sl, tp1, tp2, lot, atr_at_entry, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(timespec="seconds"),
         signal, entry, sl, tp1, tp2 or 0, lot, atr_at_entry, notes),
    )
    pid = cur.lastrowid
    c.commit()
    c.close()
    return pid


def close_position(pid: int, exit_price: float, reason: str) -> dict:
    init_db()
    c = _conn()
    row = c.execute(
        "SELECT signal, entry, lot FROM positions WHERE id=?", (pid,),
    ).fetchone()
    if not row:
        c.close()
        return {"error": "Pozicioni nuk u gjet"}
    sig, entry, lot = row
    price_diff = exit_price - entry if sig == "BUY" else entry - exit_price
    pnl_usd = price_diff * lot * 100   # gold: 1 lot = 100 oz
    c.execute(
        """UPDATE positions
           SET status='CLOSED', closed_at=?, exit_price=?, exit_reason=?, pnl_usd=?
           WHERE id=?""",
        (datetime.utcnow().isoformat(timespec="seconds"),
         exit_price, reason, round(pnl_usd, 2), pid),
    )
    c.commit()
    c.close()
    return {"pnl_usd": round(pnl_usd, 2),
            "result": "WIN" if pnl_usd > 0 else "LOSS"}


def get_open_positions() -> pd.DataFrame:
    init_db()
    c = _conn()
    df = pd.read_sql_query(
        "SELECT * FROM positions WHERE status='OPEN' ORDER BY id DESC", c,
    )
    c.close()
    return df


def log_alert(position_id: int, severity: str, signal: str,
              score: int, reasons: list, action: str) -> int:
    init_db()
    c = _conn()
    cur = c.execute(
        """INSERT INTO alerts_log
           (position_id, ts, severity, signal, score, reasons, action_recommended)
           VALUES (?,?,?,?,?,?,?)""",
        (position_id, datetime.utcnow().isoformat(timespec="seconds"),
         severity, signal, score, " | ".join(reasons), action),
    )
    aid = cur.lastrowid
    c.commit()
    c.close()
    return aid


def recent_alerts(position_id: int = None, limit: int = 10) -> pd.DataFrame:
    init_db()
    c = _conn()
    if position_id:
        df = pd.read_sql_query(
            "SELECT * FROM alerts_log WHERE position_id=? ORDER BY id DESC LIMIT ?",
            c, params=[position_id, limit],
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM alerts_log ORDER BY id DESC LIMIT ?",
            c, params=[limit],
        )
    c.close()
    return df


# ════════════════════════════════════════════════════════════════════════════
# EARLY-EXIT DETECTOR
# ════════════════════════════════════════════════════════════════════════════
def analyze_position(position: dict, df: pd.DataFrame,
                     current_price: float,
                     classifier=None, classifier_scaler=None) -> dict:
    """
    Examine an open position and look for INVALIDATION signals — patterns
    that suggest the original setup has reversed. Returns a verdict with
    severity, reasons, and recommended action.
    """
    import model as ml

    signal     = position["signal"]
    entry      = float(position["entry"])
    sl         = float(position["sl"])
    tp1        = float(position["tp1"])
    atr_entry  = float(position["atr_at_entry"]) or 5.0

    df_feat = ml.add_features(df.copy()).dropna()
    if df_feat.empty:
        return {"verdict": "OK", "score": 0, "reasons": [], "action": "HOLD"}

    latest = df_feat.iloc[-1]
    danger_score = 0
    reasons = []

    # ── 1. PRICE PROGRESSION CHECK ────────────────────────────────────────────
    if signal == "BUY":
        progress = (current_price - entry) / (tp1 - entry) if tp1 != entry else 0
        sl_progress = (entry - current_price) / (entry - sl) if entry != sl else 0
    else:  # SELL
        progress = (entry - current_price) / (entry - tp1) if entry != tp1 else 0
        sl_progress = (current_price - entry) / (sl - entry) if sl != entry else 0

    progress = max(-1, min(1.5, progress))
    sl_progress = max(0, min(1, sl_progress))

    if sl_progress >= 0.7:
        danger_score += 4
        reasons.append(f"🔴 Çmimi {int(sl_progress*100)}% drejt SL — shumë afër invalidimit")
    elif sl_progress >= 0.5:
        danger_score += 2
        reasons.append(f"🟡 Çmimi {int(sl_progress*100)}% drejt SL")

    # ── 2. MOMENTUM REVERSAL — RSI ────────────────────────────────────────────
    rsi = float(latest.get("rsi", 50))
    rsi_lag = float(df_feat["rsi"].iloc[-3]) if len(df_feat) >= 3 else rsi
    rsi_falling = rsi < rsi_lag - 5
    rsi_rising  = rsi > rsi_lag + 5

    if signal == "BUY":
        if rsi_falling and rsi < 50:
            danger_score += 2
            reasons.append(f"📉 RSI në rënie ({rsi_lag:.0f} → {rsi:.0f}) — momentum bullish po humbet")
        if rsi > 70:
            danger_score += 1
            reasons.append(f"⚠ RSI overbought ({rsi:.0f}) — rrezik reversal")
    else:  # SELL
        if rsi_rising and rsi > 50:
            danger_score += 2
            reasons.append(f"📈 RSI në rritje ({rsi_lag:.0f} → {rsi:.0f}) — momentum bearish po humbet")
        if rsi < 30:
            danger_score += 1
            reasons.append(f"⚠ RSI oversold ({rsi:.0f}) — rrezik bounce lart")

    # ── 3. MACD CROSSOVER REVERSAL ────────────────────────────────────────────
    macd      = float(latest.get("macd", 0))
    macd_sig  = float(latest.get("macd_signal", 0))
    macd_hist = float(latest.get("macd_hist", 0))
    macd_hist_prev = float(df_feat["macd_hist"].iloc[-3]) if len(df_feat) >= 3 else macd_hist

    if signal == "BUY":
        if macd < macd_sig and macd_hist < macd_hist_prev:
            danger_score += 2
            reasons.append("🔴 MACD bearish crossover — momentum kthehet poshtë")
    else:  # SELL
        if macd > macd_sig and macd_hist > macd_hist_prev:
            danger_score += 2
            reasons.append("🔴 MACD bullish crossover — momentum kthehet lart")

    # ── 4. EMA STRUCTURE BREAK ────────────────────────────────────────────────
    ema_9  = float(latest.get("ema_9",  current_price))
    ema_21 = float(latest.get("ema_21", current_price))
    ema_50 = float(latest.get("ema_50", current_price))

    if signal == "BUY":
        if current_price < ema_21:
            danger_score += 2
            reasons.append(f"⛔ Çmimi nën EMA21 (${ema_21:.2f}) — strukturë bullish e thyer")
        elif current_price < ema_9:
            danger_score += 1
            reasons.append(f"⚠ Çmimi nën EMA9 — dobësim afat-shkurtër")
    else:  # SELL
        if current_price > ema_21:
            danger_score += 2
            reasons.append(f"⛔ Çmimi mbi EMA21 (${ema_21:.2f}) — strukturë bearish e thyer")
        elif current_price > ema_9:
            danger_score += 1
            reasons.append(f"⚠ Çmimi mbi EMA9 — forcim afat-shkurtër kundër")

    # ── 5. CANDLE PATTERN REVERSAL ────────────────────────────────────────────
    if len(df_feat) >= 2:
        last2 = df_feat.iloc[-2:]
        body = float(last2["candle_body"].iloc[-1])
        wick_up = float(last2["upper_wick"].iloc[-1])
        wick_dn = float(last2["lower_wick"].iloc[-1])
        rng = float(last2["candle_range"].iloc[-1])
        if rng > 0:
            if signal == "BUY":
                # Shooting star / strong rejection wick top
                if wick_up > rng * 0.6 and body < 0:
                    danger_score += 2
                    reasons.append("🕯 Shooting Star / Bearish rejection wick — kandelë reversal")
                if body < 0 and abs(body) > rng * 0.7:
                    danger_score += 1
                    reasons.append("🕯 Kandelë e fortë bearish")
            else:  # SELL
                if wick_dn > rng * 0.6 and body > 0:
                    danger_score += 2
                    reasons.append("🕯 Hammer / Bullish rejection wick — kandelë reversal")
                if body > 0 and body > rng * 0.7:
                    danger_score += 1
                    reasons.append("🕯 Kandelë e fortë bullish")

    # ── 6. AI MODEL CHANGED ITS MIND ──────────────────────────────────────────
    if classifier is not None and classifier_scaler is not None:
        try:
            latest_X = df_feat[ml.FEATURE_COLS].iloc[[-1]]
            latest_s = classifier_scaler.transform(latest_X)
            proba = classifier.predict_proba(latest_s)[0]
            classes = classifier.classes_.tolist()
            proba_map = {c: p for c, p in zip(classes, proba)}
            p_buy  = float(proba_map.get(2, 0))
            p_sell = float(proba_map.get(0, 0))

            # Sub-model votes
            sub_votes = []
            for _name, est in classifier.named_estimators_.items():
                sub_votes.append(int(est.predict(latest_s)[0]))
            buy_n  = sum(1 for v in sub_votes if v == 2)
            sell_n = sum(1 for v in sub_votes if v == 0)

            if signal == "BUY" and sell_n >= 3:
                danger_score += 4
                reasons.append(f"🤖 AI ndryshoi mendje: {sell_n}/{len(sub_votes)} modele tani thonë SELL")
            elif signal == "SELL" and buy_n >= 3:
                danger_score += 4
                reasons.append(f"🤖 AI ndryshoi mendje: {buy_n}/{len(sub_votes)} modele tani thonë BUY")
            elif signal == "BUY" and p_sell > p_buy + 0.10:
                danger_score += 2
                reasons.append(f"🤖 Probabiliteti SELL ({p_sell*100:.0f}%) > BUY ({p_buy*100:.0f}%)")
            elif signal == "SELL" and p_buy > p_sell + 0.10:
                danger_score += 2
                reasons.append(f"🤖 Probabiliteti BUY ({p_buy*100:.0f}%) > SELL ({p_sell*100:.0f}%)")
        except Exception:
            pass

    # ── 7. VOLATILITY EXPANSION AGAINST POSITION ──────────────────────────────
    atr_now = float(latest.get("atr", atr_entry))
    if atr_now > atr_entry * 1.5:
        danger_score += 1
        reasons.append(f"📊 Volatilitet i rritur (ATR {atr_entry:.1f}→{atr_now:.1f}) — pasiguri")

    # ── 8. TIME-IN-TRADE WITHOUT PROGRESS ─────────────────────────────────────
    try:
        opened = datetime.fromisoformat(position["opened_at"])
        hours_open = (datetime.utcnow() - opened).total_seconds() / 3600
        if hours_open > 6 and progress < 0.2 and progress > -0.2:
            danger_score += 1
            reasons.append(f"⏱ Pozicioni i hapur prej {hours_open:.1f}h pa progres — stagnim")
    except Exception:
        pass

    # ── VERDICT ───────────────────────────────────────────────────────────────
    if danger_score >= 7:
        verdict = "CRITICAL"
        action  = "🚨 DIL TANI — mbylle pozicionin para SL"
    elif danger_score >= 5:
        verdict = "WARNING"
        action  = "⚠ KONSIDERO MBYLLJE — shumë shenja kundër"
    elif danger_score >= 3:
        verdict = "CAUTION"
        action  = "🟡 KUJDES — shih nga afër, lëviz SL te break-even nëse mundesh"
    elif danger_score >= 1:
        verdict = "MINOR"
        action  = "ℹ Shenja të vogla — mbaje për tani"
    else:
        verdict = "OK"
        action  = "✅ Pozicioni në rrugë të mirë"

    return {
        "verdict":      verdict,
        "score":        danger_score,
        "reasons":      reasons,
        "action":       action,
        "progress":     round(progress, 2),
        "sl_progress":  round(sl_progress, 2),
        "current_price": round(current_price, 2),
    }
