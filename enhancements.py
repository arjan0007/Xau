"""
Trading enhancements: Kill Zones, Multi-Timeframe, News filter,
Dynamic SL/TP, Trailing Stop, Position Sizing, Smart Money Concepts (SMC),
Walk-Forward Optimization.

Each function is pure and returns a "filter verdict" + reasoning so the
dashboard can show the user exactly why a trade was blocked or allowed.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
# 1. KILL ZONES — high-liquidity sessions where gold actually moves
# ════════════════════════════════════════════════════════════════════════════
KILL_ZONES = [
    {"name": "🇬🇧 London Open",        "start": 7,  "end": 10, "quality": "HIGH"},
    {"name": "🔥 NY + London Overlap", "start": 12, "end": 15, "quality": "PREMIUM"},
    {"name": "🇺🇸 NY PM",              "start": 19, "end": 21, "quality": "MEDIUM"},
]


def in_kill_zone(now_utc: datetime = None) -> dict:
    """Return whether current UTC time is in a high-liquidity session."""
    if now_utc is None:
        now_utc = datetime.utcnow()
    h = now_utc.hour
    for zone in KILL_ZONES:
        if zone["start"] <= h < zone["end"]:
            return {"in_zone": True, "zone": zone["name"], "quality": zone["quality"],
                    "minutes_left": (zone["end"] - h) * 60 - now_utc.minute}
    next_zone = None
    for zone in KILL_ZONES:
        if zone["start"] > h:
            next_zone = zone; break
    if next_zone is None:
        next_zone = KILL_ZONES[0]
        hours_to = (24 - h + next_zone["start"])
    else:
        hours_to = next_zone["start"] - h
    return {"in_zone": False, "zone": None, "quality": "DEAD",
            "next_zone": next_zone["name"], "hours_to_next": hours_to}


# ════════════════════════════════════════════════════════════════════════════
# 2. MULTI-TIMEFRAME CONFIRMATION
# ════════════════════════════════════════════════════════════════════════════
def htf_bias(df_htf: pd.DataFrame, ema_fast: int = 21, ema_slow: int = 50) -> str:
    """Higher-timeframe directional bias from EMA structure + slope."""
    if df_htf is None or len(df_htf) < ema_slow + 5:
        return "NEUTRAL"
    close = df_htf["Close"]
    fast = close.ewm(span=ema_fast, adjust=False).mean()
    slow = close.ewm(span=ema_slow, adjust=False).mean()
    slope = (slow.iloc[-1] - slow.iloc[-5]) / slow.iloc[-5]
    if fast.iloc[-1] > slow.iloc[-1] and slope > 0:
        return "BULLISH"
    if fast.iloc[-1] < slow.iloc[-1] and slope < 0:
        return "BEARISH"
    return "NEUTRAL"


def momentum_15m(df_ltf: pd.DataFrame, lookback: int = 5) -> str:
    """Short-term momentum on lower timeframe — close vs N candles ago."""
    if df_ltf is None or len(df_ltf) < lookback + 1:
        return "NEUTRAL"
    chg = df_ltf["Close"].iloc[-1] - df_ltf["Close"].iloc[-(lookback + 1)]
    if chg > 0:
        return "UP"
    if chg < 0:
        return "DOWN"
    return "NEUTRAL"


def mtf_alignment(signal: str, htf_dir: str, ltf_mom: str) -> dict:
    """Check whether 4H bias + 15M momentum agree with the 1H signal."""
    if signal == "HOLD":
        return {"aligned": False, "score": 0, "reason": "Sinjali është HOLD"}
    score = 0
    parts = []
    if signal == "BUY":
        if htf_dir == "BULLISH": score += 2; parts.append("✅ 4H bullish")
        elif htf_dir == "NEUTRAL": score += 0; parts.append("➖ 4H neutral")
        else: score -= 2; parts.append("❌ 4H bearish (kundër!)")
        if ltf_mom == "UP": score += 1; parts.append("✅ 15M momentum lart")
        elif ltf_mom == "DOWN": score -= 1; parts.append("❌ 15M momentum poshtë")
    else:  # SELL
        if htf_dir == "BEARISH": score += 2; parts.append("✅ 4H bearish")
        elif htf_dir == "NEUTRAL": score += 0; parts.append("➖ 4H neutral")
        else: score -= 2; parts.append("❌ 4H bullish (kundër!)")
        if ltf_mom == "DOWN": score += 1; parts.append("✅ 15M momentum poshtë")
        elif ltf_mom == "UP": score -= 1; parts.append("❌ 15M momentum lart")
    aligned = score >= 2
    return {"aligned": aligned, "score": score, "reason": " · ".join(parts),
            "htf": htf_dir, "ltf": ltf_mom}


# ════════════════════════════════════════════════════════════════════════════
# 3. NEWS FILTER — avoid trading around high-impact releases
# ════════════════════════════════════════════════════════════════════════════
HIGH_IMPACT_KEYWORDS = [
    "non-farm", "nfp", "cpi", "inflation", "fomc", "fed rate", "interest rate",
    "ppi", "gdp", "unemployment", "powell", "jackson hole",
]


def news_safe(events: list, now_utc: datetime = None,
              buffer_minutes: int = 30) -> dict:
    """
    Check whether we're within `buffer_minutes` of a high-impact news event.
    `events` is a list of {"time": datetime, "title": str, "impact": "High"/...}.
    """
    if not events:
        return {"safe": True, "reason": "Pa lajme të programuara", "blocker": None}
    if now_utc is None:
        now_utc = datetime.utcnow()
    delta = timedelta(minutes=buffer_minutes)
    for ev in events:
        ev_time = ev.get("time")
        if not isinstance(ev_time, datetime):
            try:
                ev_time = datetime.fromisoformat(str(ev_time))
            except Exception:
                continue
        if ev.get("impact", "").lower() != "high":
            # Also catch by keyword
            title = (ev.get("title") or "").lower()
            if not any(k in title for k in HIGH_IMPACT_KEYWORDS):
                continue
        if abs((ev_time - now_utc).total_seconds()) <= delta.total_seconds():
            mins = int((ev_time - now_utc).total_seconds() / 60)
            when = f"në {mins} min" if mins > 0 else f"para {-mins} min"
            return {"safe": False, "reason": f"⚠ Lajm high-impact {when}: {ev.get('title')}",
                    "blocker": ev.get("title")}
    return {"safe": True, "reason": "✅ Pa lajme afër", "blocker": None}


# ════════════════════════════════════════════════════════════════════════════
# 4. DYNAMIC SL/TP — based on swing structure, not just ATR
# ════════════════════════════════════════════════════════════════════════════
def find_swing_levels(df: pd.DataFrame, lookback: int = 30,
                      pivot_window: int = 3) -> dict:
    """Find most recent swing high & swing low using N-bar pivots."""
    if len(df) < lookback:
        lookback = len(df)
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    swing_highs, swing_lows = [], []
    for i in range(pivot_window, len(recent) - pivot_window):
        if highs[i] == max(highs[i - pivot_window:i + pivot_window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - pivot_window:i + pivot_window + 1]):
            swing_lows.append((i, lows[i]))
    last_high = swing_highs[-1][1] if swing_highs else float(recent["High"].max())
    last_low  = swing_lows[-1][1]  if swing_lows  else float(recent["Low"].min())
    return {"swing_high": last_high, "swing_low": last_low,
            "swing_highs": [v for _, v in swing_highs],
            "swing_lows":  [v for _, v in swing_lows]}


def dynamic_sl_tp(signal: str, entry: float, df: pd.DataFrame,
                  atr: float, min_rr: float = 1.5) -> dict:
    """Place SL beyond the relevant swing, TP at next opposite swing.
    Falls back to ATR if structure too tight."""
    sw = find_swing_levels(df)
    buffer = atr * 0.4   # small cushion past the swing
    if signal == "BUY":
        sl  = min(sw["swing_low"]  - buffer, entry - atr * 1.0)
        tp1 = max(sw["swing_high"] - buffer * 0.5, entry + atr * (min_rr + 0.5))
        tp2 = entry + (tp1 - entry) * 1.8
    elif signal == "SELL":
        sl  = max(sw["swing_high"] + buffer, entry + atr * 1.0)
        tp1 = min(sw["swing_low"]  + buffer * 0.5, entry - atr * (min_rr + 0.5))
        tp2 = entry - (entry - tp1) * 1.8
    else:
        return {"sl": entry, "tp1": entry, "tp2": entry, "rr": 0}
    rr = abs(tp1 - entry) / max(abs(entry - sl), 1e-9)
    return {"sl": round(sl, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2),
            "rr": round(rr, 2),
            "swing_high": round(sw["swing_high"], 2),
            "swing_low":  round(sw["swing_low"], 2)}


# ════════════════════════════════════════════════════════════════════════════
# 5. TRAILING STOP — break-even after TP1, then trail with structure
# ════════════════════════════════════════════════════════════════════════════
def trailing_stop_plan(signal: str, entry: float, current: float,
                       sl: float, tp1: float, atr: float) -> dict:
    """Suggest current adjusted SL based on price progression."""
    if signal not in ("BUY", "SELL"):
        return {"action": "NONE", "new_sl": sl, "reason": "S'ka pozicion"}
    progress = (current - entry) / (tp1 - entry) if tp1 != entry else 0
    if signal == "SELL":
        progress = (entry - current) / (entry - tp1) if entry != tp1 else 0

    if progress >= 1.0:
        # Hit TP1 — move SL to entry + small profit lock
        new_sl = entry + atr * 0.3 if signal == "BUY" else entry - atr * 0.3
        return {"action": "LOCK_PROFIT", "new_sl": round(new_sl, 2),
                "reason": "✅ Prek TP1 — mbyll 50%, SL te break-even+",
                "progress": round(progress, 2)}
    if progress >= 0.5:
        # Halfway → move SL to entry
        return {"action": "BREAK_EVEN", "new_sl": round(entry, 2),
                "reason": "🛡 50% e rrugës — SL te break-even",
                "progress": round(progress, 2)}
    if progress >= 0.25:
        # Quarter → trail by 1.5×ATR
        new_sl = current - atr * 1.5 if signal == "BUY" else current + atr * 1.5
        new_sl = max(new_sl, sl) if signal == "BUY" else min(new_sl, sl)
        return {"action": "TRAIL", "new_sl": round(new_sl, 2),
                "reason": "📈 Trail SL me 1.5×ATR",
                "progress": round(progress, 2)}
    return {"action": "HOLD", "new_sl": sl,
            "reason": "Pres që pozicioni të zhvillohet",
            "progress": round(progress, 2)}


# ════════════════════════════════════════════════════════════════════════════
# 6. POSITION SIZING — risk-based lot calculation
# ════════════════════════════════════════════════════════════════════════════
def position_size(balance: float, risk_pct: float, entry: float, sl: float,
                  confidence: float = 0.7,
                  recent_losses: int = 0) -> dict:
    """
    Calculate lot size so that hitting SL loses exactly `risk_pct`% of balance.
    Adjusts based on:
    - confidence (higher conf → larger size)
    - recent loss streak (auto-reduce after 2+ losses)
    """
    if entry == sl:
        return {"lot": 0, "risk_usd": 0, "reason": "SL = Entry, lot 0"}

    # Adjust risk by confidence
    if confidence >= 0.85:
        adj_risk = risk_pct * 1.5
        conf_label = "🔥 Konfidencë e lartë → risk +50%"
    elif confidence >= 0.70:
        adj_risk = risk_pct
        conf_label = "✅ Konfidencë normale → risk standard"
    else:
        adj_risk = risk_pct * 0.5
        conf_label = "⚠ Konfidencë e ulët → risk -50%"

    # Reduce after losing streak
    if recent_losses >= 3:
        adj_risk *= 0.4
        conf_label += " | ⛔ 3 humbje rresht → risk -60%"
    elif recent_losses >= 2:
        adj_risk *= 0.6
        conf_label += " | ⚠ 2 humbje rresht → risk -40%"

    risk_usd = balance * (adj_risk / 100)
    sl_distance = abs(entry - sl)         # in dollars per oz
    # Gold: 1 lot = 100 oz, so $1 move = $100 per lot
    lot = risk_usd / (sl_distance * 100)
    lot = round(max(0.01, min(lot, 10.0)), 2)
    return {
        "lot": lot, "risk_usd": round(risk_usd, 2),
        "risk_pct": round(adj_risk, 2),
        "sl_distance_usd": round(sl_distance, 2),
        "reason": conf_label,
    }


# ════════════════════════════════════════════════════════════════════════════
# 7. SMART MONEY CONCEPTS — Order Blocks, FVG, BOS, Liquidity
# ════════════════════════════════════════════════════════════════════════════
def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list:
    """Find recent bullish/bearish order blocks (last opposite candle before
    a strong impulse)."""
    if len(df) < lookback:
        return []
    recent = df.tail(lookback).reset_index(drop=False).copy()
    obs = []
    for i in range(2, len(recent) - 3):
        body_now = recent["Close"].iloc[i] - recent["Open"].iloc[i]
        body_next = recent["Close"].iloc[i + 1] - recent["Open"].iloc[i + 1]
        range_now = recent["High"].iloc[i] - recent["Low"].iloc[i]
        # Bullish OB: down candle followed by strong up candle
        if body_now < 0 and body_next > 0 and body_next > range_now * 1.5:
            obs.append({"type": "BULL_OB", "top": float(recent["High"].iloc[i]),
                        "bottom": float(recent["Low"].iloc[i]),
                        "idx": int(recent.index[i])})
        # Bearish OB: up candle followed by strong down candle
        if body_now > 0 and body_next < 0 and abs(body_next) > range_now * 1.5:
            obs.append({"type": "BEAR_OB", "top": float(recent["High"].iloc[i]),
                        "bottom": float(recent["Low"].iloc[i]),
                        "idx": int(recent.index[i])})
    return obs[-5:]  # last 5 order blocks


def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 50) -> list:
    """A 3-candle FVG: candle[i+2].low > candle[i].high  (bullish gap)
    or candle[i+2].high < candle[i].low (bearish gap)."""
    if len(df) < lookback:
        return []
    recent = df.tail(lookback).reset_index(drop=True)
    fvgs = []
    for i in range(len(recent) - 2):
        a_high = recent["High"].iloc[i]
        a_low  = recent["Low"].iloc[i]
        c_high = recent["High"].iloc[i + 2]
        c_low  = recent["Low"].iloc[i + 2]
        if c_low > a_high:
            fvgs.append({"type": "BULL_FVG", "top": float(c_low),
                         "bottom": float(a_high), "idx": i})
        if c_high < a_low:
            fvgs.append({"type": "BEAR_FVG", "top": float(a_low),
                         "bottom": float(c_high), "idx": i})
    return fvgs[-5:]


def detect_bos(df: pd.DataFrame, swing_window: int = 10) -> str:
    """Break of Structure — most recent close above last swing high (bullish)
    or below last swing low (bearish)."""
    if len(df) < swing_window * 2 + 2:
        return "NONE"
    sw = find_swing_levels(df, lookback=swing_window * 2, pivot_window=2)
    last_close = float(df["Close"].iloc[-1])
    if last_close > sw["swing_high"]:
        return "BULLISH_BOS"
    if last_close < sw["swing_low"]:
        return "BEARISH_BOS"
    return "NONE"


def smc_confluence(signal: str, df: pd.DataFrame,
                   current_price: float) -> dict:
    """Score how well current setup aligns with Smart Money Concepts."""
    if signal not in ("BUY", "SELL"):
        return {"score": 0, "details": [], "smc_ok": False}
    obs = find_order_blocks(df)
    fvgs = find_fair_value_gaps(df)
    bos = detect_bos(df)

    score = 0
    details = []
    if signal == "BUY":
        in_demand = any(o["type"] == "BULL_OB"
                        and o["bottom"] <= current_price <= o["top"] * 1.005
                        for o in obs)
        if in_demand: score += 2; details.append("✅ Brenda Bull Order Block")
        else: details.append("➖ Jashtë Order Block")
        bullish_fvg_open = any(f["type"] == "BULL_FVG"
                               and current_price < f["top"] for f in fvgs)
        if bullish_fvg_open: score += 1; details.append("✅ Bull FVG i hapur lart")
        if bos == "BULLISH_BOS": score += 2; details.append("✅ Bullish BOS konfirmuar")
        elif bos == "BEARISH_BOS": score -= 2; details.append("❌ Bearish BOS aktiv")
    else:  # SELL
        in_supply = any(o["type"] == "BEAR_OB"
                        and o["bottom"] * 0.995 <= current_price <= o["top"]
                        for o in obs)
        if in_supply: score += 2; details.append("✅ Brenda Bear Order Block")
        else: details.append("➖ Jashtë Order Block")
        bearish_fvg_open = any(f["type"] == "BEAR_FVG"
                               and current_price > f["bottom"] for f in fvgs)
        if bearish_fvg_open: score += 1; details.append("✅ Bear FVG i hapur poshtë")
        if bos == "BEARISH_BOS": score += 2; details.append("✅ Bearish BOS konfirmuar")
        elif bos == "BULLISH_BOS": score -= 2; details.append("❌ Bullish BOS aktiv")
    return {"score": score, "details": details, "smc_ok": score >= 2,
            "order_blocks": obs, "fvgs": fvgs, "bos": bos}


# ════════════════════════════════════════════════════════════════════════════
# 8. CORRELATION FILTER — block trades when DXY/SPX/Yields go wrong way
# ════════════════════════════════════════════════════════════════════════════
def correlation_filter(signal: str, corr_summary: list) -> dict:
    """Validate signal against current DXY/SPX/Yield correlation regime."""
    if not corr_summary or signal == "HOLD":
        return {"ok": True, "reason": "Korrelacionet jo të disponueshme", "warnings": []}
    warnings = []
    for c in corr_summary:
        asset = c.get("asset", "")
        corr  = c.get("correlation", 0)
        if asset == "DXY (Dollar)" and corr < -0.4:
            if signal == "BUY":
                warnings.append("⚠ DXY korr negative e fortë — verifiko DXY në rënie")
            else:
                warnings.append("⚠ DXY korr negative e fortë — verifiko DXY në rritje")
        if asset == "10Y Yield" and corr < -0.3:
            warnings.append("ℹ 10Y Yield korr negative — sheh yields para hyrjes")
    return {"ok": len(warnings) == 0, "reason": "OK" if not warnings else "Ka paralajmërime",
            "warnings": warnings}


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED TRADE GATE — combine all filters
# ════════════════════════════════════════════════════════════════════════════
def trade_gate(signal: str, confidence: float,
               df_main: pd.DataFrame,
               df_htf: pd.DataFrame = None,
               df_ltf: pd.DataFrame = None,
               events: list = None,
               corr_summary: list = None,
               check_kill_zone: bool = True,
               check_mtf: bool = True,
               check_news: bool = True,
               check_smc: bool = True,
               check_corr: bool = True) -> dict:
    """
    Run every filter and return a unified verdict:
    - allow: True/False — green light to trade
    - score: 0-10  (filter pass count + multipliers)
    - blockers: list of hard NO reasons
    - boosters: list of confluence YESs
    """
    if signal == "HOLD":
        return {"allow": False, "score": 0, "blockers": ["Sinjali është HOLD"],
                "boosters": [], "filters": {}}

    blockers, boosters = [], []
    filters = {}
    score = 0
    current = float(df_main["Close"].iloc[-1])

    # 1. Kill zone
    if check_kill_zone:
        kz = in_kill_zone()
        filters["kill_zone"] = kz
        if not kz["in_zone"]:
            blockers.append(f"⛔ Jashtë Kill Zone (sesion vdekur, prit {kz.get('next_zone','')})")
        else:
            boosters.append(f"✅ {kz['zone']} ({kz['quality']})")
            score += 2 if kz["quality"] == "PREMIUM" else 1

    # 2. Multi-timeframe alignment
    if check_mtf and df_htf is not None:
        htf_dir = htf_bias(df_htf)
        ltf_mom = momentum_15m(df_ltf) if df_ltf is not None else "NEUTRAL"
        mtf = mtf_alignment(signal, htf_dir, ltf_mom)
        filters["mtf"] = mtf
        if mtf["aligned"]:
            boosters.append(f"✅ MTF i aligned ({mtf['reason']})")
            score += 2
        else:
            blockers.append(f"⛔ MTF i pa-aligned: {mtf['reason']}")

    # 3. News
    if check_news:
        ns = news_safe(events or [])
        filters["news"] = ns
        if not ns["safe"]:
            blockers.append(ns["reason"])
        else:
            boosters.append(ns["reason"])
            score += 1

    # 4. SMC
    if check_smc:
        smc = smc_confluence(signal, df_main, current)
        filters["smc"] = smc
        if smc["smc_ok"]:
            boosters.append(f"✅ SMC i fortë (+{smc['score']}) — {'; '.join(smc['details'][:2])}")
            score += 2
        elif smc["score"] < 0:
            blockers.append(f"⛔ SMC kundër ({smc['score']}) — {'; '.join(smc['details'][:2])}")
        else:
            boosters.append(f"➖ SMC neutral ({smc['score']})")

    # 5. Correlations
    if check_corr and corr_summary:
        cf = correlation_filter(signal, corr_summary)
        filters["correlation"] = cf
        if not cf["ok"]:
            for w in cf["warnings"]:
                boosters.append(w)   # warnings, not blockers
    # Confidence boost
    if confidence >= 80: score += 2; boosters.append(f"🔥 Konfidencë e lartë ({confidence:.0f}%)")
    elif confidence >= 70: score += 1
    else: score -= 1

    allow = (len(blockers) == 0 and score >= 3)
    return {"allow": allow, "score": score, "blockers": blockers,
            "boosters": boosters, "filters": filters}


# ════════════════════════════════════════════════════════════════════════════
# 9. WALK-FORWARD OPTIMIZATION marker
# ════════════════════════════════════════════════════════════════════════════
def should_walk_forward(last_train_ts: float, candles_seen_since: int,
                        bars_per_window: int = 168) -> bool:
    """Trigger retrain after `bars_per_window` new candles (default 1 week of H1)."""
    if last_train_ts == 0:
        return True
    return candles_seen_since >= bars_per_window
