"""
Risk Calculator, Position Sizing & Pip Value tools for XAUUSD.

Gold (XAU/USD) specs:
  - Standard lot = 100 oz
  - Mini lot    = 10 oz
  - Micro lot   = 1 oz
  - Pip (tick)  = $0.01 per oz = $1 per standard lot
"""


GOLD_OZ_PER_LOT = 100  # 1 standard lot = 100 oz


def pip_value(lot_size: float, account_currency: str = "USD") -> float:
    """
    Calculate pip (0.01) value in USD for gold.
    Gold pip = $0.01 per oz. 1 lot = 100 oz → $1 per pip per lot.
    """
    return round(lot_size * GOLD_OZ_PER_LOT * 0.01, 4)


def calculate_risk(
    entry_price: float,
    stop_loss: float,
    lot_size: float,
) -> dict:
    """Calculate risk in pips and USD given entry, SL and lot size."""
    sl_distance_price = abs(entry_price - stop_loss)
    sl_pips = sl_distance_price / 0.01  # 1 pip = $0.01
    pv = pip_value(lot_size)
    risk_usd = sl_pips * pv

    return {
        "sl_distance_price": round(sl_distance_price, 2),
        "sl_pips": round(sl_pips, 1),
        "pip_value_usd": round(pv, 2),
        "risk_usd": round(risk_usd, 2),
    }


def calculate_reward(
    entry_price: float,
    take_profit: float,
    lot_size: float,
) -> dict:
    """Calculate reward in pips and USD given entry, TP and lot size."""
    tp_distance_price = abs(take_profit - entry_price)
    tp_pips = tp_distance_price / 0.01
    pv = pip_value(lot_size)
    reward_usd = tp_pips * pv

    return {
        "tp_distance_price": round(tp_distance_price, 2),
        "tp_pips": round(tp_pips, 1),
        "reward_usd": round(reward_usd, 2),
    }


def risk_reward_ratio(risk: dict, reward: dict) -> float:
    if risk["risk_usd"] == 0:
        return 0.0
    return round(reward["reward_usd"] / risk["risk_usd"], 2)


def position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> dict:
    """
    Calculate optimal lot size based on account balance and risk %.
    """
    risk_usd = account_balance * (risk_percent / 100)
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        return {"lot_size": 0, "risk_usd": 0, "risk_pct": 0}

    sl_pips = sl_distance / 0.01
    # pip_value per lot = lot * 100 * 0.01 = lot
    # risk_usd = lot_size * sl_pips * pip_value_per_lot
    # lot_size = risk_usd / (sl_pips * 1)
    lot_size = risk_usd / sl_pips
    lot_size = round(lot_size, 2)

    # Suggested levels
    std_lots  = round(lot_size, 2)
    mini_lots = round(lot_size * 10, 1)
    micro_lots= round(lot_size * 100, 0)

    return {
        "lot_size": std_lots,
        "mini_lots": mini_lots,
        "micro_lots": int(micro_lots),
        "risk_usd": round(risk_usd, 2),
        "risk_pct": risk_percent,
        "sl_pips": round(sl_pips, 0),
    }


def full_trade_plan(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> dict:
    """Full trade plan combining all calculations."""
    pos = position_size(account_balance, risk_percent, entry_price, stop_loss)
    lot = pos["lot_size"]
    risk = calculate_risk(entry_price, stop_loss, lot)
    reward = calculate_reward(entry_price, take_profit, lot)
    rr = risk_reward_ratio(risk, reward)

    direction = "BUY" if take_profit > entry_price else "SELL"

    return {
        "direction": direction,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "lot_size": lot,
        "risk_usd": risk["risk_usd"],
        "reward_usd": reward["reward_usd"],
        "risk_pct": risk_percent,
        "rr_ratio": rr,
        "sl_pips": risk["sl_pips"],
        "tp_pips": reward["tp_pips"],
        "pip_value": pip_value(lot),
        "quality": _trade_quality(rr, risk_percent),
        "balance_after_loss": round(account_balance - risk["risk_usd"], 2),
        "balance_after_win": round(account_balance + reward["reward_usd"], 2),
    }


def _trade_quality(rr: float, risk_pct: float) -> str:
    if rr >= 3.0 and risk_pct <= 2.0:
        return "🟢 Excellent"
    if rr >= 2.0 and risk_pct <= 3.0:
        return "🟡 E mirë"
    if rr >= 1.5:
        return "🟠 Mesatare"
    return "🔴 E dobët (RR < 1.5)"


def suggest_sl_tp(entry: float, signal: str, atr: float) -> dict:
    """Auto-suggest SL/TP based on ATR and signal direction."""
    sl_mult = 1.5
    tp_mult = 2.5

    if signal == "BUY":
        sl = round(entry - atr * sl_mult, 2)
        tp1 = round(entry + atr * tp_mult, 2)
        tp2 = round(entry + atr * 4.0, 2)
    elif signal == "SELL":
        sl = round(entry + atr * sl_mult, 2)
        tp1 = round(entry - atr * tp_mult, 2)
        tp2 = round(entry - atr * 4.0, 2)
    else:
        sl = tp1 = tp2 = entry

    return {
        "stop_loss": sl,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "sl_distance": round(abs(entry - sl), 2),
        "atr": round(atr, 2),
    }
