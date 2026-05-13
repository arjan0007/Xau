"""
Telegram Bot integration for XAUUSD signals.
Setup:
  1. Hap @BotFather në Telegram → /newbot → merr BOT_TOKEN
  2. Dërgo /start tek boti yt → merr CHAT_ID nga: https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Vendos BOT_TOKEN dhe CHAT_ID në config.py
"""
import requests
from datetime import datetime


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def send_message(token: str, chat_id: str, text: str,
                 parse_mode: str = "HTML") -> tuple[bool, str]:
    """Send a plain text or HTML message via Telegram."""
    if not token or not chat_id:
        return False, "Token ose Chat ID mungon."
    try:
        url = TELEGRAM_API.format(token=token, method="sendMessage")
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }, timeout=10)
        data = r.json()
        if data.get("ok"):
            return True, "Mesazhi u dërgua ✅"
        return False, data.get("description", "Gabim i panjohur")
    except Exception as e:
        return False, str(e)


def send_signal_alert(token: str, chat_id: str,
                      signal: str, price: float, confidence: float,
                      predicted_price: float | None = None,
                      supports: list = None, resistances: list = None,
                      acc: float = 0.0) -> tuple[bool, str]:
    """Send a formatted signal alert."""
    arrows = {"BUY": "📈 ⬆", "SELL": "📉 ⬇", "HOLD": "⏸ →"}
    emoji  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

    sup_str = " | ".join([f"${s:,.2f}" for s in (supports or [])[:3]]) or "—"
    res_str = " | ".join([f"${r:,.2f}" for r in (resistances or [])[:3]]) or "—"
    pred_str = f"${predicted_price:,.2f}" if predicted_price else "—"

    text = f"""
{emoji.get(signal, '⚪')} <b>XAUUSD — Sinyal i Ri</b>
━━━━━━━━━━━━━━━━━━━━
{arrows.get(signal, signal)} <b>{signal}</b>
💰 Çmimi: <code>${price:,.2f}</code>
🎯 Konfidenca: <b>{confidence}%</b>
🤖 Saktësia: {acc}%
━━━━━━━━━━━━━━━━━━━━
🔮 Parashikim: <code>{pred_str}</code>
🟢 Support: {sup_str}
🔴 Resist: {res_str}
━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}
<i>⚠ Ky nuk është këshillë financiare</i>
""".strip()

    return send_message(token, chat_id, text)


def send_anomaly_alert(token: str, chat_id: str,
                       price: float, zscore: float,
                       severity: str) -> tuple[bool, str]:
    """Send anomaly detection alert."""
    text = f"""
⚡ <b>XAUUSD — Lëvizje Anomale!</b>
━━━━━━━━━━━━━━━━━━━━
💰 Çmimi: <code>${price:,.2f}</code>
📊 Z-Score: <b>{zscore:+.2f}</b>
🚨 Severity: {severity}
━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}
<i>Lëvizje e pazakontë — trego kujdes!</i>
""".strip()
    return send_message(token, chat_id, text)


def send_sr_alert(token: str, chat_id: str,
                  price: float, level: float,
                  level_type: str) -> tuple[bool, str]:
    """Send Support/Resistance hit alert."""
    emoji = "🟢" if level_type == "Support" else "🔴"
    text = f"""
{emoji} <b>XAUUSD — {level_type} u Preku!</b>
━━━━━━━━━━━━━━━━━━━━
💰 Çmimi aktual: <code>${price:,.2f}</code>
📍 Nivel {level_type}: <code>${level:,.2f}</code>
📏 Distanca: {abs(price-level):.2f} $
━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}
""".strip()
    return send_message(token, chat_id, text)


def send_rsi_alert(token: str, chat_id: str,
                   price: float, rsi: float) -> tuple[bool, str]:
    """Send RSI overbought/oversold alert."""
    if rsi > 70:
        status = "🔴 OVERBOUGHT (RSI > 70)"
        note = "Mundësi SELL ose kujdes me BUY"
    else:
        status = "🟢 OVERSOLD (RSI < 30)"
        note = "Mundësi BUY ose kujdes me SELL"

    text = f"""
📊 <b>XAUUSD — RSI Alert!</b>
━━━━━━━━━━━━━━━━━━━━
💰 Çmimi: <code>${price:,.2f}</code>
📉 RSI(14): <b>{rsi:.1f}</b>
{status}
━━━━━━━━━━━━━━━━━━━━
💡 {note}
🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}
""".strip()
    return send_message(token, chat_id, text)


def send_ema_cross_alert(token: str, chat_id: str,
                         price: float, ema_fast: float,
                         ema_slow: float, cross_type: str) -> tuple[bool, str]:
    """Send EMA crossover alert."""
    emoji = "🟢📈" if cross_type == "Golden Cross" else "🔴📉"
    text = f"""
{emoji} <b>XAUUSD — {cross_type}!</b>
━━━━━━━━━━━━━━━━━━━━
💰 Çmimi: <code>${price:,.2f}</code>
📊 EMA9:  <code>${ema_fast:,.2f}</code>
📊 EMA21: <code>${ema_slow:,.2f}</code>
━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}
""".strip()
    return send_message(token, chat_id, text)


def test_connection(token: str, chat_id: str) -> tuple[bool, str]:
    """Test Telegram connection."""
    return send_message(token, chat_id,
                        "✅ <b>XAUUSD Bot u lidh me sukses!</b>\n🥇 Sinjalet do të vijnë këtu.")


def get_chat_id(token: str) -> str | None:
    """Auto-detect chat ID from last message."""
    try:
        url = TELEGRAM_API.format(token=token, method="getUpdates")
        r = requests.get(url, timeout=10)
        data = r.json()
        updates = data.get("result", [])
        if updates:
            return str(updates[-1]["message"]["chat"]["id"])
    except Exception:
        pass
    return None
