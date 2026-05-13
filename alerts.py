import smtplib
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Sound Alert (Browser JS) ───────────────────────────────────────────────────

SOUND_BUY = """<script>(function(){var c=new(window.AudioContext||window.webkitAudioContext)();
function b(f,d,v){var o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);
o.frequency.value=f;g.gain.value=v;o.start(c.currentTime);o.stop(c.currentTime+d);}
b(523,.15,.3);setTimeout(()=>b(659,.15,.3),160);setTimeout(()=>b(784,.25,.3),320);})();</script>"""

SOUND_SELL = """<script>(function(){var c=new(window.AudioContext||window.webkitAudioContext)();
function b(f,d,v){var o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);
o.frequency.value=f;g.gain.value=v;o.start(c.currentTime);o.stop(c.currentTime+d);}
b(784,.15,.3);setTimeout(()=>b(659,.15,.3),160);setTimeout(()=>b(523,.25,.3),320);})();</script>"""

SOUND_ALERT = """<script>(function(){var c=new(window.AudioContext||window.webkitAudioContext)();
function b(f,d,v){var o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);
o.frequency.value=f;g.gain.value=v;o.start(c.currentTime);o.stop(c.currentTime+d);}
b(880,.1,.4);setTimeout(()=>b(880,.1,.4),200);setTimeout(()=>b(1100,.2,.4),400);})();</script>"""


def play_signal_sound(signal: str):
    from streamlit.components.v1 import html
    if signal == "BUY":
        html(SOUND_BUY, height=0)
    elif signal == "SELL":
        html(SOUND_SELL, height=0)
    else:
        html(SOUND_ALERT, height=0)


def play_alert_sound():
    from streamlit.components.v1 import html
    html(SOUND_ALERT, height=0)


# ── Email Alert ────────────────────────────────────────────────────────────────

def send_email_alert(to_email, from_email, app_password,
                     signal, price, confidence):
    colors = {"BUY": "#00c853", "SELL": "#d50000", "HOLD": "#ffd600"}
    arrows = {"BUY": "↑", "SELL": "↓", "HOLD": "→"}
    color  = colors.get(signal, "#fff")
    arrow  = arrows.get(signal, "")
    ts     = datetime.now().strftime("%d %b %Y  %H:%M:%S")

    subject = f"XAUUSD Signal: {arrow} {signal} @ ${price:,.2f}"
    html_body = f"""<html><body style="font-family:Arial;background:#0d0d0d;color:#e0e0e0;padding:24px">
      <h2 style="color:#f0c040">⚡ XAUUSD Predictor — Sinyal i Ri</h2>
      <table style="border-collapse:collapse;width:100%;max-width:400px">
        <tr><td style="padding:8px;color:#aaa">Sinjali</td>
            <td style="padding:8px;font-size:24px;font-weight:bold;color:{color}">{arrow} {signal}</td></tr>
        <tr><td style="padding:8px;color:#aaa">Çmimi</td>
            <td style="padding:8px;color:#f0c040;font-size:20px">${price:,.2f}</td></tr>
        <tr><td style="padding:8px;color:#aaa">Konfidenca</td>
            <td style="padding:8px">{confidence}%</td></tr>
        <tr><td style="padding:8px;color:#aaa">Koha</td>
            <td style="padding:8px">{ts}</td></tr>
      </table>
      <p style="color:#555;font-size:11px;margin-top:24px">Ky nuk është këshillë financiare.</p>
    </body></html>"""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = from_email
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(from_email, app_password)
            s.sendmail(from_email, to_email, msg.as_string())
        return True, "Email u dërgua ✅"
    except Exception as e:
        return False, f"Gabim: {e}"


# ── Advanced Alert Checks ──────────────────────────────────────────────────────

def check_signal_changed(new_signal: str) -> bool:
    prev = st.session_state.get("last_signal")
    if prev != new_signal:
        st.session_state["last_signal"] = new_signal
        return True
    return False


def check_sr_breach(current_price: float, supports: list, resistances: list,
                    tolerance_pct: float = 0.002) -> list[dict]:
    """Check if price is near/at a S/R level."""
    alerts = []
    for s in supports:
        if abs(current_price - s) / s < tolerance_pct:
            alerts.append({"type": "Support", "level": s, "price": current_price})
    for r in resistances:
        if abs(current_price - r) / r < tolerance_pct:
            alerts.append({"type": "Resistance", "level": r, "price": current_price})
    return alerts


def check_rsi_extreme(rsi: float, prev_rsi: float = None) -> dict | None:
    """Check if RSI just crossed 70 or 30."""
    if rsi > 70:
        crossed = prev_rsi is not None and prev_rsi <= 70
        return {"type": "overbought", "rsi": rsi, "just_crossed": crossed}
    if rsi < 30:
        crossed = prev_rsi is not None and prev_rsi >= 30
        return {"type": "oversold", "rsi": rsi, "just_crossed": crossed}
    return None


def check_ema_crossover(ema_fast_now: float, ema_slow_now: float,
                        ema_fast_prev: float, ema_slow_prev: float) -> str | None:
    """Detect EMA crossover events."""
    was_above = ema_fast_prev > ema_slow_prev
    is_above  = ema_fast_now > ema_slow_now
    if not was_above and is_above:
        return "Golden Cross"   # bullish
    if was_above and not is_above:
        return "Death Cross"    # bearish
    return None
