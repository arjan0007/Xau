import anthropic
import pandas as pd
import numpy as np
from datetime import datetime


def build_market_context(
    df: pd.DataFrame,
    signal: str,
    confidence: float,
    current_price: float,
    acc: float,
    supports: list,
    resistances: list,
    fib_levels: dict,
    patterns: list,
    sentiment: str,
    news_headlines: list,
) -> str:
    """Build a rich market context string for the AI assistant."""
    latest = df.iloc[-1]

    # Calculate basic stats
    price_7d_ago = df["Close"].iloc[-min(7*24, len(df))]
    change_7d = (current_price - float(price_7d_ago)) / float(price_7d_ago) * 100
    high_7d = float(df["High"].tail(min(7*24, len(df))).max())
    low_7d  = float(df["Low"].tail(min(7*24, len(df))).min())

    sup_str = ", ".join([f"${s:,.2f}" for s in supports[:3]]) if supports else "N/A"
    res_str = ", ".join([f"${r:,.2f}" for r in resistances[:3]]) if resistances else "N/A"
    fib_str = " | ".join([f"{k:.3f}→${v:,.2f}" for k, v in fib_levels.items()]) if fib_levels else "N/A"
    pat_str = ", ".join(patterns) if patterns else "Asnjë"
    news_str = "\n".join([f"  - {h}" for h in news_headlines[:5]]) if news_headlines else "  Nuk ka lajme"

    ctx = f"""
=== XAUUSD ANALIZA LIVE — {datetime.now().strftime('%d %b %Y %H:%M')} ===

ÇMIMI:
  • Aktual:     ${current_price:,.2f}
  • Ndryshim 7d: {change_7d:+.2f}%
  • High 7d:    ${high_7d:,.2f}
  • Low 7d:     ${low_7d:,.2f}

SINJALI AI:
  • Sinjali:    {signal}
  • Konfidenca: {confidence}%
  • Saktësia:   {acc}%

INDIKATORËT TEKNIKË:
  • RSI(14):    {float(df.get('rsi', pd.Series([50])).iloc[-1] if hasattr(df, 'get') else 50):.1f}
  • MACD hist:  {float(df.get('macd_hist', pd.Series([0])).iloc[-1] if hasattr(df, 'get') else 0):.4f}
  • EMA9 > EMA21: {'Po (Bullish)' if float(df.get('ema_9', pd.Series([0])).iloc[-1] if hasattr(df, 'get') else 0) > float(df.get('ema_21', pd.Series([0])).iloc[-1] if hasattr(df, 'get') else 0) else 'Jo (Bearish)'}

NIVELET:
  • Support:    {sup_str}
  • Resistance: {res_str}
  • Fibonacci:  {fib_str}

CHART PATTERNS: {pat_str}

SENTIMENT LAJMEVE: {sentiment}
LAJMET E FUNDIT:
{news_str}
"""
    return ctx.strip()


def ask_ai(question: str, market_context: str, api_key: str, chat_history: list) -> str:
    """
    Send a question to Claude with market context.
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    if not api_key:
        return "⚠️ Vendos Anthropic API key-n në config.py për të përdorur asistentin AI."

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = f"""Ti je një analist profesional i tregut të arit (XAUUSD) dhe forex.
Ke njohuri të thella në analizën teknike dhe fundamentale.
Jep përgjigje të qarta, praktike dhe të sinqerta shqip.
Gjithmonë përmendo risk-un dhe mos jep këshilla financiare absolute.
Bazoji përgjigjet te të dhënat reale të tregut të dhëna më poshtë.

{market_context}

Rregulla:
- Përgjigju gjithmonë shqip
- Ji konkret me numra dhe nivele
- Gjithmonë përmendo "ky nuk është këshillë financiare"
- Nëse pyetja nuk lidhet me tregtinë, thuaj se je specializuar vetëm për XAUUSD
"""

    messages = chat_history.copy()
    messages.append({"role": "user", "content": question})

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "❌ API key i gabuar. Kontrollo `ANTHROPIC_API_KEY` në config.py"
    except anthropic.RateLimitError:
        return "⏱ Rate limit i arritur. Provo pas pak sekondash."
    except Exception as e:
        return f"❌ Gabim: {str(e)}"


QUICK_QUESTIONS = [
    "A duhet të blej gold tani?",
    "Ku është stop loss i mirë?",
    "Cili është target i radhës?",
    "Si është trendi aktual?",
    "A është gold overbought?",
    "Çfarë thotë RSI tani?",
    "Ku është support më i fortë?",
    "A pret ndonjë event të madh?",
]
