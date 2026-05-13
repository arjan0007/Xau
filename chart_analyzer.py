"""
Chart screenshot analyzer.

Takes an uploaded chart image (TradingView, MT4/5, OANDA, etc.), sends it to a
vision-capable LLM (Anthropic Claude), and returns a structured trade plan:
direction (BUY/SELL/WAIT), entry, SL, TP1, TP2, confidence, and reasoning.
"""

from __future__ import annotations
import base64
import json
import os
import re
from typing import Optional


SYSTEM_PROMPT = """You are an elite XAUUSD (Gold) technical analyst with 20+ years
of experience reading price-action charts. You analyze chart screenshots and
produce strict, risk-managed trade plans.

You read whatever is visible: candlesticks, trend, support/resistance, moving
averages, RSI, MACD, Bollinger Bands, volume, and chart patterns
(double tops/bottoms, head & shoulders, flags, triangles, breakouts, divergences).

You ALWAYS respond with a single JSON object — no prose outside it. Schema:

{
  "direction":   "BUY" | "SELL" | "WAIT",
  "should_enter": true | false,
  "confidence":  0-100,
  "current_price": <float or null>,
  "entry":       <float>,
  "stop_loss":   <float>,
  "take_profit_1": <float>,
  "take_profit_2": <float>,
  "risk_reward": <float>,
  "timeframe":   "<e.g. 1h, 4h, 1d>",
  "trend":       "Bullish" | "Bearish" | "Range" | "Unclear",
  "key_levels":  {"support": [<float>, ...], "resistance": [<float>, ...]},
  "patterns":    ["<pattern name>", ...],
  "indicators_read": {
      "rsi": "<value or condition>",
      "macd": "<value or condition>",
      "ma": "<position vs price>",
      "volume": "<rising/falling/normal>"
  },
  "reasoning":   "<3-5 sentences explaining the setup>",
  "warnings":    ["<reason to NOT enter, if any>", ...],
  "language":    "sq"
}

Rules:
- "should_enter" = false if signal quality is mediocre, news risk, against major
  trend, or risk:reward < 1.5.
- SL must be placed at a structural invalidation level (beyond swing high/low),
  NOT a fixed distance.
- TP1 should respect the next major S/R; TP2 the one after.
- If chart is unreadable (blurry, no candles, wrong asset), set direction "WAIT",
  should_enter false, confidence 0, and put the reason in warnings.
- Reasoning MUST be in Albanian (Shqip). Everything else stays in the JSON keys above.
- Output ONLY the JSON object. No markdown fences, no commentary.
"""


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model response, tolerant of stray fences."""
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Find the outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("Modeli nuk ktheu JSON të vlefshëm.")
    return json.loads(m.group(0))


def _build_user_prompt(current_price: Optional[float]) -> str:
    user_text = (
        "Analizo këtë screenshot të chartit të XAUUSD/Gold dhe ma kthe planin e "
        "tregtisë sipas skemës JSON të kërkuar. "
    )
    if current_price:
        user_text += f"Çmimi aktual live: ${current_price:,.2f}. "
    user_text += (
        "Shqyrto trendin, mbështetjen/rezistencën, RSI/MACD nëse duken, dhe "
        "modelet e kandelave. Vendos SL pas pikës strukturore të invalidimit. "
        "Mos hyr nëse R:R < 1.5."
    )
    return user_text


def analyze_chart(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    current_price: Optional[float] = None,
    image_mime: str = "image/png",
    model: str = "claude-sonnet-4-5",
    provider: str = "anthropic",
) -> dict:
    """
    Send the chart image to a vision LLM (Anthropic or Google Gemini) and
    parse the structured response.

    provider: "anthropic" or "gemini"
    """
    user_text = _build_user_prompt(current_price)

    if provider == "gemini":
        return _analyze_gemini(image_bytes, api_key, image_mime, model, user_text)

    return _analyze_anthropic(image_bytes, api_key, image_mime, model, user_text)


def _analyze_anthropic(image_bytes, api_key, image_mime, model, user_text):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"_error": "Mungon ANTHROPIC_API_KEY."}
    try:
        import anthropic
    except ImportError:
        return {"_error": "pip install anthropic"}

    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": image_mime, "data": b64,
                    }},
                    {"type": "text", "text": user_text},
                ],
            }],
        )
        plan = _extract_json(resp.content[0].text)
        plan["_raw_model"] = f"anthropic/{model}"
        return plan
    except Exception as e:
        return {"_error": f"Gabim Anthropic API: {e}"}


def _analyze_gemini(image_bytes, api_key, image_mime, model, user_text):
    """Use Google Gemini vision (free tier, 1500 req/day)."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"_error": "Mungon GEMINI_API_KEY. Merre falas: https://aistudio.google.com/apikey"}

    # Gemini model names differ from Claude
    model_name = model
    if "claude" in model_name.lower():
        model_name = "gemini-2.0-flash"

    import urllib.request
    import urllib.error

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": image_mime, "data": b64}},
                {"text": user_text},
            ],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1500,
            "responseMimeType": "application/json",
        },
    }

    # Auto-retry + fallback chain when servers are overloaded (503)
    fallback_chain = [model_name]
    for fb in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
        if fb not in fallback_chain:
            fallback_chain.append(fb)

    import time as _time
    last_err = None
    body_json = json.dumps(body).encode("utf-8")

    for m in fallback_chain:
        try_url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        for attempt in range(3):
            req = urllib.request.Request(
                try_url, data=body_json,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                plan = _extract_json(text)
                plan["_raw_model"] = f"gemini/{m}"
                if m != model_name:
                    plan["_fallback_used"] = f"Modeli {model_name} ishte i ngarkuar — u përdor {m}"
                return plan
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", errors="ignore")
                last_err = f"({e.code}) {msg[:200]}"
                if e.code in (503, 429, 500):
                    _time.sleep(2 * (attempt + 1))   # backoff
                    continue
                return {"_error": f"Gabim Gemini API {last_err}"}
            except Exception as e:
                last_err = str(e)
                _time.sleep(1)
                continue
        # this model failed all attempts → try next in chain

    return {"_error": f"Gemini i mbingarkuar në të gjitha modelet. Provo pas 1-2 min. Detaji: {last_err}"}


def format_decision_badge(plan: dict) -> tuple[str, str, str]:
    """Return (icon, label, color) for the headline decision."""
    if plan.get("_error"):
        return ("⚠️", "GABIM", "#ff5252")
    direction = plan.get("direction", "WAIT")
    should = plan.get("should_enter", False)
    if not should or direction == "WAIT":
        return ("⏸", "MOS HYR ENDE", "#ffd600")
    if direction == "BUY":
        return ("🟢", "HYR BUY", "#00c853")
    if direction == "SELL":
        return ("🔴", "HYR SELL", "#d50000")
    return ("⏸", "PRIT", "#ffd600")
