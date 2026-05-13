import requests
import feedparser
from datetime import datetime, timedelta, timezone
import re

# High-impact events that affect gold
GOLD_IMPACT_EVENTS = [
    "Non-Farm Payrolls", "NFP", "CPI", "Inflation", "Federal Reserve",
    "Fed", "FOMC", "Interest Rate", "GDP", "Unemployment", "PPI",
    "Retail Sales", "ISM", "PMI", "ADP", "Trade Balance",
    "Consumer Confidence", "Durable Goods", "Jackson Hole",
    "ECB", "BOE", "BOJ", "SNB", "RBA",
]


def fetch_calendar() -> list[dict]:
    """
    Fetch economic calendar events from ForexFactory RSS + fallback static list.
    Returns list of event dicts sorted by date.
    """
    events = []

    # Try ForexFactory RSS (public, no auth)
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        feed = feedparser.parse(url)
        for entry in feed.entries[:30]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            published = entry.get("published", "")
            impact = _detect_impact(title + " " + summary)
            currency = _detect_currency(title + " " + summary)

            events.append({
                "event": title[:80],
                "currency": currency,
                "impact": impact,
                "time": _parse_date(published),
                "forecast": _extract_value(summary, "Forecast"),
                "previous": _extract_value(summary, "Previous"),
                "actual": _extract_value(summary, "Actual"),
                "affects_gold": _affects_gold(title),
            })
    except Exception:
        pass

    # If RSS failed or empty, return upcoming known events
    if not events:
        events = _static_upcoming_events()

    # Sort: gold-affecting first, then by impact
    impact_order = {"High 🔴": 0, "Medium 🟡": 1, "Low ⚪": 2}
    events.sort(key=lambda x: (0 if x.get("affects_gold") else 1,
                                impact_order.get(x["impact"], 2)))
    return events


def _static_upcoming_events() -> list[dict]:
    """Fallback: next known high-impact events (approximate schedule)."""
    now = datetime.now()
    # Next approximate dates for recurring events
    events = [
        {"event": "FOMC Meeting Minutes", "currency": "USD", "impact": "High 🔴",
         "time": "Të mërkurën 20:00", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US Non-Farm Payrolls (NFP)", "currency": "USD", "impact": "High 🔴",
         "time": "E premte e parë e muajit 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US CPI (Inflation)", "currency": "USD", "impact": "High 🔴",
         "time": "~12 të muajit 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "Fed Interest Rate Decision", "currency": "USD", "impact": "High 🔴",
         "time": "Çdo 6 javë 21:00", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US GDP (Quarterly)", "currency": "USD", "impact": "High 🔴",
         "time": "Fund çereku 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "ECB Interest Rate Decision", "currency": "EUR", "impact": "High 🔴",
         "time": "Çdo 6 javë 15:15", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US PPI", "currency": "USD", "impact": "Medium 🟡",
         "time": "~14 të muajit 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US Retail Sales", "currency": "USD", "impact": "Medium 🟡",
         "time": "~17 të muajit 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": False},
        {"event": "US ADP Employment", "currency": "USD", "impact": "Medium 🟡",
         "time": "E mërkurë para NFP 15:15", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": False},
        {"event": "US ISM Manufacturing PMI", "currency": "USD", "impact": "Medium 🟡",
         "time": "1 muaji 17:00", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": False},
        {"event": "BOE Interest Rate Decision", "currency": "GBP", "impact": "High 🔴",
         "time": "Çdo 6 javë 14:00", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": True},
        {"event": "US Unemployment Claims", "currency": "USD", "impact": "Medium 🟡",
         "time": "Çdo të enjte 15:30", "forecast": "—", "previous": "—", "actual": "—", "affects_gold": False},
    ]
    return events


def _detect_impact(text: str) -> str:
    text_l = text.lower()
    high_kw = ["fomc", "nfp", "non-farm", "cpi", "interest rate", "gdp", "inflation",
               "federal reserve", "jackson hole", "ecb rate", "boe rate"]
    mid_kw = ["ppi", "retail", "adp", "ism", "pmi", "trade", "durable", "confidence"]
    if any(k in text_l for k in high_kw):
        return "High 🔴"
    if any(k in text_l for k in mid_kw):
        return "Medium 🟡"
    return "Low ⚪"


def _affects_gold(text: str) -> bool:
    text_l = text.lower()
    return any(k.lower() in text_l for k in GOLD_IMPACT_EVENTS)


def _detect_currency(text: str) -> str:
    for c in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY"]:
        if c in text.upper():
            return c
    return "USD"


def _extract_value(html: str, label: str) -> str:
    pattern = rf"{label}[:\s]*([0-9.\-+%KMB]+)"
    m = re.search(pattern, html, re.IGNORECASE)
    return m.group(1) if m else "—"


def _parse_date(date_str: str) -> str:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d %b  %H:%M")
        except Exception:
            continue
    return date_str[:16] if date_str else "—"
