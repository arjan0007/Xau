import feedparser
import requests
from datetime import datetime
import re

# RSS feeds for gold/forex news (no API key needed)
RSS_FEEDS = [
    ("Kitco News", "https://www.kitco.com/rss/kitconews.xml"),
    ("Reuters Commodities", "https://feeds.reuters.com/reuters/businessNews"),
    ("FXStreet Gold", "https://www.fxstreet.com/rss/news"),
    ("Investing.com Gold", "https://www.investing.com/rss/news_14.rss"),
]

# Sentiment keywords
POSITIVE_WORDS = [
    "rise", "rises", "rally", "rallies", "gain", "gains", "surge", "surges",
    "jump", "jumps", "high", "higher", "record", "bull", "bullish",
    "up", "upside", "buy", "buying", "strong", "strength", "boost",
    "safe haven", "demand", "inflation", "uncertainty", "war", "crisis",
    "rritje", "fitimet", "lart",
]

NEGATIVE_WORDS = [
    "fall", "falls", "drop", "drops", "decline", "declines", "plunge",
    "plunges", "sink", "sinks", "low", "lower", "bear", "bearish",
    "down", "downside", "sell", "selling", "weak", "weakness",
    "rate hike", "dollar strength", "risk-on", "recovery",
    "ulje", "bie", "poshte",
]

GOLD_KEYWORDS = [
    "gold", "xauusd", "xau", "bullion", "precious metal",
    "ar", "gold price", "yellow metal",
]


def fetch_news(max_items: int = 20) -> list[dict]:
    """Fetch latest gold-related news from RSS feeds."""
    articles = []

    for source, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "#")
                published = entry.get("published", "")

                # Filter gold-related
                text_lower = (title + " " + summary).lower()
                if any(kw in text_lower for kw in GOLD_KEYWORDS):
                    sentiment, score = analyze_sentiment(title + " " + summary)
                    articles.append({
                        "source": source,
                        "title": title[:120],
                        "summary": _clean_html(summary)[:200],
                        "link": link,
                        "published": _parse_date(published),
                        "sentiment": sentiment,
                        "score": score,
                    })
        except Exception:
            continue

    # Sort by date desc, deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        key = a["title"][:50]
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique[:max_items]


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Simple keyword-based sentiment analysis."""
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    if total == 0:
        return "Neutral ⚪", 0.0
    score = (pos - neg) / total
    if score > 0.2:
        return "Bullish 🟢", round(score, 2)
    elif score < -0.2:
        return "Bearish 🔴", round(score, 2)
    else:
        return "Neutral ⚪", round(score, 2)


def overall_sentiment(articles: list[dict]) -> tuple[str, float]:
    """Aggregate sentiment from all articles."""
    if not articles:
        return "Neutral ⚪", 0.0
    scores = [a["score"] for a in articles]
    avg = sum(scores) / len(scores)
    if avg > 0.15:
        return "Bullish 🟢", round(avg, 2)
    elif avg < -0.15:
        return "Bearish 🔴", round(avg, 2)
    return "Neutral ⚪", round(avg, 2)


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_date(date_str: str) -> str:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d %b  %H:%M")
        except Exception:
            continue
    return date_str[:16] if date_str else "—"
