# 🥇 XAUUSD AI Predictor

Aplikacion Streamlit për parashikimin e Gold (XAUUSD) me Machine Learning ensemble dhe self-learning robot.

## ✨ Features

### 🧠 AI Core
- **4-Model Ensemble** — Random Forest + Gradient Boosting + HistGB + MLP
- **Triple-Barrier Labeling** — ATR-based BUY/SELL/HOLD labels
- **Confluence Score** — 9 technical indicators voting
- **Unanimous Override** — sub-model consensus beats confluence
- **Price-Signal Sync** — predicted price always agrees with classifier votes

### 🎯 Trading Modes
- **Scalping** (5-10 signals/day, ~65% accuracy)
- **Day Trading** (2-3 signals/day, ~75% accuracy)
- **Conservative** (5-10 signals/month, ~90% accuracy)

### 🛡 Pro Filters
- **Kill Zones** — trade only London/NY sessions
- **Multi-Timeframe** — 4H bias + 1H signal + 15M momentum
- **News Filter** — block trades around NFP/CPI/FOMC
- **Smart Money Concepts** — Order Blocks, FVG, BOS
- **Correlation Filter** — DXY/SPX/Yields validation

### 💰 Risk Management
- **Dynamic SL/TP** — based on swing structure
- **Trailing Stop** — break-even after TP1
- **Position Sizing** — 1-2% risk per trade with confidence scaling
- **Auto-reduce** after losing streak

### 🚨 Position Monitor
- **Early Exit Detector** — 8 invalidation checks
- Browser notifications + sound alerts
- Telegram push notifications

### 📊 Analytics
- **Paper Trading Engine** — virtual $10k account, auto-executes signals
- **Performance Analytics** — win rate by session/day, equity curve, Sharpe
- **Monte Carlo Simulation** — 1000+ runs to estimate worst/best/median
- **Robot Brain** — self-learning, auto-tunes thresholds
- **Excel/CSV Export** — multi-sheet workbooks

## 🚀 Setup

```bash
# 1. Clone
git clone https://github.com/arjan0007/Xau.git
cd Xau

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (optional — app works without it)
cp config.py.example config.py
# Edit config.py to add API keys (Telegram, Gemini, etc.)

# 4. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Modulet

| File | Përshkrim |
|------|-----------|
| `app.py` | Main Streamlit dashboard (12 pages) |
| `model.py` | 4-model ensemble + confluence + labels |
| `technical.py` | Technical indicators (RSI, MACD, EMA, etc.) |
| `enhancements.py` | Pro filters: Kill Zones, MTF, SMC, news, sizing |
| `bot_brain.py` | Self-learning engine, auto-tune |
| `position_monitor.py` | Early exit detector |
| `paper_trading.py` | Virtual account, auto-execute signals |
| `backtest.py` | Historical simulation + Monte Carlo |
| `news.py` | RSS news + sentiment analysis |
| `calendar_data.py` | Economic calendar |
| `correlations.py` | DXY/SPX/Yield correlations |
| `seasonality.py` | Monthly/quarterly patterns |
| `journal.py` | Manual trading journal (SQLite) |
| `reports.py` | Excel/CSV export |
| `telegram_bot.py` | Telegram notifications |
| `chart_analyzer.py` | AI chart screenshot analysis |

## ⚠ Disclaimer

Ky aplikacion është **mjet edukativ/eksperimental**. Nuk është këshillë financiare. Tregjet financiare janë me rrezik të lartë — testoj me paper trading para se të rrezikoj para reale.

## 📄 License

MIT
