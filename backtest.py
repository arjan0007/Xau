import pandas as pd
import numpy as np
import model as ml


def run_backtest(df: pd.DataFrame, initial_balance: float = 10000.0,
                 lot_size: float = 0.1, spread_pips: float = 0.3):
    """
    Simulate trading based on model signals on historical data.
    Returns: summary dict + trades DataFrame
    """
    df = df.copy()
    df = ml.add_features(df)
    df["label"] = ml.make_labels(df)
    df = df.dropna()

    if len(df) < 50:
        return None, None

    # Train on first 70%, test on last 30%
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[ml.FEATURE_COLS])
    y_train = train_df["label"]

    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   min_samples_split=20, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    X_test = scaler.transform(test_df[ml.FEATURE_COLS])
    preds = model.predict(X_test)

    # Simulate trades
    # Gold spec: 1 lot = 100 oz; pip = $0.01 → pip_value = lot_size * $1
    balance = initial_balance
    trades = []
    position = None  # {"type": "BUY"/"SELL", "entry": price, "open_time": ts}
    pip_value = lot_size * 1.0   # USD per pip (0.01) per `lot_size` lots
    PIP = 0.01

    for i, (idx, row) in enumerate(test_df.iterrows()):
        pred_label = preds[i]
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = signal_map[pred_label]
        price = row["Close"]

        # Close existing position if signal reversed
        if position:
            if (position["type"] == "BUY" and signal == "SELL") or \
               (position["type"] == "SELL" and signal == "BUY") or \
               signal == "HOLD":
                entry = position["entry"]
                price_diff = (price - entry) if position["type"] == "BUY" else (entry - price)
                pnl_pips = price_diff / PIP - spread_pips      # in pips (0.01)
                pnl_usd  = pnl_pips * pip_value
                balance += pnl_usd
                trades.append({
                    "open_time": position["open_time"],
                    "close_time": idx,
                    "type": position["type"],
                    "entry": round(entry, 2),
                    "exit": round(price, 2),
                    "pnl_pips": round(pnl_pips, 2),
                    "pnl_usd": round(pnl_usd, 2),
                    "balance": round(balance, 2),
                    "result": "WIN" if pnl_usd > 0 else "LOSS",
                })
                position = None

        # Open new position
        if signal in ("BUY", "SELL") and position is None:
            position = {"type": signal, "entry": price, "open_time": idx}

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return None, None

    total_trades = len(trades_df)
    wins = (trades_df["result"] == "WIN").sum()
    losses = total_trades - wins
    win_rate = round(wins / total_trades * 100, 1)
    total_pnl = round(trades_df["pnl_usd"].sum(), 2)
    avg_win = round(trades_df[trades_df["result"] == "WIN"]["pnl_usd"].mean(), 2) if wins > 0 else 0
    avg_loss = round(trades_df[trades_df["result"] == "LOSS"]["pnl_usd"].mean(), 2) if losses > 0 else 0

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for bal in trades_df["balance"]:
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_loss = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Sharpe (simplified)
    returns = trades_df["pnl_usd"]
    sharpe = round(returns.mean() / (returns.std() + 1e-9), 2)

    summary = {
        "total_trades": total_trades,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": round(max_dd, 2),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "final_balance": round(balance, 2),
    }

    return summary, trades_df


def monte_carlo(trades_df, initial_balance: float = 10000.0, runs: int = 1000):
    """Monte Carlo: randomly reorder the historical trade P&L stream
    `runs` times to estimate worst/best/median outcomes."""
    if trades_df is None or trades_df.empty:
        return None
    pnls = trades_df["pnl_usd"].values
    finals = []
    max_dds = []
    for _ in range(runs):
        shuffled = np.random.permutation(pnls)
        bal = initial_balance
        peak = bal
        dd = 0
        for p in shuffled:
            bal += p
            if bal > peak: peak = bal
            cur_dd = (peak - bal) / peak * 100 if peak > 0 else 0
            if cur_dd > dd: dd = cur_dd
        finals.append(bal)
        max_dds.append(dd)
    finals = np.array(finals)
    return {
        "runs": runs,
        "final_p5":  round(float(np.percentile(finals, 5)), 2),
        "final_p50": round(float(np.percentile(finals, 50)), 2),
        "final_p95": round(float(np.percentile(finals, 95)), 2),
        "dd_p5":  round(float(np.percentile(max_dds, 5)), 2),
        "dd_p50": round(float(np.percentile(max_dds, 50)), 2),
        "dd_p95": round(float(np.percentile(max_dds, 95)), 2),
        "prob_profit": round(float((finals > initial_balance).mean() * 100), 2),
        "finals": finals.tolist(),
    }
