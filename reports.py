"""
Export & reporting utilities — pulls together everything the user did and
spits it out as CSV / Excel-friendly DataFrames that the UI can offer as
download buttons.
"""
import pandas as pd
import io
from datetime import datetime


def export_all_to_excel() -> bytes:
    """Bundle paper trades, journal, bot brain into a multi-sheet workbook."""
    import paper_trading as pt
    import bot_brain as bb
    import journal as jrn

    sheets = {}
    try:
        sheets["Paper Trades"] = pt.get_closed_trades(limit=10000)
    except Exception:
        sheets["Paper Trades"] = pd.DataFrame()
    try:
        sheets["AI Predictions"] = bb.get_recent_predictions(limit=10000)
    except Exception:
        sheets["AI Predictions"] = pd.DataFrame()
    try:
        sheets["Manual Journal"] = jrn.get_all_trades()
    except Exception:
        sheets["Manual Journal"] = pd.DataFrame()

    try:
        s = pt.get_stats()
        sheets["Summary"] = pd.DataFrame([
            ["Starting balance", f"${s.get('starting',0):,.2f}"],
            ["Current balance",  f"${s.get('balance',0):,.2f}"],
            ["Total trades",     s.get("trades", 0)],
            ["Win rate",         f"{s.get('win_rate',0)}%"],
            ["Profit factor",    s.get("profit_factor", 0)],
            ["Total P&L",        f"${s.get('total_pnl',0):,.2f}"],
            ["Max drawdown",     f"{s.get('max_dd_pct',0)}%"],
            ["Sharpe ratio",     s.get("sharpe", 0)],
            ["Return",           f"{s.get('return_pct',0)}%"],
            ["Best trade",       f"${s.get('best_trade',0):,.2f}"],
            ["Worst trade",      f"${s.get('worst_trade',0):,.2f}"],
            ["Generated at",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ], columns=["Metric", "Value"])
    except Exception:
        sheets["Summary"] = pd.DataFrame()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            try:
                df.to_excel(writer, sheet_name=name[:31], index=False)
            except Exception:
                pd.DataFrame().to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def export_csv(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")
