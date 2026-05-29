import json
import os
from datetime import datetime, timezone

import model as ml
from market_data import fetch_data
from storage import data_path


DEFAULT_TARGETS = "day_trading:1h:21d"


def _parse_targets(raw: str):
    targets = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        pieces = [p.strip() for p in item.split(":")]
        if len(pieces) != 3:
            raise ValueError(f"Invalid training target: {item}. Use mode:interval:period")
        targets.append(tuple(pieces))
    if not targets:
        raise ValueError("No training targets configured.")
    return targets


def run_training():
    targets = _parse_targets(os.environ.get("XAUUSD_TRAIN_TARGETS", DEFAULT_TARGETS))
    results = []

    for mode, interval, period in targets:
        print(f"[train] mode={mode} interval={interval} period={period}", flush=True)
        ml.set_trading_mode(mode)
        ml.set_model_tag(f"{mode}_{interval}")

        df = fetch_data(interval, period)
        if df.empty:
            raise RuntimeError(f"No market data returned for {mode}:{interval}:{period}")

        model, scaler, accuracy = ml.train(df)
        result = {
            "mode": mode,
            "interval": interval,
            "period": period,
            "rows": int(len(df)),
            "accuracy": float(accuracy),
            "model_path": ml.MODEL_PATH,
            "scaler_path": ml.SCALER_PATH,
            "price_model_path": ml.PRICE_MODEL_PATH,
            "price_scaler_path": ml.PRICE_SCALER_PATH,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[train] done accuracy={accuracy} rows={len(df)}", flush=True)
        results.append(result)

    status = {
        "ok": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "targets": results,
    }
    with open(data_path("training_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print("[train] all targets complete", flush=True)
    return status


if __name__ == "__main__":
    run_training()
