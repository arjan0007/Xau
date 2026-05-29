import os
import time
from datetime import datetime, timezone

from train_job import run_training


def _enabled() -> bool:
    return os.environ.get("XAUUSD_BACKGROUND_TRAIN", "1").strip().lower() not in {"0", "false", "no", "off"}


def _interval_seconds() -> int:
    raw = os.environ.get("XAUUSD_BACKGROUND_RETRAIN_HOURS", "4")
    try:
        hours = max(float(raw), 0.25)
    except ValueError:
        hours = 4.0
    return int(hours * 3600)


def main():
    if not _enabled():
        print("[background-trainer] disabled", flush=True)
        return

    interval = _interval_seconds()
    print(f"[background-trainer] started interval={interval}s", flush=True)

    while True:
        started = datetime.now(timezone.utc).isoformat()
        try:
            print(f"[background-trainer] training cycle started {started}", flush=True)
            run_training()
            print("[background-trainer] training cycle complete", flush=True)
        except Exception as exc:
            print(f"[background-trainer] training failed: {exc}", flush=True)

        time.sleep(interval)


if __name__ == "__main__":
    main()
