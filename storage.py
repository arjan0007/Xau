import os
from pathlib import Path


def data_path(filename: str) -> str:
    base_dir = (
        os.environ.get("XAUUSD_DATA_DIR")
        or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
        or Path(__file__).resolve().parent
    )
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path / filename)
