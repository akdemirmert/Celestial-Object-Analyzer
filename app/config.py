"""Application configuration loaded from config.json with env overrides."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

_DEFAULTS = {
    "astrometry_api_key": "",
    "astrometry_timeout_s": 180,
    "max_image_dim": 2400,
    "port": 8777,
}


def load_config() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    env_key = os.environ.get("ASTROMETRY_API_KEY")
    if env_key:
        cfg["astrometry_api_key"] = env_key
    return cfg


CONFIG = load_config()
