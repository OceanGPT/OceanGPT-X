from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (directory containing this file's parent)
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

# Ensure YOLOv5 modules are found before any other imports.
# Some environments (e.g. vlm conda env) have a third-party `utils` package
# in site-packages that shadows yolov5's `utils`.
_yolov5_dir = os.getenv("YOLOV5_DIR") or "./yolov5"
_yolov5_abs = _project_root / _yolov5_dir
if _yolov5_abs.exists():
    sys.path.insert(0, str(_yolov5_abs.resolve()))


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    app_title: str = os.getenv("APP_TITLE", "Marine Image Retrieval API")
    app_version: str = os.getenv("APP_VERSION", "0.5.0")

    # Inference
    device: str = os.getenv("DEVICE", "cuda")
    topk: int = _get_int("TOPK", 5)
    threshold: float = _get_float("THRESHOLD", 0.90)
    router_threshold: float = _get_float("ROUTER_THRESHOLD", 0.5)
    use_oceanclip: bool = _get_bool("USE_OCEANCLIP", True)

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 8000)

    # YOLOv5 source code (git clone https://github.com/ultralytics/yolov5)
    yolov5_dir: str = os.getenv("YOLOV5_DIR", "./yolov5")

    # Data files (downloaded by scripts/download_assets.py)
    model_dir: str = os.getenv("MODEL_DIR", "./downloaded_assets/data/bioclip")
    metadata: str = os.getenv("METADATA", "./downloaded_assets/data/metadata/metadata.jsonl")
    index_dir: str = os.getenv("INDEX_DIR", "./downloaded_assets/data/faiss")

    # Model weights (downloaded by scripts/download_assets.py)
    router_model_path: str = os.getenv(
        "ROUTER_MODEL_PATH",
        "./downloaded_assets/models/cls_bio_sonar/best.pt",
    )
    sonar_cls_path: str = os.getenv(
        "SONAR_CLS_PATH",
        "./downloaded_assets/models/sonar/best.pt",
    )
    fish_coral_cls_path: str = os.getenv(
        "FISH_CORAL_CLS_PATH",
        "./downloaded_assets/models/fish_coral_cls/best.pt",
    )
    fish_model_path: str = os.getenv(
        "FISH_MODEL_PATH",
        "./downloaded_assets/models/fish_detector/best.pt",
    )
    coral_model_path: str = os.getenv(
        "CORAL_MODEL_PATH",
        "./downloaded_assets/models/coral_detector/best.pt",
    )
    oceanclip_checkpoint: str = os.getenv(
        "OCEANCLIP_CHECKPOINT",
        "./downloaded_assets/models/oceanclip-bio/epoch_50.pt",
    )
    oceanclip_terms_path: str = os.getenv(
        "OCEANCLIP_TERMS_PATH",
        "./downloaded_assets/models/oceanclip-bio/terms.txt",
    )

    def ensure_yolov5_path(self) -> None:
        yolov5_dir = str(Path(self.yolov5_dir))
        if yolov5_dir not in sys.path:
            sys.path.append(yolov5_dir)

    def as_dict(self) -> dict:
        return {
            "app_title": self.app_title,
            "app_version": self.app_version,
            "device": self.device,
            "topk": self.topk,
            "threshold": self.threshold,
            "router_threshold": self.router_threshold,
            "use_oceanclip": self.use_oceanclip,
            "host": self.host,
            "port": self.port,
            "yolov5_dir": self.yolov5_dir,
            "model_dir": self.model_dir,
            "metadata": self.metadata,
            "index_dir": self.index_dir,
            "router_model_path": self.router_model_path,
            "sonar_cls_path": self.sonar_cls_path,
            "fish_coral_cls_path": self.fish_coral_cls_path,
            "fish_model_path": self.fish_model_path,
            "coral_model_path": self.coral_model_path,
            "oceanclip_checkpoint": self.oceanclip_checkpoint,
            "oceanclip_terms_path": self.oceanclip_terms_path,
        }


settings = Settings()


def print_settings_summary() -> None:
    print("[Config] Current settings:")
    for k, v in settings.as_dict().items():
        print(f"  - {k}: {v}")
