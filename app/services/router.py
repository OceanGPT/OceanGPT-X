"""
Router binary classifier: sonar vs biological.
Uses YOLOv11-cls model (ultralytics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from app.core.config import settings
from app.core.state import state


def load_router_model(model_path: str):
    """Load the router model, returning the YOLO instance and class mapping."""
    model = YOLO(model_path)
    class_names = model.names  # {0: 'bio', 1: 'sonar'}
    return model, class_names


def run_router_classification(image_path: Path) -> Dict[str, Any]:
    """Classify a single image as sonar or biological."""
    if state.router_model is None:
        raise RuntimeError("Router model not initialized")

    results = state.router_model.predict(str(image_path), verbose=False)
    probs = results[0].probs

    sonar_idx = None
    for idx, name in state.router_class_names.items():
        if name == "sonar":
            sonar_idx = idx
            break
    if sonar_idx is None:
        raise RuntimeError("Router model has no 'sonar' class")

    sonar_prob = float(probs.data[sonar_idx].item())
    is_sonar = sonar_prob >= settings.router_threshold

    top1_idx = int(probs.top1)
    top1_prob = float(probs.top1conf.item())
    top1_label = state.router_class_names.get(top1_idx, str(top1_idx))

    predicted_type = "sonar" if is_sonar else "biological"
    stage = "sonar_routed" if is_sonar else "bio_routed"

    router_module = {
        "enabled": True,
        "predicted_type": predicted_type,
        "confidence": round(sonar_prob if is_sonar else (1.0 - sonar_prob), 4),
        "model_name": "yolo11n-cls",
        "raw_top1_label": top1_label,
        "raw_top1_confidence": round(top1_prob, 4),
        "sonar_probability": round(sonar_prob, 4),
        "threshold": settings.router_threshold,
    }

    return {
        "stage": stage,
        "predicted_type": predicted_type,
        "confidence": router_module["confidence"],
        "router_module": router_module,
    }
