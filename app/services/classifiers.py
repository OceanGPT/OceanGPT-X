from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from app.core.config import settings

from utils.augmentations import classify_transforms, letterbox  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402


def prepare_image_for_classification(image_path: Path, device: str) -> torch.Tensor:
    img_pil = Image.open(image_path).convert("RGB")
    im = np.array(img_pil)[:, :, ::-1].copy()
    img_tensor = classify_transforms(224)(im).unsqueeze(0).to(device)
    return img_tensor


def process_classification_result(model_result, model_names: dict) -> Dict[str, Any]:
    pred = model_result[0].pred[0] if hasattr(model_result[0], "pred") else model_result[0]
    probs = torch.softmax(pred, dim=0)
    top1_idx = probs.argmax().item()
    top1_conf = probs[top1_idx].item()

    all_probs = probs.cpu().numpy()
    all_labels = [model_names[i] for i in range(len(all_probs))]

    return {
        "primary_label": model_names[top1_idx],
        "confidence": top1_conf,
        "all_labels": all_labels,
        "all_probabilities": all_probs.tolist(),
    }


def predict_with_yolo_classifier(model, image: Image.Image, device: str, model_name: str) -> Dict[str, Any]:
    im = np.array(image)[:, :, ::-1].copy()
    img_tensor = classify_transforms(224)(im).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    logits = output[0] if isinstance(output, (list, tuple)) else output

    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(f"{model_name} output is not a Tensor, got {type(logits)}")

    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    elif logits.ndim == 1:
        pass
    else:
        raise RuntimeError(
            f"{model_name} output shape {tuple(logits.shape)} is not a classification logits shape. "
            f"Expected [1, C] or [C]."
        )

    probs = torch.softmax(logits, dim=-1)
    top1_idx = int(torch.argmax(probs).item())
    top1_prob = float(probs[top1_idx].item())

    all_probs = probs.detach().cpu().numpy()
    class_names = model.names if hasattr(model, "names") else {}
    all_labels = [class_names[i] if i in class_names else str(i) for i in range(len(all_probs))]

    return {
        "model_type": "yolo",
        "model_name": model_name,
        "primary_label": class_names[top1_idx] if top1_idx in class_names else str(top1_idx),
        "confidence": top1_prob,
        "all_labels": all_labels,
        "all_probabilities": all_probs.tolist(),
    }


def predict_with_yolo_detector(model, image: Image.Image, device: str, model_name: str) -> Dict[str, Any]:
    img0 = np.array(image)
    img0 = img0[:, :, ::-1].copy()

    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    imgsz = 640

    img = letterbox(img0, new_shape=imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    if pred is None or len(pred) == 0 or pred[0] is None or len(pred[0]) == 0:
        return {
            "model_type": "yolo_detector",
            "model_name": model_name,
            "primary_label": None,
            "confidence": 0.0,
            "all_labels": [],
            "detections": [],
            "note": "No detection above confidence threshold.",
        }

    det = pred[0]
    class_names = model.names if hasattr(model, "names") else {}

    detections = []
    all_labels = []

    for row in det:
        conf = float(row[4].item())
        cls_id = int(row[5].item())
        label = class_names[cls_id] if cls_id in class_names else str(cls_id)

        detections.append({
            "label": label,
            "confidence": conf,
        })

        if label not in all_labels:
            all_labels.append(label)

    best_row = det[:, 4].argmax()
    best_conf = float(det[best_row, 4].item())
    best_cls = int(det[best_row, 5].item())
    primary_label = class_names[best_cls] if best_cls in class_names else str(best_cls)

    return {
        "model_type": "yolo_detector",
        "model_name": model_name,
        "primary_label": primary_label,
        "confidence": best_conf,
        "all_labels": all_labels,
        "detections": detections,
        "note": "Primary label selected from highest-confidence detection.",
    }
