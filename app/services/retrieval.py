from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

from app.core.config import settings
from app.core.state import state


def resolve_device(preferred_device: str) -> str:
    if preferred_device.startswith("cuda") and torch.cuda.is_available():
        return preferred_device
    return "cpu"


def load_bioclip(device: str):
    out = open_clip.create_model_from_pretrained(f"local-dir:{settings.model_dir}")

    if isinstance(out, (tuple, list)) and len(out) == 2:
        model, preprocess = out
    elif isinstance(out, (tuple, list)) and len(out) == 3:
        model, preprocess, _ = out
    else:
        raise RuntimeError("Unexpected return from create_model_from_pretrained")

    model = model.to(device).eval()
    return model, preprocess


def load_index():
    index_path = Path(settings.index_dir) / "index.faiss"
    id_map_path = Path(settings.index_dir) / "id_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map not found: {id_map_path}")

    index = faiss.read_index(str(index_path))
    with id_map_path.open("r", encoding="utf-8") as f:
        id_map = json.load(f)

    return index, id_map


def load_metadata():
    metadata_path = Path(settings.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    id2meta: Dict[str, Dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "id" in rec:
                id2meta[rec["id"]] = rec
    return id2meta


@torch.no_grad()
def encode_single_image(model, preprocess, image_path: Path, device: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img_t)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def simplify_topk_result(
    rank: int,
    faiss_i: int,
    sim: float,
    id_map: List[str],
    id2meta: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if int(faiss_i) < 0:
        return None

    img_id = id_map[int(faiss_i)]
    meta = id2meta.get(img_id, {})

    return {
        "rank": rank,
        "id": img_id,
        "similarity": float(sim),
        "labels": meta.get("preferred_labels", []),
    }


def build_retrieval_module(
    sims: np.ndarray,
    idxs: np.ndarray,
    id_map: List[str],
    id2meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    topk_results: List[Dict[str, Any]] = []

    for rank, (faiss_i, sim) in enumerate(zip(idxs, sims), start=1):
        item = simplify_topk_result(rank, int(faiss_i), float(sim), id_map, id2meta)
        if item is not None:
            topk_results.append(item)

    best_idx = int(idxs[0]) if len(idxs) > 0 and int(idxs[0]) >= 0 else None
    best_sim = float(sims[0]) if len(sims) > 0 else None

    retrieval_module = {
        "enabled": True,
        "db_hit": False,
        "threshold": settings.threshold,
        "top1_similarity": best_sim,
        "top1_id": None,
        "top1_labels": [],
        "topk": topk_results,
    }

    if best_idx is None:
        return retrieval_module

    top1_id = id_map[best_idx]
    top1_meta = id2meta.get(top1_id, {})

    retrieval_module["top1_id"] = top1_id
    retrieval_module["top1_labels"] = top1_meta.get("preferred_labels", [])
    retrieval_module["db_hit"] = best_sim is not None and best_sim >= settings.threshold

    return retrieval_module


def run_faiss_query(image_path: Path) -> Dict[str, Any]:
    if (
        state.model is None
        or state.preprocess is None
        or state.index is None
        or state.id_map is None
        or state.id2meta is None
        or state.runtime_device is None
    ):
        raise RuntimeError("Service not initialized correctly")

    q_feat = encode_single_image(
        state.model,
        state.preprocess,
        image_path,
        device=state.runtime_device,
    )
    sims, idxs = state.index.search(q_feat, settings.topk)
    sims, idxs = sims[0], idxs[0]

    retrieval_module = build_retrieval_module(sims, idxs, state.id_map, state.id2meta)

    top1_meta = {}
    if retrieval_module["top1_id"] is not None:
        top1_meta = state.id2meta.get(retrieval_module["top1_id"], {})

    return {
        "device": state.runtime_device,
        "retrieval_module": retrieval_module,
        "top1_meta": top1_meta,
    }
