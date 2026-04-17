from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from PIL import Image

from app.core.config import settings
from app.core.state import state
from app.services.oceanclip_service import predict_with_oceanclip
from app.services.classifiers import predict_with_yolo_classifier, predict_with_yolo_detector
from app.services.retrieval import run_faiss_query
from app.services.router import run_router_classification


def generate_request_id() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{now}_{short_id}"


def infer_image_type_from_meta(meta: Dict[str, Any]) -> str:
    domain = meta.get("domain")
    if domain in ("sonar", "biological"):
        return domain
    return "unknown"


def build_default_modules() -> Dict[str, Any]:
    return {
        "retrieval": None,
        "router": None,
        "sonar": None,
        "fish_coral": None,
        "fish": None,
        "coral": None,
        "oceanclip": None,
        "fusion": None,
    }


def build_final_result(
    status: str,
    source: Optional[str],
    image_type: Optional[str],
    primary_label: Optional[str],
    all_labels: Optional[List[str]],
    confidence: Optional[float],
    display_text: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "image_type": image_type,
        "primary_label": primary_label,
        "all_labels": all_labels or [],
        "confidence": confidence,
        "display_text": display_text,
        "note": note,
    }


def extract_name_from_bioclip_term(term: Optional[str]) -> str:
    if not term:
        return ""

    s = str(term).strip()
    if not s:
        return ""

    if ">" in s:
        parts = [p.strip() for p in s.split(">") if p.strip()]
        if parts:
            return parts[-1]

    return s


def _normalize_label(label: str) -> str:
    if not label:
        return ""
    return label.strip().lower().replace("_", " ").replace("  ", " ")


def _labels_match(detector_label: str, oceanclip_term: str) -> bool:
    d = _normalize_label(detector_label)
    o = _normalize_label(extract_name_from_bioclip_term(oceanclip_term))
    if not d or not o:
        return False
    if d == o:
        return True
    d_genus = d.split()[0] if d.split() else ""
    o_genus = o.split()[0] if o.split() else ""
    return bool(d_genus and o_genus and d_genus == o_genus)


def build_candidate(
    source: str,
    image_type: str,
    primary_label: Optional[str],
    all_labels: Optional[List[str]],
    confidence: Optional[float],
    note: str,
) -> Optional[Dict[str, Any]]:
    if not primary_label:
        return None

    return {
        "source": source,
        "image_type": image_type,
        "primary_label": primary_label,
        "all_labels": all_labels or [],
        "confidence": float(confidence) if confidence is not None else 0.0,
        "note": note,
    }


def build_sonar_candidate(sonar_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sonar_result:
        return None

    return build_candidate(
        source="sonar_model",
        image_type="sonar",
        primary_label=sonar_result.get("primary_label"),
        all_labels=sonar_result.get("all_labels"),
        confidence=sonar_result.get("confidence"),
        note="Candidate from sonar YOLO classifier.",
    )


def build_fish_candidate(fish_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not fish_result:
        return None

    return build_candidate(
        source=fish_result.get("model_name", "fish_model"),
        image_type="fish",
        primary_label=fish_result.get("primary_label"),
        all_labels=fish_result.get("all_labels"),
        confidence=fish_result.get("confidence"),
        note="Candidate from fish YOLO detector.",
    )


def build_coral_candidate(coral_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not coral_result:
        return None

    return build_candidate(
        source=coral_result.get("model_name", "coral_model"),
        image_type="coral",
        primary_label=coral_result.get("primary_label"),
        all_labels=coral_result.get("all_labels"),
        confidence=coral_result.get("confidence"),
        note="Candidate from coral YOLO detector.",
    )


def build_oceanclip_candidate(oceanclip_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not oceanclip_result:
        return None

    primary_match = oceanclip_result.get("primary_match")
    if not isinstance(primary_match, dict):
        return None

    primary_term = primary_match.get("term")
    primary_label = extract_name_from_bioclip_term(primary_term)

    matches = oceanclip_result.get("matches") or []
    all_labels = []
    for m in matches:
        if isinstance(m, dict) and "term" in m:
            all_labels.append(m["term"])

    return build_candidate(
        source="oceanclip",
        image_type="biological",
        primary_label=primary_label,
        all_labels=all_labels,
        confidence=oceanclip_result.get("confidence"),
        note="Candidate from OceanCLIP.",
    )


def fuse_by_highest_confidence(candidates: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    valid_candidates = [c for c in candidates if c is not None and c.get("primary_label")]

    if not valid_candidates:
        return {
            "selected_source": None,
            "image_type": "unknown",
            "primary_label": None,
            "confidence": None,
            "all_labels": [],
            "reason": "No valid prediction candidate was produced.",
            "display_text": "Result: Undetermined",
            "candidates": [],
            "domain_decision": "unknown",
        }

    best = max(valid_candidates, key=lambda x: x.get("confidence", 0.0))

    merged_labels = []
    for c in valid_candidates:
        for label in c.get("all_labels", []):
            if label not in merged_labels:
                merged_labels.append(label)

    return {
        "selected_source": best["source"],
        "image_type": best["image_type"],
        "primary_label": best["primary_label"],
        "confidence": best["confidence"],
        "all_labels": merged_labels,
        "reason": (
            f"Selected candidate with highest confidence: "
            f"source={best['source']}, label={best['primary_label']}, confidence={best['confidence']:.4f}."
        ),
        "display_text": f"Result: {best['primary_label']}",
        "candidates": valid_candidates,
        "domain_decision": "unknown_fallback",
    }


def fuse_sonar(sonar_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    candidate = build_sonar_candidate(sonar_result)
    if not candidate:
        return {
            "selected_source": None,
            "image_type": "sonar",
            "primary_label": None,
            "confidence": None,
            "all_labels": [],
            "reason": "Router predicted sonar but sonar model produced no result.",
            "display_text": "Result: Undetermined",
            "candidates": [],
            "domain_decision": "sonar",
        }

    return {
        "selected_source": candidate["source"],
        "image_type": "sonar",
        "primary_label": candidate["primary_label"],
        "confidence": candidate["confidence"],
        "all_labels": candidate["all_labels"],
        "reason": (
            f"Router predicted sonar. Sonar classifier: "
            f"label={candidate['primary_label']}, confidence={candidate['confidence']:.4f}."
        ),
        "display_text": f"Result: {candidate['primary_label']}",
        "candidates": [candidate],
        "domain_decision": "sonar",
    }


def _determine_bio_image_type(
    fish_coral_result: Optional[Dict[str, Any]],
    oceanclip_result: Optional[Dict[str, Any]],
) -> str:
    if fish_coral_result and fish_coral_result.get("primary_label") in ("fish", "coral"):
        return fish_coral_result["primary_label"]
    if oceanclip_result:
        if oceanclip_result.get("is_fish"):
            return "fish"
        if oceanclip_result.get("is_coral"):
            return "coral"
    return "biological"


def fuse_biological(
    fish_result: Optional[Dict[str, Any]],
    coral_result: Optional[Dict[str, Any]],
    oceanclip_result: Optional[Dict[str, Any]],
    fish_coral_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    image_type = _determine_bio_image_type(fish_coral_result, oceanclip_result)

    detector_candidate = None
    if image_type == "fish":
        detector_candidate = build_fish_candidate(fish_result)
    elif image_type == "coral":
        detector_candidate = build_coral_candidate(coral_result)

    oceanclip_candidate = build_oceanclip_candidate(oceanclip_result)

    # Cross-validate detector and OceanCLIP: check if the detector's label
    # matches any of OceanCLIP's top terms (genus-level or exact match).
    matched_term = None
    if detector_candidate and oceanclip_result:
        det_label = detector_candidate["primary_label"]
        for match in oceanclip_result.get("matches", []):
            if isinstance(match, dict) and match.get("term"):
                if _labels_match(det_label, match["term"]):
                    matched_term = match["term"]
                    break

    if oceanclip_candidate:
        if matched_term:
            best = {
                "source": "oceanclip+detector",
                "image_type": image_type,
                "primary_label": matched_term,
                "confidence": oceanclip_candidate["confidence"],
            }
            reason = (
                f"Biological domain. Detector and OceanCLIP agree: "
                f"detector={detector_candidate['primary_label']}, "
                f"OceanCLIP match={matched_term}, "
                f"similarity={oceanclip_candidate['confidence']:.4f}."
            )
        else:
            best = oceanclip_candidate
            reason = (
                f"Biological domain. OceanCLIP selected as primary: "
                f"label={best['primary_label']}, similarity={best['confidence']:.4f}."
            )
    elif detector_candidate and (detector_candidate.get("confidence") or 0.0) >= 0.5:
        best = detector_candidate
        reason = (
            f"Biological domain. Detector selected (OceanCLIP unavailable): "
            f"source={best['source']}, label={best['primary_label']}, confidence={best['confidence']:.4f}."
        )
    elif detector_candidate:
        best = detector_candidate
        reason = (
            f"Biological domain. Detector selected (low confidence, OceanCLIP unavailable): "
            f"source={best['source']}, label={best['primary_label']}, confidence={best['confidence']:.4f}."
        )
    else:
        return {
            "selected_source": None,
            "image_type": image_type,
            "primary_label": None,
            "confidence": None,
            "all_labels": [],
            "reason": "Biological domain but no valid candidate produced.",
            "display_text": "Result: Undetermined",
            "candidates": [],
            "domain_decision": "biological",
        }

    candidates = [c for c in [detector_candidate, oceanclip_candidate] if c is not None]
    merged_labels: List[str] = []
    for c in candidates:
        for label in c.get("all_labels", []):
            if label not in merged_labels:
                merged_labels.append(label)

    final_label = best["primary_label"]

    return {
        "selected_source": best["source"],
        "image_type": image_type,
        "primary_label": final_label,
        "confidence": best["confidence"],
        "all_labels": merged_labels,
        "reason": reason,
        "display_text": f"Result: {final_label}",
        "candidates": candidates,
        "domain_decision": "biological",
    }


def run_full_pipeline(image_path: Path) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    faiss_result = run_faiss_query(image_path)
    retrieval_module = faiss_result["retrieval_module"]
    top1_meta = faiss_result["top1_meta"]

    modules = build_default_modules()
    modules["retrieval"] = retrieval_module

    if retrieval_module["db_hit"]:
        labels = retrieval_module["top1_labels"]
        primary_label = labels[0] if labels else None
        image_type = infer_image_type_from_meta(top1_meta)
        confidence = retrieval_module["top1_similarity"]

        final_result = build_final_result(
            status="success",
            source="database",
            image_type=image_type,
            primary_label=primary_label,
            all_labels=labels,
            confidence=confidence,
            display_text=f"Result: {primary_label}",
            note="Matched from retrieval database.",
        )

        return {
            "success": True,
            "stage": "db_hit",
            "final_result": final_result,
            "modules": modules,
        }

    oceanclip_result = None
    router_result = run_router_classification(image_path)
    modules["router"] = router_result["router_module"]

    if (
        settings.use_oceanclip
        and state.oceanclip_model is not None
        and state.oceanclip_text_features is not None
    ):
        oceanclip_result = predict_with_oceanclip(
            image=image,
            model=state.oceanclip_model,
            preprocess=state.oceanclip_preprocess,
            text_features=state.oceanclip_text_features,
            terms=state.oceanclip_terms,
            device=state.runtime_device,
            topk=5,
        )
        modules["oceanclip"] = oceanclip_result

    router_predicted_type = router_result.get("predicted_type", "unknown")

    sonar_result = None
    fish_result = None
    coral_result = None
    fish_coral_result = None

    if router_predicted_type == "sonar":
        sonar_result = predict_with_yolo_classifier(
            state.sonar_model,
            image,
            state.runtime_device,
            "MergedData_7",
        )
        modules["sonar"] = sonar_result
    else:
        fish_coral_result = predict_with_yolo_classifier(
            state.fish_coral_model,
            image,
            state.runtime_device,
            "fish_coral_cls",
        )
        modules["fish_coral"] = fish_coral_result

        bio_subtype = fish_coral_result.get("primary_label", "fish")

        if bio_subtype == "fish":
            fish_result = predict_with_yolo_detector(
                state.fish_model,
                image,
                state.runtime_device,
                "merge_fish_small",
            )
            modules["fish"] = fish_result
        else:
            coral_result = predict_with_yolo_detector(
                state.coral_model,
                image,
                state.runtime_device,
                "Coral_one2",
            )
            modules["coral"] = coral_result

    if router_predicted_type == "sonar":
        fusion_result = fuse_sonar(sonar_result)
    else:
        fusion_result = fuse_biological(
            fish_result,
            coral_result,
            oceanclip_result,
            fish_coral_result,
        )

    modules["fusion"] = fusion_result

    final_result = build_final_result(
        status="success",
        source=fusion_result["selected_source"],
        image_type=fusion_result["image_type"],
        primary_label=fusion_result["primary_label"],
        all_labels=fusion_result["all_labels"],
        confidence=fusion_result["confidence"],
        display_text=fusion_result["display_text"],
        note=fusion_result["reason"],
    )

    return {
        "success": True,
        "stage": "multi_model_fused",
        "final_result": final_result,
        "modules": modules,
    }
