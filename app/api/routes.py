from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.state import state
from app.services.pipeline import generate_request_id, run_full_pipeline

router = APIRouter()


@router.get("/health")
def health():
    ready = all([
        state.model is not None,
        state.preprocess is not None,
        state.index is not None,
        state.id_map is not None,
        state.id2meta is not None,
        state.runtime_device is not None,
        state.router_model is not None,
        state.sonar_model is not None,
        state.fish_coral_model is not None,
        state.fish_model is not None,
        state.coral_model is not None,
    ])

    return {
        "success": ready,
        "service": settings.app_title,
        "version": settings.app_version,
        "device": state.runtime_device,
        "ready": ready,
        "router_model_loaded": state.router_model is not None,
        "router_classes": state.router_class_names,
        "sonar_model_loaded": state.sonar_model is not None,
        "fish_coral_model_loaded": state.fish_coral_model is not None,
        "fish_model_loaded": state.fish_model is not None,
        "coral_model_loaded": state.coral_model is not None,
        "oceanclip_loaded": state.oceanclip_model is not None,
        "oceanclip_terms_count": len(state.oceanclip_terms) if state.oceanclip_terms else 0,
        "use_oceanclip": settings.use_oceanclip,
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    request_id = generate_request_id()
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    tmp_dir = tempfile.mkdtemp(prefix="predict_")
    tmp_path = Path(tmp_dir) / f"input{suffix}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_full_pipeline(tmp_path)

        return JSONResponse(content={
            "success": True,
            "request_id": request_id,
            "message": "Prediction completed.",
            "filename": file.filename,
            "content_type": file.content_type,
            "result": result,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "request_id": request_id,
                "message": f"Inference error: {str(e)}",
            },
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
