from __future__ import annotations

import os

from app.core.config import settings
from app.core.state import state
from app.services.oceanclip_service import (
    load_oceanclip_model,
    load_oceanclip_text_features,
    load_terms_from_txt,
)
from app.services.retrieval import load_bioclip, load_index, load_metadata, resolve_device
from app.services.router import load_router_model

from models.experimental import attempt_load  # noqa: E402


def load_all_resources() -> None:
    """Load all models, indices, and metadata into state at startup."""
    state.runtime_device = resolve_device(settings.device)
    print(f"[Startup] Using device: {state.runtime_device}")

    # 1. Retrieval module
    state.model, state.preprocess = load_bioclip(state.runtime_device)
    print("[Startup] BioCLIP retrieval model loaded.")

    state.index, state.id_map = load_index()
    print("[Startup] FAISS index loaded.")

    state.id2meta = load_metadata()
    print("[Startup] Metadata loaded.")

    # 2. Router module
    print("[Startup] Loading router model...")
    state.router_model, state.router_class_names = load_router_model(settings.router_model_path)
    print(f"[Startup] Router model loaded. Classes: {state.router_class_names}")

    # 3. Sonar classifier
    print("[Startup] Loading sonar classification model...")
    state.sonar_model = attempt_load(settings.sonar_cls_path, device=state.runtime_device)
    state.sonar_model.eval()
    print(f"[Startup] Sonar model loaded. Classes: {state.sonar_model.names}")

    # 4. Fish/Coral binary classification
    print("[Startup] Loading fish/coral classifier...")
    state.fish_coral_model = attempt_load(settings.fish_coral_cls_path, device=state.runtime_device)
    state.fish_coral_model.eval()
    print(f"[Startup] Fish/Coral model loaded. Classes: {state.fish_coral_model.names}")

    # 5. fish detector
    print("[Startup] Loading fish detection model...")
    state.fish_model = attempt_load(settings.fish_model_path, device=state.runtime_device)
    state.fish_model.eval()
    print(f"[Startup] Fish model loaded. Classes: {state.fish_model.names}")

    # 6. coral detector
    print("[Startup] Loading coral detection model...")
    state.coral_model = attempt_load(settings.coral_model_path, device=state.runtime_device)
    state.coral_model.eval()
    print(f"[Startup] Coral model loaded. Classes: {state.coral_model.names}")

    # 7. OceanCLIP (optional)
    print("[Startup] Loading OceanCLIP finetuned model...")
    try:
        (
            state.oceanclip_model,
            state.oceanclip_preprocess,
            state.oceanclip_tokenizer,
        ) = load_oceanclip_model(settings.oceanclip_checkpoint, state.runtime_device)

        if state.oceanclip_model is not None:
            print(f"[DEBUG] OCEANCLIP_TERMS_PATH = '{settings.oceanclip_terms_path}'")
            print(f"[DEBUG] OCEANCLIP_TERMS_PATH type = {type(settings.oceanclip_terms_path)}")
            print(f"[DEBUG] File exists: {os.path.exists(settings.oceanclip_terms_path)}")

            state.oceanclip_terms = load_terms_from_txt(
                settings.oceanclip_terms_path,
                max_terms=1000,
            )

            if state.oceanclip_terms:
                (
                    state.oceanclip_text_features,
                    state.oceanclip_terms,
                ) = load_oceanclip_text_features(
                    state.oceanclip_model,
                    state.oceanclip_terms,
                    state.runtime_device,
                )
                print(f"[Startup] OceanCLIP text features computed for {len(state.oceanclip_terms)} terms")
            else:
                print("[Startup] Warning: No terms loaded for OceanCLIP")
        else:
            print("[Startup] Warning: OCEANCLIP_MODEL is None")

    except Exception as e:
        print(f"[Startup] Warning: Failed to load OceanCLIP: {e}")
        state.oceanclip_model = None
        state.oceanclip_preprocess = None
        state.oceanclip_tokenizer = None
        state.oceanclip_text_features = None
        state.oceanclip_terms = []
