from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppState:
    model: Any = None
    preprocess: Any = None
    index: Any = None
    id_map: Optional[List[str]] = None
    id2meta: Optional[Dict[str, Dict[str, Any]]] = None
    runtime_device: Optional[str] = None

    router_model: Any = None
    router_class_names: dict = field(default_factory=dict)  # {0: 'bio', 1: 'sonar'}

    sonar_model: Any = None
    fish_coral_model: Any = None
    fish_model: Any = None
    coral_model: Any = None

    oceanclip_model: Any = None
    oceanclip_preprocess: Any = None
    oceanclip_tokenizer: Any = None
    oceanclip_text_features: Any = None
    oceanclip_terms: List[str] = field(default_factory=list)


state = AppState()
