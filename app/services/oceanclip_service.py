from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from PIL import Image


def load_oceanclip_model(checkpoint_path: str, device: str):
    try:
        import open_clip

        print(f"[OceanCLIP] Loading model from {checkpoint_path} using standard open_clip")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=False)

        print("[OceanCLIP] Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"[OceanCLIP] Checkpoint loaded, type: {type(checkpoint)}")

        if "state_dict" in checkpoint:
            print("[OceanCLIP] Loading from 'state_dict' key")
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            print("[OceanCLIP] Loading from 'model_state_dict' key")
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("[OceanCLIP] Loading checkpoint as direct state_dict")
            model.load_state_dict(checkpoint)

        model = model.to(device).eval()
        print("[OceanCLIP] Model loaded successfully")

        return model, preprocess, open_clip.tokenize

    except Exception as e:
        print(f"\n[OceanCLIP] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_default_terms() -> List[str]:
    return [
        "coral", "fish", "marine", "invertebrate",
        "Acanthastrea", "Acropora", "Alveopora", "Amphiprion",
        "Canthigaster", "Carcharhinus", "Catalaphyllia", "Chaetodon",
        "Chelmon", "Chromis", "Cleidopus", "Colpophyllia",
        "Coradion", "Cyphastrea", "Dendrogyra", "Discosoma",
        "Euphyllia", "Forcipiger", "Heliofungia", "Heniochus",
        "Hydnophora", "Leptoseris", "Meandrina", "Millepora",
        "Monocentris", "Montastraea", "Montipora", "Myripristis",
        "Naso", "Palythoa", "Paraluteres", "Platax",
        "Pocillopora", "Porites", "Pseudanthias", "Rhodactis",
        "Ricordea", "Seriatopora", "Siganus", "Toxotes",
    ]


def load_terms_from_txt(terms_path: str, max_terms: int = 1000) -> List[str]:
    if not os.path.exists(terms_path):
        print(f"[OceanCLIP] Terms file not found: {terms_path}, using default terms")
        return get_default_terms()

    terms: List[str] = []
    seen = set()

    with open(terms_path, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if not term:
                continue
            if term not in seen:
                seen.add(term)
                terms.append(term)
            if len(terms) >= max_terms:
                break

    if not terms:
        print("[OceanCLIP] Terms file is empty, using default terms")
        return get_default_terms()

    print(f"[OceanCLIP] Loaded {len(terms)} terms from {terms_path}")
    return terms


def load_oceanclip_text_features(model, terms: List[str], device: str):
    import open_clip

    text_tokens = open_clip.tokenize(terms).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, terms


@torch.no_grad()
def predict_with_oceanclip(
    image: Image.Image,
    model,
    preprocess,
    text_features: torch.Tensor,
    terms: List[str],
    device: str,
    topk: int = 5,
) -> Dict[str, Any]:
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    img_features = model.encode_image(img_tensor)
    img_features /= img_features.norm(dim=-1, keepdim=True)

    similarity = (img_features @ text_features.T).squeeze(0)
    scores, indices = similarity.topk(min(topk, len(terms)))

    matches = []
    for i in range(len(indices)):
        idx = indices[i].item()
        matches.append({
            "term": terms[idx],
            "similarity": scores[i].item(),
        })

    top_term = matches[0]["term"].lower() if matches else ""

    fish_keywords = [
        "fish", "pisces", "ichthys", "shark", "ray", "eel", "grouper", "snapper",
        "amphiprion", "canthigaster", "carcharhinus", "chaetodon", "chelmon",
        "chromis", "cleidopus", "coradion", "forcipiger", "heniochus", "monocentris",
        "myripristis", "naso", "paraluteres", "platax", "pseudanthias", "siganus", "toxotes",
    ]
    coral_keywords = [
        "coral", "anthozoa", "acropora", "alveopora", "catalaphyllia", "colpophyllia",
        "cyphastrea", "dendrogyra", "discosoma", "euphyllia", "heliofungia", "hydnophora",
        "leptoseris", "meandrina", "millepora", "montastraea", "montipora", "palythoa",
        "pocillopora", "porites", "rhodactis", "ricordea", "seriatopora",
    ]

    is_fish = any(keyword in top_term for keyword in fish_keywords)
    is_coral = any(keyword in top_term for keyword in coral_keywords)

    if not is_fish and not is_coral:
        for match in matches[:3]:
            term_lower = match["term"].lower()
            if any(keyword in term_lower for keyword in fish_keywords):
                is_fish = True
                break
            if any(keyword in term_lower for keyword in coral_keywords):
                is_coral = True
                break

    return {
        "model_type": "oceanclip",
        "matches": matches,
        "primary_match": matches[0] if matches else None,
        "is_fish": is_fish,
        "is_coral": is_coral,
        "confidence": matches[0]["similarity"] if matches else 0.0,
    }
