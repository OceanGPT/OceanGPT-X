"""Download all model weights and data files from Hugging Face.

The Hugging Face repository is:
    https://huggingface.co/zhemaxiya/marine-image-api-models

This script downloads everything needed to run the service except:
    - YOLOv5 source code (clone from https://github.com/ultralytics/yolov5)

Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --download-dir ./models
    python scripts/download_assets.py --repo-id your-org/your-repo --download-dir ./models
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

# Default Hugging Face repository
DEFAULT_REPO_ID = "zhemaxiya/marine-image-api-models"
DEFAULT_DOWNLOAD_DIR = Path("./downloaded_assets")

# Files to download.
# Keys are descriptive (shown in output).
# Each value is (repo_file_path, local_relative_path).
FILES = {
    # --- Model weights ---
    "ROUTER_MODEL_PATH": ("cls_bio_sonar/best.pt", "models/cls_bio_sonar/best.pt"),
    "SONAR_CLS_PATH": ("sonar/best.pt", "models/sonar/best.pt"),
    "FISH_CORAL_CLS_PATH": ("fish_coral_cls/best.pt", "models/fish_coral_cls/best.pt"),
    "FISH_MODEL_PATH": ("fish_detector/best.pt", "models/fish_detector/best.pt"),
    "CORAL_MODEL_PATH": ("coral_detector/best.pt", "models/coral_detector/best.pt"),
    "OCEANCLIP_CHECKPOINT": ("oceanclip-bio/epoch_50.pt", "models/oceanclip-bio/epoch_50.pt"),
    "OCEANCLIP_TERMS_PATH": ("oceanclip-bio/terms.txt", "models/oceanclip-bio/terms.txt"),

    # --- BioCLIP base model (for FAISS retrieval feature encoding) ---
    "bioclip_model": ("bioclip/open_clip_pytorch_model.bin", "data/bioclip/open_clip_pytorch_model.bin"),
    "bioclip_config": ("bioclip/open_clip_config.json", "data/bioclip/open_clip_config.json"),

    # --- FAISS retrieval index ---
    "faiss_index": ("faiss/index.faiss", "data/faiss/index.faiss"),
    "faiss_id_map": ("faiss/id_map.json", "data/faiss/id_map.json"),

    # --- Metadata ---
    "metadata": ("metadata/metadata.jsonl", "data/metadata/metadata.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model weights and data files from Hugging Face.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID (default: %(default)s)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Local directory to store downloaded files (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.download_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, str] = {}

    for name, (repo_file, local_relpath) in FILES.items():
        local_subdir = base_dir / Path(local_relpath).parent
        local_subdir.mkdir(parents=True, exist_ok=True)

        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=repo_file,
            repo_type="model",
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        downloaded[name] = str(local_path)
        print(f"[OK] {name}: {local_path}")

    print("\n---")
    print("All assets downloaded. Paths are relative to the project root.")
    print("Only override environment variables if you use non-default paths.")


if __name__ == "__main__":
    main()
