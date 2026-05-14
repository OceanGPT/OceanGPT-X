"""Download all model weights and data files from Hugging Face.

All assets are hosted in the OceanGPT-X collection:
    https://huggingface.co/collections/zjunlp/oceangpt-x

Individual repositories:
    - zjunlp/Ocean-router    : Router + Fish/Coral binary classifier
    - zjunlp/Ocean-yolo      : Fish, Coral, Sonar detectors
    - zjunlp/OceanCLIP-0.15B : OceanCLIP fine-tuned weights + BioCLIP base
    - zjunlp/Ocean-FAISS     : FAISS index + metadata

This script downloads everything needed to run the service except:
    - YOLOv5 source code (clone from https://github.com/ultralytics/yolov5)

Usage:
    python scripts/download_assets.py
    python scripts/download_assets.py --download-dir ./models
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_DOWNLOAD_DIR = Path("./downloaded_assets")

# Files to download, organized by source repository.
# Each entry: (repo_id, repo_file_path, local_relative_path)
FILES = [
    # --- Ocean-router (zjunlp/Ocean-router) ---
    ("zjunlp/Ocean-router", "cls_bio_sonar/best.pt", "models/cls_bio_sonar/best.pt"),
    ("zjunlp/Ocean-router", "fish_coral_cls/best.pt", "models/fish_coral_cls/best.pt"),

    # --- Ocean-yolo (zjunlp/Ocean-yolo) ---
    ("zjunlp/Ocean-yolo", "fish_detector/best.pt", "models/fish_detector/best.pt"),
    ("zjunlp/Ocean-yolo", "coral_detector/best.pt", "models/coral_detector/best.pt"),
    ("zjunlp/Ocean-yolo", "sonar_detector/best.pt", "models/sonar_detector/best.pt"),

    # --- OceanCLIP-0.15B (zjunlp/OceanCLIP-0.15B) ---
    ("zjunlp/OceanCLIP-0.15B", "oceanclip-bio/epoch_50.pt", "models/oceanclip-bio/epoch_50.pt"),
    ("zjunlp/OceanCLIP-0.15B", "oceanclip-bio/terms.txt", "models/oceanclip-bio/terms.txt"),
    ("zjunlp/OceanCLIP-0.15B", "bioclip/open_clip_pytorch_model.bin", "data/bioclip/open_clip_pytorch_model.bin"),
    ("zjunlp/OceanCLIP-0.15B", "bioclip/open_clip_config.json", "data/bioclip/open_clip_config.json"),

    # --- Ocean-FAISS (zjunlp/Ocean-FAISS) ---
    ("zjunlp/Ocean-FAISS", "faiss/index.faiss", "data/faiss/index.faiss"),
    ("zjunlp/Ocean-FAISS", "faiss/id_map.json", "data/faiss/id_map.json"),
    ("zjunlp/Ocean-FAISS", "metadata/metadata.jsonl", "data/metadata/metadata.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model weights and data files from the OceanGPT-X Hugging Face collection.",
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

    print(f"Downloading {len(FILES)} files to {base_dir.absolute()}\n")

    for repo_id, repo_file, local_relpath in FILES:
        local_subdir = base_dir / Path(local_relpath).parent
        local_subdir.mkdir(parents=True, exist_ok=True)

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            repo_type="model",
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] {local_relpath}")

    print("\n---")
    print("All assets downloaded. Paths are relative to the project root.")
    print("Only override environment variables if you use non-default paths.")


if __name__ == "__main__":
    main()
