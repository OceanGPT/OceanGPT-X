# Marine Image Retrieval API

A FastAPI-based marine image recognition service supporting:

- FAISS retrieval with BioCLIP features
- Multi-model result fusion with cross-validation
- Sonar image and biological image (fish & coral) classification

## Architecture

```
Input Image
  |
  v
[1] FAISS Retrieval (BioCLIP features, similarity >= 0.90 -> return directly)
  |  No match
  v
[2] Router Binary Classifier (sonar / biological, YOLOv11-cls)
  |
  +-- sonar --> [3a] Sonar Classifier (15 classes, YOLOv5) -> fusion
  |
  +-- biological --> [3b] Fish/Coral Binary Classifier (YOLOv5)
                          |
                          +-- fish  -> Fish Detector (YOLOv5)  --+
                          +-- coral -> Coral Detector (YOLOv5) --+-> fusion (+ OceanCLIP)
  v
[4] Fusion: cross-validate OceanCLIP species match with detector;
            fall back to highest-confidence candidate
```

## Quick Start

### 1. Install Dependencies

```bash
conda env create -f environment.yml
conda activate marine-api
```

### 2. Clone YOLOv5 Source Code

Required for loading YOLOv5-format models (sonar, fish, coral):

```bash
git clone https://github.com/ultralytics/yolov5 /path/to/yolov5
```

### 3. Download Models & Data

All model weights and data files are hosted on Hugging Face:
**[zhemaxiya/marine-image-api-models](https://huggingface.co/zhemaxiya/marine-image-api-models)**

```bash
python scripts/download_assets.py
```

This downloads:
- 7 model weights (Router, Sonar, Fish/Coral, Fish, Coral, OceanCLIP checkpoint + terms)
- BioCLIP base model for feature encoding
- FAISS retrieval index
- Metadata for image lookup

Custom download directory:

```bash
python scripts/download_assets.py --download-dir ./my-models
```

### 4. Configure Environment (Optional)

All paths default to the `downloaded_assets/` directory created by the download script.
Only set environment variables if you use custom paths:

```bash
export YOLOV5_DIR=/path/to/yolov5
```

To adjust inference parameters:

```bash
export THRESHOLD=0.85
export TOPK=10
```

### 5. Start the Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Development mode with auto-reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Streamlit Demo

Launch the interactive web demo:

```bash
streamlit run streamlit/demo.py
```

## API Endpoints

### Health Check

```
GET /health
```

### Prediction

```
POST /predict
```

| Field | Type  | Description    |
|-------|-------|----------------|
| file  | image | Image to classify |

**Example:**

```bash
curl -X POST http://localhost:8000/predict -F "file=@test/soner_cube.png"
```

Or open `http://localhost:8000/docs` for interactive API documentation.

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `THRESHOLD` | `0.90` | FAISS retrieval similarity threshold |
| `ROUTER_THRESHOLD` | `0.5` | Probability threshold for sonar routing |
| `USE_OCEANCLIP` | `true` | Enable OceanCLIP species identification |
| `TOPK` | `5` | Number of FAISS retrieval results |
| `DEVICE` | `cuda` | Computation device (`cuda` or `cpu`) |

## Project Structure

```
app/
  api/          # FastAPI routes (/health, /predict)
  core/         # Configuration and global state
  services/     # Model loading, retrieval, classification, fusion
  main.py       # Application entry point
scripts/
  download_assets.py  # One-command model + data downloader
streamlit/
  demo.py       # Streamlit demo UI
test/           # Sample test images
```

## Note

This repository does not include model weights or data files. Download them via `scripts/download_assets.py` and configure paths in `.env`.
