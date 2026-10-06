"""
FastAPI backend — Cashew Nut QC System
POST /predict?model=resnet50        — multipart image → JSON detections
GET  /health                        — device / model status
GET  /models                        — list available classifier models
"""
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Cashew QC API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipelines: dict = {}

# ── Model registry ────────────────────────────────────────────────────────────
def _model_registry():
    from inference_pipeline import _EFFNET_PATH, _RESNET_PATH
    return [
        {
            "id":          "resnet50",
            "name":        "ResNet-50",
            "input_size":  224,
            "path":        _RESNET_PATH,
            "available":   _RESNET_PATH.exists(),
            "description": "Baseline classifier (224px)",
        },
        {
            "id":          "efficientnet_b3",
            "name":        "EfficientNet-B3",
            "input_size":  300,
            "path":        _EFFNET_PATH,
            "available":   _EFFNET_PATH.exists(),
            "description": "Upgraded classifier (300px)",
        },
    ]


def _default_model_id() -> str:
    from inference_pipeline import _EFFNET_PATH
    return "efficientnet_b3" if _EFFNET_PATH.exists() else "resnet50"


def get_pipeline(model_id: str | None = None):
    if model_id is None:
        model_id = _default_model_id()

    reg = {m["id"]: m for m in _model_registry()}
    if model_id not in reg:
        raise HTTPException(400, f"Unknown model: {model_id}. Use: {list(reg)}")
    info = reg[model_id]
    if not info["available"]:
        raise HTTPException(404, f"Model '{model_id}' not found on disk — train it first")

    if model_id not in _pipelines:
        from inference_pipeline import CashewPipeline
        log.info("Loading pipeline: %s (%dpx)", model_id, info["input_size"])
        _pipelines[model_id] = CashewPipeline(
            classifier_path=str(info["path"]),
            classifier_size=info["input_size"],
        )
        log.info("Pipeline ready: %s", model_id)

    return _pipelines[model_id], model_id


@app.on_event("startup")
async def _startup():
    get_pipeline()  # preload default


@app.get("/health")
async def health():
    import torch
    p, mid = get_pipeline()
    cuda = torch.cuda.is_available()
    return {
        "status":    "ok",
        "device":    str(p.device),
        "cuda":      cuda,
        "gpu_name":  torch.cuda.get_device_name(0) if cuda else None,
        "model":     mid,
    }


@app.get("/models")
async def list_models():
    reg = _model_registry()
    default = _default_model_id()
    return {
        "models":  [
            {k: v for k, v in m.items() if k != "path"}
            for m in reg
        ],
        "default": default,
    }


@app.post("/predict")
async def predict(
    file:  UploadFile = File(...),
    model: str        = Query(None, description="Model ID: resnet50 | efficientnet_b3"),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")

    h, w = img.shape[:2]
    p, model_used = get_pipeline(model)

    t0 = time.perf_counter()
    preds = p.predict(img)
    ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "latency_ms":  ms,
        "image_size":  {"width": w, "height": h},
        "count":       len(preds),
        "model_used":  model_used,
        "predictions": preds,
    }


# Static files must be mounted last
_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
