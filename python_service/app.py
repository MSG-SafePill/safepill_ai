import os
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from python_service.inference import YoloInferenceEngine
from python_service.ocr_pipeline import OcrPipeline

app = FastAPI(title="SafePill Python Inference Service")


class InferRequest(BaseModel):
    imagePath: str


class PillCandidate(BaseModel):
    pillName: str
    confidence: float


class InferResponse(BaseModel):
    requestId: str
    status: str
    candidates: list[PillCandidate]


class OcrRequest(BaseModel):
    imagePath: str
    useDetections: bool = True


class OcrCandidate(BaseModel):
    text: str
    normalizedText: str
    confidence: float
    regionIndex: int


class OcrResponse(BaseModel):
    requestId: str
    status: str
    candidates: list[OcrCandidate]


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt"


@lru_cache(maxsize=1)
def get_engine() -> YoloInferenceEngine:
    model_path = Path(os.getenv("SAFEPILL_MODEL_PATH", str(_default_model_path())))
    conf_threshold = float(os.getenv("SAFEPILL_CONF_THRESHOLD", "0.25"))
    iou_threshold = float(os.getenv("SAFEPILL_IOU_THRESHOLD", "0.7"))
    device = os.getenv("SAFEPILL_DEVICE")
    return YoloInferenceEngine(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )


@lru_cache(maxsize=1)
def get_ocr_pipeline() -> OcrPipeline:
    use_gpu = os.getenv("SAFEPILL_OCR_GPU", "false").lower() == "true"
    return OcrPipeline(use_gpu=use_gpu)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    predictions = get_engine().infer(image_path)
    return InferResponse(
        requestId=str(uuid4()),
        status="ok" if predictions else "no_detection",
        candidates=[PillCandidate(**candidate) for candidate in predictions],
    )


@app.post("/ocr", response_model=OcrResponse)
def ocr(request: OcrRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    detections = get_engine().detect(image_path) if request.useDetections else None
    candidates = get_ocr_pipeline().extract(image_path=image_path, detections=detections)

    return OcrResponse(
        requestId=str(uuid4()),
        status="ok" if candidates else "no_text",
        candidates=[OcrCandidate(**candidate) for candidate in candidates],
    )
