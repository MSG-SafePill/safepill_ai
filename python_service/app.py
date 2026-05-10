import os
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from python_service.chatbot import ChatbotService
from python_service.db import fetch_pill_catalog
from python_service.fusion import PillFusionService
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


class IdentifiedPill(BaseModel):
    pillName: str
    confidence: float
    ocrScore: float
    detectionScore: float
    matchedText: str | None = None


class IdentifyRequest(BaseModel):
    imagePath: str
    topK: int = 5


class IdentifyResponse(BaseModel):
    requestId: str
    status: str
    detections: list[PillCandidate]
    ocrCandidates: list[OcrCandidate]
    identifiedPills: list[IdentifiedPill]


class ChatRequest(BaseModel):
    question: str
    identifiedPills: list[str] = []
    imagePath: str | None = None


class ChatResponse(BaseModel):
    requestId: str
    status: str
    answer: str
    referencedPills: list[str]


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


@lru_cache(maxsize=1)
def get_fusion_service() -> PillFusionService:
    return PillFusionService()


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService()


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


@app.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")
    if request.topK < 1 or request.topK > 20:
        raise HTTPException(status_code=400, detail="topK must be between 1 and 20.")

    detections = get_engine().detect(image_path)
    ocr_candidates = get_ocr_pipeline().extract(image_path=image_path, detections=detections)
    pill_catalog = fetch_pill_catalog()
    identified = get_fusion_service().rank_candidates(
        detections=detections,
        ocr_candidates=ocr_candidates,
        pill_catalog=pill_catalog,
        top_k=request.topK,
    )
    return IdentifyResponse(
        requestId=str(uuid4()),
        status="ok" if identified else "no_match",
        detections=[PillCandidate(pillName=str(det["pillName"]), confidence=float(det["confidence"])) for det in detections],
        ocrCandidates=[OcrCandidate(**candidate) for candidate in ocr_candidates],
        identifiedPills=[IdentifiedPill(**item) for item in identified],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    identified_pills = list(request.identifiedPills)

    if request.imagePath:
        identify_result = identify(IdentifyRequest(imagePath=request.imagePath, topK=5))
        identified_pills = [item.pillName for item in identify_result.identifiedPills]

    if not identified_pills:
        raise HTTPException(status_code=400, detail="identifiedPills or imagePath must be provided.")

    try:
        result = get_chatbot_service().answer(request.question, identified_pills)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        requestId=str(uuid4()),
        status="ok",
        answer=str(result["answer"]),
        referencedPills=[str(name) for name in result["referencedPills"]],
    )
