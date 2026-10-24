import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from python_service.chatbot import ChatbotService
from python_service.db import fetch_pill_catalog
from python_service.fusion import PillFusionService
from python_service.inference import YoloInferenceEngine
from python_service.interaction_analysis import InteractionAnalysisService
from python_service.medication_matcher import MedicationMatcher
from python_service.multi_pill_classifier import MultiPillClassificationService
from python_service.pill_classifier import PillClassificationService
from python_service.ocr_pipeline import OcrPipeline
from python_service.prescription_ocr import PrescriptionOcrParser
from python_service.vision_runtime import (
    ImageTooLarge, MAX_IMAGE_BYTES, VisionTimingMiddleware, logger, save_upload, timed_stage,
)

_vision_slots = BoundedSemaphore(1)
_startup_failures: dict[str, str] = {}


def vision_slot():
    # Reject overlap instead of accumulating stale work after the caller times out.
    if not _vision_slots.acquire(blocking=False):
        raise HTTPException(503, "AI server is busy. Please retry shortly.", headers={"Retry-After": "2"})
    try:
        yield
    except HTTPException:
        raise
    except ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Required AI model or image file is unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(400, "Unable to process image. Check the image format and request.") from exc
    except Exception as exc:
        logger.error("vision stage=request_failed error_type=%s", type(exc).__name__)
        raise HTTPException(502, "AI image processing failed. Please retry.") from exc
    finally:
        _vision_slots.release()


def _ready_service(name, loader):
    if name in _startup_failures:
        raise HTTPException(503, f"AI {name} is unavailable. Check server startup logs.")
    return loader()


def _preload_vision_models():
    _startup_failures.clear()
    for name, loader in (
        ("detection", get_engine),
        ("classification", get_pill_classification_service),
        ("ocr", lambda: get_ocr_pipeline().warmup()),
    ):
        try:
            with timed_stage(f"startup_{name}"):
                loader()
        except Exception as exc:
            _startup_failures[name] = type(exc).__name__
            logger.error("vision model=%s startup_failed=%s; restore model/dependencies and restart", name, type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading completes before accepting requests; one instance per server worker.
    await run_in_threadpool(_preload_vision_models)
    yield


app = FastAPI(title="SafePill Python Inference Service", lifespan=lifespan)
app.add_middleware(VisionTimingMiddleware)


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
    itemId: int | None = None
    itemType: str | None = None
    manufacturer: str | None = None
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
    contextItems: list[dict] = []
    userProfile: dict | None = None
    imagePath: str | None = None


class ChatResponse(BaseModel):
    requestId: str
    status: str
    answer: str
    referencedPills: list[str]


class InteractionIngredient(BaseModel):
    name: str
    dosage: str | None = None


class InteractionItem(BaseModel):
    itemName: str
    itemType: str
    ingredients: list[InteractionIngredient] = []
    intakeTimes: list[str] = []
    efficacy: str | None = None
    precautions: str | None = None


class InteractionRuleInput(BaseModel):
    itemNameA: str | None = None
    itemNameB: str | None = None
    ingredientNameA: str
    ingredientNameB: str
    riskLevel: str | None = None
    description: str | None = None


class InteractionAnalyzeRequest(BaseModel):
    items: list[InteractionItem]
    interactionRules: list[InteractionRuleInput] = []
    userProfile: dict | None = None


class InteractionWarning(BaseModel):
    title: str | None = None
    severity: str | None = None
    items: list[str] = []
    reason: str | None = None


class InteractionEvidence(BaseModel):
    source: str | None = None
    text: str | None = None


class InteractionAnalyzeResponse(BaseModel):
    requestId: str
    status: str
    riskLevel: str
    summary: str
    warnings: list[InteractionWarning]
    recommendations: list[str]
    scheduleRecommendations: list[str] = []
    foodWarnings: list[str] = []
    consultationGuidance: list[str] = []
    evidence: list[InteractionEvidence]
    disclaimer: str


class PrescriptionOcrRequest(BaseModel):
    imagePath: str


class ScheduleSuggestion(BaseModel):
    takeTime: str
    daysOfWeek: list[str]
    dosage: str
    mealTiming: str | None = None


class OcrMedicationMatchCandidate(BaseModel):
    itemId: int
    itemType: str
    itemName: str
    manufacturer: str | None = None
    score: float


class PrescriptionOcrItem(BaseModel):
    medicineName: str
    rawText: str
    dosage: str | None = None
    frequency: str | None = None
    mealTiming: str | None = None
    days: str | None = None
    scheduleSuggestions: list[ScheduleSuggestion] = []
    matchCandidates: list[OcrMedicationMatchCandidate] = []
    confidence: float


class PrescriptionOcrResponse(BaseModel):
    requestId: str
    status: str
    items: list[PrescriptionOcrItem]
    rawCandidates: list[OcrCandidate]


class MedicationMatchRequest(BaseModel):
    keywords: list[str]
    topK: int = 5


class MedicationMatchCandidate(BaseModel):
    itemId: int
    itemType: str
    itemName: str
    manufacturer: str | None = None
    score: float


class MedicationMatchResult(BaseModel):
    keyword: str
    candidates: list[MedicationMatchCandidate]


class MedicationMatchResponse(BaseModel):
    requestId: str
    status: str
    results: list[MedicationMatchResult]


class PillClassificationCandidate(BaseModel):
    classLabel: str
    medicineName: str
    manufacturer: str
    mark: str
    shape: str
    color: str
    score: float
    reason: str


class PillClassificationResponse(BaseModel):
    success: bool
    candidates: list[PillClassificationCandidate]


class PillBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectedPillClassification(BaseModel):
    pillIndex: int
    box: PillBox
    candidates: list[PillClassificationCandidate]


class MultiPillClassificationResponse(BaseModel):
    success: bool
    detectedCount: int
    detectedPills: list[DetectedPillClassification]


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "pill_detect" / "yolo11n" / "best.pt"


def _default_pill_cls_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "pill_cls" / "yolo11n" / "best.pt"


def _default_pill_cls_mapping_path() -> Path:
    return Path(__file__).resolve().parent.parent / "mappings" / "pill_class_mapping.json"


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
def get_medication_matcher() -> MedicationMatcher:
    return MedicationMatcher()


@lru_cache(maxsize=1)
def get_chatbot_service() -> ChatbotService:
    return ChatbotService()


@lru_cache(maxsize=1)
def get_interaction_analysis_service() -> InteractionAnalysisService:
    return InteractionAnalysisService()


@lru_cache(maxsize=1)
def get_prescription_parser() -> PrescriptionOcrParser:
    return PrescriptionOcrParser()


@lru_cache(maxsize=1)
def get_pill_classification_service() -> PillClassificationService:
    model_path = Path(os.getenv("SAFEPILL_PILL_CLS_MODEL_PATH", str(_default_pill_cls_model_path())))
    mapping_path = Path(os.getenv("SAFEPILL_PILL_CLS_MAPPING_PATH", str(_default_pill_cls_mapping_path())))
    device = os.getenv("SAFEPILL_DEVICE")
    return PillClassificationService(model_path=model_path, mapping_path=mapping_path, device=device)


@lru_cache(maxsize=1)
def get_multi_pill_classification_service() -> MultiPillClassificationService:
    return MultiPillClassificationService(_ready_service("classification", get_pill_classification_service))


@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(upload: UploadFile) -> Path:
    if upload.size is not None and upload.size > MAX_IMAGE_BYTES:
        raise ImageTooLarge("Image must be 30 MB or smaller.")
    suffix = Path(upload.filename or "").suffix or ".jpg"
    return save_upload(upload.file, suffix)


@app.post("/infer", response_model=InferResponse, dependencies=[Depends(vision_slot)])
def infer(request: InferRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    predictions = _ready_service("detection", get_engine).infer(image_path)
    return InferResponse(
        requestId=str(uuid4()),
        status="ok" if predictions else "no_detection",
        candidates=[PillCandidate(**candidate) for candidate in predictions],
    )


@app.post("/ocr", response_model=OcrResponse, dependencies=[Depends(vision_slot)])
def ocr(request: OcrRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    detections = _ready_service("detection", get_engine).detect(image_path) if request.useDetections else None
    with timed_stage("ocr_total"):
        candidates = _ready_service("ocr", get_ocr_pipeline).extract(image_path=image_path, detections=detections)

    return OcrResponse(
        requestId=str(uuid4()),
        status="ok" if candidates else "no_text",
        candidates=[OcrCandidate(**candidate) for candidate in candidates],
    )


@app.post("/identify", response_model=IdentifyResponse, dependencies=[Depends(vision_slot)])
def identify(request: IdentifyRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")
    if request.topK < 1 or request.topK > 20:
        raise HTTPException(status_code=400, detail="topK must be between 1 and 20.")

    detections = _ready_service("detection", get_engine).detect(image_path)
    with timed_stage("ocr_total"):
        ocr_candidates = _ready_service("ocr", get_ocr_pipeline).extract(image_path=image_path, detections=detections)
    with timed_stage("catalog_fetch"):
        pill_catalog = fetch_pill_catalog()
    with timed_stage("catalog_rank"):
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


@app.post("/identify-upload", response_model=IdentifyResponse, dependencies=[Depends(vision_slot)])
def identify_upload(image: UploadFile = File(...), topK: int = 5):
    image_path = _save_upload(image)
    try:
        return identify(IdentifyRequest(imagePath=str(image_path), topK=topK))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/classify-pill", response_model=PillClassificationResponse, dependencies=[Depends(vision_slot)])
def classify_pill(image: UploadFile = File(...)):
    image_path = _save_upload(image)
    try:
        try:
            candidates = _ready_service("classification", get_pill_classification_service).classify(image_path=image_path, top_k=3)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return PillClassificationResponse(
            success=bool(candidates),
            candidates=[PillClassificationCandidate(**candidate) for candidate in candidates],
        )
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/classify-pills", response_model=MultiPillClassificationResponse, dependencies=[Depends(vision_slot)])
def classify_pills(image: UploadFile = File(...)):
    image_path = _save_upload(image)
    try:
        try:
            detected_pills = get_multi_pill_classification_service().classify(image_path=image_path, top_k=3)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return MultiPillClassificationResponse(
            success=bool(detected_pills),
            detectedCount=len(detected_pills),
            detectedPills=[DetectedPillClassification(**item) for item in detected_pills],
        )
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/prescription-ocr", response_model=PrescriptionOcrResponse, dependencies=[Depends(vision_slot)])
def prescription_ocr(request: PrescriptionOcrRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    with timed_stage("ocr_total"):
        ocr_candidates = _ready_service("ocr", get_ocr_pipeline).extract_prescription_medicines(image_path=image_path)
    with timed_stage("prescription_parse"):
        parsed_items = get_prescription_parser().parse(ocr_candidates)
    with timed_stage("catalog_match"):
        match_results = {
            str(result["keyword"]): result["candidates"]
            for result in get_medication_matcher().match(
                [str(item["medicineName"]) for item in parsed_items],
                top_k=5,
            )
        } if parsed_items else {}
    for item in parsed_items:
        item["matchCandidates"] = match_results.get(str(item["medicineName"]), [])
    return PrescriptionOcrResponse(
        requestId=str(uuid4()),
        status="ok" if parsed_items else "no_text",
        items=[PrescriptionOcrItem(**item) for item in parsed_items],
        rawCandidates=[OcrCandidate(**candidate) for candidate in ocr_candidates],
    )


@app.post("/prescription-ocr-upload", response_model=PrescriptionOcrResponse, dependencies=[Depends(vision_slot)])
def prescription_ocr_upload(image: UploadFile = File(...)):
    image_path = _save_upload(image)
    try:
        return prescription_ocr(PrescriptionOcrRequest(imagePath=str(image_path)))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    identified_pills = list(request.identifiedPills)

    if request.imagePath:
        identify_result = identify(IdentifyRequest(imagePath=request.imagePath, topK=5))
        identified_pills = [item.pillName for item in identify_result.identifiedPills]

    if not identified_pills and request.contextItems:
        identified_pills = [str(item.get("itemName")) for item in request.contextItems if item.get("itemName")]

    if not identified_pills:
        raise HTTPException(status_code=400, detail="identifiedPills or imagePath must be provided.")

    try:
        result = get_chatbot_service().answer(
            request.question,
            identified_pills,
            context_items=request.contextItems,
            user_profile=request.userProfile,
        )
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


@app.post("/medication-match", response_model=MedicationMatchResponse)
def medication_match(request: MedicationMatchRequest):
    if not request.keywords:
        raise HTTPException(status_code=400, detail="keywords is required.")
    results = get_medication_matcher().match(request.keywords, request.topK)
    return MedicationMatchResponse(
        requestId=str(uuid4()),
        status="ok" if results else "no_match",
        results=[MedicationMatchResult(**result) for result in results],
    )


@app.post("/interaction/analyze", response_model=InteractionAnalyzeResponse)
def analyze_interaction(request: InteractionAnalyzeRequest):
    try:
        result = get_interaction_analysis_service().analyze(
            items=[item.model_dump() for item in request.items],
            interaction_rules=[rule.model_dump() for rule in request.interactionRules],
            user_profile=request.userProfile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return InteractionAnalyzeResponse(
        requestId=str(uuid4()),
        status="ok",
        riskLevel=str(result["riskLevel"]),
        summary=str(result["summary"]),
        warnings=[InteractionWarning(**warning) for warning in result["warnings"]],
        recommendations=[str(item) for item in result["recommendations"]],
        scheduleRecommendations=[str(item) for item in result.get("scheduleRecommendations", [])],
        foodWarnings=[str(item) for item in result.get("foodWarnings", [])],
        consultationGuidance=[str(item) for item in result.get("consultationGuidance", [])],
        evidence=[InteractionEvidence(**item) for item in result["evidence"]],
        disclaimer=str(result["disclaimer"]),
    )
