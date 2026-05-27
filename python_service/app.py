import os
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from python_service.chatbot import ChatbotService
from python_service.db import fetch_pill_catalog
from python_service.fusion import PillFusionService
from python_service.inference import YoloInferenceEngine
from python_service.interaction_analysis import InteractionAnalysisService
from python_service.medication_matcher import MedicationMatcher
from python_service.ocr_pipeline import OcrPipeline
from python_service.prescription_ocr import PrescriptionOcrParser

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


@app.get("/health")
def health():
    return {"status": "ok"}


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await upload.read())
        return Path(temp_file.name)


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


@app.post("/identify-upload", response_model=IdentifyResponse)
async def identify_upload(image: UploadFile = File(...), topK: int = 5):
    image_path = await _save_upload(image)
    try:
        return identify(IdentifyRequest(imagePath=str(image_path), topK=topK))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/prescription-ocr", response_model=PrescriptionOcrResponse)
def prescription_ocr(request: PrescriptionOcrRequest):
    image_path = Path(request.imagePath)
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")

    ocr_candidates = get_ocr_pipeline().extract(image_path=image_path, detections=None)
    parsed_items = get_prescription_parser().parse(ocr_candidates)
    match_results = {
        str(result["keyword"]): result["candidates"]
        for result in get_medication_matcher().match(
            [str(item["medicineName"]) for item in parsed_items],
            top_k=5,
        )
    }
    for item in parsed_items:
        item["matchCandidates"] = match_results.get(str(item["medicineName"]), [])
    return PrescriptionOcrResponse(
        requestId=str(uuid4()),
        status="ok" if parsed_items else "no_text",
        items=[PrescriptionOcrItem(**item) for item in parsed_items],
        rawCandidates=[OcrCandidate(**candidate) for candidate in ocr_candidates],
    )


@app.post("/prescription-ocr-upload", response_model=PrescriptionOcrResponse)
async def prescription_ocr_upload(image: UploadFile = File(...)):
    image_path = await _save_upload(image)
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
