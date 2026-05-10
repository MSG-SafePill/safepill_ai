import argparse
import json
from pathlib import Path

from python_service.inference import YoloInferenceEngine
from python_service.ocr_pipeline import OcrPipeline


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR pipeline for SafePill")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default=str(default_model_path()), help="YOLO model path")
    parser.add_argument("--use-detections", action="store_true", help="Crop OCR regions with YOLO detections")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    detections = None
    if args.use_detections:
        yolo = YoloInferenceEngine(model_path=Path(args.model))
        detections = yolo.detect(image_path)

    pipeline = OcrPipeline()
    result = pipeline.extract(image_path=image_path, detections=detections)
    print(json.dumps({"status": "ok" if result else "no_text", "candidates": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
