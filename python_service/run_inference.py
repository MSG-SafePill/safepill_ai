import argparse
import json
from pathlib import Path

from python_service.inference import YoloInferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference for SafePill")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt"),
        help="Path to YOLO model (.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--device", default=None, help="Inference device (e.g., cpu, 0)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    engine = YoloInferenceEngine(
        model_path=Path(args.model),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    predictions = engine.infer(image_path)
    print(json.dumps({"status": "ok" if predictions else "no_detection", "candidates": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
