import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification model for SafePill pill MVP.")
    parser.add_argument("--data", default="datasets/pill_mvp/yolo_cls")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--imgsz", default=224, type=int)
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="pill_mvp")
    parser.add_argument("--output-model", default="models/pill_cls/best.pt", type=Path)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    save_dir = Path(result.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise SystemExit(f"Training finished, but best.pt was not found: {best_model}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, args.output_model)
    print(f"Copied best model to {args.output_model}")


if __name__ == "__main__":
    main()
