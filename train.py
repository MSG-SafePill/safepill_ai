import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO for multi-pill product classification."
    )
    parser.add_argument("--data", default="data.yaml", help="Path to YOLO data yaml.")
    parser.add_argument("--weights", default="yolov8s.pt", help="Base model weights.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for training (e.g. 0, 0,1, cpu). Default: auto (GPU if available).",
    )
    parser.add_argument("--project", default="SafePill_AI")
    parser.add_argument("--name", default="yolov8s_multiclass")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Disable result plots.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str | int:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data yaml not found: {data_path}")

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not detected. Training will run on CPU.")

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        device=resolve_device(args.device),
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        name=args.name,
        plots=not args.no_plots,
        cache=args.cache,
    )

    print("Training completed.")
    print(f"best.pt: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()