import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a new YOLO11 pill detector from random initialization.")
    parser.add_argument("--model", default="yolo11n.yaml", help="YOLO11 detection architecture YAML; .pt weights are not accepted")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data.yaml"), help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--project", default="runs/detect", help="Output project directory")
    parser.add_argument("--name", default="pill_yolo11n_scratch", help="Run name")
    parser.add_argument("--output-model", default="models/pill_detect/yolo11n/best.pt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).name.lower().startswith("yolo11") or Path(args.model).suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit("--model must be a YOLO11 architecture YAML; trained .pt weights are not allowed.")
    if not Path(args.data).is_file():
        raise SystemExit(f"Detection data YAML not found: {args.data}")
    if args.output_model.exists():
        raise SystemExit(f"Refusing to overwrite existing model: {args.output_model}")
    import torch
    from ultralytics import YOLO

    print("==================================================")
    if torch.cuda.is_available():
        print(f"GPU 인식 완료: {torch.cuda.get_device_name(0)}")
    else:
        print("경고: GPU를 찾을 수 없습니다. CPU 학습은 오래 걸릴 수 있습니다.")
    print("==================================================\n")

    model = YOLO(args.model)

    print("[SafePill] YOLO11 검출 모델을 처음부터 학습합니다.")

    result = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        name=args.name,
        plots=True,
        cache=False,
        pretrained=False,
        resume=False,
    )

    best_model = Path(result.save_dir) / "weights" / "best.pt"
    if not best_model.is_file():
        raise SystemExit(f"Training finished, but best.pt was not found: {best_model}")
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, args.output_model)
    print(f"New YOLO11 detector saved to {args.output_model}")


if __name__ == "__main__":
    main()
