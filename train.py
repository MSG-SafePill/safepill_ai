import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for SafePill.")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained YOLO model file")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data.yaml"), help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--project", default="SafePill_AI", help="Output project directory")
    parser.add_argument("--name", default="yolov8s_full_run", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("==================================================")
    if torch.cuda.is_available():
        print(f"✅ GPU 인식 완료: {torch.cuda.get_device_name(0)}")
    else:
        print("🚨 경고: GPU를 찾을 수 없습니다. CPU로 학습하면 너무 오래 걸립니다!")
    print("==================================================\n")

    model = YOLO(args.model)

    print("🚀 [Safe Pill] 실전 AI 학습을 시작합니다! (모니터 끄고 푹 주무셔도 됩니다)")

    model.train(
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
    )

    print("✨ 기나긴 학습이 드디어 완료되었습니다!")
    print(f"📂 앱에 넣을 최종 모델 파일: [{args.project}/{args.name}/weights/best.pt]")


if __name__ == "__main__":
    main()