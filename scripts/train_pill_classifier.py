import argparse
import hashlib
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a new YOLO11n pill classifier from random initialization.")
    parser.add_argument("--data", default="datasets/pill_mvp/yolo_cls")
    parser.add_argument("--model", default="yolo11n-cls.yaml", help="YOLO11 classification architecture YAML; .pt weights are not accepted")
    parser.add_argument("--mapping", default="mappings/pill_class_mapping.json", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=224, type=int)
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="pill_mvp_yolo11n")
    parser.add_argument("--output-model", default="models/pill_cls/yolo11n/best.pt", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite-output", action="store_true", help="Explicitly allow replacing the output model.")
    parser.add_argument("--check-only", action="store_true", help="Validate separate train/val images without loading or downloading a model.")
    return parser.parse_args()


def validate_dataset(data: Path, mapping_path: Path) -> dict[str, dict[str, int]]:
    if not all((data / split).is_dir() for split in ("train", "val")):
        raise ValueError(f"Classification train/val directories not found: {data}")
    with mapping_path.open(encoding="utf-8") as source:
        labels = set(json.load(source))
    if not labels:
        raise ValueError("Medicine class mapping is empty.")
    counts = {}
    hashes = {}
    for split in ("train", "val"):
        classes = {path.name for path in (data / split).iterdir() if path.is_dir()}
        if classes != labels:
            raise ValueError(f"{split} classes must match mapping: expected={sorted(labels)}, found={sorted(classes)}")
        counts[split] = {}
        hashes[split] = set()
        for label in sorted(labels):
            images = [path for path in (data / split / label).rglob("*")
                      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                raise ValueError(f"No images found in {data / split / label}")
            counts[split][label] = len(images)
            for image in images:
                with image.open("rb") as source:
                    hashes[split].add(hashlib.file_digest(source, "sha256").hexdigest())
    if hashes["train"] & hashes["val"]:
        raise ValueError("Train/val contain identical images. Use independently captured validation images, not copies of the three demo images.")
    return counts


def main() -> None:
    args = parse_args()
    if not Path(args.model).name.lower().startswith("yolo11") or Path(args.model).suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit("--model must be a YOLO11 classification architecture YAML; trained .pt weights are not allowed.")
    if args.output_model.exists() and not args.overwrite_output:
        raise SystemExit(f"Refusing to overwrite existing model: {args.output_model}. Use a separate --output-model for comparisons.")
    try:
        counts = validate_dataset(Path(args.data), args.mapping)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"model": args.model, "output": str(args.output_model), "images": counts}, ensure_ascii=False))
    if args.check_only:
        print("Input layout and exact-duplicate checks passed; recognition accuracy has not been evaluated.")
        return
    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=False,
        resume=False,
    )

    save_dir = Path(result.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise SystemExit(f"Training finished, but best.pt was not found: {best_model}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, args.output_model)
    print(f"Copied best model to {args.output_model}")
    print("New YOLO11 classifier saved. Validate on held-out images before use.")


if __name__ == "__main__":
    main()
