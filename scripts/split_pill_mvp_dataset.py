import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split pill MVP images into YOLO classification folders.")
    parser.add_argument("--raw-dir", default="datasets/pill_mvp/raw", type=Path)
    parser.add_argument("--output-dir", default="datasets/pill_mvp/yolo_cls", type=Path)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(images: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    train_count = max(1, int(len(images) * train_ratio))
    val_count = max(1, int(len(images) * val_ratio)) if len(images) >= 3 else 0

    if train_count + val_count >= len(images) and len(images) >= 3:
        train_count = len(images) - 2
        val_count = 1

    return {
        "train": images[:train_count],
        "val": images[train_count : train_count + val_count],
        "test": images[train_count + val_count :],
    }


def copy_split(split_name: str, class_label: str, images: list[Path], output_dir: Path) -> None:
    target_dir = output_dir / split_name / class_label
    target_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, target_dir / image.name)


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise SystemExit(f"Raw dataset directory not found: {args.raw_dir}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    summary: dict[str, dict[str, int]] = {}
    class_dirs = sorted(path for path in args.raw_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found under: {args.raw_dir}")

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if len(images) < 2:
            raise SystemExit(f"Class {class_dir.name} needs at least 2 images.")

        random.shuffle(images)
        splits = split_images(images, args.train_ratio, args.val_ratio)
        summary[class_dir.name] = {}
        for split_name, split_files in splits.items():
            copy_split(split_name, class_dir.name, split_files, args.output_dir)
            summary[class_dir.name][split_name] = len(split_files)

    for class_label, counts in summary.items():
        print(
            f"{class_label}: train={counts['train']}, "
            f"val={counts['val']}, test={counts['test']}"
        )


if __name__ == "__main__":
    main()
