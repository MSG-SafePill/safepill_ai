import argparse
import random
import shutil
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO train/val dataset from image+label folders."
    )
    parser.add_argument("--images-dir", required=True, help="Source image root directory.")
    parser.add_argument("--labels-dir", required=True, help="Source label root directory.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--class-names-file",
        help="Optional text file with class names (one per line). Creates output data.yaml.",
    )
    parser.add_argument(
        "--data-yaml-path",
        default="data.yaml",
        help="Path of generated data.yaml (relative to output-dir or absolute).",
    )
    return parser.parse_args()


def collect_pairs(
    images_dir: Path, labels_dir: Path, extensions: Sequence[str]
) -> list[tuple[Path, Path, Path]]:
    ext_set = {ext.lower() for ext in extensions}
    pairs: list[tuple[Path, Path, Path]] = []

    for image_path in images_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in ext_set:
            continue

        rel = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path, rel))

    return pairs


def copy_pairs(
    pairs: Sequence[tuple[Path, Path, Path]],
    output_dir: Path,
    split_name: str,
) -> None:
    images_root = output_dir / "images" / split_name
    labels_root = output_dir / "labels" / split_name

    for image_src, label_src, rel in tqdm(pairs, desc=f"{split_name} copy"):
        image_dst = images_root / rel
        label_dst = (labels_root / rel).with_suffix(".txt")

        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)


def write_data_yaml(
    output_dir: Path, class_names_file: Path, data_yaml_path_arg: str
) -> Path:
    class_names = [
        line.strip() for line in class_names_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not class_names:
        raise ValueError(f"Class names file is empty: {class_names_file}")

    data_yaml_path = Path(data_yaml_path_arg)
    if not data_yaml_path.is_absolute():
        data_yaml_path = output_dir / data_yaml_path

    content = "\n".join(
        [
            f"train: {output_dir / 'images' / 'train'}",
            f"val: {output_dir / 'images' / 'val'}",
            "",
            f"nc: {len(class_names)}",
            "names:",
            *[f"  - {name}" for name in class_names],
            "",
        ]
    )
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(content, encoding="utf-8")
    return data_yaml_path


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir not found: {labels_dir}")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    pairs = collect_pairs(images_dir, labels_dir, args.extensions)
    if not pairs:
        raise RuntimeError("No valid image/label pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Found pairs: {len(pairs)} (train={len(train_pairs)}, val={len(val_pairs)})")
    copy_pairs(train_pairs, output_dir, "train")
    copy_pairs(val_pairs, output_dir, "val")

    print(f"Dataset prepared at: {output_dir}")

    if args.class_names_file:
        yaml_path = write_data_yaml(
            output_dir=output_dir,
            class_names_file=Path(args.class_names_file),
            data_yaml_path_arg=args.data_yaml_path,
        )
        print(f"Generated data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
