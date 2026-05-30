import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def _write_data_yaml(base_dir: Path, class_map_path: Path, yaml_path: Path) -> None:
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    classes = class_map.get("classes", [])
    names = [str(item["name"]) for item in classes]

    lines = [
        "# YOLO dataset config (auto-generated)",
        f"train: {((base_dir / 'images' / 'train').as_posix())}",
        f"val: {((base_dir / 'images' / 'val').as_posix())}",
        "",
        f"nc: {len(names)}",
        f"names: {json.dumps(names, ensure_ascii=False)}",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_data(pairs: list[tuple[Path, Path]], split_name: str, base_dir: Path) -> None:
    for img_src, txt_src in tqdm(pairs, desc=f"{split_name} 복사 중"):
        img_dst = base_dir / "images" / split_name / img_src.name
        txt_dst = base_dir / "labels" / split_name / txt_src.name
        shutil.copy2(img_src, img_dst)
        shutil.copy2(txt_src, txt_dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split image/label pairs into YOLO train/val dataset.")
    parser.add_argument(
        "--image-dir",
        default=r"C:\AIData\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\원천데이터\경구약제조합_5000종",
        help="Source image root directory",
    )
    parser.add_argument(
        "--label-dir",
        default=r"C:\AIData\labels_txt",
        help="Directory containing converted YOLO txt labels",
    )
    parser.add_argument(
        "--dataset-dir",
        default=r"C:\AIData\pill_dataset",
        help="Output dataset root directory",
    )
    parser.add_argument(
        "--class-map",
        default=r"C:\AIData\labels_txt\classes.json",
        help="Class mapping json from convert_json.py",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio between 0 and 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--data-yaml",
        default=str(Path(__file__).resolve().parent / "data.yaml"),
        help="Path to write generated data.yaml for YOLO training",
    )
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    dataset_dir = Path(args.dataset_dir)
    class_map_path = Path(args.class_map)
    data_yaml_path = Path(args.data_yaml)

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_images = [
        image_path
        for image_path in image_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    print(f"총 {len(all_images)}개의 알약 사진을 찾았습니다.")

    valid_pairs: list[tuple[Path, Path]] = []
    for image_path in all_images:
        txt_path = label_dir / f"{image_path.stem}.txt"
        if txt_path.exists():
            valid_pairs.append((image_path, txt_path))

    print(f"정답지와 짝이 맞는 데이터는 총 {len(valid_pairs)}개입니다.")
    if not valid_pairs:
        raise RuntimeError("No valid image/label pairs found. Check --image-dir and --label-dir.")

    rng = random.Random(args.seed)
    rng.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * (1 - args.val_ratio))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Train: {len(train_pairs)}개 / Val: {len(val_pairs)}개")
    _copy_data(train_pairs, "train", dataset_dir)
    _copy_data(val_pairs, "val", dataset_dir)
    _write_data_yaml(base_dir=dataset_dir, class_map_path=class_map_path, yaml_path=data_yaml_path)

    print(f"\n✨ 데이터셋 구성 완료: {dataset_dir}")
    print(f"📘 data.yaml 생성 완료: {data_yaml_path}")


if __name__ == "__main__":
    main()
