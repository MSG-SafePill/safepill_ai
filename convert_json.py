import argparse
import json
from pathlib import Path

from tqdm import tqdm


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_image_size(data: dict) -> tuple[float, float] | None:
    image_sources = []
    if isinstance(data.get("images"), list) and data["images"]:
        image_sources.append(data["images"][0])
    if isinstance(data.get("image"), dict):
        image_sources.append(data["image"])

    for source in image_sources:
        width = _safe_float(source.get("width"))
        height = _safe_float(source.get("height"))
        if width and height and width > 0 and height > 0:
            return width, height
    return None


def _extract_bbox(annotation: dict) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = (_safe_float(v) for v in bbox)
    elif isinstance(bbox, dict):
        x = _safe_float(bbox.get("x"))
        y = _safe_float(bbox.get("y"))
        w = _safe_float(bbox.get("w", bbox.get("width")))
        h = _safe_float(bbox.get("h", bbox.get("height")))
    else:
        return None

    if None in (x, y, w, h):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _category_id_to_name(data: dict) -> dict[int, str]:
    mapping: dict[int, str] = {}
    categories = data.get("categories")
    if not isinstance(categories, list):
        return mapping

    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        name = category.get("name")
        if category_id is None or not name:
            continue
        try:
            mapping[int(category_id)] = str(name).strip()
        except (TypeError, ValueError):
            continue
    return mapping


def _extract_label(annotation: dict, category_map: dict[int, str]) -> str | None:
    for field in ("category_name", "class_name", "label", "item_name", "pill_name", "name"):
        value = annotation.get(field)
        if value:
            return str(value).strip()

    category_id = annotation.get("category_id", annotation.get("class_id"))
    if category_id is not None:
        try:
            return category_map.get(int(category_id))
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON labels to YOLO TXT labels (multi-class).")
    parser.add_argument(
        "--json-dir",
        default=r"C:\AIData\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\라벨링데이터\경구약제조합_5000종",
        help="Directory containing source JSON annotation files",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\AIData\labels_txt",
        help="Directory where YOLO txt labels will be written",
    )
    parser.add_argument(
        "--class-map-output",
        default=r"C:\AIData\labels_txt\classes.json",
        help="Path to save class name to class id mapping",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    class_map_output = Path(args.class_map_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    class_map_output.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.rglob("*.json"))
    print(f"총 {len(json_files)}개의 정답지를 찾았습니다. 멀티클래스 YOLO 라벨로 변환합니다.")

    class_to_id: dict[str, int] = {}
    defective_bbox_count = 0
    unknown_label_count = 0

    for file_path in tqdm(json_files):
        with file_path.open("r", encoding="utf-8") as source_file:
            data = json.load(source_file)

        image_size = _extract_image_size(data)
        if image_size is None:
            defective_bbox_count += 1
            continue
        img_w, img_h = image_size
        category_map = _category_id_to_name(data)

        lines: list[str] = []
        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []

        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue

            bbox = _extract_bbox(annotation)
            if bbox is None:
                defective_bbox_count += 1
                continue

            label_name = _extract_label(annotation, category_map)
            if not label_name:
                unknown_label_count += 1
                continue

            class_id = class_to_id.setdefault(label_name, len(class_to_id))
            x, y, w, h = bbox
            x_center = (x + (w / 2.0)) / img_w
            y_center = (y + (h / 2.0)) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        txt_path = output_dir / f"{file_path.stem}.txt"
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    class_map_output.write_text(
        json.dumps(
            {
                "num_classes": len(class_to_id),
                "classes": [{"id": class_id, "name": name} for name, class_id in sorted(class_to_id.items(), key=lambda item: item[1])],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✨ 변환 완료! 라벨 폴더: {output_dir}")
    print(f"📘 클래스 매핑 저장: {class_map_output}")
    print(f"🔢 총 클래스 수: {len(class_to_id)}")
    print(f"🚨 잘못된 bbox/이미지 크기 데이터: {defective_bbox_count}개")
    print(f"🚨 클래스 정보가 없어 제외된 어노테이션: {unknown_label_count}개")


if __name__ == "__main__":
    main()