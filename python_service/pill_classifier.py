import json
from pathlib import Path
from typing import Any


class PillClassificationService:
    def __init__(self, model_path: Path, mapping_path: Path, device: str | None = None):
        if not model_path.exists():
            raise FileNotFoundError(f"Pill classification model not found: {model_path}")

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.mapping = self._load_mapping(mapping_path)
        self.device = device

    @staticmethod
    def _load_mapping(mapping_path: Path) -> dict[str, dict[str, Any]]:
        if not mapping_path.exists():
            return {}
        with mapping_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return {str(label): value for label, value in data.items()}

    def classify(self, image_path: Path, top_k: int = 3) -> list[dict[str, Any]]:
        results = self.model.predict(str(image_path), device=self.device, verbose=False)
        if not results:
            return []

        result = results[0]
        if result.probs is None:
            return []

        labels = [int(index) for index in result.probs.top5[:top_k]]
        scores = [float(score) for score in result.probs.top5conf[:top_k]]
        names = result.names or {}

        candidates = []
        for class_index, score in zip(labels, scores, strict=False):
            class_label = str(names.get(class_index, class_index))
            metadata = self.mapping.get(class_label, {})
            candidates.append(
                {
                    "classLabel": class_label,
                    "medicineName": str(metadata.get("medicineName", "확인 필요")),
                    "manufacturer": str(metadata.get("manufacturer", "확인 필요")),
                    "mark": str(metadata.get("mark", "확인 필요")),
                    "shape": str(metadata.get("shape", "확인 필요")),
                    "color": str(metadata.get("color", "확인 필요")),
                    "score": score,
                    "reason": "이미지 분류 결과 가장 유사한 후보입니다.",
                }
            )
        return candidates
