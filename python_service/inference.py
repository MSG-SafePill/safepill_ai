from pathlib import Path

from ultralytics import YOLO


class YoloInferenceEngine:
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = self._load_model()

    def _load_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")
        return YOLO(str(self.model_path))

    def detect(self, image_path: Path) -> list[dict[str, float | int | str]]:
        results = self._model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[dict[str, float | int | str]] = []

        for box in result.boxes:
            class_idx = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "pillName": str(names[class_idx]),
                    "confidence": round(confidence, 4),
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                }
            )
        return detections

    def infer(self, image_path: Path) -> list[dict[str, float | str]]:
        detections = self.detect(image_path)
        best_by_label: dict[str, float] = {}

        for det in detections:
            label = str(det["pillName"])
            confidence = float(det["confidence"])
            prev = best_by_label.get(label)
            if prev is None or confidence > prev:
                best_by_label[label] = confidence

        return [
            {"pillName": label, "confidence": round(confidence, 4)}
            for label, confidence in sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)
        ]
