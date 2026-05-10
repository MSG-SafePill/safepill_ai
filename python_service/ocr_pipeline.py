import re
from functools import lru_cache
from pathlib import Path

import cv2
import easyocr
import numpy as np


class OcrPipeline:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool = False, padding: int = 8) -> None:
        self.languages = languages or ["ko", "en"]
        self.use_gpu = use_gpu
        self.padding = padding

    @lru_cache(maxsize=1)
    def _reader(self) -> easyocr.Reader:
        return easyocr.Reader(self.languages, gpu=self.use_gpu)

    def extract(
        self,
        image_path: Path,
        detections: list[dict[str, float | int | str]] | None = None,
        min_chars: int = 2,
    ) -> list[dict[str, float | str | int]]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to open image: {image_path}")

        if detections:
            regions = [self._clip_bbox(det, image.shape[1], image.shape[0]) for det in detections]
        else:
            h, w = image.shape[:2]
            regions = [{"x1": 0, "y1": 0, "x2": w, "y2": h}]

        raw_candidates: list[dict[str, float | str | int]] = []
        for region_idx, region in enumerate(regions):
            crop = image[region["y1"] : region["y2"], region["x1"] : region["x2"]]
            if crop.size == 0:
                continue
            processed = self._preprocess(crop)
            # easyocr output: [([x1,y1],[x2,y2],[x3,y3],[x4,y4]), text, confidence]
            ocr_results = self._reader().readtext(processed, detail=1, paragraph=False)
            for _, text, confidence in ocr_results:
                normalized = self._normalize_text(text)
                if len(normalized) < min_chars:
                    continue
                raw_candidates.append(
                    {
                        "text": text.strip(),
                        "normalizedText": normalized,
                        "confidence": round(float(confidence), 4),
                        "regionIndex": region_idx,
                    }
                )

        return self._deduplicate(raw_candidates)

    def _clip_bbox(self, det: dict[str, float | int | str], width: int, height: int) -> dict[str, int]:
        x1 = max(0, int(det["x1"]) - self.padding)
        y1 = max(0, int(det["y1"]) - self.padding)
        x2 = min(width, int(det["x2"]) + self.padding)
        y2 = min(height, int(det["y2"]) + self.padding)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _normalize_text(self, text: str) -> str:
        compact = re.sub(r"\s+", "", text.strip())
        return re.sub(r"[^0-9A-Za-z가-힣]", "", compact).upper()

    def _deduplicate(self, candidates: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
        best_by_token: dict[str, dict[str, float | str | int]] = {}
        for item in candidates:
            token = str(item["normalizedText"])
            prev = best_by_token.get(token)
            if prev is None or float(item["confidence"]) > float(prev["confidence"]):
                best_by_token[token] = item
        return sorted(best_by_token.values(), key=lambda item: float(item["confidence"]), reverse=True)
