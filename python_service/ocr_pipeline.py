import re
from pathlib import Path
from threading import RLock

import cv2
import easyocr
import numpy as np

from python_service.image_io import read_image
from python_service.vision_runtime import timed_stage


class OcrPipeline:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool = False, padding: int = 8) -> None:
        self.languages = languages or ["ko", "en"]
        self.use_gpu = use_gpu
        self.padding = padding
        self._reader_instance = None
        self._lock = RLock()

    def _reader(self) -> easyocr.Reader:
        with self._lock:
            if self._reader_instance is None:
                with timed_stage("ocr_model_load"):
                    self._reader_instance = easyocr.Reader(self.languages, gpu=self.use_gpu)
            return self._reader_instance

    def warmup(self) -> None:
        self._reader()

    def extract(
        self,
        image_path: Path,
        detections: list[dict[str, float | int | str]] | None = None,
        min_chars: int = 2,
    ) -> list[dict[str, float | str | int]]:
        with timed_stage("ocr_image_decode"):
            image = read_image(image_path)

        if not detections:
            image = self._downscale(image)

        if detections:
            regions = [self._clip_bbox(det, image.shape[1], image.shape[0]) for det in detections]
        else:
            h, w = image.shape[:2]
            regions = [{"x1": 0, "y1": 0, "x2": w, "y2": h}]

        raw_candidates: list[dict[str, float | str | int]] = []
        seen_regions: set[tuple[int, int, int, int]] = set()
        for region_idx, region in enumerate(regions):
            region_key = tuple(region[key] for key in ("x1", "y1", "x2", "y2"))
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)
            crop = image[region["y1"] : region["y2"], region["x1"] : region["x2"]]
            if crop.size == 0:
                continue
            with timed_stage("ocr_preprocess"):
                processed = self._preprocess(crop)
            # easyocr output: [([x1,y1],[x2,y2],[x3,y3],[x4,y4]), text, confidence]
            with self._lock, timed_stage("ocr_inference"):
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

    def extract_prescription_medicines(self, image_path: Path) -> list[dict[str, float | str | int]]:
        """Read the product-name column at its original resolution.

        Pharmacy printouts often put ingredients, instructions, and the receipt
        beside the medicine names. Reading the whole page as one region makes
        those unrelated strings look like additional prescriptions.
        """
        with timed_stage("ocr_image_decode"):
            image = read_image(image_path)
        height, width = image.shape[:2]
        x1, x2 = int(width * 0.08755), int(width * 0.36765)
        y1, y2 = int(height * 0.18675), int(height * 0.79366)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        with timed_stage("ocr_preprocess"):
            grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        with self._lock, timed_stage("ocr_inference"):
            ocr_results = self._reader().readtext(grayscale, detail=1, paragraph=False)

        candidates: dict[str, dict[str, float | str | int]] = {}
        for _, text, confidence in ocr_results:
            medicine_name = self._prescription_medicine_name(text)
            if medicine_name is None:
                continue
            candidate = {
                "text": medicine_name,
                "normalizedText": self._normalize_text(medicine_name),
                "confidence": round(float(confidence), 4),
                "regionIndex": 0,
            }
            key = str(candidate["normalizedText"])
            previous = candidates.get(key)
            if previous is None or float(candidate["confidence"]) > float(previous["confidence"]):
                candidates[key] = candidate
        return list(candidates.values())

    @staticmethod
    def _prescription_medicine_name(text: str) -> str | None:
        cleaned = re.sub(r"\s+", "", text.strip())
        if not cleaned or cleaned.startswith(("(", "[")):
            return None
        if re.search(r"(?:흰색|백색|청색|분홍색|녹색|황색|미황색|연녹색|경질|연질|복약|운전)", cleaned):
            return None
        # Parentheses on these printouts contain ingredients, not another drug.
        title = re.split(r"[(_]", cleaned, maxsplit=1)[0]
        if not re.match(r"^[가-힣][가-힣A-Za-z0-9.-]{2,}$", title):
            return None
        if not re.search(r"(?:캡[슐술]|캠술|서방정|장용정|정|시럽|현탁액|주사|산|액)(?:\d+(?:\.\d+)?(?:mg|g|밀리그램))?$", title, re.IGNORECASE):
            return None
        return re.sub(r"(?:캡술|캠술)$", "캡슐", title)

    def _downscale(self, image: np.ndarray, max_dimension: int = 1800) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_dimension:
            return image
        scale = max_dimension / longest
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

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
