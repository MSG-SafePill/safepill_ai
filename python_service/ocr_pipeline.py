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
        self.min_side = 1280
        self.max_side = 1920

    @lru_cache(maxsize=1)
    def _reader(self) -> easyocr.Reader:
        return easyocr.Reader(self.languages, gpu=self.use_gpu)

    def extract(
        self,
        image_path: Path,
        detections: list[dict[str, float | int | str]] | None = None,
        min_chars: int = 2,
        min_text_height_px: int = 0,
        min_text_height_ratio: float = 0.0,
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
            crop = self._resize_for_ocr(crop)
            region_candidates = self._extract_with_variants(
                image=crop,
                min_chars=min_chars,
                min_text_height_px=min_text_height_px,
                min_text_height_ratio=min_text_height_ratio,
            )
            for candidate in region_candidates:
                candidate["regionIndex"] = region_idx
            raw_candidates.extend(region_candidates)

        return self._deduplicate(raw_candidates)

    def _clip_bbox(self, det: dict[str, float | int | str], width: int, height: int) -> dict[str, int]:
        x1 = max(0, int(det["x1"]) - self.padding)
        y1 = max(0, int(det["y1"]) - self.padding)
        x2 = min(width, int(det["x2"]) + self.padding)
        y2 = min(height, int(det["y2"]) + self.padding)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _extract_with_variants(
        self,
        image: np.ndarray,
        min_chars: int,
        min_text_height_px: int,
        min_text_height_ratio: float,
    ) -> list[dict[str, float | str | int]]:
        variants = self._build_variants(image)
        collected: list[dict[str, float | str | int]] = []
        image_height = image.shape[0]
        for index, processed in enumerate(variants):
            ocr_results = self._reader().readtext(processed, detail=1, paragraph=False)
            for bbox, text, confidence in ocr_results:
                normalized = self._normalize_text(text)
                if len(normalized) < min_chars:
                    continue
                bbox_height, bbox_width = self._bbox_size(bbox)
                text_height_ratio = bbox_height / float(max(image_height, 1))
                if min_text_height_px > 0 and bbox_height < min_text_height_px:
                    continue
                if min_text_height_ratio > 0 and text_height_ratio < min_text_height_ratio:
                    continue
                collected.append(
                    {
                        "text": text.strip(),
                        "normalizedText": normalized,
                        "confidence": round(float(confidence), 4),
                        "textHeightPx": int(round(bbox_height)),
                        "textWidthPx": int(round(bbox_width)),
                        "textHeightRatio": round(float(text_height_ratio), 6),
                    }
                )

            if self._has_confident_candidate(collected):
                break

            # Use a stronger pass only when the simpler image is too noisy.
            if index >= 1 and collected:
                break
        return collected

    def _resize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest < self.min_side:
            ratio = self.min_side / float(longest)
        elif longest > self.max_side:
            ratio = self.max_side / float(longest)
        else:
            return image
        resized = cv2.resize(
            image,
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA,
        )
        return resized

    def _build_variants(self, image: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.bilateralFilter(clahe, d=5, sigmaColor=45, sigmaSpace=45)
        sharpened = cv2.addWeighted(denoised, 1.35, cv2.GaussianBlur(denoised, (0, 0), 1.0), -0.35, 0)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return [
            image,
            sharpened,
            adaptive,
            otsu,
        ]

    def _has_confident_candidate(self, candidates: list[dict[str, float | str | int]]) -> bool:
        if not candidates:
            return False
        return any(float(candidate["confidence"]) >= 0.5 for candidate in candidates)

    def _normalize_text(self, text: str) -> str:
        compact = re.sub(r"\s+", "", text.strip())
        return re.sub(r"[^0-9A-Za-z가-힣]", "", compact).upper()

    def _bbox_size(self, bbox: list[list[float]] | tuple[tuple[float, float], ...]) -> tuple[float, float]:
        points = np.array(bbox, dtype=np.float32)
        if points.size == 0:
            return 0.0, 0.0
        xs = points[:, 0]
        ys = points[:, 1]
        width = float(np.max(xs) - np.min(xs))
        height = float(np.max(ys) - np.min(ys))
        return max(height, 0.0), max(width, 0.0)

    def _deduplicate(self, candidates: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
        best_by_token: dict[str, dict[str, float | str | int]] = {}
        for item in candidates:
            token = str(item["normalizedText"])
            prev = best_by_token.get(token)
            if prev is None or float(item["confidence"]) > float(prev["confidence"]):
                best_by_token[token] = item
        return sorted(best_by_token.values(), key=lambda item: float(item["confidence"]), reverse=True)
