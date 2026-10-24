from pathlib import Path
from typing import Any

import cv2
import numpy as np

from python_service.pill_classifier import PillClassificationService
from python_service.image_io import read_image
from python_service.vision_runtime import timed_stage


class MultiPillClassificationService:
    def __init__(self, classifier: PillClassificationService):
        self.classifier = classifier

    def classify(self, image_path: Path, top_k: int = 3) -> list[dict[str, Any]]:
        with timed_stage("classification_image_decode"):
            image = read_image(image_path)

        with timed_stage("pill_regions"):
            boxes = self._detect_candidate_boxes(image)
        if not boxes:
            height, width = image.shape[:2]
            boxes = [(0, 0, width, height)]

        detected_pills = []
        for index, box in enumerate(boxes, start=1):
            crop = self._crop_with_padding(image, box)
            # Preserve the previous JPEG codec/default quality, without a disk round trip.
            with timed_stage("pill_crop_prepare"):
                encoded_ok, encoded = cv2.imencode(".jpg", crop)
                if not encoded_ok:
                    raise ValueError("Unable to encode pill image.")
                decoded_crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded_crop is None:
                    raise ValueError("Unable to decode pill crop.")
            candidates = self.classifier.classify_image(decoded_crop, top_k=top_k)

            x1, y1, x2, y2 = box
            detected_pills.append(
                {
                    "pillIndex": index,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "candidates": candidates,
                }
            )
        return detected_pills

    def _detect_candidate_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        image_area = height * width

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        corner_samples = np.concatenate(
            [
                blurred[: max(8, height // 20), : max(8, width // 20)].reshape(-1, 3),
                blurred[: max(8, height // 20), -max(8, width // 20) :].reshape(-1, 3),
                blurred[-max(8, height // 20) :, : max(8, width // 20)].reshape(-1, 3),
                blurred[-max(8, height // 20) :, -max(8, width // 20) :].reshape(-1, 3),
            ],
            axis=0,
        )
        background_color = np.median(corner_samples, axis=0)
        color_distance = np.linalg.norm(blurred.astype(np.float32) - background_color, axis=2)

        color_mask = ((color_distance > 28) | ((saturation > 35) & (value > 45))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.0015:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 18 or h < 18:
                continue
            aspect = max(w / h, h / w)
            if aspect > 6.0:
                continue
            fill_ratio = area / float(w * h)
            if fill_ratio < 0.12:
                continue
            boxes.append((x, y, x + w, y + h))

        return sorted(boxes, key=lambda item: (item[1], item[0]))

    @staticmethod
    def _crop_with_padding(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = max(8, int((x2 - x1) * 0.18))
        pad_y = max(8, int((y2 - y1) * 0.18))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        return image[y1:y2, x1:x2]
