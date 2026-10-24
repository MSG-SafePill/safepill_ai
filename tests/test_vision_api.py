"""FastAPI contract/concurrency tests with mocked inference and no database access."""
import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

REQUIRED = ("fastapi", "httpx", "ultralytics", "cv2", "easyocr", "dotenv", "openai")
MISSING = [name for name in REQUIRED if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, "AI/API test dependencies missing: " + ", ".join(MISSING))
class VisionApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from python_service import app
        self.module = app
        app._startup_failures.clear()
        self.preload = patch.object(app, "_preload_vision_models")
        self.preload.start()
        self.addCleanup(self.preload.stop)
        self.client = TestClient(app.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.files = {"image": ("test.jpg", b"test-fixture", "image/jpeg")}

    def test_response_contract_and_tempfile_cleanup(self):
        paths = []

        class Classifier:
            def classify(self, image_path, top_k):
                paths.append(image_path)
                return []

        with patch.object(self.module, "get_multi_pill_classification_service", return_value=Classifier()):
            response = self.client.post("/classify-pills", files=self.files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": False, "detectedCount": 0, "detectedPills": []})
        self.assertTrue(paths)
        self.assertFalse(paths[0].exists())
        self.assertIn("x-request-id", response.headers)

    def test_health_responds_during_inference_and_overlap_is_rejected(self):
        started, finish = threading.Event(), threading.Event()

        class SlowClassifier:
            def classify(self, **kwargs):
                started.set()
                if not finish.wait(5):
                    raise RuntimeError("test timed out")
                return []

        with patch.object(self.module, "get_multi_pill_classification_service", return_value=SlowClassifier()), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.client.post, "/classify-pills", files=self.files)
            try:
                self.assertTrue(started.wait(2))
                health = pool.submit(self.client.get, "/health").result(timeout=1)
                self.assertEqual(health.json(), {"status": "ok"})
                busy = pool.submit(self.client.post, "/classify-pills", files=self.files).result(timeout=1)
                self.assertEqual(busy.status_code, 503)
            finally:
                finish.set()
            self.assertEqual(first.result(timeout=2).status_code, 200)

    def test_missing_model_returns_503(self):
        self.module._startup_failures["classification"] = "FileNotFoundError"
        try:
            response = self.client.post("/classify-pill", files=self.files)
        finally:
            self.module._startup_failures.clear()
        self.assertEqual(response.status_code, 503)

    def test_invalid_image_releases_capacity(self):
        with patch.object(self.module, "get_multi_pill_classification_service", side_effect=ValueError("invalid image")):
            response = self.client.post("/classify-pills", files=self.files)
        self.assertEqual(response.status_code, 400)
        self.assertTrue(self.module._vision_slots.acquire(blocking=False))
        self.module._vision_slots.release()

    def test_upload_limit_returns_413_without_inference(self):
        with patch.object(self.module, "MAX_IMAGE_BYTES", 5), \
             patch.object(self.module, "get_multi_pill_classification_service") as service:
            response = self.client.post("/classify-pills", files=self.files)
        self.assertEqual(response.status_code, 413)
        service.assert_not_called()

    def test_prescription_upload_returns_only_medicine_titles(self):
        class PrescriptionOcr:
            def extract_prescription_medicines(self, image_path):
                self.image_path = image_path
                return [
                    {"text": "코푸정", "normalizedText": "코푸정", "confidence": 0.55, "regionIndex": 0},
                    {"text": "레세티잘정", "normalizedText": "레세티잘정", "confidence": 0.73, "regionIndex": 0},
                ]

        class NoMatches:
            def match(self, keywords, top_k):
                return []

        ocr = PrescriptionOcr()
        with patch.object(self.module, "get_ocr_pipeline", return_value=ocr), \
             patch.object(self.module, "get_medication_matcher", return_value=NoMatches()):
            response = self.client.post("/prescription-ocr-upload", files=self.files)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["medicineName"] for item in response.json()["items"]],
            ["코푸정", "레세티잘정"],
        )
        self.assertFalse(ocr.image_path.exists())


if __name__ == "__main__":
    unittest.main()
