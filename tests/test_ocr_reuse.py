"""Control-flow tests with a fake OCR reader; these do not measure OCR accuracy."""
import importlib.util
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for array fixtures")
class OcrReuseTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.reader = Mock()
        self.reader.readtext.return_value = [(None, " CLP DE1 ", 0.9)]
        self.factory = Mock(return_value=self.reader)
        self.image_io = SimpleNamespace(read_image=Mock(return_value=self.image))
        source = Path(__file__).resolve().parents[1] / "python_service" / "ocr_pipeline.py"
        spec = importlib.util.spec_from_file_location("ocr_under_test", source)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "cv2": Mock(), "easyocr": SimpleNamespace(Reader=self.factory),
            "python_service.image_io": self.image_io,
        }):
            spec.loader.exec_module(self.module)
        self.pipeline = self.module.OcrPipeline()
        self.pipeline._preprocess = lambda image: image

    def test_parallel_first_load_constructs_one_reader(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            readers = list(pool.map(lambda _: self.pipeline._reader(), range(32)))
        self.assertTrue(all(reader is self.reader for reader in readers))
        self.factory.assert_called_once_with(["ko", "en"], gpu=False)

    def test_duplicate_regions_are_read_once_with_original_region_index(self):
        box = {"x1": 10, "y1": 10, "x2": 40, "y2": 40}
        result = self.pipeline.extract(Path("fixture.png"), [box, dict(box)])
        self.reader.readtext.assert_called_once()
        self.assertEqual(result, [{"text": "CLP DE1", "normalizedText": "CLPDE1", "confidence": 0.9, "regionIndex": 0}])

    def test_distinct_regions_are_not_dropped(self):
        self.pipeline.extract(Path("fixture.png"), [
            {"x1": 10, "y1": 10, "x2": 40, "y2": 40},
            {"x1": 60, "y1": 60, "x2": 80, "y2": 80},
        ])
        self.assertEqual(self.reader.readtext.call_count, 2)

    def test_no_detections_reads_full_image_and_reuses_model(self):
        self.pipeline.warmup()
        for _ in range(2):
            self.pipeline.extract(Path("fixture.png"))
        self.factory.assert_called_once()
        self.assertEqual(self.reader.readtext.call_count, 2)
        self.assertEqual(self.reader.readtext.call_args.args[0].shape, self.image.shape)

    def test_prescription_returns_product_titles_without_receipt_or_form_descriptions(self):
        from python_service.prescription_ocr import PrescriptionOcrParser

        self.reader.readtext.return_value = [
            (None, "레세티잘정(레보세티리진염산염", 0.72),
            (None, "(주)종근당", 0.9),
            (None, "흰색 서방정", 0.95),
            (None, "종근당세파흘러캠술(세파클러", 0.13),
            (None, "성모약국", 0.98),
        ]
        self.module.cv2.cvtColor.side_effect = lambda crop, _: crop

        candidates = self.pipeline.extract_prescription_medicines(Path("fixture.png"))
        items = PrescriptionOcrParser().parse(candidates)

        self.assertEqual(
            [item["medicineName"] for item in items],
            ["레세티잘정", "종근당세파흘러캡슐"],
        )
        self.assertLess(self.reader.readtext.call_args.args[0].shape[0], self.image.shape[0])


if __name__ == "__main__":
    unittest.main()
