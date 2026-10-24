"""Validate configuration failures using a fake model, never real predictions."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from python_service.pill_classifier import PillClassificationService


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_path = Path(self.temp.name) / "best.pt"
        self.model_path.touch()
        self.mapping_path = Path(self.temp.name) / "mapping.json"
        self.mapping_path.write_text(json.dumps({label: {} for label in ("40792", "29002", "34342")}), encoding="utf-8")

    def test_three_medicine_labels_are_accepted(self):
        model = SimpleNamespace(task="classify", names={0: "40792", 1: "29002", 2: "34342"})
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}):
            service = PillClassificationService(self.model_path, self.mapping_path)
        self.assertEqual(set(service.mapping), {"40792", "29002", "34342"})

    def test_detection_model_cannot_be_used_as_medicine_classifier(self):
        model = SimpleNamespace(task="detect", names={0: "pill"})
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}):
            with self.assertRaisesRegex(ValueError, "classification model"):
                PillClassificationService(self.model_path, self.mapping_path)

    def test_generic_pretrained_labels_cannot_masquerade_as_medicines(self):
        model = SimpleNamespace(task="classify", names={0: "tench", 1: "goldfish"})
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}):
            with self.assertRaisesRegex(ValueError, "mapping"):
                PillClassificationService(self.model_path, self.mapping_path)


if __name__ == "__main__":
    unittest.main()
