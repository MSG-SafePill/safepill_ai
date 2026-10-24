"""Verify both training entry points start from architecture YAML without old weights."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import train
from scripts import train_pill_classifier


class ScratchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        weights = self.run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"new-training-result")
        self.yolo = Mock()
        self.yolo.return_value.train.return_value = SimpleNamespace(save_dir=self.run_dir)

    def test_detection_uses_new_yolo11_architecture_and_no_pretrained_weights(self):
        data = self.root / "data.yaml"
        data.write_text("train: images/train\nval: images/val\nnames: [pill]\n", encoding="utf-8")
        output = self.root / "detector" / "best.pt"
        args = ["train.py", "--data", str(data), "--output-model", str(output)]
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch("sys.argv", args), patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=self.yolo), "torch": torch}):
            train.main()
        self.yolo.assert_called_once_with("yolo11n.yaml")
        self.assertFalse(self.yolo.return_value.train.call_args.kwargs["pretrained"])
        self.assertFalse(self.yolo.return_value.train.call_args.kwargs["resume"])
        self.assertEqual(output.read_bytes(), b"new-training-result")

    def test_classification_uses_new_yolo11_architecture_and_no_pretrained_weights(self):
        data = self.root / "dataset"
        labels = ("40792", "29002", "34342")
        for split in ("train", "val"):
            for label in labels:
                folder = data / split / label
                folder.mkdir(parents=True)
                (folder / "image.jpg").write_bytes(f"{split}:{label}".encode())
        mapping = self.root / "mapping.json"
        mapping.write_text(json.dumps({label: {} for label in labels}), encoding="utf-8")
        output = self.root / "classifier" / "best.pt"
        args = ["train_pill_classifier.py", "--data", str(data), "--mapping", str(mapping), "--output-model", str(output)]
        with patch("sys.argv", args), patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=self.yolo)}):
            train_pill_classifier.main()
        self.yolo.assert_called_once_with("yolo11n-cls.yaml")
        self.assertFalse(self.yolo.return_value.train.call_args.kwargs["pretrained"])
        self.assertFalse(self.yolo.return_value.train.call_args.kwargs["resume"])
        self.assertEqual(output.read_bytes(), b"new-training-result")


if __name__ == "__main__":
    unittest.main()
