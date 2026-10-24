"""Dataset-layout checks only; fixture bytes are not training or inference samples."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_pill_classifier import validate_dataset


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = self.root / "mapping.json"
        self.labels = ("40792", "34342", "29002")
        self.mapping.write_text(json.dumps({label: {} for label in self.labels}), encoding="utf-8")
        self.data = self.root / "dataset"

    def make_layout(self):
        for split in ("train", "val"):
            for label in self.labels:
                folder = self.data / split / label
                folder.mkdir(parents=True)
                (folder / "fixture.jpg").write_bytes(f"fixture:{split}:{label}".encode())

    def test_absent_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "directories not found"):
            validate_dataset(self.data, self.mapping)

    def test_separate_images_for_all_three_classes_pass_layout_check(self):
        self.make_layout()
        counts = validate_dataset(self.data, self.mapping)
        self.assertEqual(counts, {split: {label: 1 for label in self.labels} for split in ("train", "val")})

    def test_wrong_classes_are_rejected(self):
        self.make_layout()
        (self.data / "train" / "pill").mkdir()
        with self.assertRaisesRegex(ValueError, "classes must match"):
            validate_dataset(self.data, self.mapping)

    def test_empty_class_is_rejected(self):
        self.make_layout()
        (self.data / "val" / "40792" / "fixture.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "No images"):
            validate_dataset(self.data, self.mapping)

    def test_copying_demo_images_into_validation_is_rejected(self):
        self.make_layout()
        image = self.data / "train" / "40792" / "fixture.jpg"
        (self.data / "val" / "40792" / "copy.jpg").write_bytes(image.read_bytes())
        with self.assertRaisesRegex(ValueError, "identical images"):
            validate_dataset(self.data, self.mapping)


if __name__ == "__main__":
    unittest.main()
