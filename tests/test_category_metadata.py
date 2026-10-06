"""Category metadata checks require no model or Detectron2 dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CategoryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "instances.json"

    def load(self, categories):
        self.path.write_text(json.dumps({"categories": categories}))
        module_path = ROOT / "ovusis/data/datasets/category_metadata.py"
        self.assertTrue(module_path.is_file(), "annotation-based metadata reader is missing")
        spec = importlib.util.spec_from_file_location("category_metadata_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_category_metadata(self.path)

    def test_padded_string_ids_remain_strings_in_coco_order(self):
        metadata = self.load([{"id": "010", "name": "Train only"}, {"id": "002", "name": "Shared"}])
        self.assertEqual(metadata["thing_classes"], ["Shared", "Train only"])
        self.assertEqual(metadata["thing_dataset_id_to_contiguous_id"], {"002": 0, "010": 1})

    def test_integer_ids_remain_integers_in_coco_order(self):
        metadata = self.load([{"id": 10, "name": "Train only"}, {"id": 2, "name": "Shared"}])
        self.assertEqual(metadata["thing_dataset_id_to_contiguous_id"], {2: 0, 10: 1})

    def test_unpadded_string_ids_use_same_sorting_as_coco(self):
        metadata = self.load([{"id": "2", "name": "Two"}, {"id": "10", "name": "Ten"}])
        self.assertEqual(metadata["thing_classes"], ["Ten", "Two"])

    def test_colors_preserve_annotations_or_are_deterministic(self):
        first = self.load([{"id": "002", "name": "Shared"},
                           {"id": "010", "name": "Train only", "color": [1, 2, 3]}])
        second = self.load([{"id": "010", "name": "Train only", "color": [1, 2, 3]},
                            {"id": "002", "name": "Shared"}])
        self.assertEqual(first, second)
        self.assertEqual(first["thing_colors"][1], [1, 2, 3])
        self.assertEqual(len(first["thing_colors"][0]), 3)
        self.assertTrue(all(0 <= value <= 255 for value in first["thing_colors"][0]))

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate.*ID"):
            self.load([{"id": 1, "name": "One"}, {"id": 1, "name": "Two"}])

    def test_duplicate_names_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate.*name"):
            self.load([{"id": 1, "name": "Same"}, {"id": 2, "name": "Same"}])

    def test_mixed_id_types_fail(self):
        with self.assertRaisesRegex(ValueError, "type"):
            self.load([{"id": 1, "name": "One"}, {"id": "002", "name": "Two"}])

    def test_empty_categories_fail(self):
        with self.assertRaisesRegex(ValueError, "categories"):
            self.load([])

    def test_invalid_category_fields_fail(self):
        for category in [None, {"name": "One"}, {"id": True, "name": "One"},
                         {"id": 1, "name": ""}, {"id": 1, "name": 42},
                         {"id": "", "name": "One"}, {"id": 1.5, "name": "One"},
                         {"id": 1, "name": "One", "color": [1, 2, 300]}]:
            with self.subTest(category=category):
                with self.assertRaises(ValueError):
                    self.load([category])


if __name__ == "__main__":
    unittest.main()
