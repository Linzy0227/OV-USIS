"""Integration checks using real Detectron2 catalogs and COCO loading."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "openvocab_ovusis_train_a": "train",
    "openvocab_ovusis_val_b": "train",
    "openvocab_ovusis_val_a": "val",
    "openvocab_ovusis_train_b": "val",
}


def write_fixture(root, integer_ids=False):
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for split, categories in [
        ("train", [{"id": "010", "name": "Train only"}, {"id": "002", "name": "Shared"}]),
        ("val", [{"id": "021", "name": "Test last"}, {"id": "002", "name": "Shared"},
                 {"id": "011", "name": "Test first"}]),
    ]:
        if integer_ids:
            categories = [dict(c, id=int(c["id"])) for c in categories]
        (root / split).mkdir(exist_ok=True)
        document = {
            "categories": categories,
            "images": [{"id": 1, "file_name": "fixture.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2 if integer_ids else "002",
                             "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0,
                             "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]]}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(document))


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("requires the ovs environment and compiled Detectron2")
        fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(fixture.cleanup)
        cls.root = Path(fixture.name)
        write_fixture(cls.root)
        isolated_environment = patch.dict(os.environ, {"OVUSIS_DATASET_ROOT": str(cls.root)})
        isolated_environment.start()
        cls.addClassCleanup(isolated_environment.stop)
        sys.path.insert(0, str(ROOT / "detectron2"))
        from detectron2.data import DatasetCatalog, MetadataCatalog
        cls.datasets = DatasetCatalog
        cls.metadata = MetadataCatalog
        spec = importlib.util.spec_from_file_location(
            "ovusis_test_datasets", ROOT / "ovusis/data/datasets/__init__.py",
            submodule_search_locations=[str(ROOT / "ovusis/data/datasets")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        cls.module = package.register_ovusis_instances

    def test_classes_and_a_b_mapping_come_from_annotations(self):
        for name, split in NAMES.items():
            with self.subTest(name=name):
                metadata = self.metadata.get(name)
                self.assertEqual(metadata.image_root, str(self.root / split))
                self.assertEqual(metadata.json_file, str(
                    self.root / "annotations" / f"instances_{split}.json"))
                expected = ["Shared", "Train only"] if split == "train" else ["Shared", "Test first", "Test last"]
                self.assertEqual(metadata.thing_classes, expected)
                self.assertEqual(len(metadata.thing_colors), len(expected))
                self.assertEqual(metadata.thing_dataset_id_to_contiguous_id["002"], 0)

    def test_default_root_is_project_dataset(self):
        with patch.dict(os.environ, {"OVUSIS_DATASET_ROOT": "", "DETECTRON2_DATASETS": ""}):
            self.assertEqual(self.module.default_dataset_root(), ROOT / "dataset")

    def test_repeated_registration_preserves_loader_and_rejects_new_paths(self):
        name = "openvocab_ovusis_train_a"
        loader = self.datasets[name]
        self.module.register_all_ovusis_instances(self.root)
        self.assertIs(self.datasets[name], loader)
        with self.assertRaises(ValueError):
            self.module.register_all_ovusis_instances("/tmp/another-ovusis-dataset")
        self.assertIs(self.datasets[name], loader)

    def check_loading(self, integer_ids):
        old_loaders = {name: self.datasets[name] for name in NAMES}
        old_metadata = {name: self.metadata.get(name).as_dict() for name in NAMES}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root, integer_ids)
            try:
                for name in NAMES:
                    self.datasets.remove(name)
                    self.metadata.remove(name)
                self.module.register_all_ovusis_instances(root)
                for name, split in NAMES.items():
                    records = self.datasets.get(name)
                    self.assertEqual(records[0]["file_name"], str(root / split / "fixture.jpg"))
                    self.assertEqual(records[0]["annotations"][0]["category_id"], 0)
                    raw_id = 2 if integer_ids else "002"
                    self.assertEqual(self.metadata.get(name).thing_dataset_id_to_contiguous_id[raw_id], 0)
                    self.assertEqual(self.metadata.get(name).thing_classes[0], "Shared")
            finally:
                for name in NAMES:
                    if name in self.datasets.list():
                        self.datasets.remove(name)
                    if name in self.metadata.list():
                        self.metadata.remove(name)
                    self.datasets.register(name, old_loaders[name])
                    metadata = dict(old_metadata[name])
                    metadata.pop("name", None)
                    self.metadata.get(name).set(**metadata)

    def test_string_ids_survive_real_coco_loading(self):
        self.check_loading(integer_ids=False)

    def test_integer_ids_survive_real_coco_loading(self):
        self.check_loading(integer_ids=True)


if __name__ == "__main__":
    unittest.main()
