"""Register both OV-USIS evaluation directions from one dataset root."""

import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances

from .category_metadata import load_category_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS = {
    "openvocab_ovusis_train_a": "train",
    "openvocab_ovusis_val_b": "train",
    "openvocab_ovusis_val_a": "val",
    "openvocab_ovusis_train_b": "val",
}


def default_dataset_root():
    return Path(os.environ.get("OVUSIS_DATASET_ROOT") or
                os.environ.get("DETECTRON2_DATASETS") or PROJECT_ROOT / "dataset").expanduser()


def register_all_ovusis_instances(root):
    root = Path(root).expanduser().resolve()
    pending = []
    for name, split in SPLITS.items():
        image_root = str(root / split)
        json_file = str(root / "annotations" / f"instances_{split}.json")
        if name in DatasetCatalog.list():
            metadata = MetadataCatalog.get(name)
            if (metadata.get("image_root") != image_root or
                    metadata.get("json_file") != json_file):
                raise ValueError(f"Dataset {name} is already registered with different paths")
            continue
        pending.append((name, split, json_file, image_root))
    if not pending:
        return
    metadata = {
        split: load_category_metadata(root / "annotations" / f"instances_{split}.json")
        for split in {split for _, split, _, _ in pending}
    }
    for name, split, json_file, image_root in pending:
        register_coco_instances(name, metadata[split], json_file, image_root)


# Keep imports and --help available before the dataset has been downloaded.
_root = default_dataset_root()
if all((_root / "annotations" / f"instances_{split}.json").is_file()
       for split in ("train", "val")):
    register_all_ovusis_instances(_root)
