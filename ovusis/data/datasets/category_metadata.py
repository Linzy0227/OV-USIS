"""Read instance category names, IDs, and visualization colors from COCO JSON."""

import hashlib
import json
from pathlib import Path


def load_category_metadata(json_file):
    path = Path(json_file)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    categories = document.get("categories") if isinstance(document, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{path}: expected a nonempty categories list")

    ids = set()
    names = set()
    id_type = None
    items = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"{path}: each category must be an object")
        category_id = category.get("id")
        name = category.get("name")
        if type(category_id) not in (int, str) or category_id == "":
            raise ValueError(f"{path}: category ID must be an integer or nonempty string")
        if id_type is not None and type(category_id) is not id_type:
            raise ValueError(f"{path}: all category IDs must have the same type")
        id_type = type(category_id)
        if category_id in ids:
            raise ValueError(f"{path}: Duplicate category ID: {category_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: category name must be a nonempty string")
        name = name.strip()
        if name in names:
            raise ValueError(f"{path}: Duplicate category name: {name}")
        color = category.get("color")
        if color is None:
            color = list(hashlib.sha256(str(category_id).encode("utf-8")).digest()[:3])
        if (not isinstance(color, list) or len(color) != 3 or
                any(type(value) is not int or not 0 <= value <= 255 for value in color)):
            raise ValueError(f"{path}: category color must contain three integers in [0, 255]")
        ids.add(category_id)
        names.add(name)
        items.append({"id": category_id, "name": name, "color": color})

    # Match load_coco_json exactly: sort raw IDs without changing their type.
    items.sort(key=lambda category: category["id"])
    return {
        "thing_classes": [category["name"] for category in items],
        "thing_colors": [category["color"] for category in items],
        "thing_dataset_id_to_contiguous_id": {
            category["id"]: index for index, category in enumerate(items)
        },
    }
