"""Summarize per-category AP tables using the train/test COCO category names.

This aggregates existing evaluation results; it does not evaluate predictions.
Only the Python standard library is required.
"""

import argparse
import json
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ("AP", "AP50", "AP75")


def load_category_names(path):
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    categories = document.get("categories") if isinstance(document, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{path}: expected a nonempty COCO categories list")
    names = set()
    for category in categories:
        name = category.get("name") if isinstance(category, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: category name must be a nonempty string")
        name = name.strip()
        if name in names:
            raise ValueError(f"{path}: Duplicate category name: {name}")
        names.add(name)
    return names


def parse_result_table(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if parts == ["category", *METRICS]:
                continue
            if all(part and set(part) <= set("-: ") for part in parts):
                continue
            location = f"{path}: line {number}"
            if len(parts) != 4 or not parts[0]:
                raise ValueError(f"{location}: expected category, AP, AP50, AP75")
            name = parts[0]
            if name in rows:
                raise ValueError(f"{location}: Duplicate result category: {name}")
            try:
                values = tuple(float(part) for part in parts[1:])
            except ValueError as error:
                raise ValueError(f"{location}: invalid AP value") from error
            if any(not math.isnan(value) and (
                not math.isfinite(value) or not 0 <= value <= 100
            ) for value in values):
                raise ValueError(f"{location}: AP values must be in [0, 100] or nan")
            rows[name] = values
    if not rows:
        raise ValueError(f"{path}: No metric rows found")
    return rows


def summarize_metrics(rows, train_names, test_names):
    missing = test_names - rows.keys()
    unknown = rows.keys() - test_names
    if missing or unknown:
        details = []
        if missing:
            details.append("Missing categories: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("Unknown categories: " + ", ".join(sorted(unknown)))
        raise ValueError("; ".join(details))
    groups = {
        "intersection": train_names & test_names,
        "novel": test_names - train_names,
        "all": test_names,
    }
    summary = {}
    for group, names in groups.items():
        entry = {"count": len(names), "valid_counts": {}}
        for index, metric in enumerate(METRICS):
            values = [rows[name][index] for name in sorted(names)
                      if not math.isnan(rows[name][index])]
            entry[metric] = math.fsum(values) / len(values) if values else None
            entry["valid_counts"][metric] = len(values)
        summary[group] = entry
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("a", "b"), default="a",
                        help="A: train -> val; B: val -> train (default: a)")
    parser.add_argument("--results", type=Path,
                        help="per-category table (default: results_<split>.txt)")
    parser.add_argument("--train-json", type=Path, help="training COCO annotations")
    parser.add_argument("--test-json", type=Path, help="evaluation COCO annotations")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    args = parser.parse_args(argv)
    if (args.train_json is None) != (args.test_json is None):
        parser.error("--train-json and --test-json must be supplied together")
    dataset_root = Path(os.environ.get("OVUSIS_DATASET_ROOT") or
                        os.environ.get("DETECTRON2_DATASETS") or
                        PROJECT_ROOT / "dataset").expanduser().resolve()
    train_json = dataset_root / "annotations/instances_train.json"
    test_json = dataset_root / "annotations/instances_val.json"
    if args.split == "b":
        train_json, test_json = test_json, train_json
    train_json = args.train_json or train_json
    test_json = args.test_json or test_json
    results = args.results or PROJECT_ROOT / f"results_{args.split}.txt"
    try:
        summary = summarize_metrics(parse_result_table(results),
                                    load_category_names(train_json),
                                    load_category_names(test_json))
        if args.output:
            args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Results: {results}\nTrain annotations: {train_json}\nTest annotations: {test_json}")
    print(f"\n{'Group':<15} {'Classes':>7} {'mAP':>10} {'mAP50':>10} {'mAP75':>10}")
    for group, entry in summary.items():
        values = [f"{entry[metric]:.4f}" if entry[metric] is not None else "N/A"
                  for metric in METRICS]
        print(f"{group:<15} {entry['count']:>7} " + " ".join(f"{v:>10}" for v in values))
        if any(entry["valid_counts"][metric] != entry["count"] for metric in METRICS):
            print("  Valid classes: " + ", ".join(
                f"{metric}={entry['valid_counts'][metric]}" for metric in METRICS))


if __name__ == "__main__":
    main()
