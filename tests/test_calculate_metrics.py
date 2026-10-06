"""Run with: python -m unittest discover -s tests -v."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.train = self.directory / "train.json"
        self.test = self.directory / "test.json"
        self.results = self.directory / "results.txt"
        self.output = self.directory / "summary.json"
        self.write_categories(self.train, ["Seen", "Train only"])
        self.write_categories(self.test, ["Seen", "Novel one", "Novel two"])
        self.results.write_text(
            "| category | AP | AP50 | AP75 |\n"
            "|:---|---:|---:|---:|\n"
            "| Seen | 60 | 80 | 70 |\n"
            "| Novel one | 20 | 40 | 30 |\n"
            "| Novel two | 10 | 30 | 20 |\n", encoding="utf-8"
        )

    def write_categories(self, path, names):
        path.write_text(json.dumps({"categories": [
            {"id": i, "name": name} for i, name in enumerate(names)
        ]}), encoding="utf-8")

    def run_metrics(self, *extra):
        return subprocess.run([
            sys.executable, str(ROOT / "calculate_metrics.py"),
            "--results", str(self.results), "--train-json", str(self.train),
            "--test-json", str(self.test), "--output", str(self.output), *extra,
        ], cwd=self.directory, capture_output=True, text=True)

    def summary(self):
        result = self.run_metrics()
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.output.read_text())

    def test_custom_dataset_groups_and_class_weighted_total(self):
        # Catch hardcoded class names and an unweighted mean of the two groups.
        summary = self.summary()
        self.assertEqual(summary["intersection"]["count"], 1)
        self.assertEqual(summary["intersection"]["AP"], 60)
        self.assertEqual(summary["novel"]["count"], 2)
        self.assertEqual(summary["novel"]["AP"], 15)
        self.assertEqual(summary["all"]["AP"], 30)
        self.assertEqual(summary["all"]["AP50"], 50)
        self.assertEqual(summary["all"]["AP75"], 40)

    def test_nan_is_excluded_per_metric_and_json_stays_valid(self):
        self.results.write_text(
            "| Seen | nan | 80 | 70 |\n"
            "| Novel one | 20 | nan | 30 |\n"
            "| Novel two | 10 | 30 | 20 |\n"
        )
        summary = self.summary()
        self.assertIsNone(summary["intersection"]["AP"])
        self.assertEqual(summary["all"]["AP"], 15)
        self.assertEqual(summary["all"]["AP50"], 55)
        self.assertEqual(summary["all"]["valid_counts"]["AP"], 2)

    def test_empty_novel_group_is_reported(self):
        self.write_categories(self.train, ["Seen", "Novel one", "Novel two"])
        summary = self.summary()
        self.assertEqual(summary["novel"]["count"], 0)
        self.assertIsNone(summary["novel"]["AP"])

    def test_missing_category_fails_instead_of_changing_denominator(self):
        self.results.write_text("| Seen | 60 | 80 | 70 |\n")
        result = self.run_metrics()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing", result.stderr)
        self.assertFalse(self.output.exists())

    def test_unknown_category_fails(self):
        with self.results.open("a") as stream:
            stream.write("| Unknown | 10 | 20 | 30 |\n")
        result = self.run_metrics()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown", result.stderr)

    def test_duplicate_result_fails(self):
        with self.results.open("a") as stream:
            stream.write("| Seen | 90 | 90 | 90 |\n")
        result = self.run_metrics()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate", result.stderr)

    def test_invalid_values_and_malformed_rows_fail(self):
        for row in ["| Seen | oops | 80 | 70 |", "| Seen | inf | 80 | 70 |",
                    "| Seen | 101 | 80 | 70 |", "| Seen | -1 | 80 | 70 |",
                    "| Seen | 60 | 80 |"]:
            with self.subTest(row=row):
                self.results.write_text(row + "\n")
                result = self.run_metrics()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("line 1", result.stderr)

    def test_empty_table_fails(self):
        self.results.write_text("| category | AP | AP50 | AP75 |\n")
        result = self.run_metrics()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No metric rows", result.stderr)

    def test_duplicate_annotation_names_fail(self):
        self.write_categories(self.test, ["Seen", "Seen"])
        result = self.run_metrics()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate", result.stderr)

    def test_partial_annotation_override_fails(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "calculate_metrics.py"),
            "--train-json", str(self.train),
        ], cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("together", result.stderr)

    def test_split_presets_and_environment_root_work_from_another_directory(self):
        annotations = self.directory / "annotations"
        annotations.mkdir()
        self.write_categories(annotations / "instances_train.json", ["Seen", "Train only"])
        self.write_categories(annotations / "instances_val.json", ["Seen", "Novel one", "Novel two"])
        env = dict(os.environ, OVUSIS_DATASET_ROOT=str(self.directory),
                   DETECTRON2_DATASETS="/tmp/unused-ovusis-fallback")
        for split in ["a", "b"]:
            with self.subTest(split=split):
                if split == "b":
                    self.results.write_text("| Seen | 60 | 80 | 70 |\n| Train only | 40 | 60 | 50 |\n")
                result = subprocess.run([
                    sys.executable, str(ROOT / "calculate_metrics.py"), "--split", split,
                    "--results", str(self.results), "--output", str(self.output),
                ], cwd=self.directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(self.output.read_text())
                self.assertEqual(summary["all"]["AP"], 30 if split == "a" else 50)
                self.assertEqual(summary["novel"]["count"], 2 if split == "a" else 1)


if __name__ == "__main__":
    unittest.main()
