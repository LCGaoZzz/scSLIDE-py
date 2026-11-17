"""Tests for design tooling, NOT tests of scSLIDE or R parity."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from compare_labelled import compare


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, left, right):
        a, b = self.root / "a.csv", self.root / "b.csv"
        a.write_text(left, encoding="utf-8")
        b.write_text(right, encoding="utf-8")
        return a, b

    def test_id_reordering(self):
        a, b = self.pair("id,A,B\nx,1,2\ny,3,4\n", "id,B,A\ny,4,3\nx,2,1\n")
        result = compare(a, b)
        self.assertTrue(result["matched_after_id_alignment"])
        self.assertFalse(result["raw_row_order_equal"])
        self.assertFalse(result["raw_column_order_equal"])

    def test_wrong_value(self):
        a, b = self.pair("id,A\nx,1\n", "id,A\nx,2\n")
        self.assertEqual(compare(a, b)["mismatch_count"], 1)

    def test_integer_precision(self):
        a, b = self.pair("id,A\nx,9007199254740992\n", "id,A\nx,9007199254740993\n")
        self.assertFalse(compare(a, b)["matched_after_id_alignment"])

    def test_duplicate_columns(self):
        a, b = self.pair("id,A,A\nx,1,2\n", "id,A,B\nx,1,2\n")
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_duplicate_rows(self):
        a, b = self.pair("id,A\nx,1\nx,2\n", "id,A\nx,1\n")
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_missing_axis(self):
        a, b = self.pair("id,A,B\nx,1,2\n", "id,A\nx,1\n")
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_ragged(self):
        a, b = self.pair("id,A,B\nx,1\n", "id,A\nx,1\n")
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_nonfinite(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            a, b = self.pair("id,A\nx," + value + "\n", "id,A\nx,1\n")
            with self.assertRaises(ValueError):
                compare(a, b)

    def test_numeric_requires_rationale(self):
        a, b = self.pair("id,A\nx,1\n", "id,A\nx,1\n")
        with self.assertRaises(ValueError):
            compare(a, b, mode="numeric", atol=0.0, rtol=0.0)

    def test_numeric_tolerance(self):
        a, b = self.pair("id,A\nx,1\n", "id,A\nx,1.1\n")
        result = compare(a, b, mode="numeric", atol=0.2, rtol=0.0,
                         tolerance_note="tool unit test, not scientific tolerance")
        self.assertTrue(result["matched_after_id_alignment"])
        with self.assertRaises(ValueError):
            compare(a, b, mode="numeric", atol=-1.0, rtol=0.0, tolerance_note="test")

    def test_no_overwrite(self):
        a, b = self.pair("id,A\nx,1\n", "id,A\nx,1\n")
        out = self.root / "report.json"
        out.write_text("original", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).with_name("compare_labelled.py")),
                   "--reference", str(a), "--candidate", str(b), "--out", str(out)]
        proc = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(out.read_text(), "original")

    def test_design_fixture_hand_count(self):
        path = Path(__file__).resolve().parents[1] / "fixtures/lmc_contract.json"
        fixture = json.loads(path.read_text())
        expected = fixture["known_answer_counts_landmark_by_sample"]
        result = [[0] * len(fixture["sample_order"]) for _ in fixture["landmark_ids"]]
        for cell, sample, neighbors in zip(fixture["query_ids"], fixture["sample_for_query"], fixture["neighbor_index"]):
            if cell in fixture["landmark_ids"]:
                continue
            s = fixture["sample_order"].index(sample)
            for neighbor in neighbors[:fixture["k"]]:
                result[neighbor - 1][s] += 1
        self.assertEqual(result, expected)
        self.assertEqual(sum(map(sum, result)), 10)
        csv_path = path.with_name("lmc_known_answer.csv")
        with csv_path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual([[int(v) for v in row[1:]] for row in rows[1:]], expected)


if __name__ == "__main__":
    unittest.main()
