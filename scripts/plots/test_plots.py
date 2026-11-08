"""Small input-contract and end-to-end checks for plotting only."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from common import ORIGINAL, Run, num, read_csv

SCRIPTS = Path(__file__).resolve().parent


def fixture(directory, workloads=("example",)):
    directory.mkdir(parents=True)
    rows = []
    for workload in workloads:
        for policy, elapsed in ((ORIGINAL, 200), ("linear", 100), ("future-policy", 150)):
            rows.append(dict(engine="rust", snapshots="False", workload=workload,
                policy=policy, budget_mib=1, hit_rate=.75, baseline_tool_time_ms=1000,
                baseline_tool_cost_usd=0, net_time_saved_ms=1000-elapsed,
                modeled_request_ms=elapsed, gross_hit_time_ms=900, net_cost_saved_usd=0,
                rust_policy_ns_per_request=1000))
    write_rows(directory, rows)
    return rows


def write_rows(directory, rows):
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, source, engine="rust", snapshots="off"):
        return argparse.Namespace(input=source, output=self.root / "plots", engine=engine, snapshots=snapshots)

    def test_duplicates_are_rejected(self):
        source = self.root / "input"
        rows = fixture(source)
        write_rows(source, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Run(self.args(source))

    def test_bad_columns_and_nonfinite_values(self):
        path = self.root / "bad.csv"
        path.write_text("other\n1\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            read_csv(path, ["hit_rate"])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            num({"value": "nan"}, "value")

    def test_engines_and_snapshot_modes_stay_separate(self):
        source = self.root / "input"
        rows = fixture(source)
        for snapshots in ("True", "False"):
            rows.append({**rows[0], "engine": "python-reference", "snapshots": snapshots})
        write_rows(source, rows)
        rust = Run(self.args(source))
        reference = Run(self.args(source, "python-reference", "on"))
        self.assertEqual(len(rust.rows), 3)
        self.assertEqual(len(reference.rows), 1)
        self.assertNotEqual(rust.output, reference.output)

    def test_batch_directories_arithmetic_and_zero_costs(self):
        one, three = self.root / "run-one", self.root / "run-three"
        fixture(one)
        fixture(three, ("one", "two", "three"))
        result = subprocess.run([sys.executable, str(SCRIPTS / "plots_all.py"),
            "--inputs", str(one), str(three), "--output", str(self.root / "plots"), "--skip", "paper"],
            cwd=self.root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skip cost_saved: no defined normalized values", result.stdout)
        for source in (one, three):
            gallery = self.root / "plots" / source.name
            self.assertTrue((gallery / "index.html").exists())
            self.assertEqual(len(list(gallery.rglob("*.png"))), 5)
            csv_path = gallery / "rust-snapshots-off/tradeoffs/vs_original.csv"
            rows = read_csv(csv_path, ["time_reduction_vs_original"])
            for r in rows:
                expected = {ORIGINAL: 0, "linear": .5, "future-policy": .25}[r["policy"]]
                self.assertAlmostEqual(num(r, "time_reduction_vs_original"), expected)
            metadata = json.loads(csv_path.with_suffix(".json").read_text())
            self.assertIn(str(source / "summary.csv"), metadata["sources_sha256"])
        self.assertTrue((self.root / "plots/index.html").exists())


if __name__ == "__main__":
    unittest.main()
