#!/usr/bin/env python3
"""Run every implemented replay policy and independently score its decisions.

Run from any directory after `cargo build --release --bins`. Outputs are kept
separate from the historical results. Real TCG traces are not included.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tvcache_eval.evaluator import evaluate
from tvcache_eval.policies import ReferencePolicy
from tvcache_eval.trace import load_trace, read_jsonl

SWEEP = {
    "balanced": [0, 1, 3, 5, 8, 10],
    "expensive_reuse": [0, 1, 2, 4, 6, 7],
    "popularity_shift": [0, 1, 3, 5, 8, 10],
    "memory_pressure": [0, 133, 331, 661, 991, 1321],
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_rust_bytes(trace, decisions, report):
    """Also check Rust's full-entry accounting (Python subtracts unused handles)."""
    resident = {}
    used = peak = 0
    for event, d in zip(trace.events, decisions, strict=True):
        assert (d["event_index"], d["node_id"]) == (event["event_index"], event["node_id"])
        assert d["hit"] == (d["node_id"] in resident)
        assert not d["store_snapshot"]
        for victim in d["evict"]:
            used -= resident.pop(victim)
        if d["admit"]:
            assert not d["hit"]
            size = trace.nodes[d["node_id"]]["cache_entry_bytes"]
            resident[d["node_id"]] = size
            used += size
        assert used <= report["capacity_bytes"]
        peak = max(peak, used)
    assert (used, peak) == (report["final_bytes"], report["peak_bytes_after_enforcement"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/compatibility-2026-11-07")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    validation = []
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dump(output / "run.json", {"git_revision": revision, "budgets_mib": SWEEP,
         "inputs": "data/synthetic only; real TCG traces unavailable",
         "costs": "synthetic fixture costs; default zero cache overheads; uncalibrated",
         "external_snapshot_budget_bytes": None})
    for dataset, budgets in SWEEP.items():
        trace = load_trace(ROOT / "data/synthetic" / dataset)
        validation.append({"workload": dataset, **trace.validation})
        for mib in budgets:
            run_dir = output / dataset / f"{mib}mib"
            run_dir.mkdir(parents=True, exist_ok=True)
            commands = [(["all"], run_dir / "rust"),
                        (["linear", "configs/lfu-persistent.json"], run_dir / "rust-persistent")]
            for mode, decisions_dir in commands:
                command = [str(ROOT / "target/release/replay"), f"data/synthetic/{dataset}",
                           mode[0], str(mib), *mode[1:], f"--decisions={decisions_dir}"]
                reports = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
                dump(decisions_dir / "replay.json", reports)
                for report in reports:
                    label = "lfu-persistent" if len(mode) > 1 else report["policy"]
                    decisions = list(read_jsonl(decisions_dir / f"{report['policy']}.decisions.jsonl"))
                    check_rust_bytes(trace, decisions, report)
                    scored = evaluate(trace, memory_budget_bytes=mib * 1024**2, decisions=decisions)
                    metrics = scored["metrics"]
                    assert metrics["hits"] == report["hits"]
                    assert math.isclose(metrics["gross_hit_time_ms"], report["gross_tool_ms_saved"], rel_tol=1e-9, abs_tol=1e-7)
                    assert math.isclose(metrics["gross_hit_cost_usd"], report["gross_tool_cost_usd_saved"], rel_tol=1e-9, abs_tol=1e-7)
                    if mib == 0:
                        assert metrics["hits"] == metrics["peak_memory_bytes"] == 0
                    if mib == budgets[-1]:
                        assert metrics["hits"] == trace.validation["repeat_accesses"]
                    dump(decisions_dir / f"{label}.scored.json", scored)
                    rows.append({"engine": "rust", "workload": dataset, "budget_mib": mib,
                                 "policy": label, "snapshots": False,
                                 "rust_peak_bytes": report["peak_bytes_after_enforcement"],
                                 "rust_final_bytes": report["final_bytes"],
                                 "rust_policy_ns_per_request": report["policy_ns"] / report["requests"],
                                 "rust_bypassed_missing_parent": report["bypassed_missing_parent"],
                                 "rust_peak_bytes_before_enforcement": report["peak_bytes_before_enforcement"],
                                 **metrics})
            for snapshots in (False, True):
                for name in ("none", "lru", "lfu"):
                    scored = evaluate(trace, memory_budget_bytes=mib * 1024**2,
                                      policy=ReferencePolicy(name, snapshots=snapshots))
                    metrics = scored["metrics"]
                    if name == "none" or mib == 0:
                        assert metrics["hits"] == metrics["peak_memory_bytes"] == 0
                    if name == "none":
                        assert metrics["net_time_saved_ms"] == metrics["net_cost_saved_usd"] == 0
                    if name != "none" and mib == budgets[-1]:
                        assert metrics["hits"] == trace.validation["repeat_accesses"]
                    dump(run_dir / f"python-{name}-snapshots-{snapshots}.json", scored)
                    rows.append({"engine": "python-reference", "workload": dataset,
                                 "budget_mib": mib, "policy": name, "snapshots": snapshots, **metrics})
            # Save after every budget so partial output remains inspectable.
            with (output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{dataset} {mib} MiB: Rust policies scored and reference runs passed ({len(rows)} total)", flush=True)
    dump(output / "validation.json", validation)
    dump(output / "completion.json", {"complete": True, "runs": len(rows),
         "rust_runs": sum(r["engine"] == "rust" for r in rows),
         "python_reference_runs": sum(r["engine"] == "python-reference" for r in rows),
         "checks": ["trace hashes and causal features", "decision residency and budgets",
                    "Rust full-entry peak/final bytes", "hit and gross savings agreement",
                    "zero-budget and no-cache controls", "full-capacity repeat ceiling"]})
    print(f"Completed {len(rows)} runs. Reports: {output}")


if __name__ == "__main__":
    main()
