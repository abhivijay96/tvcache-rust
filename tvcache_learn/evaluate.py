"""Score policies on one split with tvcache_eval (net metrics), for comparison.

python -m tvcache_learn.evaluate SPLIT.json --split test --out results/learned.csv \\
    --policies lru lfu linear fitted=linear:configs/linear-fitted.json small=mlp:models/small.json

A policy is a replay mode (fifo, lru, lfu, linear) or NAME=MODE:FILE, where FILE
is LinearPolicy weights (linear) or a tvcache-mlp-v1 model (mlp).
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import budget_rule, ensure_replay, load_split, run_replay, score

FIELDS = ["trace", "budget_mib", "policy", "hit_rate", "tool_time_saved_ms",
          "saved_fraction_of_achievable", "tool_cost_saved_usd", "gross_tool_ms_saved",
          "peak_memory_bytes", "evicted_entries", "policy_ns_per_request"]


def parse_policy(spec):
    name, _, rest = spec.partition("=")
    if not rest:
        return name, name, None
    mode, _, path = rest.partition(":")
    if mode not in ("linear", "mlp") or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=linear:FILE or NAME=mlp:FILE, got {spec}")
    return name, mode, path


def run_one(job):
    trace, mib, (name, mode, path) = job
    report, log = run_replay(trace, mode, mib, model=path, decisions=True)
    metrics = score(trace, mib, log)
    ceiling = json.loads((Path(trace) / "summary.json").read_text())["repeat_tool_time_ms"]
    return {"trace": Path(trace).name, "budget_mib": mib, "policy": name,
            "hit_rate": metrics["hit_rate"], "tool_time_saved_ms": metrics["tool_time_saved_ms"],
            "saved_fraction_of_achievable": metrics["tool_time_saved_ms"] / ceiling if ceiling else None,
            "tool_cost_saved_usd": metrics["tool_cost_saved_usd"],
            "gross_tool_ms_saved": report["gross_tool_ms_saved"],
            "peak_memory_bytes": metrics["peak_memory_bytes"],
            "evicted_entries": report["evicted_entries"],
            "policy_ns_per_request": report["policy_ns"] / report["requests"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path)
    parser.add_argument("--split-name", "--split", dest="which", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--policies", nargs="+", type=parse_policy,
                        default=[parse_policy(p) for p in ("fifo", "lru", "lfu", "linear")])
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument("--budgets-mib", nargs="+", type=int, help="absolute budgets; override fractions")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    splits, budgets = load_split(args.split)
    if args.fractions or args.budgets_mib:
        budgets = budget_rule(args.fractions, args.budgets_mib)
    traces = splits[args.which]
    if not traces:
        parser.error(f"split manifest has no {args.which} traces")
    ensure_replay()
    jobs = [(str(t), m, p) for t in traces for m in budgets(t)
            for p in args.policies]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    print(f"{'policy':<16}{'runs':>5}{'hit rate':>10}{'saved/achievable':>18}{'policy ns/req':>15}")
    for name, rs in by_policy.items():
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{name:<16}{len(rs):>5}{mean('hit_rate'):>10.3f}"
              f"{mean('saved_fraction_of_achievable'):>18.3f}{mean('policy_ns_per_request'):>15.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
