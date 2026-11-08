"""Dump eviction candidates from Rust replay for every split/trace/budget/policy.

python -m tvcache_learn.collect SPLIT.json OUT_DIR [--policies lru lfu big=mlp:big.json] [--per-eviction 32]

Writes OUT_DIR/<split>/<trace>.<policy>.<mib>mib.jsonl. The behavior policy only
decides which states the cache visits; labels come from the trace's future.
Several behavior policies give the model a wider range of cache states; a
trained model as behavior policy shows it the states it creates itself
(DAgger-style bootstrapping). A policy is a replay mode or NAME=MODE:FILE.
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import budget_rule, ensure_replay, load_split, run_replay
from .evaluate import parse_policy


def jobs(splits, budgets, policies, out):
    names = Counter(d.name for dirs in splits.values() for d in dirs)
    if dup := [n for n, c in names.items() if c > 1]:
        raise ValueError(f"trace directory names must be unique across splits: {dup}")
    for split, dirs in splits.items():
        for trace in dirs:
            for mib in budgets(trace):
                for policy in policies:
                    yield split, trace, mib, policy, out / split / f"{trace.name}.{policy[0]}.{mib}mib.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--policies", nargs="+", type=parse_policy, default=[parse_policy("lru")])
    parser.add_argument("--affected", action="store_true",
                        help="also dump subtree node IDs (needed for loss:H / minloss:H labels)")
    parser.add_argument("--fractions", nargs="+", type=float,
                        help="budgets as fractions of unique entry bytes (default: manifest)")
    parser.add_argument("--budgets-mib", nargs="+", type=int, help="absolute budgets; override fractions")
    parser.add_argument("--per-eviction", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true", help="overwrite existing dumps")
    args = parser.parse_args(argv)
    splits, budgets = load_split(args.split)
    if args.fractions or args.budgets_mib:
        budgets = budget_rule(args.fractions, args.budgets_mib)
    todo = [j for j in jobs(splits, budgets, args.policies, args.out)
            if args.force or not j[-1].exists()]
    ensure_replay()

    def run(job):
        split, trace, mib, (name, mode, model), path = job
        if mode == "original":
            raise ValueError("the original enforcer builds no candidates to dump")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        report, _ = run_replay(trace, mode, mib, model=model, extra=[
            f"--dump-candidates={partial}", f"--dump-per-eviction={args.per_eviction}",
            f"--dump-seed={args.seed}", *(["--dump-affected"] if args.affected else [])])
        partial.rename(path)
        return f"{split:<5} {trace.name:<24} {name:<14} {mib:>6} MiB  hit {report['hit_rate']:.3f}  evictions {report['evicted_entries']}"

    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(run, todo):
            print(line, flush=True)
    print(f"{len(todo)} dumps written under {args.out}")


if __name__ == "__main__":
    main()
