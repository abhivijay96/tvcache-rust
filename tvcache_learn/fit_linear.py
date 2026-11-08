"""Fit LinearPolicy weights by replay search (docs/offline-learned-policy-plan.md).

python -m tvcache_learn.fit_linear SPLIT.json --out configs/linear-fitted.json

Each weight vector is replayed on every train trace and budget; the objective is
the mean over runs of savings as a fraction of the trace's achievable savings
(repeat_tool_time_ms), so traces count equally. Seeded random search, then
coordinate refinement with shrinking steps, on train. The top candidates are
re-scored on val, and the best on val is written. Vectors are L1-normalized:
positive rescaling does not change the ranking.

Objectives: net (tvcache_eval tool time saved, including prefix replay on
misses), gross (replay's hit-time sum), or hit (hit rate).
"""

import argparse
import csv
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import budget_rule, ensure_replay, load_split, run_replay, score
from .train import LINEAR_WEIGHT_NAMES


def normalize(w):
    w = np.asarray(w, dtype=float)
    total = np.abs(w).sum()
    return w / total if total > 0 else w


def run_one(job):
    trace, mib, weights, objective = job
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(zip(LINEAR_WEIGHT_NAMES, weights)), f)
    try:
        report, log = run_replay(trace, "linear", mib, model=f.name, decisions=objective == "net")
    finally:
        os.unlink(f.name)
    if objective == "hit":
        return report["hit_rate"]
    ceiling = json.loads((Path(trace) / "summary.json").read_text())["repeat_tool_time_ms"]
    saved = score(trace, mib, log)["tool_time_saved_ms"] if objective == "net" else report["gross_tool_ms_saved"]
    return saved / ceiling if ceiling else 0.0


class Objective:
    def __init__(self, runs, objective, pool):
        self.runs, self.objective, self.pool, self.cache = runs, objective, pool, {}

    def __call__(self, vectors):
        """Mean objective for each vector; evaluates all runs of all vectors in parallel."""
        fresh = [v for v in {tuple(np.round(v, 12)) for v in vectors} if v not in self.cache]
        jobs = [(str(t), m, list(v), self.objective) for v in fresh for t, m in self.runs]
        values = list(self.pool.map(run_one, jobs))
        for i, v in enumerate(fresh):
            per_run = values[i * len(self.runs):(i + 1) * len(self.runs)]
            self.cache[v] = float(np.mean(per_run))
        return [self.cache[tuple(np.round(v, 12))] for v in vectors]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="CSV of every evaluated vector (default: next to --out)")
    parser.add_argument("--objective", choices=["net", "gross", "hit"], default="net")
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument("--budgets-mib", nargs="+", type=int, help="absolute budgets; override fractions")
    parser.add_argument("--samples", type=int, default=64, help="random vectors")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--refine", type=int, default=3, help="best random vectors to refine")
    parser.add_argument("--top", type=int, default=5, help="train-best vectors re-scored on val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    splits, budgets = load_split(args.split)
    if args.fractions or args.budgets_mib:
        budgets = budget_rule(args.fractions, args.budgets_mib)
    runs = {name: [(t, m) for t in dirs for m in budgets(t)]
            for name, dirs in splits.items()}
    if not runs["val"]:
        parser.error("split manifest needs val traces for selection")
    ensure_replay()
    rng = np.random.default_rng(args.seed)
    with ProcessPoolExecutor(args.workers) as pool:
        train = Objective(runs["train"], args.objective, pool)
        start = [normalize(np.ones(9))] + [normalize(rng.normal(size=9)) for _ in range(args.samples)]
        values = train(start)
        print(f"random search: {len(start)} vectors x {len(runs['train'])} runs; "
              f"default {values[0]:.4f}, best {max(values):.4f}", flush=True)
        seeds = [start[i] for i in np.argsort(values)[::-1][:args.refine]]
        for n, w in enumerate(seeds):
            current = train([w])[0]
            for step in args.steps:
                improved = True
                while improved:
                    improved = False
                    moves = [normalize(w + sign * step * np.eye(9)[i]) for i in range(9) for sign in (1, -1)]
                    scores = train(moves)
                    if max(scores) > current + 1e-9:
                        w, current, improved = moves[int(np.argmax(scores))], max(scores), True
            print(f"refined seed {n}: train {current:.4f}", flush=True)
        ranked = sorted(train.cache.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
        val = Objective(runs["val"], args.objective, pool)
        val_scores = val([np.array(v) for v, _ in ranked])
    best = int(np.argmax(val_scores))
    weights = dict(zip(LINEAR_WEIGHT_NAMES, ranked[best][0]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=2) + "\n")
    log = args.log or args.out.with_suffix(".search.csv")
    with log.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*LINEAR_WEIGHT_NAMES, f"train_{args.objective}", f"val_{args.objective}"])
        val_of = {v: s for (v, _), s in zip(ranked, val_scores)}
        for v, s in sorted(train.cache.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow([*v, s, val_of.get(v, "")])
    print(f"default all-ones: train {train.cache[tuple(np.round(normalize(np.ones(9)), 12))]:.4f}")
    print(f"selected: train {ranked[best][1]:.4f}, val {val_scores[best]:.4f}")
    print(f"wrote {args.out} and {log}")


if __name__ == "__main__":
    main()
