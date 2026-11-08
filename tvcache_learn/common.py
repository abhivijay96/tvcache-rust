"""Shared helpers: split manifests, budgets, running the Rust replay, and scoring."""

import functools
import json
import math
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPLAY = REPO / "target" / "release" / "replay"
MIB = 1024 ** 2


def load_split(path):
    """Split manifest: {"train": [trace dirs], "val": [...], "test": [...]}.

    Relative trace paths resolve against the manifest's directory. Default
    budgets come from "budgets_mib" (absolute) or "budget_fractions" (of each
    trace's unique_cache_entry_bytes); returns (splits, budget rule).
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    splits = {}
    for name in ("train", "val", "test"):
        dirs = [Path(d) if Path(d).is_absolute() else (path.parent / d).resolve()
                for d in manifest.get(name, [])]
        for d in dirs:
            if not (d / "events.jsonl").exists():
                raise FileNotFoundError(f"{name} trace has no events.jsonl: {d}")
        splits[name] = dirs
    if not splits["train"]:
        raise ValueError("split manifest needs a nonempty train list")
    return splits, budget_rule(manifest.get("budget_fractions"), manifest.get("budgets_mib"))


def budget_rule(fractions=None, mib=None):
    """trace_dir -> sorted integer MiB budgets (replay takes whole MiB).

    Absolute `mib` wins over `fractions`; with neither, 10/25/50% of the trace.
    """
    if mib:
        return lambda trace_dir: sorted(set(mib))
    fractions = fractions or [0.1, 0.25, 0.5]

    def rule(trace_dir):
        summary = json.loads((Path(trace_dir) / "summary.json").read_text())
        total = summary["unique_cache_entry_bytes"]
        return sorted({max(1, math.ceil(f * total / MIB)) for f in fractions})
    return rule


def ensure_replay():
    if not REPLAY.exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "replay"], cwd=REPO, check=True)
    return REPLAY


def run_replay(trace_dir, mode, mib, model=None, extra=(), decisions=False):
    """Run one replay; returns (report, decisions list or None)."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(ensure_replay()), str(trace_dir), mode, str(mib)]
        if model is not None:
            cmd.append(str(model))
        if decisions:
            cmd.append(f"--decisions={tmp}")
        cmd.extend(extra)
        done = subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
        reports = json.loads(done.stdout)
        if len(reports) != 1:
            raise ValueError(f"expected one report from {mode}, got {len(reports)}")
        log = None
        if decisions:
            files = list(Path(tmp).glob("*.decisions.jsonl"))
            with files[0].open() as stream:
                log = [json.loads(line) for line in stream]
        return reports[0], log


@functools.lru_cache(maxsize=8)
def load_trace(trace_dir):
    from tvcache_eval.trace import load_trace as load
    return load(trace_dir)


def score(trace_dir, mib, log):
    """Net metrics from tvcache_eval for a replay decision log."""
    from tvcache_eval.evaluator import evaluate
    return evaluate(load_trace(str(trace_dir)), memory_budget_bytes=mib * MIB,
                    decisions=log)["metrics"]
