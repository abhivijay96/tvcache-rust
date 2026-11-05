"""python3 -m tvcache_eval {validate,run,score}."""

import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from .costs import CostModel
from .evaluator import EvaluationError, evaluate
from .policies import ReferencePolicy
from .trace import TraceError, load_trace, read_jsonl


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def jsonl_sink(stack, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open("w"))
    return lambda row: stream.write(json.dumps(row, allow_nan=False) + "\n")


def bytes_from_mib(value):
    try:
        amount = float(value)
        if amount < 0:
            raise ValueError()
        return int(amount * 1024 ** 2)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("expected a finite, nonnegative MiB value")


def protect_inputs(inputs, outputs):
    protected = {Path(path).resolve() for path in inputs}
    for output in outputs:
        if Path(output).resolve() in protected:
            raise EvaluationError(f"output would overwrite an input: {output}")


def write_reports(directory, reports):
    directory = Path(directory)
    dump_json(directory / "results.json", reports)
    rows = [{"workload": r["workload"], "policy": r["policy"],
             "memory_budget_bytes": r["memory_budget_bytes"],
             "external_budget_bytes": r["external_budget_bytes"], **r["metrics"]} for r in reports]
    if rows:
        columns = sorted(set().union(*(r.keys() for r in rows)))
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print("workload          policy  budget MiB   hit rate   time saved (s)   net USD saved   peak MiB")
    for report in reports:
        m = report["metrics"]
        hit_rate = f"{100 * m['hit_rate']:.2f}%" if m["hit_rate"] is not None else "n/a"
        print(f"{report['workload']:<17} {report['policy']:<7} "
              f"{report['memory_budget_bytes'] / 1024**2:>10.2f} {hit_rate:>10} "
              f"{m['tool_time_saved_ms'] / 1000:>16.3f} {m['net_cost_saved_usd']:>15.6f} "
              f"{m['peak_memory_bytes'] / 1024**2:>10.3f}")
    print(f"Reports: {directory / 'results.json'} and {directory / 'summary.csv'}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="check trace integrity, graph, causal features, and summary")
    validate.add_argument("traces", nargs="+", type=Path)
    validate.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="sweep reference policies and memory budgets")
    run.add_argument("traces", nargs="+", type=Path)
    run.add_argument("--policies", nargs="+", choices=["none", "lru", "lfu"], default=["none", "lru", "lfu"])
    run.add_argument("--budgets-mib", nargs="+", type=bytes_from_mib, default=[1024**2, 8*1024**2, 64*1024**2])
    run.add_argument("--no-snapshots", action="store_true", help="cache values without snapshot handles")
    run.add_argument("--write-decisions", action="store_true", help="write decisions for independent rescoring")
    run.add_argument("--write-events", action="store_true", help="write per-access accounting JSONL")
    score = commands.add_parser("score", help="validate and score a Rust/simulator decision log")
    score.add_argument("trace", type=Path)
    score.add_argument("decisions", type=Path)
    score.add_argument("--budget-mib", type=bytes_from_mib, required=True)
    score.add_argument("--write-events", action="store_true")
    score.add_argument("--no-cache", action="store_true", help="rescore the cache-disabled baseline without lookup/policy overhead")
    for command in (run, score):
        command.add_argument("--cost-model", type=Path)
        command.add_argument("--external-budget-mib", type=bytes_from_mib, default=None,
                             help="optional separate cap on external snapshot storage")
        command.add_argument("--output", type=Path, default=Path("results/evaluation"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            if args.output:
                protect_inputs([p / name for p in args.traces for name in (
                    "nodes.jsonl", "events.jsonl", "summary.json", "manifest.json")], [args.output])
            results = [{"workload": path.name, **load_trace(path).validation} for path in args.traces]
            if args.output:
                dump_json(args.output, results)
            print(json.dumps(results, indent=2, allow_nan=False))
            return 0
        costs = CostModel.load(args.cost_model) if args.cost_model else CostModel()
        paths = [args.trace] if args.command == "score" else args.traces
        inputs = [path / name for path in paths for name in (
            "nodes.jsonl", "events.jsonl", "summary.json", "manifest.json")]
        if args.cost_model:
            inputs.append(args.cost_model)
        if args.command == "score":
            inputs.append(args.decisions)
        outputs = [args.output / "results.json", args.output / "summary.csv"]
        if args.command == "score" and args.write_events:
            outputs.append(args.output / "scored.events.jsonl")
        elif args.command == "run":
            for path in paths:
                for name in args.policies:
                    for budget in args.budgets_mib:
                        stem = f"{path.name}.{name}.{budget}bytes"
                        if args.write_decisions:
                            outputs.append(args.output / f"{stem}.decisions.jsonl")
                        if args.write_events:
                            outputs.append(args.output / f"{stem}.events.jsonl")
        protect_inputs(inputs, outputs)
        reports = []
        if args.command == "score":
            trace = load_trace(args.trace)
            with ExitStack() as stack:
                sink = jsonl_sink(stack, args.output / "scored.events.jsonl") if args.write_events else None
                reports.append(evaluate(trace, memory_budget_bytes=args.budget_mib,
                                        external_budget_bytes=args.external_budget_mib,
                                        decisions=read_jsonl(args.decisions), costs=costs, event_sink=sink,
                                        cache_enabled=not args.no_cache))
        else:
            for path in args.traces:
                trace = load_trace(path)
                for name in args.policies:
                    for budget in args.budgets_mib:
                        stem = f"{trace.name}.{name}.{budget}bytes"
                        with ExitStack() as stack:
                            decisions = jsonl_sink(stack, args.output / f"{stem}.decisions.jsonl") if args.write_decisions else None
                            events = jsonl_sink(stack, args.output / f"{stem}.events.jsonl") if args.write_events else None
                            reports.append(evaluate(trace, memory_budget_bytes=budget,
                                                    external_budget_bytes=args.external_budget_mib,
                                                    policy=ReferencePolicy(name, snapshots=not args.no_snapshots),
                                                    costs=costs, decision_sink=decisions, event_sink=events))
        write_reports(args.output, reports)
        return 0
    except (TraceError, EvaluationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
