"""Matched-budget comparisons against the repository's original enforcer."""
from common import ORIGINAL, Run, color, label, legend, main, num, ordered, percentage


def render(args):
    """Plot time improvement over the original enforcer and policy callback overhead."""
    run = Run(args)
    figures = []
    fig, panels = run.grid("Modeled execution time vs original enforcer")
    exported = []
    for ax, workload in panels:
        rows = [r for r in run.rows if r["workload"] == workload]
        baseline = {num(r, "budget_mib"): r for r in rows if r["policy"] == ORIGINAL}
        if not baseline:
            raise ValueError(f"{workload}: original-enforcer rows are required; reference engines cannot be substituted")
        for policy in ordered(rows):
            points = sorted([r for r in rows if r["policy"] == policy], key=lambda r: num(r, "budget_mib"))
            values = []
            for r in points:
                budget = num(r, "budget_mib")
                if budget not in baseline:
                    raise ValueError(f"{workload}: original baseline missing at {budget} MiB")
                original_ms = num(baseline[budget], "modeled_request_ms")
                if original_ms <= 0:
                    raise ValueError("original modeled time must be positive for relative comparison")
                value = 1 - num(r, "modeled_request_ms") / original_ms
                values.append(value)
                exported.append({**r, "time_reduction_vs_original": value})
            ax.plot([num(r, "budget_mib") for r in points], values, marker="o", ms=4,
                    color=color(policy), lw=2, label=label(policy))
        ax.axhline(0, color="#333333", lw=1)
        ax.set_xlabel("Cache capacity (MiB)")
        ax.set_ylabel("Time reduction vs original (higher is better)")
        percentage(ax)
    legend(fig, panels[0][0])
    figures.append(run.save(fig, "tradeoffs/vs_original", exported,
        "Matched workload, capacity, and engine. Positive values mean lower modeled total tool-work time.\n"
        "Original is this repository's enforcer, not a reproduction of the paper's end-to-end system; ties may vary between runs."))
    fig, panels = run.grid("Rust policy callback cost per request")
    overhead_rows = []
    for ax, workload in panels:
        rows = [r for r in run.rows if r["workload"] == workload and num(r, "budget_mib") > 0]
        for policy in ordered(rows):
            if policy == ORIGINAL:
                continue  # The original enforcer does not use the timed CachePolicy callbacks.
            points = sorted([r for r in rows if r["policy"] == policy], key=lambda r: num(r, "budget_mib"))
            ax.plot([num(r, "budget_mib") for r in points],
                    [num(r, "rust_policy_ns_per_request") / 1000 for r in points],
                    marker="o", ms=4, lw=2, color=color(policy), label=label(policy))
            overhead_rows.extend({**r, "plotted_microseconds_per_request": num(r, "rust_policy_ns_per_request") / 1000}
                                 for r in points)
        ax.set_xlabel("Cache capacity (MiB)")
        ax.set_ylabel("Policy callback time (µs/request)")
        ax.set_yscale("symlog", linthresh=1)
    legend(fig, panels[0][0])
    figures.append(run.save(fig, "tradeoffs/policy_overhead", overhead_rows,
        "Measured Rust callbacks only; candidate construction and engine work excluded. Original enforcer is not instrumented here.\n"
        "Single local runs; diagnostic timings, not isolated microbenchmark estimates."))
    return figures


if __name__ == "__main__":
    main(render)
