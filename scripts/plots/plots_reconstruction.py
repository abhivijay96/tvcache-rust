"""Show how reconstruction reduces gross cache-hit savings."""
from common import Run, color, label, main, num, ordered, percentage


def render(args):
    """Plot gross savings, reconstruction penalty, and net savings at the smallest nonzero capacity."""
    run = Run(args)
    fig, panels = run.grid("Where gross hit savings go")
    exported = []
    for ax, workload in panels:
        rows = [r for r in run.rows if r["workload"] == workload and num(r, "budget_mib") > 0]
        if not rows:
            raise ValueError(f"{workload}: needs a nonzero budget")
        budget = min(num(r, "budget_mib") for r in rows)
        rows = [r for r in rows if num(r, "budget_mib") == budget]
        policies = ordered(rows)
        rows = sorted(rows, key=lambda r: policies.index(r["policy"]))
        for i, r in enumerate(rows):
            baseline = num(r, "baseline_tool_time_ms")
            if baseline <= 0:
                raise ValueError("baseline tool time must be positive")
            gross = num(r, "gross_hit_time_ms") / baseline
            net = num(r, "net_time_saved_ms") / baseline
            ax.bar(i, gross, color=color(r["policy"]), alpha=.24)
            ax.plot([i, i], [net, gross], color=color(r["policy"]), lw=3)
            ax.scatter(i, net, color=color(r["policy"]), edgecolor="white", s=70, zorder=3)
            exported.append({**r, "gross_fraction": gross, "net_fraction": net, "penalty_fraction": gross-net})
        ax.set_xticks(range(len(rows)), [label(r["policy"]) for r in rows], rotation=25, ha="right")
        ax.set_title(f"{workload.replace('_', ' ')} · {budget:g} MiB")
        ax.set_ylabel("Fraction of uncached tool time")
        ax.axhline(0, lw=.8, color="#888888")
        percentage(ax)
    return [run.save(fig, "reconstruction/gross_vs_net", exported,
        f"{run.mode}. Pale bar = gross hit savings; dot = net savings; connecting line = reconstruction + modeled overhead penalty.\n"
        "Each workload uses its smallest nonzero tested capacity; synthetic costs, not training wall-clock savings.")]


if __name__ == "__main__":
    main(render)
