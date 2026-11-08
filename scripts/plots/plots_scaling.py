"""Plot measured local burst-load results, separate from simulation metrics."""
import matplotlib.pyplot as plt
from common import Run, legend, main, num, read_csv


def render(args):
    """Plot lookup latency and achieved burst throughput versus concurrent clients."""
    run = Run(args, summary=False)
    run.mode = "measured local burst load"
    rows = read_csv(run.source / "load-summary.csv", ["tasks", "finds", "mean_ms", "p95_ms", "finds_per_second"])
    seen = set()
    for r in rows:
        key = num(r, "tasks"), num(r, "finds")
        if key in seen:
            raise ValueError(f"duplicate load cell: {key}")
        seen.add(key)
        for field in ("tasks", "finds", "mean_ms", "p95_ms", "finds_per_second"):
            if num(r, field) <= 0:
                raise ValueError(f"load {field} must be positive")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Local cache lookup scaling", fontsize=17, y=.98)
    for i, tasks in enumerate(sorted({num(r, "tasks") for r in rows})):
        points = sorted([r for r in rows if num(r, "tasks") == tasks], key=lambda r: num(r, "finds"))
        x = [num(r, "finds") for r in points]
        c = plt.get_cmap("tab10")(i % 10)
        axes[0].plot(x, [num(r, "p95_ms") for r in points], "o-", color=c, label=f"{tasks:g} tasks · p95")
        axes[0].plot(x, [num(r, "mean_ms") for r in points], "--", color=c, alpha=.65)
        axes[1].plot(x, [num(r, "finds_per_second") for r in points], "o-", color=c, label=f"{tasks:g} tasks")
    for ax in axes:
        ax.set_xlabel("Concurrent lookup clients")
        ax.set_xscale("log", base=2)
    axes[0].set_ylabel("Lookup completion time (ms)")
    axes[0].set_title("Solid = p95; dashed = mean")
    axes[1].set_ylabel("Achieved burst throughput (lookups/s)")
    axes[1].set_title("Closed-loop burst workload")
    legend(fig, axes[0])
    return [run.save(fig, "scaling/local_load", rows,
        "Measured local cache lookups; eviction disabled. One run per cell; graph sizes and setup workers differ by task count.\n"
        "Burst throughput is not controlled offered RPS; these curves cannot reproduce the paper's open-loop scaling experiment.")]


if __name__ == "__main__":
    main(render)
