"""Keep snapshot-enabled Python experiments separate from Rust policies."""
from common import Run, color, label, legend, main, num, percentage, read_csv


def render(args):
    """Plot modeled snapshot benefit and external storage at the smallest nonzero budget."""
    run = Run(args, summary=False)
    run.mode = "python-reference, paired snapshot-off/on"
    rows = [r for r in read_csv(run.source / "summary.csv", ["engine", "snapshots", "workload", "policy", "budget_mib"])
            if r["engine"] == "python-reference" and r["policy"] != "none" and num(r, "budget_mib") > 0]
    if not rows:
        raise ValueError("snapshot plot requires Python reference rows with snapshots both on and off")
    run.rows = rows
    fig, panels = run.grid("Snapshot benefit · Python references")
    storage_fig, storage_panels = run.grid("External snapshot storage · Python references")
    storage_axes = dict((workload, ax) for ax, workload in storage_panels)
    exported = []
    storage_rows = []
    for ax, workload in panels:
        available = [r for r in rows if r["workload"] == workload]
        budget = min(num(r, "budget_mib") for r in available)
        available = [r for r in available if num(r, "budget_mib") == budget]
        storage_ax = storage_axes[workload]
        for i, policy in enumerate(sorted({r["policy"] for r in available})):
            pair = [r for r in available if r["policy"] == policy]
            if len(pair) != 2 or {r["snapshots"] for r in pair} != {"False", "True"}:
                raise ValueError(f"{workload}/{policy}: expected one snapshot-on and one snapshot-off row")
            pair.sort(key=lambda r: r["snapshots"] == "True")
            if any(num(r, "baseline_tool_time_ms") <= 0 for r in pair):
                raise ValueError(f"{workload}: positive baseline time required for snapshot benefit")
            ax.plot([0, 1], [num(r, "net_time_saved_ms") / num(r, "baseline_tool_time_ms") for r in pair],
                    marker="o", color=color(policy), label=label(policy), lw=2)
            storage_ax.bar(i, num(pair[1], "peak_external_snapshot_bytes") / 1024**3,
                           color=color(policy), label=label(policy), width=.55)
            exported.extend({**r, "plotted_net_fraction": num(r, "net_time_saved_ms") / num(r, "baseline_tool_time_ms")}
                            for r in pair)
            storage_rows.append({**pair[1], "plotted_external_gib": num(pair[1], "peak_external_snapshot_bytes") / 1024**3})
        ax.set_title(f"{workload.replace('_', ' ')} · {budget:g} MiB")
        ax.set_xticks([0, 1], ["Values only", "Values + snapshots"])
        ax.set_xlim(-.15, 1.15)
        ax.set_ylabel("Modeled net time saved")
        percentage(ax)
        storage_ax.set_title(f"{workload.replace('_', ' ')} · {budget:g} MiB")
        storage_ax.set_xticks(range(len({r['policy'] for r in available})),
                             [label(p) for p in sorted({r['policy'] for r in available})])
        storage_ax.set_ylabel("Peak external snapshot storage (GiB)")
        storage_ax.set_ylim(bottom=0)
    legend(fig, panels[0][0])
    caption = ("Python independent-entry reference policies only; smallest nonzero tested memory capacity per workload.\n"
               "External cap and overheads follow input reports; current audit uses unlimited external storage and zero storage prices.")
    return [run.save(fig, "snapshots/net_benefit", exported, caption),
            run.save(storage_fig, "snapshots/external_storage", storage_rows, caption)]


if __name__ == "__main__":
    main(render)
