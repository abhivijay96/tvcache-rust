"""Capacity curves for one engine/mode, including the no-cache control."""
import json
import matplotlib.pyplot as plt

from common import Run, color, label, legend, main, num, ordered, percentage


def render(args):
    """Plot hit rate, modeled time savings, and modeled USD savings versus capacity."""
    run = Run(args)
    validation_path = run.source / "validation.json"
    validation = {r["workload"]: r for r in json.loads(validation_path.read_text())} if validation_path.exists() else {}
    figures = []
    metrics = [("hit_rate", "Value hit rate", "Hit rate", lambda r: num(r, "hit_rate")),
               ("time_saved", "Modeled tool-time savings", "Fraction of baseline time saved",
                lambda r: num(r, "net_time_saved_ms") / num(r, "baseline_tool_time_ms")),
               ("cost_saved", "Modeled cost savings", "Fraction of baseline USD saved",
                lambda r: num(r, "net_cost_saved_usd") / num(r, "baseline_tool_cost_usd"))]
    for name, title, ylabel, metric in metrics:
        fig, panels = run.grid(f"{title} vs cache capacity")
        exported = []
        for ax, workload in panels:
            rows = [r for r in run.rows if r["workload"] == workload]
            for policy in ordered(rows):
                points = sorted([r for r in rows if r["policy"] == policy], key=lambda r: num(r, "budget_mib"))
                # Undefined normalized metrics remain missing; never convert unknown/free costs into savings.
                denominator = "baseline_tool_cost_usd" if name == "cost_saved" else "baseline_tool_time_ms"
                points = [r for r in points if name == "hit_rate" or num(r, denominator) > 0]
                if not points:
                    continue
                ax.plot([num(r, "budget_mib") for r in points], [metric(r) for r in points],
                        marker="o", ms=4, lw=2, color=color(policy), label=label(policy))
                exported.extend({**r, "plotted_value": metric(r)} for r in points)
            if name == "hit_rate" and workload in validation:
                ax.axhline(validation[workload]["repeat_fraction"], ls=":", color="#777777", lw=1.4,
                           label="Full-cache repeat ceiling")
            ax.set_xlabel("Cache capacity (MiB)")
            ax.set_ylabel(ylabel)
            percentage(ax)
            if name == "hit_rate":
                ax.set_ylim(0, 1.01)
            else:
                ax.axhline(0, color="#999999", lw=.7)
        legend(fig, panels[0][0])
        if not exported:
            plt.close(fig)
            print(f"skip {name}: no defined normalized values")
            continue
        figures.append(run.save(fig, f"capacity/{name}", exported,
            f"{run.mode}. Synthetic estimates; net includes reconstruction. Cache overheads/prices follow the input cost model.\n"
            "Separate workload capacity axes; points are single runs, without confidence intervals."))
    return figures


if __name__ == "__main__":
    main(render)
