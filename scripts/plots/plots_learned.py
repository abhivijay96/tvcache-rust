"""Policy comparison plots for a learned-policy validation run (val-summary.csv)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, num, read_csv

KIND_COLORS = {"baseline": "#2a78d6", "learned": "#eb6834"}
FAMILIES = ["balanced", "expensive_reuse", "memory_pressure", "popularity_shift"]
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.grid": True, "axes.grid.axis": "x",
    "grid.alpha": .25, "axes.axisbelow": True, "ytick.left": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})


def summary(path):
    rows = read_csv(path, ["policy", "kind", "family", "budget_mib", "hit_rate_mean",
                           "tool_time_saved_ms_mean", "tool_cost_saved_usd_mean",
                           "policy_ns_per_request_mean"])
    return {(r["family"], r["budget_mib"], r["policy"]): r for r in rows}


def dots(ax, policies, kinds, values, title, xlabel, log=False):
    y = range(len(policies))
    ax.hlines(y, 0 if not log else min(v for v in values if v > 0) / 2, values,
              color="#dddddd", lw=1, zorder=1)
    ax.scatter(values, y, s=36, c=[KIND_COLORS[kinds[p]] for p in policies],
               edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_yticks(list(y), policies, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    if log:
        ax.set_xscale("log")
    else:
        lo, hi = min(values), max(values)
        pad = (hi - lo) * .08
        ax.set_xlim(lo - pad, hi + pad)


def legend(fig):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="white", label=k)
               for k, c in KIND_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(.99, .995))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "results/learned-1g-val")
    p.add_argument("--output", type=Path, help="default results/plots/INPUT_NAME")
    args = p.parse_args()
    out = args.output or ROOT / "results/plots" / args.input.name
    out.mkdir(parents=True, exist_ok=True)
    data = summary(args.input / "val-summary.csv")

    overall = {pol: r for (fam, bud, pol), r in data.items() if fam == "ALL" and bud == "ALL"}
    kinds = {pol: r["kind"] for pol, r in overall.items()}
    order = sorted(overall, key=lambda pol: -num(overall[pol], "tool_time_saved_ms_mean"))
    runs = int(next(iter(overall.values()))["runs"])
    note = "Synthetic 1 GiB traces, validation split. Mean over {} runs per policy ({} traces x 3 budgets: 512/1024/2048 MiB)."

    # Group 1: four headline metrics, all families and budgets.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    metric = lambda field, scale=1: [num(overall[pol], field) * scale for pol in order]
    dots(axes[0, 0], order, kinds, metric("tool_time_saved_ms_mean", 1 / 3.6e6),
         "Net tool time saved", "hours per trace (higher is better)")
    dots(axes[0, 1], order, kinds, metric("tool_cost_saved_usd_mean"),
         "Net tool cost saved", "USD per trace (higher is better)")
    dots(axes[1, 0], order, kinds, metric("hit_rate_mean", 100),
         "Hit rate", "% of tool calls served from cache (higher is better)")
    overhead = metric("policy_ns_per_request_mean", 1e-3)
    ax = axes[1, 1]
    dots(ax, order, kinds, [v if v > 0 else float("nan") for v in overhead],
         "Policy overhead", "µs per request, log scale (lower is better)", log=True)
    for i, v in enumerate(overhead):
        if v <= 0:
            ax.annotate("not instrumented", (0, i), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", color=MUTED, fontsize=8)
    fig.suptitle("Learned vs baseline eviction policies: overall", x=.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.text(.01, -.01, note.format(runs, runs // 3) + " Policies sorted by net time saved.", color=MUTED, fontsize=8)
    legend(fig)
    fig.savefig(out / "learned_overall.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "learned_overall.pdf", bbox_inches="tight")
    plt.close(fig)

    # Group 2: hit rate per workload family, averaged over budgets.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, fam in zip(axes.flat, FAMILIES):
        dots(ax, order, kinds, [num(data[fam, "ALL", pol], "hit_rate_mean") * 100 for pol in order],
             fam, "hit rate, % (higher is better)")
    fig.suptitle("Hit rate by workload", x=.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(.01, -.01, note.format(runs // 4, runs // 12) +
             " Same policy order as the overall figure; x-axes differ per panel.", color=MUTED, fontsize=8)
    legend(fig)
    fig.savefig(out / "learned_hit_rate_by_workload.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "learned_hit_rate_by_workload.pdf", bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
