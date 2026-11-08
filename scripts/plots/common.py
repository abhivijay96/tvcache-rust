"""Shared input, styling, and output helpers; no simulation happens here."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/compatibility-2026-11-07"
REFERENCES = Path(__file__).parent / "references/tvcache-v1"
ORIGINAL = "original-ttl-and-watermark"
POLICIES = [ORIGINAL, "fifo", "linear", "lru", "lfu", "lfu-persistent", "none"]
LABELS = {ORIGINAL: "Original enforcer", "fifo": "FIFO", "linear": "Linear TCG",
          "lru": "LRU", "lfu": "LFU", "lfu-persistent": "Persistent LFU", "none": "No cache"}
COLORS = dict(zip(POLICIES, ["#303846", "#999999", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#666666"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="benchmark directory")
    p.add_argument("--output", type=Path, help="run output directory; default results/plots/INPUT_NAME")
    p.add_argument("--engine", choices=["rust", "python-reference"], default="rust")
    p.add_argument("--snapshots", choices=["off", "on"], default="off", help="reference-policy mode")
    return p


def read_csv(path, required):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def num(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('workload', 'row')} / {row.get('policy', '')}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def ordered(rows):
    names = {r["policy"] for r in rows}
    return [p for p in POLICIES if p in names] + sorted(names - set(POLICIES))


def label(policy):
    return LABELS.get(policy, policy)


def color(policy):
    return COLORS.get(policy, plt.get_cmap("tab10")(int(hashlib.sha256(policy.encode()).hexdigest()[:6], 16) % 10))


class Run:
    def __init__(self, args, *, summary=True):
        self.source = args.input.resolve()
        base_output = (args.output or ROOT / "results/plots" / self.source.name).resolve()
        if base_output == self.source or base_output in self.source.parents:
            raise ValueError("output must be a separate plot directory, not the input or its parent")
        self.output = base_output / f"{args.engine}-snapshots-{args.snapshots}"
        if args.engine == "rust" and args.snapshots == "on":
            raise ValueError("Rust replay does not admit snapshots; use --engine python-reference")
        self.engine = args.engine
        self.mode = f"{args.engine}, snapshots {args.snapshots}"
        self.rows = []
        if summary:
            rows = read_csv(self.source / "summary.csv", ["engine", "snapshots", "workload", "policy", "budget_mib"])
            for row in rows:
                if row["snapshots"] not in {"True", "False"}:
                    raise ValueError("snapshots column must contain True or False")
                if row["engine"] == args.engine and (row["snapshots"] == "True") == (args.snapshots == "on"):
                    self.rows.append(row)
            if not self.rows:
                raise ValueError(f"no rows for {self.mode} in {self.source}")
            seen = set()
            for row in self.rows:
                key = row["workload"], row["policy"], num(row, "budget_mib")
                if key in seen:
                    raise ValueError(f"duplicate workload/policy/budget: {key}")
                if key[2] < 0:
                    raise ValueError("budget_mib must be nonnegative")
                seen.add(key)

    def grid(self, title):
        workloads = sorted({r["workload"] for r in self.rows})
        columns = min(2, len(workloads))
        fig, axes = plt.subplots(math.ceil(len(workloads)/columns), columns,
                                 figsize=(12, 4.0*math.ceil(len(workloads)/columns)), squeeze=False)
        fig.suptitle(title, fontsize=17, y=.98)
        for ax in axes.flat[len(workloads):]:
            ax.set_visible(False)
        for ax, name in zip(axes.flat, workloads):
            ax.set_title(name.replace("_", " "))
        return fig, list(zip(axes.flat, workloads))

    def save(self, fig, stem, rows, caption, *, bottom=.14, top=.84, wspace=.27):
        path = self.output / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.text(.02, .016, caption, fontsize=8, color="#4B5563", va="bottom")
        fig.subplots_adjust(top=top, bottom=bottom, left=.09, right=.97, hspace=.45, wspace=wspace)
        for extension in ("png", "pdf"):
            fig.savefig(path.with_suffix("." + extension), dpi=180, bbox_inches="tight")
        plt.close(fig)
        with path.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
        sources = [self.source / n for n in ("summary.csv", "validation.json", "run.json", "load-summary.csv")]
        sources += sorted(REFERENCES.glob("*.csv")) if stem.startswith("paper/") else []
        provenance = {"caption": caption, "input_directory": str(self.source), "mode": self.mode,
                      "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}}
        path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(path.with_suffix(".png"))
        return path.with_suffix(".png")


def legend(fig, ax):
    entries = {}
    for panel in [ax, *fig.axes]:
        handles, names = panel.get_legend_handles_labels()
        entries.update(zip(names, handles))
    names, handles = list(entries), list(entries.values())
    if handles:
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .94),
                   ncol=min(4, len(names)), frameon=False, fontsize=9)


def percentage(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def main(render):
    try:
        render(parser(render.__doc__).parse_args())
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")
