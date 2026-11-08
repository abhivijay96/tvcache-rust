#!/usr/bin/env python3
"""Generate plots and per-run galleries for one or more benchmark directories."""
import html
import json
from pathlib import Path
import sys

from common import ROOT, parser, read_csv
import plots_capacity
import plots_tradeoffs
import plots_reconstruction
import plots_snapshots
import plots_scaling
import plots_paper

PLOTS = {"capacity": plots_capacity, "tradeoffs": plots_tradeoffs,
         "reconstruction": plots_reconstruction, "snapshots": plots_snapshots,
         "scaling": plots_scaling, "paper": plots_paper}


def gallery(directory, figures, source, skipped):
    cards, markdown = [], [f"# Plots: {source.name}", "", f"Input: `{source}`", "",
        "Paper panels use unmatched real workloads. Local replay estimates are not measured training speedups.", ""]
    for path in figures:
        relative = path.relative_to(directory).as_posix()
        caption = json.loads(path.with_suffix(".json").read_text())["caption"]
        stem = relative[:-4]
        cards.append(f'<article><h2>{html.escape(stem)}</h2><a href="{relative}"><img src="{relative}" loading="lazy"></a>'
                     f'<p>{html.escape(caption)}</p><p><a href="{stem}.pdf">PDF</a> · '
                     f'<a href="{stem}.csv">Plot data</a> · <a href="{stem}.json">Provenance</a></p></article>')
        markdown.extend([f"## {stem}", "", f"![{stem}]({relative})", "", caption, "",
                         f"[PDF]({stem}.pdf) · [Data]({stem}.csv) · [Provenance]({stem}.json)", ""])
    if skipped:
        markdown.extend(["Skipped: " + "; ".join(skipped), ""])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text("\n".join(markdown))
    page = '<!doctype html><meta charset="utf-8"><title>TVCache plots</title><style>' \
        'body{font:16px system-ui;background:#f3f5f8;color:#253047;max-width:1400px;margin:32px auto;padding:0 24px}' \
        'article{background:white;border:1px solid #dde2ea;border-radius:12px;padding:24px;margin:24px 0}' \
        'img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.5;white-space:pre-line}a{color:#0072b2}</style>'
    page += f'<h1>TVCache · {html.escape(source.name)}</h1><p>Input: {html.escape(str(source))}</p>'
    page += '<p>Paper panels provide context across unmatched workloads. Replay estimates are not measured training speedups.</p>'
    page += "".join(cards)
    if skipped:
        page += '<p>Skipped: ' + html.escape("; ".join(skipped)) + '</p>'
    (directory / "index.html").write_text(page)


def main():
    p = parser(__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, help="multiple benchmark directories; replaces --input")
    p.add_argument("--skip", nargs="+", choices=list(PLOTS), default=[])
    args = p.parse_args()
    sources = [v.resolve() for v in (args.inputs or [args.input])]
    if len({s.name for s in sources}) != len(sources):
        p.error("input directory names must be unique to avoid output collisions")
    output_root = (args.output or ROOT / "results/plots").resolve()
    for source in sources:
        args.input, args.output = source, output_root / source.name
        figures, skipped = [], []
        rows = read_csv(source / "summary.csv", ["engine"])
        for name, module in PLOTS.items():
            reason = None
            if name in args.skip:
                reason = "requested"
            elif name == "scaling" and not (source / "load-summary.csv").exists():
                reason = "load-summary.csv absent"
            elif name == "snapshots" and not any(r["engine"] == "python-reference" for r in rows):
                reason = "Python reference data absent"
            elif name == "tradeoffs" and args.engine != "rust":
                reason = "requires original Rust enforcer"
            if reason:
                skipped.append(f"{name}: {reason}")
                print(f"skip {name}: {reason}")
                continue
            figures.extend(module.render(args))
        gallery(args.output, figures, source, skipped)
        print(f"Gallery: {args.output / 'index.html'}")
    output_root.mkdir(parents=True, exist_ok=True)
    links = [f'<li><a href="{html.escape(d.name)}/index.html">{html.escape(d.name)}</a></li>'
             for d in sorted(output_root.iterdir()) if d.is_dir() and (d / "index.html").is_file()]
    (output_root / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Benchmark plots</title>'
        '<h1>Benchmark plot galleries</h1><ul>' + ''.join(links) + '</ul>')
    runs = [d for d in sorted(output_root.iterdir()) if d.is_dir() and (d / "index.html").is_file()]
    (output_root / "README.md").write_text("# Benchmark plot galleries\n\n[Open browser index](index.html)\n\n" +
        "\n".join(f"- [{d.name}]({d.name}/README.md) · [Browser gallery]({d.name}/index.html)" for d in runs) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")
