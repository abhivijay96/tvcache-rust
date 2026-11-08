# Benchmark plots

Run from any working directory. Inputs and outputs are organized per benchmark
run; no benchmark or simulator is rerun by these scripts.

```sh
# From the repository root (use an existing environment with matplotlib):
python3 scripts/plots/plots_all.py

# Explicit directories; each input gets its own output directory:
python3 scripts/plots/plots_all.py \
  --inputs results/compatibility-2026-11-07 results/another-run \
  --output results/plots

# One plot family:
python3 scripts/plots/plots_capacity.py --input results/compatibility-2026-11-07

# Python reference curves stay separate from Rust; separate mode directories:
python3 scripts/plots/plots_capacity.py --input results/compatibility-2026-11-07 \
  --engine python-reference --snapshots on
```

An isolated environment was created for this run. To set up or reproduce it:

```sh
uv venv scripts/plots/.venv
uv pip install --python scripts/plots/.venv/bin/python -r scripts/plots/requirements.lock
scripts/plots/.venv/bin/python scripts/plots/plots_all.py
```

`requirements.lock` records the exact versions used; `requirements.txt` is the
small direct-dependency declaration if you prefer your existing environment.
`plots_all.py --output` names the gallery root; individual scripts' `--output`
names one run's output directory. By default, both produce the layout below.

```text
results/
  compatibility-2026-11-07/     # inputs stay together
    summary.csv
    validation.json           # optional: full-cache repeat ceilings
    run.json                  # optional: input provenance
    load-summary.csv          # optional: measured server load cells
  plots/
    index.html                # all generated run galleries
    compatibility-2026-11-07/
      index.html              # image gallery, data/PDF/provenance links
      README.md               # same figures for Markdown viewers
      rust-snapshots-off/
        capacity/
        tradeoffs/
        reconstruction/
        snapshots/
        scaling/
        paper/
```

Every figure has `.png`, vector `.pdf`, the plotted input rows in `.csv`, and
caption/source SHA-256 hashes in `.json`. Open `results/plots/index.html` in a
browser, or browse the per-run Markdown gallery. Filenames are stable for easy
regeneration; different runs and selected engine/snapshot modes do not overwrite
one another. The gallery reflects the most recent batch invocation; older
figure files remain on disk but are not silently added to it.

| Script | Figures |
| --- | --- |
| `plots_capacity.py` | Hit rate and modeled time/USD savings against MiB; original included |
| `plots_tradeoffs.py` | Matched-budget time improvement over original; callback CPU cost |
| `plots_reconstruction.py` | Gross versus net savings at smallest nonzero budget |
| `plots_snapshots.py` | Python-reference snapshot savings and external storage |
| `plots_scaling.py` | Measured mean/p95 lookup latency and burst throughput |
| `plots_paper.py` | Versioned paper hit-rate / median-speedup references beside local estimates |
| `plots_all.py` | All applicable families, run galleries, and top-level index |

## Input contract and checks

`summary.csv` uses the schema produced by `scripts/benchmark_compatibility.py`:
engine, snapshot mode, workload, policy, budget, evaluator metrics, and Rust
timing columns. Older `results/lru-lfu/summary.csv` has a different schema;
regenerate with the compatibility runner rather than inventing its missing net
metrics. New datasets and policy names are discovered from rows. No four-panel
or six-policy requirement is hardcoded into the layout.

Missing files/columns, nonfinite metric values, duplicate selected experiment
keys, and absent matched original baselines produce clear errors. Undefined
normalized savings for a zero baseline are omitted, not converted to zero; an
entirely undefined figure is skipped with a message.
Optional load data and Python reference data are skipped by the batch runner
when absent; individual scripts fail clearly if their own inputs are missing.
Use `--skip FAMILY ...` to request a subset. Unknown costs must be encoded and
handled upstream; these scripts never fill blank costs or create measurements.

The original enforcer's callback CPU time is not instrumented, so it is omitted
from that figure. Other overhead values exclude candidate construction. Curves
are single observations without uncertainty bars; intermediate segments connect
tested capacities rather than claim extra measurements. Snapshot figures always
use Python reference policies regardless of the primary engine selection; their
captions identify this explicitly. No Python curve is merged into a Rust policy.

## Comparing with the paper

References are in [`references/tvcache-v1`](references/tvcache-v1/README.md),
including exact source locations. Paper Figure 5 motivates hit-rate plots, Table
2 motivates speedup context, and Figure 8 motivates server scaling measurements.
Our current inputs support capacity curves and aggregate modeled work savings;
they do not supply matched training epochs, real TCG workloads, measured median
tool execution speedup, training rewards, GPU makespan, or process RSS. Those
require further measurements. The original repository enforcer is labeled
"Original enforcer" and must not be read as the paper's published result.

Paper panels keep real published measurements and synthetic replay estimates on
separate axes. In particular, local `speedup` divides total uncached tool time
by total modeled request time; the paper's Table 2 divides median tool-call
execution times. Local burst throughput also differs from controlled offered
RPS. These figures provide context, not a claim of reproducing or beating the
paper. Adding another paper/version means adding its own reference directory
with provenance and a plot using compatible metric definitions.
