# Handoff: add a 1536 MiB budget to the 1 GiB val results, then plot vs budget

Branch: `learned-1g-budgets`. Run this on the unicorn Slurm cluster.

## Goal

`results/learned-1g-val/` already has every policy at 512, 1024 and 2048 MiB on
the 12 val traces. The user wants plots with **memory budget on the x-axis**
at 512 / 1024 / 1536 / 2048 MiB, so:

1. Run all 17 policies on the 12 val traces at **1536 MiB only** (204 runs).
2. Merge those rows into `val-runs.csv` and rebuild `val-summary.csv`.
3. Plot metrics vs budget (spec below).

Cost: about 30 s single-threaded per run on an M-series Mac (5 s for fifo, up to
about 55 s for lfu and the big models). That's about 2 CPU-hours in total, or
a few minutes on one 32-CPU node.

## What is where

| Thing | Location |
| --- | --- |
| Replay simulator | `cargo build --release --bin replay` gives `target/release/replay` |
| Evaluator | `tvcache_learn/evaluate.py` (runs replay, rescores with `tvcache_eval`); **stdlib only**, no torch needed |
| Trained models | `results/learned-1g-val/models/` (force-added; `results/*` is gitignored) |
| Existing results | `results/learned-1g-val/val-runs.csv`, `val-summary.csv`, `README.md` (schema) |
| Val traces | **Not in git.** 258 MB; the user copies them over (step 1) |
| Plot script | `scripts/plots/plots_learned.py` (current per-policy dot plots) |

## Cluster notes

- Slurm is only on PATH in a login shell: `bash -l`. The user's template is
  `~/cc.sbatch` (`-p aimi -M unicorn`); `default_partition` is the big one.
- No Rust installed. Install user-level:
  `curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env`.
- `python3.12` and `uv` are available. Matplotlib only for plotting:
  `uv venv .venv && uv pip install matplotlib`.
- `evaluate.py --workers` defaults to `os.cpu_count()`, which on a node counts
  **all** cores, not your allocation. Always pass `--workers $SLURM_CPUS_PER_TASK`.

## Steps

### 1. Traces (the user runs this from the Mac)

```sh
rsync -a ~/dev/school/sysml/sysml_final/.claude/worktrees/trace-1g/target/traces/1g/val/ \
  unicorn:~/scratch/tvcache-1g/traces/val/
```

### 2. Split manifest

`load_split` requires a nonempty, existing `train` list. For eval only, reuse
the val dirs as `train` (it is unused). Write `~/scratch/tvcache-1g/split.json`:

```json
{
  "train": ["traces/val/balanced_1g_s11"],
  "val": ["traces/val/balanced_1g_s11", "traces/val/balanced_1g_s12", "traces/val/balanced_1g_s13",
          "traces/val/expensive_reuse_1g_s11", "traces/val/expensive_reuse_1g_s12", "traces/val/expensive_reuse_1g_s13",
          "traces/val/memory_pressure_1g_s11", "traces/val/memory_pressure_1g_s12", "traces/val/memory_pressure_1g_s13",
          "traces/val/popularity_shift_1g_s11", "traces/val/popularity_shift_1g_s12", "traces/val/popularity_shift_1g_s13"],
  "budgets_mib": [1536]
}
```

(Relative paths resolve against the manifest's directory.)

### 3. Build and sanity check (before the full run)

The Linux build must reproduce the Mac numbers. Run one known row on the login
node or a small interactive job:

```sh
cargo build --release --bin replay
python3.12 -m tvcache_learn.evaluate ~/scratch/tvcache-1g/split.json --split val \
  --budgets-mib 512 --workers 4 --out /tmp/check.csv \
  --policies lfu-persistent=linear:configs/lfu-persistent.json
grep balanced_1g_s11 /tmp/check.csv
```

Expected hit rates at 512 MiB on `balanced_1g_s11` (verified on the Mac):
lfu-persistent 0.710126, lin-belady 0.697926, small-benefit 0.499482. If they
differ, stop and tell the user.

### 4. The full 1536 MiB run (sbatch)

Policy specs. These were verified to reproduce the stored val rows:

```sh
M=results/learned-1g-val/models
POLICIES="fifo lru lfu linear original
  lfu-persistent=linear:configs/lfu-persistent.json
  lin-belady=mlp:$M/linear-belady.json lin-logdist=mlp:$M/linear-logdist.json lin-reuse=mlp:$M/linear-reuse.json
  small-belady=mlp:$M/small-belady.json small-logdist=mlp:$M/small-logdist.json small-reuse=mlp:$M/small-reuse.json
  small-benefit=mlp:$M/small-reuse-benefit.json
  big-belady=mlp:$M/big-belady.json big-logdist=mlp:$M/big-logdist.json big-reuse=mlp:$M/big-reuse.json
  big-benefit=mlp:$M/big-reuse-benefit.json"
```

`lfu-persistent` is **not** a replay mode. It is the linear policy with
`configs/lfu-persistent.json`. `original` is a valid replay mode.

```sh
#!/bin/bash
#SBATCH -M unicorn
#SBATCH -p default_partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=tvc-1536
#SBATCH --output=logs/tvc-1536-%j.log
cd ~/dev/tvcache-rust   # wherever the clone is
# paste POLICIES from above here
python3.12 -m tvcache_learn.evaluate ~/scratch/tvcache-1g/split.json --split val \
  --budgets-mib 1536 --workers $SLURM_CPUS_PER_TASK \
  --out results/learned-1g-val/val-1536-raw.csv --policies $POLICIES
```

Expect 204 rows. Memory per worker is not measured; 64G for 32 workers is a
guess. Check `sacct` after the run and adjust if it is OOM-killed.

### 5. Merge into val-runs.csv and rebuild val-summary.csv

`evaluate.py` writes 11 columns. `val-runs.csv` adds these (see the README):

- `family`, `seed`: parse from the trace name `<family>_1g_s<seed>`; seed has no
  leading zero (`11`).
- `kind`: `learned` for the `lin-`, `small-` and `big-` policies, else `baseline`.
- `model_features`, `model_hidden`, `model_target`, `model_output`: copy per
  policy from the existing rows (fifo, lru, lfu, lfu-persistent, linear and
  original are blank; note lin-belady is also blank in the existing file).

Column order: `family,seed,kind,trace,budget_mib,policy,hit_rate,tool_time_saved_ms,saved_fraction_of_achievable,tool_cost_saved_usd,gross_tool_ms_saved,peak_memory_bytes,evicted_entries,policy_ns_per_request,model_features,model_hidden,model_target,model_output`.

The result is 612 + 204 = 816 rows. Then rebuild `val-summary.csv` with the
same columns: `policy,kind,family,budget_mib,runs`, then `_mean` and `_std` for
hit_rate, saved_fraction_of_achievable, tool_time_saved_ms, tool_cost_saved_usd
and policy_ns_per_request. Group by (policy, family, budget), plus `ALL`
aggregates over family, over budget, and over both. Check std: rebuild the
current summary from the current 612 rows first and diff it against the
committed file. They should match (sample vs population std: pick whichever
reproduces it).

Update the README's run counts and budget list.

### 6. Plots: budget on the x-axis

Extend `scripts/plots/plots_learned.py` (keep its style: blue `#2a78d6` is
baseline, orange `#eb6834` is learned, recessive grid, light theme, PNG + PDF,
footnote saying "synthetic traces, validation split").

- **Group 1 (2x2):** net tool time saved (hours per trace), net tool cost saved
  (USD per trace), hit rate (%), policy overhead (µs per request, log y). x = budget
  MiB at 512/1024/1536/2048, family=ALL.
- **Group 2 (2x2):** hit rate vs budget for balanced, expensive_reuse,
  memory_pressure, popularity_shift.
- 17 lines per panel is unreadable. Show a subset as lines with markers:
  lfu-persistent, lfu, lru, original, fifo, big-belady, small-belady,
  lin-belady. Within each kind, vary line style or marker, not hue, and label
  line ends directly. Mention in the footnote that the rest are in the CSV.
- `original` has `policy_ns_per_request = 0` (not instrumented). Drop it from
  the overhead panel and say so.
- Output to `results/plots/learned-1g-val/`.

### 7. Hand back

Commit `val-runs.csv`, `val-summary.csv`, `README.md` (force-add under
`results/`), the plot script, and the plots, on this branch. **Do not push.**
The user pushes themselves.
