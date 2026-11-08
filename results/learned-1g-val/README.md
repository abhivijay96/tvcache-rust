# Learned vs baseline eviction policies on the 1 GiB synthetic traces (validation)

Validation results, ready for plotting. **These are validation numbers, not
final results:** the val set was also used to pick each model's best epoch, so
learned models are slightly favored. The test seeds (14-18) are reserved and
untouched; quote final numbers only from a single run on them.

## Setup

- **Traces:** the 1 GiB synthetic traces from `scripts/gen_1g_splits.py`
  (branch `gen-1g-splits`, commit `4f80e12`). Each family uses its original
  settings with 4x tasks and rollouts and 128x tool output sizes (memory_pressure
  sizes unchanged): about 40k accesses and 2-5k unique entries, 3.2-4.9 GiB in
  total. Families: balanced, expensive_reuse, memory_pressure, popularity_shift.
  - Train: seeds 1-5 per family, 20 traces (used to train the learned models).
  - Val: seeds 11-13 per family, 12 traces (**every row in `val-*.csv`**).
- **Budgets:** 512, 1024 and 2048 MiB, about 10-60% of each trace's total.
- **Simulator:** Rust replay (`src/replay.rs`), result-only (no snapshots), with
  server-style subtree eviction. Each run's decision log is rescored by
  `tvcache_eval`, so time and cost saved are **net** (they include re-running a
  missing prefix on a miss).

## Files

| File | Contents |
| --- | --- |
| `val-runs.csv` | One row per (val trace, budget, policy): 612 runs |
| `val-summary.csv` | Mean and std across traces, per policy x family x budget, plus `ALL` aggregates |
| `baseline-budget-sweep-train.csv` | Earlier sanity sweep: baselines only, train seeds 1-3, budgets 256-5120 MiB (good for full budget curves) |
| `models/*.json` | Trained models (tvcache-mlp-v1); `*.linear-weights.json` for the linear models |

### Columns (`val-runs.csv`)

| Column | Meaning |
| --- | --- |
| `family`, `seed`, `trace` | Workload family, generator seed, trace name |
| `kind` | `baseline` or `learned` |
| `budget_mib` | Cache memory budget |
| `policy` | Policy name (see below) |
| `hit_rate` | Fraction of tool calls served from cache |
| `tool_time_saved_ms` | Net tool time saved vs no cache |
| `saved_fraction_of_achievable` | `tool_time_saved_ms` / trace's maximum possible savings (every repeat a free hit). **The main metric; comparable across traces.** |
| `tool_cost_saved_usd` | Net synthetic dollar cost saved |
| `gross_tool_ms_saved` | Rust replay's gross sum over hits (overstates net; for reference) |
| `peak_memory_bytes` | Peak cache occupancy |
| `evicted_entries` | Entries evicted, counting subtree members |
| `policy_ns_per_request` | Time inside the policy per request (decision overhead) |
| `model_features`, `model_hidden`, `model_target`, `model_output` | Learned models only: input feature set, hidden widths, training target, output transform |

`val-summary.csv` has `<metric>_mean` and `<metric>_std` (std across traces)
for the main metrics, with `family=ALL` and/or `budget_mib=ALL` aggregates.

## Policies

Baselines: `fifo`, `lru`, `lfu` (in-cache counts), `lfu-persistent` (counts survive
eviction), `linear` (untuned all-ones linear TCG policy), `original` (the server's
TTL and watermark enforcer; it breaks ties nondeterministically).

Learned: `<size>-<target>`.

| Size | Inputs | Hidden layers | Parameters |
| --- | --- | --- | --- |
| `lin` | 9 basic features | none | 10 |
| `small` | 9 basic features | 16 | 177 |
| `big` | 40 rich features | 128-128-64 | 30,081 |

| Target | Learns to |
| --- | --- |
| `belady` | Pick, among an eviction's candidates, the one used farthest in the future |
| `logdist` | Predict log of the number of calls until next use |
| `reuse` | Predict "used again within 1,000 calls" |
| `benefit` | `reuse`, then score = P(reuse) x runtime / bytes (cost-aware variant) |

Training data: at every eviction of LRU and LFU replays on the train traces,
8 sampled candidates with their features, labelled from the trace's future
(7.3M rows). Details: `docs/learned-policies.md`.

## Headline (all val runs, mean)

| Policy | Hit rate | Saved / achievable |
| --- | --- | --- |
| lfu-persistent | 0.823 | **0.768** |
| big-belady | **0.828** | 0.765 |
| small-belady | **0.828** | 0.760 |
| lfu | 0.810 | 0.753 |
| lin-belady | 0.825 | 0.750 |
| lru | 0.778 | 0.720 |
| original | 0.771 | 0.710 |
| fifo | 0.732 | 0.669 |
| linear (untuned) | 0.788 | 0.647 |

- Learned models roughly tie the best baseline (lfu-persistent) and beat
  LRU, LFU, FIFO and the original policy. They are best on popularity_shift
  (big models 0.834-0.836 vs lfu-persistent 0.808).
- Bigger is better: big > small > linear for every target.
- Learned models have the highest hit rates but slightly less time saved: their
  labels ignore tool cost.
- Training the linear policy raised it from 0.647 (all-ones) to 0.742-0.750.
- The `benefit` variants underperform (about 0.60), likely because they count only
  the root's runtime against the whole subtree's bytes. Not yet confirmed.
- Policy overhead: big models about 0.8 ms per request, small about 9 µs, linear
  about 3 µs, lfu-persistent about 2 µs.

## Plotting suggestions

- Bar chart: `saved_fraction_of_achievable_mean` per policy, `family=ALL`,
  `budget_mib=ALL`, with std error bars.
- Per-family grouped bars at 1024 MiB, which shows the popularity_shift win.
- Budget curves: `baseline-budget-sweep-train.csv` (256-5120 MiB) for baselines;
  learned models only have 512/1024/2048.
- Hit rate vs saved fraction scatter, which shows that hit rate alone misleads.
- Label all of it "synthetic traces, validation split".

## Reproduce

```sh
P=.venv/bin/python   # torch + numpy
$P -m tvcache_learn.collect SPLIT.json runs/1g/dumps --policies lru lfu --per-eviction 8
$P -m tvcache_learn.dataset runs/1g/dumps runs/1g/data
$P -m tvcache_learn.train runs/1g/data --model big --target belady --epochs 20 --patience 4 --out big-belady.json
$P -m tvcache_learn.evaluate SPLIT.json --split val --out val.csv --policies lru lfu big=mlp:big-belady.json
```

`SPLIT.json` lists the train and val trace directories with `"budgets_mib": [512, 1024, 2048]`.
