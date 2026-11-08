# Learned policies: neural scorers and fitted linear weights

All learned policies score each legal eviction action and evict the lowest
score, like `LinearPolicy`. Training happens in Python (`tvcache_learn`); the
Rust `MlpPolicy` (`src/policy/mlp.rs`) runs exported weights with a hand-written
forward pass, so the server needs no ML runtime.

## Pipeline

Use the venv with torch and numpy (`python3.12 -m venv .venv && .venv/bin/pip install torch numpy`).
A split manifest lists trace directories for `train`, `val` and `test`, plus
default `budget_fractions` of each trace's unique entry bytes. See
`configs/learn/smoke-split.json`, which is for smoke tests only.

```sh
P=.venv/bin/python
# 1. Dump eviction candidates (features as the policy sees them) from Rust replay.
$P -m tvcache_learn.collect configs/learn/smoke-split.json runs/x/dumps --policies lru lfu
# 2. Add future-reuse labels; writes runs/x/data/{train,val,test}.npz. Dumps can then be deleted.
$P -m tvcache_learn.dataset runs/x/dumps runs/x/data
# 3. Train and export (tvcache-mlp-v1 JSON).
$P -m tvcache_learn.train runs/x/data --model small --target reuse:256 --out runs/x/models/small.json
$P -m tvcache_learn.train runs/x/data --model big --target belady --out runs/x/models/big.json
# 4. Fit LinearPolicy weights directly by replay search (train), selected on val.
$P -m tvcache_learn.fit_linear configs/learn/smoke-split.json --out runs/x/models/linear-fitted.json
# 5. Compare on test with tvcache_eval net metrics.
$P -m tvcache_learn.evaluate configs/learn/smoke-split.json --split test --out runs/x/test.csv \
    --policies lru lfu linear fitted=linear:runs/x/models/linear-fitted.json \
               small=mlp:runs/x/models/small.json big=mlp:runs/x/models/big.json
```

Run any single model in replay with `cargo run --release --bin replay -- DATASET mlp MIB MODEL.json`.

## Features

`src/policy/features.rs` defines both feature sets, and the dump and the
policies call the same functions, so training and serving inputs are identical.
`basic` is `LinearPolicy`'s nine bounded features. `rich` (40) appends log-scaled
history, costs, sizes, subtree size, missing-value flags, cache pressure and
hashed tool buckets. Defaults are basic for linear and small models, rich for big.
Exported models record feature names; Rust refuses a model whose names differ.

## Dumps and labels

`replay --dump-candidates=FILE` writes a header line and then one row per
sampled candidate per eviction: `--dump-per-eviction` (default 32) are sampled,
always including the chosen victim. The behavior policy only decides which cache
states are visited; using several (`--policies lru lfu`) widens coverage.

The label is the distance, in events, to the candidate root's next access. A
rollout reaches a node through its whole stateful prefix, so this is also the
first future use of anything in the removed subtree. Roots never used again are
censored past the end of the trace. Datasets store raw distances, so targets are
chosen at train time:

| Target | Loss | Score meaning |
| --- | --- | --- |
| `reuse:H` | BCE | logit of "reused within H events" |
| `logdist` | MSE | `-log1p(distance)` |
| `belady` | per-eviction softmax | imitate Belady's farthest-next-use victim |

`--output prob_benefit_per_byte` turns a `reuse:H` logit into expected runtime
saved per reclaimed byte. Validation reports victim accuracy and victim quality
(`log1p(d_victim) / log1p(d_max)`, where 1.0 is Belady among the sampled
candidates) for every target, plus the behavior policy as a reference.

A linear model on basic features also writes `*.linear-weights.json`, which the
existing `linear` replay mode loads. It decides identically to the same model
loaded as an MLP.

## Linear weight search

`fit_linear` follows [the offline plan](offline-learned-policy-plan.md):
L1-normalized vectors, seeded random search starting from the all-ones default,
coordinate refinement with shrinking steps, and selection of the top train
vectors on val. The objective is the mean over runs of savings as a fraction of
achievable savings (`repeat_tool_time_ms`). `net` scores decision logs with
`tvcache_eval`; `gross` and `hit` use the replay report and are much faster.

## Caveats

- Replay is result-only (no snapshots), so these policies do not yet choose snapshots.
- Policy time grows with model size times resident candidates. A 128-128-64 network
  measured about 0.5 ms per request on `memory_pressure`. Candidate sampling inside
  the policy is the planned fix if this matters.
- Replay budgets are whole MiB, so very small traces get coarse budgets.
