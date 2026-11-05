# Naive linear TCG retention policy

```sh
cargo run --bin replay -- data/synthetic/balanced linear 1
cargo run --bin replay -- data/synthetic/balanced all 1 configs/linear-default.json
```

`all` runs original, FIFO, and linear independently. `both` retains its existing
meaning of original plus FIFO. An optional fourth argument supplies linear
weights for `linear` or `all`; omitted weight fields default to 1. Set a term to
0 to disable it. Unknown names and nonfinite weights are rejected. Negative
weights are allowed to reverse a feature's preference. Reports include the
effective weights in `policy_parameters`.

For each legal eviction candidate, compute `score = sum(weight_i * feature_i)`.
Evict the LOWEST score, breaking equal scores by task/node ID. Default weights
are all 1: this is an untuned starting heuristic, not a learned reuse estimate.
Admission accepts candidates that individually fit both total budgets. There is
no score threshold; an incoming entry may be evicted immediately after insertion.

| Weight / feature | Transformation |
| --- | --- |
| frequency | access count / (1 + access count) |
| recency | 100 / (100 + events since direct access) |
| runtime | estimated runtime ms / (1000 + runtime ms) |
| dollar_cost | estimated USD / (0.001 + USD) |
| inverse_size | 1024 / (1024 + reclaimable resident bytes) |
| depth | node depth / (1 + node depth) |
| branching | observed immediate children / (1 + children) |
| subtree_frequency | subtree accesses / (1 + subtree accesses) |
| subtree_recency | 100 / (100 + events since subtree access) |

Inputs are bounded to [0, 1], so large byte/time units cannot dominate solely
because of their units. These fixed scales do not depend on future workload
statistics. Unknown runtime/cost/recency contributes zero. Larger values favor
retention with positive weights; the size feature favors smaller removal actions.

The policy uses root metadata and whole-action reclaimable bytes; its subtree
frequency/recency captures descendant traffic. It does not sum descendant dollar
or runtime benefits, estimate prefix replay costs, or model snapshot savings.
Depth and branching are naive proxies whose signs/weights need evaluation.

Replay computes history causally in `src/policy/history.rs`. The access callback
receives pre-access features; candidate features are refreshed after the current
request is observed, so a freshly accessed entry has age zero. Counts persist
across eviction and include misses/bypasses. Every request updates all its path
prefixes' subtree statistics. Observed branching includes read-only children;
task namespaces remain isolated. This history is evaluator metadata, outside
the simulated budget, and can grow with distinct observed paths.

The runner reconstructs these fields rather than copying the trace's frozen
pre-access features. It currently does not pass through the trace's `extra` map;
new custom features need an explicit, causal update rule. Only already-resolved
node cost estimates are exposed, using the fixture-estimate assumptions in
[replay.md](replay.md). Snapshot modeling remains excluded.

Implementation: `src/policy/linear.rs`. Public API: `LinearPolicy::default()` or
`LinearPolicy::new(LinearWeights { ... })`, usable as `Box<dyn CachePolicy>`.
No resident ranking state is held by this policy; the engine supplies current
features and enforces graph/legal-action constraints.
