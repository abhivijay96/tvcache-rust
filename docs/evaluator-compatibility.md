# Evaluator compatibility audit (2026-11-07)

Weitao's evaluator (`9560bd4`, `tvcache_eval/`) accepts the current Rust
replay's four synthetic datasets and decision format. It can independently
score every implemented Rust replay policy: original TTL/watermark, FIFO,
linear, LRU, LFU, and persistent LFU via its linear configuration.

## Reproduce

From the repository root:

```sh
cargo test
cargo build --release --bins
python3 scripts/benchmark_compatibility.py --output results/compatibility-2026-11-07
# Optional live checks and 32/64/128-task load matrix (port 7878 must be free):
python3 scripts/benchmark_server.py --output results/server-checks
```

The runner uses the existing five workload-specific budgets plus zero, for
144 Rust runs and 144 Python reference runs. It validates trace hashes and
causal features; checks decision residency and both memory accounting models;
compares Rust hits and gross time/USD savings against independent scoring;
and checks zero-budget, no-cache, and full-capacity controls. Each Rust run
retains its decision log, original report, and scored report. `summary.csv`
combines the results, and `completion.json` is written only after all checks pass.

To score one existing Rust decision log directly:

```sh
python3 -m tvcache_eval score data/synthetic/balanced \
  results/compatibility-2026-11-07/balanced/1mib/rust/linear.decisions.jsonl \
  --budget-mib 1 --output results/score-example
```

## What is and is not comparable

| Aspect | Rust replay | Python evaluator/reference |
| --- | --- | --- |
| Input | Synthetic catalog and ordered events | Same; additionally validates hashes, summaries, causal features |
| Policy execution | Actual Rust policies and graph | `run` implements only independent-entry none/LRU/LFU; `score` accepts any valid decision log |
| Eviction | Removes subtrees; insertion needs a parent | Reference policies remove entries independently; logical prefix directory survives eviction |
| Memory | Full `cache_entry_bytes` | Subtracts snapshot handle bytes when no snapshot is stored |
| Capacity enforcement | Inserts, then evicts; transient overshoot possible | Checks final residency per event; cannot establish absence of transient overshoot |
| State/cost | Result-only gross hit savings | Lazy state per rollout; misses reconstruct prefixes and optionally restore snapshots |
| Snapshots | Replay never admits snapshots | References can admit snapshots, with create/restore costs and separate external storage |
| Serving | Server currently uses its original enforcer | Scoring Rust policies does not install them in the server |

Hit counts and gross savings must agree when rescoring the same decisions.
Net savings can be lower because the evaluator charges reconstruction. Python
reference LRU/LFU curves need not match Rust LRU/LFU curves: their eviction,
admission, and memory semantics differ. The audit retains Rust full-entry
occupancy in `rust_peak_bytes`/`rust_final_bytes`; unprefixed occupancy columns
use the evaluator's accounting. No evaluator semantics were changed to force
agreement.

Snapshot-enabled reference sweeps use an unlimited external snapshot budget.
Specify `--external-budget-mib` with `python3 -m tvcache_eval run` for a bounded
snapshot experiment. Default cache overheads and storage prices are zero
placeholders. Runtime and USD costs come from synthetic fixtures. None of
these results establish real training speedup, GPU makespan, production cost,
or performance on actual TCG workloads.

The current client uses an adaptive snapshot threshold and can insert executed
ancestors while reconstructing a missing prefix. The evaluator's references
admit snapshot candidates by default and never implicitly admit replayed
ancestors. It also omits TTL expiry, sandbox leases/deletion, and warm fork banks.
Decision-log scoring checks the explicit Rust replay decisions; it does not
establish full equivalence with every client/server lifecycle behavior.

The live load matrix uses cache lookups with eviction disabled (`none` here
means no eviction, whereas the evaluator's `none` means no caching). It covers
32/64/128 tasks and 2/4/8/16 rollouts per task at depth five, seed 12345. The
32-task audit used eight dataset-build workers, the 64-task run used 32, and
the 128-task run used 128 to shorten setup. Each uses a fresh server. These are local server/client measurements,
not a serving-policy comparison: alternate `CachePolicy` implementations are
currently wired into offline replay only.

## Other current inputs and code

The six legacy files in
`/mnt/c/Users/hanso/OneDrive/Documents/ChatGPT/tvcache/traces/synthetic`
are single-stream JSONL fixtures, not the evaluator's directory format.
They use opaque call strings and unknown (`null`) USD costs; the current
evaluator requires numeric tool costs. They need an explicit conversion and
unknown-cost strategy before entering this evaluator. The audit separately
checks their documented no-cache and unlimited result-only expectations.

`tests/test_tvclient.py` targets older `AsyncStatefulExecutor`/`ToolCallEnv`
exports. The current sibling `tvclient` provides `TvcacheToolExecutor` and
`Sandbox`. `tests/test_tvclient_current.py` exercises that API's exact hits,
stateless keys, branch reconstruction, snapshot restores, fork-cost reporting,
and released leases against a live server, using in-memory mock sandboxes:

```sh
# Fresh server in another terminal:
cargo run --release -- --eviction-policy=none
PYTHONPATH=../tvclient python3 tests/test_tvclient_current.py
```

The original client integration test still requires migration; the new smoke
test does not cover all of its fork-bank, deletion-manager, or concurrency cases.
The sandbox lifecycle test includes a TTL leg and must run with
`--eviction-policy=ttl_and_watermark`.

## Remaining data dependency

Real TCG traces have not arrived. Ordered accesses with task/rollout IDs,
stateful prefixes, entry sizes, tool costs/runtimes, and snapshot measurements
are needed to validate and calibrate the model. A final graph with aggregate
hit counts cannot reconstruct the chronological stream. Offline-learned and
online-adaptive policies are currently plans, not executable benchmarks.

See [saved results](../results/compatibility-2026-11-07/README.md).
