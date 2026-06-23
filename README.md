# tvcache

An in-memory cache server for **agentic RL rollouts**. It stores the result of
each tool call, keyed by the sequence of state-mutating calls that produced it,
so that identical rollout prefixes execute once instead of once per rollout. It
also tracks **sandbox snapshots** so a rollout can resume from the deepest
cached state (forking it) rather than replaying from scratch.

This is the reference implementation of **TVCACHE**, from
[*TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents*](https://arxiv.org/abs/2602.10986)
(Abhishek Vijaya Kumar, Bhaskar Kataria, Byungsoo Oh, Emaad Manzoor, Rachee
Singh), **ICML 2026** — [arXiv:2602.10986](https://arxiv.org/abs/2602.10986).

> In RL post-training of LLM agents, calls to external tools take several
> seconds or even minutes, leaving allocated GPUs idle and inflating
> post-training time and cost. While many tool invocations repeat across
> parallel rollouts and could in principle be cached, naively caching their
> outputs for reuse is incorrect since tool outputs depend on the environment
> state induced by prior agent interactions. We present TVCACHE, a stateful
> tool-value cache for LLM agent post-training. TVCACHE maintains a tree of
> observed tool-call sequences and performs longest-prefix matching for cache
> lookups: a hit occurs only when the agent's full tool history matches a
> previously executed sequence, guaranteeing identical environment state. On
> three diverse workloads — terminal-based tasks, SQL generation, and video
> understanding — TVCACHE achieves cache hit rates of up to 70% and reduces
> median tool call execution time by up to 6.9X, with no degradation in
> post-training reward accumulation.


## Mental model

State for one task is a **Tool Call Graph (TCG)** — a prefix tree (trie):

- **Path** = the canonical serialization of the *stateful chain* (the tool calls
  so far whose `will_mutate_state()` is true). Stateless calls are looked up
  against this prefix but never extend it.
- **Node** (`src/tcg/node.rs`) = one call on that chain. It holds the cached
  `value`, an optional `sandbox_id` (a cold snapshot handle for resuming),
  `runtime`, `ttl`, hit counters (`number_of_cache_hits`, `number_of_lpm_hits`),
  and its byte size.
- **`find` is longest-prefix-match** (`src/tcg/tool_call_graph.rs`): it returns
  the exact-path `value` if present; otherwise the **deepest ancestor that owns a
  sandbox snapshot**, so the client forks from there and executes only the
  missing suffix.

Around the graph the server tracks:

- **Sandbox references** — `find` increments a snapshot's refcount (the rollout
  now holds it); the client `unref`s when done. Zero-ref victims are drainable.
- **Eviction** — a background loop (`src/cache_policy_enforcer.rs`) enforces TTL
  + memory watermarks; marked sandboxes flow to the deletion loop via drain.
- **Fork-cost EMA** — a per-task moving average of client-reported fork times,
  shared across workers so each can size its snapshot threshold.

## Architecture

- Tokio TCP server on `0.0.0.0:7878` (`src/main.rs`); one spawned task per
  connection, requests handled sequentially on each connection
  (`src/request_handler.rs`).
- Shared state is `Arc<Mutex<TVCache>>`, where `TVCache` is
  `HashMap<task_id, Arc<Mutex<ToolCallGraph>>>` (`src/tvcache.rs`).
- **Lock ordering** (`src/types.rs`): take the cache-map lock only to look up /
  insert a TCG handle, release it, *then* lock that one TCG. Never hold two TCG
  locks at once; cross-TCG operations iterate a snapshot and lock one at a time.

## Build & run

```bash
cargo run                       # listens on 0.0.0.0:7878
cargo run -- --eviction-policy=ttl_and_watermark \
             --high-watermark-mb=1024 --low-watermark-mb=768
RUST_LOG=debug cargo run        # tracing via EnvFilter
cargo test
```

`--eviction-policy` accepts `ttl_and_watermark` (default) or `none`.

## Wire protocol

Binary, length-prefixed framing (`src/protocol_primitives.rs`). Each request
starts with a `u8` tag; the response starts with its own `u8` tag. `tvclient`
is the reference encoder/decoder.

| Tag | Request | Purpose |
| --- | --- | --- |
| 1 | `Insert` | Store a value and/or sandbox snapshot at a path. |
| 2 | `Find` | LPM lookup; leases the returned sandbox (refcount++). |
| 3 | `DumpAll` | JSON dump of every TCG (debug). |
| 4 | `UnrefSandbox` | Release a leased sandbox (refcount--). |
| 5 | `MarkSandboxesEvicted` | Mark victims for deletion. |
| 6 | `ListSandboxCandidates` | Least-useful sandboxes (ascending lpm score) for eviction. |
| 7 | `DrainReclaimableSandboxes` | Hand zero-ref victims to the deletion loop. |
| 8 | `ListHotSandboxes` | Most-useful sandboxes (descending lpm score) for pre-forking. |
| 9 | `ReportForkCost` | Fold a fork-time sample into the task EMA. |
| 10 | `GetForkCost` | Read the task's fork-cost EMA. |

## Roadmap

1. Concurrent writes to a TCG
2. Intelligent lookup to identify equivalent states
3. Representing parallel tool-calls during transitions.
