# Original-versus-FIFO replay

From the repository root:

```sh
cargo run --bin replay -- data/synthetic/balanced both 1
cargo run --bin replay -- data/synthetic/balanced original 1
cargo run --bin replay -- data/synthetic/balanced fifo 1
```

Arguments: dataset directory, `original|fifo|both|linear|all`, and integer capacity in MiB.
`linear` selects the [linear heuristic](linear-policy.md); `all` includes all
three policies. A fourth argument optionally supplies linear weights as JSON.
JSON goes to stdout. Use 0 for a no-cache control and 16 to hold all balanced
dataset entries. `cargo run` without `--bin` still starts the normal server.

The original run calls `eviction::ttl_and_watermark::enforce` directly, rather
than reimplementing its ranking. Both runs use `ToolCallGraph::find/insert`,
the same trace, the same budget, and subtree removal. FIFO uses `CachePolicy`.
Missing ancestors prevent insertion, as in the server; the runner does not
execute missing prefixes to reconstruct a sandbox.

## Controlled experiment assumptions

- Result-only: no TTLs, admitted snapshots, restores, forks, or snapshot deletion.
- Charges use trace `cache_entry_bytes`, not RSS or payload marker length. This
  estimate may contain synthetic handle overhead even though no snapshot is
  admitted. Both runs use the same estimates. External snapshot bytes and model
  overhead are excluded.
- Both modes insert first and enforce after each insertion, rather than every
  5 seconds as in the server. This is an explicit alternative to the contract's
  pre-insertion lifecycle: `on_insert` can receive an over-budget context.
  FIFO supports this; future policies must support this mode before using it.
  Transient and post-enforcement peaks are reported separately.
- Oversized entries bypass insertion in both modes, unlike the server's existing
  admission behavior. Each run begins with an empty cache.
- The original code breaks equal-hit ties using HashMap iteration, so its
  results can vary across processes. FIFO is deterministic for a given trace.
- Savings are gross sums of synthetic fixture costs on result hits, not measured
  training speedup or net savings. Unknown dollar costs propagate to null.
- Candidate features are refreshed from causal request history, including
  ancestor/subtree accesses. Counts persist across evictions; see the
  [linear policy feature definitions](linear-policy.md).

`src/replay.rs` implements replay; `src/bin/replay.rs` is the CLI. The manifest
format is checked. Event indices must increase and times must not decrease.
The catalog and events are loaded into memory outside the simulated byte budget.

Run `cargo test` for a hand-calculated case distinguishing original from FIFO,
zero/large-budget controls, input ordering, and the existing adapter tests.
