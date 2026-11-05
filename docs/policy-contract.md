# Policy contract (starting API)

## Example implementation: FIFO

`src/policy/fifo.rs` provides `FifoPolicy`, a small reference implementation:

```rust
use tvcache::policy::{fifo::FifoPolicy, CachePolicy};
let policy: Box<dyn CachePolicy> = Box::new(FifoPolicy::default());
assert_eq!(policy.name(), "fifo");
```

It admits candidates whose charge fits each total budget (eviction may still be
needed), and selects the oldest inserted root among the supplied legal actions.
Hits do not refresh age. Removal callbacks clear state, and reinsertion puts an
entry at the back of the queue. For a subtree action, the root determines age;
notify removal for all affected descendants. Untracked roots are skipped. This
example prioritizes clarity over performance. The separate [replay runner](replay.md)
exercises it against the original enforcer; production server dispatch is unchanged.
Run `cargo test policy::fifo` for its tests.

`src/policy/mod.rs` exports `CachePolicy` through the `tvcache` library. Implement
this trait for LRU, LFU, TCG heuristics, or a predictor; select implementations
with `Box<dyn CachePolicy>`. No model runtime is required by the interface.

## Existing server behavior

`src/eviction/mod.rs` defines configuration and eviction result types, not a
pluggable policy trait. `src/cache_policy_enforcer.rs` dispatches `none` or
`ttl_and_watermark`. `src/eviction/ttl_and_watermark.rs` expires subtrees and,
above the high watermark, ranks subtree roots by cumulative result-hit counts.
The configured low watermark is currently unused. This is not conventional LFU:
victims are subtrees, and their scores aggregate descendant hits.

The new contract does not replace that production path yet. Integrating it there
requires supplying legal candidates and callbacks at server lookup/insert/removal
sites. Keep server snapshot lease and deletion semantics in the engine.

## Ownership and lifecycle

One policy instance belongs to one cache run. Construct a fresh instance for each
benchmark. The engine owns storage, resident statistics, budget definitions,
dependency checks, and metrics. The policy owns its ordering, frequency counters,
heuristic weights, or learned model. It must not inspect future trace records.

1. Resolve a request and determine its actual outcome. Call `on_access` exactly
   once, with features from BEFORE the access. This includes misses and bypasses;
   LRU should refresh resident entries on hits, while LFU must document whether
   counts are resident-only or persist across eviction. Increment engine-maintained
   observed statistics only after capturing the pre-access features.
2. On a miss, execute (or simulate execution), then call `admission` with the
   resulting entry and the engine-computed incremental memory charge. Estimated
   costs need not be available on the first lookup: use `None`, not zero.
3. Bypass a rejected entry or one that individually exceeds a relevant budget
   before evicting anything. For updates, compute incremental charges correctly;
   do not count an existing resident allocation twice.
4. If space is needed, supply only legal `EvictionCandidate`s to `choose_victim`.
   `incoming` is the admission candidate, or `None` for background pressure.
   Validate the returned index. `None` means stop; never insert over budget. The
   engine should return/report invalid selections, not loop indefinitely.
5. Apply the selected atomic removal, update storage/accounting, and call
   `on_remove` once for each affected entry. Rebuild candidates after EVERY action
   because subtree candidates may overlap. Only supply actions with positive
   reclaimable bytes in a resource under pressure. If no action can make progress,
   bypass the insertion; earlier removals are not rolled back by this contract.
6. Insert only once the candidate fits, update accounting, then call `on_insert`.
   All mutation callbacks receive the post-mutation cache context. For replacing
   an entry, call `on_remove(Replaced)` before `on_insert`.

For initial result-only replay, an action removes a single entry, and `affected`
contains only its ID. For server integration it may remove a subtree; include
every affected entry and aggregate only bytes actually reclaimable under the
chosen budget. Pinned entries or dependencies can make an entire action illegal.
Delayed sandbox deletion must not be reported as immediately freed external bytes.
This starting API handles removal of whole entries/subtrees; independent snapshot
replacement without removing the entry will need an explicit update callback.

## Metadata and features

`EntryId` includes task ID and node ID. Never key solely by command text.
`EntryMeta` includes path, parent, depth, mutation flag, resident size estimate,
tool runtime/cost estimates, and optional snapshot costs. `AccessFeatures` includes
pre-access frequency, recency, observed branching, subtree access information,
and an optional named numeric feature map for future predictors. Unknown recency
is `None`. Use `AccessContext.event_index` for deterministic ordering; resolve
equal scores with a stable ID/order, not HashMap iteration order.

Policies can maintain state from callbacks and combine it with candidate features.
The engine must maintain candidate features as accesses occur, including subtree
statistics for ancestors, rather than reuse an entry's stale insertion features.
Do not populate observed branching from the complete final graph. Future labels
and training targets must stay outside the online policy input.

`MemoryCharge` separates resident bytes and external snapshot bytes. The engine
chooses what is charged: `cache_entry_bytes` in the supplied fixtures is an entry
estimate, not measured RSS, and external snapshot sizes are separate. Capacity
for an unused resource can be `u64::MAX`; zero means zero capacity. Policy/model
state overhead is not included and should be measured separately by the evaluator.

## Weitao trace adapter

`policy::synthetic::TraceCatalog` reads `nodes.jsonl` for `tvcache-synthetic-v1`.
`read_jsonl::<AccessRecord>` reads `events.jsonl`; callers can also deserialize
one line at a time to stream large workloads. Check the enclosing manifest's
format is `tvcache-synthetic-v1` before using the adapter. `resolve` joins by
task/node ID and rejects unknown IDs, premature accesses, and invalid costs.
The evaluator must check monotonically increasing event indices/times as it
streams. `rollout_id` is available to its stateful replay logic.

Only expose the current resolved entry and causal event features to policies,
not the complete catalog. Static runtime and dollar fields are synthetic fixture
estimates. In real traces, measured future execution times must be replaced by
estimates available when the decision occurs. The adapter does not treat a
snapshot candidate as an admitted snapshot and does not calculate net savings.

## Team handoff

- James: implement `CachePolicy` for LRU/LFU; document miss-counting and tie rules.
- Wu: implement heuristics/predictor using the same callbacks and metadata.
- Evaluator: own residency, construct legal candidates, invoke lifecycle, and
  measure hits, savings, memory, and policy overhead. The trait returns decisions,
  not metrics, so all policies are measured consistently.
- Trace owner: preserve ordering and causal `features`; provide estimates and
  missing-value semantics for any future real-data adapter.

Run `cargo test --test policy_contract` to validate the adapter against all four
checked-in synthetic datasets and the trait's runtime interchangeability.
