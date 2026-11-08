# Plan: online-adaptive TCG policy

Status: proposed future work; no implementation or performance claim. Companion:
[offline-learned fixed policy](offline-learned-policy-plan.md).

## Goal and hypothesis

Start a new training run from generalizable offline coefficients `w0`, then make
small evidence-based updates as reuse patterns and execution costs change.
Minimize cumulative cost over the entire run, including learning overhead and
unsuccessful adaptation, rather than only the cost after convergence.

Start with eviction ranking. Admission thresholds and snapshot decisions are
later extensions. If no validated offline artifact exists, use the current
heuristic as an explicitly untrained starting point and validate it first.

## Recommended first design

Use a fixed active policy for each evaluation epoch and a small set of shadow
challengers. Challengers receive the same ordered requests but maintain their
own simulated cache state. Periodically compare their costs and promote a
candidate only when there is enough evidence that it improves on the active
policy. Keep `w0` as a persistent comparison and fallback.

This is online selection/refinement of coefficient vectors using rolling replay.
It is a deliberately modest first learner; it does not require per-decision
reinforcement learning or an unbiased counterfactual reward from each eviction.

The cost objective and causal-feature rules are the same as in the offline plan.
Begin with result-only tool-work cost, then extend to net monetary or measured
wall-clock cost when the evaluator can support it.

## Telemetry, histograms, and triggers

Collect bounded rolling statistics for all requests, including misses and
bypasses. Track estimated miss cost, realized execution cost when available,
request counts, action scores, reclaimed bytes, subtree activity, evict/reinsert
churn, capacity pressure, and policy overhead. Keep bounded ghost metadata for
recently evicted entries to observe subsequent reuse without retaining values.

Use logarithmic cost/size bins with explicit unknown-cost buckets. Per bin, keep
counts, total estimated cost, hit/miss counts, and cost-estimate coverage.
Aggregate at a few useful levels, such as tool family, depth band, or subtree
activity band, rather than creating unbounded per-path histograms.

Sort bins by **total avoidable-cost opportunity**, using current estimates and
clearly labeling that quantity as a diagnostic. An expensive miss can be a
first access or a dependency-constrained miss and may not be avoidable. Raw
total miss cost is useful for prioritizing investigation, not a benefit label.

Example: if high-cost, frequently reused subtrees account for rising miss cost,
propose challengers that increase runtime/subtree-reuse weights. Whether those
challengers help must be determined by replay: correlated features and competing
entries mean a histogram alone cannot identify the correct coefficients.

Schedule checks every configurable number of requests, with a minimum elapsed
interval. Trigger an experiment when a sustained cost-rate increase, distribution
shift, or miss-cost concentration exceeds a configured tolerance, or at a slower
periodic check. Normalize cost by requests and account for workload mix so higher
traffic alone does not trigger adaptation. Require minimum sample counts, use
hysteresis/cooldowns, and cap updates and learning CPU per epoch. Calibrate these
settings on training/validation runs, not the eventual test stream.

## Adaptation loop

1. **Observe:** serve with `w_active`, update causal request history, and collect
   an epoch of telemetry. Keep normalization fixed in the first version so
   coefficient changes have a consistent meaning.
2. **Propose:** generate a small set of bounded perturbations around `w_active`,
   emphasizing dimensions suggested by diagnostics while including occasional
   broader probes. Include `w_active` and `w0`. Normalize nonzero vectors and
   constrain distance from the active vector and from `w0`.
3. **Screen:** replay candidates on a bounded past window. Start all comparisons
   from the same compatible saved state, or give all of them the same unscored
   warm-up prefix followed by a scored segment. Do not compare a cold challenger
   against a warm active cache. Historical screening selects candidates; it does
   not establish future improvement.
4. **Confirm:** freeze the shortlisted candidates and run independent shadow
   caches on the next, previously unseen request epoch. Use causal cost estimates
   for each request and record prediction uncertainty/coverage. Measure paired
   cost differences over blocks; do not treat correlated requests as independent
   samples. Require adequate data and a practical margin beyond estimated noise
   and learning overhead. Limit repeated selection and reuse of confirmation data.
5. **Promote:** if a challenger passes, atomically install its coefficients at an
   epoch boundary and record the reason, measurements, and policy version.
   Retain the real cache contents. Do not replace them with shadow state or assume
   shadow-resident entries exist in the real cache. Measure the transition cost.
6. **Monitor:** compare subsequent costs with fixed-policy shadows. Revert to the
   previous vector or `w0` after sustained regression. A rollback changes future
   decisions; it cannot undo prior evictions or recover their cost.

No reliable winner means no coefficient update. Freeze learning during sparse
feedback or excessive overhead while continuing to serve with the active policy.
Persist updates within the run by default; exporting a new cross-run default
requires the offline validation process.

## Feedback limitations and safeguards

Observed live cost is conditional on the active policy. A cache hit hides the
execution cost that a challenger miss would have paid. Shadow evaluation must
use prior estimates and declare uncertainty; it cannot claim to have measured
all counterfactual costs. Miss-only telemetry also biases the data, so retain the
logical request stream and bounded reuse history across eviction.

Replaying a common trace assumes cache choices do not change the logical workload.
In real training, latency can affect scheduling, timeouts, and even the future
request stream. Validate promising adaptive policies in controlled end-to-end
runs with paired workloads/seeds. Shadow gains are provisional until then.

Do not add deliberate costly live exploration in the first version. If shadow
replay proves too inaccurate or expensive, a later experiment can compare a
small library of coefficient vectors with controlled randomized assignment at
independent cache/run boundaries. Account for cache carryover and log assignment
probabilities; a naive per-request bandit would ignore delayed eviction effects.

Bound telemetry, ghost history, replay buffers, and shadow-cache metadata.
Measure their memory separately from cached values and include their operational
cost in the learning budget. The current evaluator's unbounded history is not a
production-ready design for this purpose.

## Thresholds: distinguish two different uses

**Adaptation trigger:** a threshold on observed cost or drift decides when to
evaluate new coefficients. It does not directly decide which entries to retain.

**Admission/value threshold:** a later extension may retain an incoming entry
only when its expected benefit exceeds the aggregate opportunity cost of making
room. That requires calibrated benefit estimates and a legal eviction plan.
The current weighted score is ordinal, not dollars or milliseconds, so applying
a dollar threshold to it would be meaningless.

A score histogram or quantile may approximate a ranking cutoff, but varying
entry sizes, overlapping subtree removals, and separate resource budgets prevent
using a simple item-count percentile as a memory-capacity guarantee. Keep exact
legal-action and byte-accounting checks in the engine; rebuild candidates after
each removal. Evaluate admission changes separately from weight adaptation.

## Integration work

- Reuse `LinearWeights` and causal features; add versioned active/previous/default
  coefficient state and bounded diagnostics in a proposed adaptive wrapper.
- Add an evaluator/controller outside the decision path for window replay,
  challenger state, confirmation, and promotion. Avoid training under TCG locks.
- Extend telemetry explicitly: existing callbacks provide outcomes and estimated
  metadata, but do not supply a complete realized-cost feedback channel or all
  shadow replay state. Add execution-completion/cost observations and evaluator
  hooks rather than inferring unavailable measurements.
- Choose and test lifecycle semantics before deployment. Current replay inserts
  before enforcing; the general contract makes space first. An adaptive policy
  must be evaluated in its actual deployment mode.
- Keep legality, TTLs, dependency handling, sandbox leases, and actual reclaimable
  bytes in the engine. Learning may change rankings, never these constraints.
- Log initial coefficients, feature/cost versions, epoch boundaries, challenger
  seeds, updates, and rollback events so adaptation can be reproduced.

## Evaluation and milestones

1. **Instrumentation:** causal cost accounting, bounded histograms/ghost state,
   saved replay state or common warm-up support, and adaptation-log schema.
2. **Replay prototype:** start every test run from the same offline `w0`; train
   and confirm only on prefixes already observed by the learner. Tune the learner's
   own settings on separate runs.
3. **Stress evaluation:** stable workloads, abrupt/gradual phase changes, rare
   expensive calls, changing cost estimates, cold starts, constrained capacity,
   and sparse/unknown costs. Include cases where adapting should do nothing.
4. **Shadow pilot:** verify overhead and estimated-versus-realized cost calibration
   before activating updates in the server.
5. **Controlled active pilot:** bounded updates, transition accounting, rollback,
   and comparison against the frozen `w0` over complete runs.

Report cumulative primary cost, net gain after learning overhead, worst-window
regression, adaptation delay, coefficient movement, churn, update/rollback counts,
memory, and decision latency. Compare against frozen `w0`, the untuned linear
heuristic, and available conventional policies. A hindsight best-fixed vector
can be an explicitly noncausal reference, never a deployable baseline.

Proceed only if adaptation improves whole-run cost over frozen `w0` on held-out
changing workloads without unacceptable regressions on stable workloads. Set
numerical improvement, regression, and overhead limits before final evaluation.
If no consistent benefit survives overhead and uncertainty, keep the fixed
policy and use the telemetry for periodic offline retraining instead.
