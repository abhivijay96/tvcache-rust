# Plan: offline-learned, fixed TCG policy

Status: proposed future work; no implementation or performance claim. Companion:
[online adaptation plan](online-adaptive-policy-plan.md).

## Goal and hypothesis

Learn one set of retention coefficients from several agent-training runs, then
freeze those coefficients for use in new runs. The hypothesis is that a shared
combination of reuse, execution cost, size, and TCG structure generalizes better
than the current all-ones heuristic.

“Fixed” refers to coefficients, feature transformations, and decision rules.
Frequency, recency, and observed graph statistics still evolve during a run.
Retraining later produces a new explicitly versioned policy artifact; it does
not silently change an active run's policy.

## Starting point and scope

Build on [LinearPolicy](linear-policy.md) and the [policy contract](policy-contract.md).
Start with the existing nine bounded features and lowest-score subtree eviction:

`retention_score(action) = sum_i weight_i * feature_i(action)`

Higher scores favor retention. Initially preserve the current admission rule and
feature transformations so that experiments isolate coefficient learning.
Depth and branching are hypotheses, not inherently positive signals: allow their
weights to become negative or zero. Use stable task/node IDs to break ties.

The first experiment covers result caching under the existing replay assumptions.
Snapshot selection, restore/fork costs, and prefix reconstruction are a separate
extension. Production integration is a later milestone; the server does not yet
dispatch through `CachePolicy`.

## What to minimize

Choose the primary objective before tuning. For the first experiment, use total
simulated tool execution time on misses/bypasses, at a fixed memory capacity.
This is a tool-work proxy, not elapsed training time: concurrent tool runtimes
cannot simply be added to estimate wall-clock duration.

For a fixed trace with known per-request costs, minimizing residual tool cost is
equivalent to maximizing gross saved cost. Extend the evaluator to report the
no-cache total and residual cost explicitly. If costs are missing, report
coverage and use a documented estimate or exclude that objective for that trace;
never treat missing cost as free execution.

Report USD, hit rate, memory, and policy CPU time separately. If the target later
becomes a combined monetary objective, explicitly price execution, storage, and
policy overhead in the same units. If the target becomes end-to-end training
time, use measured scheduling/critical-path effects. The runtime and USD feature
weights are learned ranking parameters, not implicit unit conversions.

Train across runs and capacities using an objective such as:

`J(w) = mean_over_runs_and_capacities(C(w) / C(no_cache))`

Exclude zero-denominator cases from the ratio and report them separately. Give
runs equal weight initially so a large run does not dominate. Also report total
cost weighted by request volume and worst-workload regressions. Select any
regression constraint and evaluation capacities before examining the test set.

## Data and causal features

1. Collect complete ordered request streams from multiple training runs,
   preserving task/rollout IDs, configuration, agent checkpoint or training
   phase, memory capacity, entry sizes, and cost observations available at each
   point. Include hits, misses, and bypasses, not just resident-cache events.
2. Partition whole runs into training, validation, and final test sets. Keep
   related task instances/seeds grouped; additionally hold out workload families
   or checkpoints to test transfer. Do not randomly split requests from the
   same run across sets.
3. Reconstruct features causally for every replay. Runtime estimates must come
   from prior observations or a declared cold-start prior, rather than a tool's
   later execution. Do not expose the final graph or future reuse to the policy.
4. Use the existing synthetic datasets for smoke checks and controlled stress
   cases. They alone cannot establish generalization to real training runs.

Keep training targets and full replay streams outside the serving policy. Record
the split manifest and feature schema alongside every experiment.

## Learning procedure

Use direct replay-based coefficient optimization first: eviction involves discrete
rankings and dependencies, so there is no need to assume a differentiable loss.

1. Measure the all-ones linear heuristic, FIFO, the original subtree-hit policy,
   and no-cache/large-capacity controls. Add LRU/LFU comparisons when available.
2. Search bounded coefficient vectors with seeded random search, then refine
   promising vectors using coordinate perturbations at progressively smaller
   step sizes. Tune on training runs; select the final candidate on validation.
3. Normalize each nonzero vector, for example to `sum(abs(w_i)) = 1`. Positive
   rescaling does not change this ranking policy, so searching scaled duplicates
   wastes work. Keep the zero-vector tie-order policy only as an explicit control.
4. Evaluate each vector with a fresh policy/cache and identical trace, capacity,
   cost model, and lifecycle mode. Cache experiment results by these inputs.
5. Prefer a stable near-best vector over a narrow optimum: inspect performance
   under nearby coefficients, feature ablations, and held-out capacities.
6. Freeze the selected artifact and evaluate it once on untouched test runs.
   Report paired per-run differences and uncertainty across independent runs;
   repeat the nondeterministic original baseline as needed.

The first search should remain small enough to understand: nine existing weights,
fixed scales, no learned admission threshold, and a recorded compute budget.

## Making the policy more TCG-aware

The current scorer combines root costs with subtree activity and action bytes;
it does not estimate the full benefit lost when a subtree is removed. After the
coefficient-only experiment, evaluate these additions one at a time:

- Recent subtree request rate and changes in that rate, including training-phase
  drift that lifetime counts may conceal.
- Estimated incremental execution loss for the entire removal action, using
  causally estimated descendant demand and costs.
- Shared-prefix reuse and the cost to reconstruct from the nearest surviving
  usable ancestor, once replay models reconstruction and snapshot availability.
- Missing-cost indicators and explicit cold-start estimates.

Score legal removal actions, not independent nodes that may be impossible to
remove. Sum distinct affected request costs only when the execution model
justifies it; nested prefixes and snapshots must not receive duplicate credit.
Recompute candidates after each eviction because subtree actions overlap.

New aggregate features require engine support: `affected` currently provides IDs,
not all descendant costs, and the replay runner does not pass through `extra`.
Any learned scales must be fit on training data and frozen with the weights.

## Deliverables and implementation sequence

1. **Evaluation foundation:** split manifests, residual-cost accounting, causal
   real-trace estimates, capacity sweeps, and reproducible baseline reports.
2. **Offline trainer:** a replay-driven search tool with seeds, bounded search,
   run-level validation, and coefficient/feature ablation reports.
3. **Fixed artifact:** weights compatible with the current JSON config plus a
   separate manifest containing feature version, cost objective, training data
   version, lifecycle mode, validation results, and supported resource budgets.
4. **Generalization report:** held-out runs/families/phases, regression cases,
   policy overhead, and sensitivity to budget and coefficient perturbations.
5. **Optional extension:** action-level cost features, then snapshot-aware replay
   and production integration, each with a fresh evaluation.

Before production integration, resolve the replay/contract lifecycle difference:
replay currently inserts and then enforces, whereas the contract normally makes
space before insertion. Validate the chosen behavior explicitly; do not transfer
benchmark results across those modes without measurement.

## Acceptance and failure criteria

Predeclare a practical improvement target, acceptable per-workload regression,
and overhead budget after measuring baseline variability. Promote only if the
frozen policy improves held-out primary cost within those limits, preserves
graph/budget correctness, and yields reproducible decisions.

If gains vanish on unseen runs, retain the heuristic and report the failure.
Possible next experiments are richer action features or a small number of
explicit workload profiles; avoid selecting profiles using future run behavior.
The fixed artifact, if successful, becomes the starting point and fallback for
the online policy.
