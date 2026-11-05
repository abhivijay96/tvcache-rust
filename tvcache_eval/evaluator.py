"""Independent accounting for cache decisions, including lazy state reconstruction.

Policies choose admissions and victims; this module checks those choices and owns
all hit, execution, snapshot, memory, and cost accounting.
"""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import zip_longest
from types import MappingProxyType

from .costs import CostModel
from .trace import number


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    event_index: int
    node_id: str
    hit: bool
    admit: bool = False
    evict: tuple = ()
    store_snapshot: bool = False

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or set(row) - cls.__dataclass_fields__.keys():
            raise EvaluationError("unknown decision fields")
        if not isinstance(row.get("evict", []), (list, tuple)):
            raise EvaluationError("evict must be a list of node IDs")
        try:
            return cls(**{**row, "evict": tuple(row.get("evict", ()))})
        except TypeError as exc:
            raise EvaluationError(f"invalid decision: {exc}") from exc


@dataclass(frozen=True)
class Entry:
    node_id: str
    memory_bytes: int
    external_bytes: int
    has_snapshot: bool
    last_access: int
    frequency: int


@dataclass(frozen=True)
class Context:
    # A policy sees only resident entries and the current access, not future nodes.
    entries: object
    memory_budget_bytes: int
    external_budget_bytes: int | None
    memory_bytes: int
    external_bytes: int
    cost_estimates: object


def entry_sizes(node, store_snapshot):
    snapshot = node["snapshot_candidate"]
    memory = node["cache_entry_bytes"]
    if snapshot and not store_snapshot:
        memory -= snapshot["handle_bytes"]
    return memory, snapshot["external_bytes"] if store_snapshot and snapshot else 0


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Totals:
    """Additive accounting; groups use exactly the same formulas as the full run."""

    def __init__(self):
        self.values = defaultdict(int)
        self.latencies = []

    def add(self, row):
        for key, value in row.items():
            if key not in {"event_index", "node_id", "tool", "task_id", "phase", "rollout_id"}:
                self.values[key] += value
                if not math.isfinite(self.values[key]):
                    raise EvaluationError(f"event {row['event_index']}: nonfinite accumulated {key}")
        self.latencies.append(row["modeled_request_ms"])

    def report(self):
        v = self.values
        for key in ("requests", "hits", "misses", "admissions", "admission_rejections", "evictions",
                    "snapshot_creations", "snapshot_restores", "replayed_calls", "executed_calls",
                    "baseline_tool_time_ms", "baseline_tool_cost_usd", "executed_tool_time_ms",
                    "executed_tool_cost_usd", "gross_hit_time_ms", "gross_hit_cost_usd",
                    "requested_value_bytes", "hit_value_bytes", "modeled_request_ms",
                    "modeled_request_cost_usd", "memory_bytes_after", "external_bytes_after",
                    "replay_time_ms", "replay_cost_usd", "snapshot_create_ms", "snapshot_restore_ms",
                    "snapshot_create_usd", "snapshot_restore_usd", "cache_overhead_ms", "cache_overhead_usd"):
            v[key] += 0
        result = dict(v)
        result.update({
            "hit_rate": ratio(v["hits"], v["requests"]),
            "byte_hit_rate": ratio(v["hit_value_bytes"], v["requested_value_bytes"]),
            "tool_time_saved_ms": v["baseline_tool_time_ms"] - v["executed_tool_time_ms"],
            "tool_cost_saved_usd": v["baseline_tool_cost_usd"] - v["executed_tool_cost_usd"],
            "net_time_saved_ms": v["baseline_tool_time_ms"] - v["modeled_request_ms"],
            "net_cost_saved_usd": v["baseline_tool_cost_usd"] - v["modeled_request_cost_usd"],
            "average_memory_bytes": ratio(v["memory_bytes_after"], v["requests"]),
            "average_external_bytes": ratio(v["external_bytes_after"], v["requests"]),
            "request_latency_p50_ms": percentile(self.latencies, 0.50),
            "request_latency_p95_ms": percentile(self.latencies, 0.95),
            "request_latency_p99_ms": percentile(self.latencies, 0.99),
        })
        result["tool_time_saved_fraction"] = ratio(result["tool_time_saved_ms"], v["baseline_tool_time_ms"])
        result["tool_cost_saved_fraction"] = ratio(result["tool_cost_saved_usd"], v["baseline_tool_cost_usd"])
        result["speedup"] = ratio(v["baseline_tool_time_ms"], v["modeled_request_ms"])
        # Sums of occupancy samples are internal; report their averages instead.
        result.pop("memory_bytes_after", None)
        result.pop("external_bytes_after", None)
        return result


def evaluate(trace, *, memory_budget_bytes, policy=None, decisions=None, costs=None,
             external_budget_bytes=None, decision_sink=None, event_sink=None, cache_enabled=None):
    """Run a policy OR validate one decision per access from another simulator.

    Sinks are optional callables receiving JSON-compatible dictionaries. A policy
    implements decide(node, event, context) -> Decision. The cache starts empty.
    """
    number(memory_budget_bytes, "memory_budget_bytes", integer=True)
    if external_budget_bytes is not None:
        number(external_budget_bytes, "external_budget_bytes", integer=True)
    if (policy is None) == (decisions is None):
        raise EvaluationError("provide exactly one of policy or decisions")
    if cache_enabled is None:
        cache_enabled = getattr(policy, "cache_enabled", True)
    if not isinstance(cache_enabled, bool):
        raise EvaluationError("cache_enabled must be boolean")
    costs = costs or CostModel()
    entries, live_depths = {}, {}
    memory = external = peak_memory = peak_external = 0
    memory_byte_ms = external_byte_ms = 0.0
    total, groups = Totals(), {name: defaultdict(Totals) for name in ("tool", "task_id", "phase")}
    rollout_times, rollout_baselines = defaultdict(float), defaultdict(float)
    estimated = {node_id: {field: costs.estimate(node, field) for field in (
        "runtime_ms", "tool_cost_usd", "snapshot_create_ms", "snapshot_restore_ms",
        "snapshot_create_usd", "snapshot_restore_usd")}
        for node_id, node in trace.nodes.items()}
    supplied = iter(decisions) if decisions is not None else None
    previous_time = trace.events[0]["time_ms"] if trace.events else 0
    # Explicit pairing catches both missing and surplus decision records.
    pairs = zip_longest(trace.events, supplied) if supplied is not None else ((e, None) for e in trace.events)
    for event, decision in pairs:
        if event is None:
            raise EvaluationError("extra decision records after end of trace")
        index, node_id = event["event_index"], event["node_id"]
        node = trace.nodes[node_id]
        estimate = estimated[node_id]
        context = Context(MappingProxyType(entries.copy()), memory_budget_bytes,
                          external_budget_bytes, memory, external, MappingProxyType(estimate))
        if policy is not None:
            # The same estimates must drive cost-aware choices and accounting.
            policy_node = {**node, "runtime_ms": estimate["runtime_ms"],
                           "tool_cost_usd": estimate["tool_cost_usd"]}
            if node["snapshot_candidate"] is not None:
                policy_node["snapshot_candidate"] = {
                    **node["snapshot_candidate"], "create_ms": estimate["snapshot_create_ms"],
                    "restore_ms": estimate["snapshot_restore_ms"],
                }
            decision = policy.decide(policy_node, event, context)
        if isinstance(decision, dict):
            decision = Decision.from_dict(decision)
        if not isinstance(decision, Decision):
            raise EvaluationError(f"event {index}: missing/invalid decision")

        def invalid(message):
            raise EvaluationError(f"event {index}: {message}")

        if (type(decision.event_index) is not int or decision.event_index != index
                or decision.node_id != node_id):
            invalid("decision does not match trace event_index/node_id")
        if not all(isinstance(x, bool) for x in (decision.hit, decision.admit, decision.store_snapshot)):
            invalid("hit/admit/store_snapshot must be boolean")
        if decision.hit != (node_id in entries):
            invalid("claimed hit/miss disagrees with cache residency")
        if not cache_enabled and (decision.hit or decision.admit or decision.evict or decision.store_snapshot):
            invalid("cache-disabled baseline cannot perform cache operations")
        if decision.admit and decision.hit:
            invalid("admission on a hit")
        if decision.store_snapshot and (not decision.admit or node["snapshot_candidate"] is None):
            invalid("snapshot requires admission of a snapshot candidate")
        if not isinstance(decision.evict, (tuple, list)) or not all(isinstance(x, str) for x in decision.evict):
            invalid("evict must be a list of node IDs")
        if len(set(decision.evict)) != len(decision.evict):
            invalid("duplicate eviction victim")
        if any(victim not in entries for victim in decision.evict):
            invalid("eviction victim is not resident")
        new_memory, new_external = entry_sizes(node, decision.store_snapshot) if decision.admit else (0, 0)
        resulting_memory = memory - sum(entries[x].memory_bytes for x in decision.evict) + new_memory
        resulting_external = external - sum(entries[x].external_bytes for x in decision.evict) + new_external
        if resulting_memory > memory_budget_bytes:
            invalid(f"memory budget exceeded: {resulting_memory} > {memory_budget_bytes}")
        if external_budget_bytes is not None and resulting_external > external_budget_bytes:
            invalid("external snapshot budget exceeded")

        # State reconstruction uses the pre-decision cache. Victims are removed
        # after serving the request, before the new entry is allocated.
        delta = event["time_ms"] - previous_time
        memory_byte_ms += memory * delta
        external_byte_ms += external * delta
        previous_time = event["time_ms"]
        row = dict(event_index=index, node_id=node_id, tool=node["tool_call"]["tool"],
                   task_id=event["task_id"], phase=event.get("phase", 0), rollout_id=event["rollout_id"],
                   requests=1, hits=int(decision.hit), misses=int(not decision.hit),
                   admissions=int(decision.admit), admission_rejections=int(not decision.hit and not decision.admit),
                   evictions=len(decision.evict), snapshot_creations=int(decision.store_snapshot),
                   snapshot_restores=0, replayed_calls=0, executed_calls=0,
                   baseline_tool_time_ms=estimate["runtime_ms"], baseline_tool_cost_usd=estimate["tool_cost_usd"],
                   gross_hit_time_ms=estimate["runtime_ms"] if decision.hit else 0,
                   gross_hit_cost_usd=estimate["tool_cost_usd"] if decision.hit else 0,
                   requested_value_bytes=node["value_bytes"], hit_value_bytes=node["value_bytes"] if decision.hit else 0,
                   executed_tool_time_ms=0.0, executed_tool_cost_usd=0.0,
                   replay_time_ms=0.0, replay_cost_usd=0.0,
                   snapshot_create_ms=0.0, snapshot_restore_ms=0.0,
                   snapshot_create_usd=0.0, snapshot_restore_usd=0.0)
        if not decision.hit:
            live_depth = live_depths.get(event["rollout_id"], 0)
            chain, ancestor = [], node["parent_id"]
            while ancestor is not None:
                chain.append(ancestor)
                ancestor = trace.nodes[ancestor]["parent_id"]
            restore = next((x for x in chain if x in entries and entries[x].has_snapshot
                            and trace.nodes[x]["depth"] > live_depth), None)
            if restore is not None:
                live_depth = trace.nodes[restore]["depth"]
                row["snapshot_restores"] = 1
                row["snapshot_restore_ms"] = estimated[restore]["snapshot_restore_ms"]
                row["snapshot_restore_usd"] = estimated[restore]["snapshot_restore_usd"]
            replay = [x for x in reversed(chain) if trace.nodes[x]["depth"] > live_depth]
            row["replayed_calls"] = len(replay)
            row["replay_time_ms"] = sum(estimated[x]["runtime_ms"] for x in replay)
            row["replay_cost_usd"] = sum(estimated[x]["tool_cost_usd"] for x in replay)
            row["executed_calls"] = 1 + len(replay)
            row["executed_tool_time_ms"] = estimate["runtime_ms"] + row["replay_time_ms"]
            row["executed_tool_cost_usd"] = estimate["tool_cost_usd"] + row["replay_cost_usd"]
            live_depths[event["rollout_id"]] = node["depth"] - int(not node["tool_call"]["mutates_state"])
        if decision.store_snapshot:
            row["snapshot_create_ms"] = estimate["snapshot_create_ms"]
            row["snapshot_create_usd"] = estimate["snapshot_create_usd"]
        row["cache_overhead_ms"] = (costs.lookup_ms * cache_enabled + costs.policy_ms * cache_enabled
                                    + costs.insert_ms * decision.admit + costs.eviction_ms * len(decision.evict))
        row["cache_overhead_usd"] = (costs.lookup_usd * cache_enabled + costs.policy_usd * cache_enabled
                                     + costs.insert_usd * decision.admit + costs.eviction_usd * len(decision.evict))
        row["modeled_request_ms"] = (row["executed_tool_time_ms"] + row["snapshot_create_ms"]
                                     + row["snapshot_restore_ms"] + row["cache_overhead_ms"])
        row["modeled_request_cost_usd"] = (row["executed_tool_cost_usd"] + row["snapshot_create_usd"]
                                           + row["snapshot_restore_usd"] + row["cache_overhead_usd"])
        if decision.hit:
            old = entries[node_id]
            entries[node_id] = Entry(node_id, old.memory_bytes, old.external_bytes,
                                     old.has_snapshot, index, old.frequency + 1)
        for victim in decision.evict:
            del entries[victim]
        if decision.admit:
            entries[node_id] = Entry(node_id, new_memory, new_external, decision.store_snapshot, index, 1)
        memory, external = resulting_memory, resulting_external
        peak_memory, peak_external = max(peak_memory, memory), max(peak_external, external)
        row["memory_bytes_after"], row["external_bytes_after"] = memory, external
        total.add(row)
        for field, by_value in groups.items():
            by_value[str(row[field])].add(row)
        rollout_times[event["rollout_id"]] += row["modeled_request_ms"]
        rollout_baselines[event["rollout_id"]] += row["baseline_tool_time_ms"]
        if event["is_last"]:
            live_depths.pop(event["rollout_id"], None)
        if decision_sink:
            decision_sink(asdict(decision))
        if event_sink:
            event_sink(row)

    metrics = total.report()
    duration = trace.events[-1]["time_ms"] - trace.events[0]["time_ms"] if trace.events else 0
    gib_hour_in_byte_ms = (1024 ** 3) * 3_600_000
    storage_cost = (memory_byte_ms * costs.memory_usd_per_gib_hour
                    + external_byte_ms * costs.snapshot_storage_usd_per_gib_hour) / gib_hour_in_byte_ms
    metrics.update({
        "peak_memory_bytes": peak_memory, "final_memory_bytes": memory,
        "peak_external_snapshot_bytes": peak_external, "final_external_snapshot_bytes": external,
        "peak_memory_utilization": ratio(peak_memory, memory_budget_bytes),
        "average_memory_utilization": ratio(metrics["average_memory_bytes"] or 0, memory_budget_bytes),
        "trace_duration_ms": duration, "memory_byte_ms": memory_byte_ms,
        "external_snapshot_byte_ms": external_byte_ms,
        "time_weighted_average_memory_bytes": ratio(memory_byte_ms, duration),
        "time_weighted_average_external_bytes": ratio(external_byte_ms, duration),
        "storage_cost_usd": storage_cost,
        "net_cost_saved_usd": metrics["net_cost_saved_usd"] - storage_cost,
        "modeled_total_cost_usd": total.values["modeled_request_cost_usd"] + storage_cost,
        "rollouts": len(rollout_times),
        "rollout_latency_p50_ms": percentile(list(rollout_times.values()), 0.5),
        "rollout_latency_p95_ms": percentile(list(rollout_times.values()), 0.95),
        "baseline_rollout_latency_p50_ms": percentile(list(rollout_baselines.values()), 0.5),
        "baseline_rollout_latency_p95_ms": percentile(list(rollout_baselines.values()), 0.95),
    })
    for key, value in metrics.items():
        if value is not None and not math.isfinite(value):
            raise EvaluationError(f"nonfinite metric {key}; estimates exceed numeric range")
    return {
        "format": "tvcache-evaluation-v1", "workload": trace.name,
        "policy": getattr(policy, "name", "decision-log"),
        "memory_budget_bytes": memory_budget_bytes, "external_budget_bytes": external_budget_bytes,
        "cache_enabled": cache_enabled,
        "cost_model": costs.as_dict(), "trace_validation": trace.validation,
        "validation": {"valid": True, "decisions_checked": len(trace.events), "starts_empty": True},
        "assumptions": [
            "Synthetic estimates unless cost_model.calibrated is true; zero overheads are placeholders.",
            "Sequential per-access model with live sandbox state per rollout; no concurrency or GPU makespan model.",
            "Exact task-scoped value hits are lazy; misses restore the deepest resident ancestor ahead of live state and replay the suffix.",
            "Entries bundle values and optional snapshots; eviction retains the logical prefix directory, whose memory is excluded.",
            "Only explicit admissions create snapshots; reconstruction does not implicitly cache replayed ancestors.",
            "Storage billing uses logical trace timestamps through the last access, excluding any retention tail.",
            "Grouped net cost excludes global storage billing; grouped replay charges belong to the triggering request.",
        ],
        "metrics": metrics,
        "by_tool": {k: v.report() for k, v in sorted(groups["tool"].items())},
        "by_task": {k: v.report() for k, v in sorted(groups["task_id"].items())},
        "by_phase": {k: v.report() for k, v in sorted(groups["phase"].items())},
    }
