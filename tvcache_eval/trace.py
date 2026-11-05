"""Load the project's synthetic format and check graph and causal-history invariants."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


class TraceError(ValueError):
    pass


def number(value, label, *, integer=False, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (integer and not isinstance(value, int))):
        raise TraceError(f"{label}: expected a finite {'integer' if integer else 'number'} >= {minimum}")
    return value


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise TraceError(f"{path}: {exc}") from exc


def read_jsonl(path):
    try:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise TraceError(f"{path}:{line_number}: empty JSONL record")
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise TraceError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise TraceError(f"{path}:{line_number}: expected an object")
                yield row
    except OSError as exc:
        raise TraceError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class Trace:
    name: str
    nodes: dict
    events: tuple
    manifest: dict
    validation: dict


def load_trace(directory, *, verify_hashes=True):
    """Validate before evaluation; never silently sort or repair malformed traces."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files_sha256", {}), dict):
        raise TraceError("manifest and files_sha256 must be objects")
    checked_hashes = []
    if manifest and manifest.get("format") != "tvcache-synthetic-v1":
        raise TraceError("unsupported trace format")
    if verify_hashes:
        for name, expected in manifest.get("files_sha256", {}).items():
            if name not in {"nodes.jsonl", "events.jsonl", "summary.json"}:
                raise TraceError(f"unexpected manifest file: {name}")
            try:
                actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            except OSError as exc:
                raise TraceError(str(exc)) from exc
            if actual != expected:
                raise TraceError(f"SHA-256 mismatch: {name}")
            checked_hashes.append(name)

    nodes, keys = {}, set()
    for node in read_jsonl(directory / "nodes.jsonl"):
        required = {"node_id", "task_id", "parent_id", "key", "depth", "first_seen_event",
                    "tool_call", "runtime_ms", "tool_cost_usd", "value_bytes",
                    "cache_entry_bytes", "snapshot_candidate"}
        if required - node.keys():
            raise TraceError(f"node missing fields: {sorted(required - node.keys())}")
        node_id = node["node_id"]
        if not isinstance(node_id, str) or not isinstance(node["task_id"], str):
            raise TraceError("node_id and task_id must be strings")
        if node_id in nodes:
            raise TraceError(f"duplicate node_id: {node_id}")
        for field in ("runtime_ms", "tool_cost_usd", "value_bytes", "cache_entry_bytes",
                      "depth", "first_seen_event"):
            number(node[field], f"{node_id}.{field}", integer=field not in {"runtime_ms", "tool_cost_usd"})
        call = node["tool_call"]
        if not isinstance(call, dict) or not isinstance(call.get("mutates_state"), bool) or not isinstance(call.get("tool"), str):
            raise TraceError(f"{node_id}: invalid tool_call")
        key = node["key"]
        if not isinstance(key, list) or not key or not all(isinstance(x, str) for x in key):
            raise TraceError(f"{node_id}: key must be a nonempty string list")
        if node["depth"] != len(key):
            raise TraceError(f"{node_id}: depth/key mismatch")
        if key[-1] != json.dumps(call, sort_keys=True, separators=(",", ":")):
            raise TraceError(f"{node_id}: key does not encode tool_call canonically")
        identity = (node["task_id"], tuple(key))
        if identity in keys:
            raise TraceError(f"{node_id}: duplicate task-scoped cache key")
        keys.add(identity)
        snapshot = node["snapshot_candidate"]
        if snapshot is not None:
            if not call["mutates_state"] or not isinstance(snapshot, dict):
                raise TraceError(f"{node_id}: only stateful calls can have snapshots")
            for field in ("create_ms", "restore_ms", "external_bytes", "handle_bytes"):
                number(snapshot.get(field), f"{node_id}.snapshot.{field}", integer=field.endswith("bytes"))
        minimum = node["value_bytes"] + (snapshot["handle_bytes"] if snapshot else 0)
        if node["cache_entry_bytes"] < minimum:
            raise TraceError(f"{node_id}: entry size omits value or snapshot handle")
        nodes[node_id] = node
    for node_id, node in nodes.items():
        parent_id = node["parent_id"]
        if parent_id is None:
            if node["depth"] != 1:
                raise TraceError(f"{node_id}: root depth must be 1")
        else:
            if not isinstance(parent_id, str) or parent_id not in nodes:
                raise TraceError(f"{node_id}: missing parent")
            parent = nodes[parent_id]
            if (parent["task_id"] != node["task_id"] or not parent["tool_call"]["mutates_state"]
                    or parent["key"] != node["key"][:-1]):
                raise TraceError(f"{node_id}: invalid stateful parent/prefix")

    events = tuple(read_jsonl(directory / "events.jsonl"))
    counts, subtree_counts = Counter(), Counter()
    last, subtree_last, last_time, first_seen = {}, {}, {}, {}
    children = defaultdict(set)
    rollouts, closed = {}, set()
    task_rollouts, tool_accesses = Counter(), Counter()
    phase_rollouts = defaultdict(Counter)
    previous_time = 0
    for index, event in enumerate(events):
        try:
            node = nodes[event["node_id"]]
            rollout_id = event["rollout_id"]
            if not isinstance(rollout_id, str):
                raise TraceError("rollout_id must be a string")
            number(event["event_index"], "event_index", integer=True)
            number(event["step"], "step", integer=True)
            number(event["time_ms"], "time_ms")
            phase = event.get("phase", 0)
            number(phase, "phase", integer=True)
            if event["event"] != "access" or event["event_index"] != index:
                raise TraceError("expected consecutive, zero-based access events")
            if not isinstance(event["is_last"], bool):
                raise TraceError("is_last must be boolean")
            if event["time_ms"] < previous_time:
                raise TraceError("timestamps go backwards")
            previous_time = event["time_ms"]
            if rollout_id in closed:
                raise TraceError("event after rollout is_last")
            step, task_id, parent_id, rollout_phase = rollouts.get(rollout_id, (0, node["task_id"], None, phase))
            if (event["step"] != step or event["task_id"] != task_id
                    or node["task_id"] != task_id or node["parent_id"] != parent_id
                    or phase != rollout_phase):
                raise TraceError("rollout step/task/stateful-prefix mismatch")
            if step == 0:
                task_rollouts[task_id] += 1
                phase_rollouts[phase][task_id] += 1
            tool_accesses[node["tool_call"]["tool"]] += 1
            node_id = node["node_id"]
            expected_features = {
                "access_count_before": counts[node_id],
                "last_access_index": last.get(node_id),
                "events_since_last_access": index - last[node_id] if node_id in last else None,
                "time_since_last_access_ms": event["time_ms"] - last_time[node_id] if node_id in last else None,
                "observed_children_before": len(children[node_id]),
                "observed_parent_children_before": len(children[parent_id]),
                "stateful_prefix_depth": node["depth"] - 1,
                "subtree_access_count_before": subtree_counts[node_id],
                "events_since_subtree_access": index - subtree_last[node_id] if node_id in subtree_last else None,
            }
            features = event.get("features", {})
            if not isinstance(features, dict):
                raise TraceError("features must be an object")
            for field, expected in expected_features.items():
                if field in features and features[field] != expected:
                    raise TraceError(f"noncausal/incorrect feature {field}: {features[field]} != {expected}")
            first_seen.setdefault(node_id, index)
            counts[node_id] += 1
            last[node_id], last_time[node_id] = index, event["time_ms"]
            # Root siblings are tracked per task, not as children of a shared null root.
            if parent_id is not None:
                children[parent_id].add(node_id)
            ancestor = node_id
            while ancestor is not None:
                subtree_counts[ancestor] += 1
                subtree_last[ancestor] = index
                ancestor = nodes[ancestor]["parent_id"]
            if event["is_last"]:
                closed.add(rollout_id)
                rollouts.pop(rollout_id, None)
            else:
                next_parent = node_id if node["tool_call"]["mutates_state"] else parent_id
                rollouts[rollout_id] = (step + 1, task_id, next_parent, phase)
        except (KeyError, TypeError, TraceError) as exc:
            raise TraceError(f"event {index}: {exc}") from exc
    if rollouts:
        raise TraceError("unterminated rollouts: missing is_last")
    for node_id, node in nodes.items():
        if first_seen.get(node_id) != node["first_seen_event"]:
            raise TraceError(f"{node_id}: first_seen_event mismatch or unused node")

    repeated = len(events) - len(nodes)
    stats = {"events": len(events), "unique_nodes": len(nodes), "rollouts": len(closed),
             "uncached_tool_time_ms": sum(nodes[e["node_id"]]["runtime_ms"] for e in events),
             "uncached_tool_cost_usd": sum(nodes[e["node_id"]]["tool_cost_usd"] for e in events),
             "repeat_accesses": repeated, "repeat_fraction": repeated / len(events) if events else 0,
             "repeat_tool_time_ms": sum((counts[k]-1)*n["runtime_ms"] for k, n in nodes.items()),
             "repeat_tool_cost_usd": sum((counts[k]-1)*n["tool_cost_usd"] for k, n in nodes.items()),
             "unique_tasks_observed": len(task_rollouts),
             "read_only_nodes": sum(not n["tool_call"]["mutates_state"] for n in nodes.values()),
             "snapshot_candidate_nodes": sum(n["snapshot_candidate"] is not None for n in nodes.values()),
             "unique_cache_entry_bytes": sum(n["cache_entry_bytes"] for n in nodes.values()),
             "unique_external_snapshot_bytes": sum((n["snapshot_candidate"] or {}).get("external_bytes", 0) for n in nodes.values())}
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            raise TraceError("summary must be an object")
        for field, actual in stats.items():
            if field in summary:
                number(summary[field], f"summary {field}")
                tolerance = 1e-8 if field == "repeat_fraction" else 1e-6
                if not math.isclose(summary[field], actual, rel_tol=1e-9, abs_tol=tolerance):
                    raise TraceError(f"summary {field} mismatch")
        for field, actual in (("task_rollouts", task_rollouts), ("tool_accesses", tool_accesses)):
            if field in summary and summary[field] != dict(actual):
                raise TraceError(f"summary {field} mismatch")
        if "phases" in summary:
            if not isinstance(summary["phases"], list):
                raise TraceError("summary phases must be a list")
            observed_phases = set()
            for row in summary["phases"]:
                if not isinstance(row, dict):
                    raise TraceError("summary phase must be an object")
                phase = number(row.get("phase"), "summary phase", integer=True)
                if phase in observed_phases:
                    raise TraceError("duplicate summary phase")
                observed_phases.add(phase)
                actual = dict(phase_rollouts.get(phase, {}))
                if row.get("rollouts") != sum(actual.values()) or row.get("task_rollouts") != actual:
                    raise TraceError(f"summary phase {phase} rollout mismatch")
            if set(phase_rollouts) - observed_phases:
                raise TraceError("summary missing observed phase")
    return Trace(directory.name, nodes, events, manifest,
                 {**stats, "valid": True, "sha256_verified": sorted(checked_hashes)})
