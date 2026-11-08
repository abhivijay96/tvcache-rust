#!/usr/bin/env python3
"""Generate policy-independent synthetic TVCache-style traces (Python 3.10+)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

VERSION = "1.0.0"
FORMAT = "tvcache-synthetic-v1"
OUTPUT_FILES = ("nodes.jsonl", "events.jsonl", "summary.json", "manifest.json")

DEFAULT_CONFIG = {
    "seed": 42, "num_tasks": 24, "num_rollouts": 1000,
    "templates_per_task": 8, "shared_prefix_length": 3,
    "min_stateful_calls": 6, "max_stateful_calls": 10, "branch_factor": 3,
    "hot_template_probability": 0.65, "task_zipf_exponent": 1.1,
    "read_only_probability": 0.3, "event_interval_ms": 1.0,
    "inter_rollout_gap_ms": 10.0, "node_overhead_bytes": 256,
    "phases": [
        {"fraction": 1.0, "hot_task_fraction": 0.25,
         "hot_task_probability": 0.8, "hot_task_offset": 0}
    ],
    "snapshot": {
        "mode": "cost_threshold", "probability": 0.5,
        "size_bytes": {"min": 1048576, "max": 67108864},
        "handle_bytes": 64,
        "create_ms_per_mib": 2.0, "restore_ms_per_mib": 1.0
    },
    "tool_profiles": [
        {"name": "shell", "mutates_state": True, "weight": 3.0,
         "runtime_ms": {"min": 2.0, "max": 100.0},
         "cost_usd": {"min": 0.0, "max": 0.0001},
         "value_bytes": {"min": 128, "max": 8192}},
        {"name": "compile", "mutates_state": True, "weight": 1.0,
         "runtime_ms": {"min": 200.0, "max": 10000.0},
         "cost_usd": {"min": 0.0001, "max": 0.01},
         "value_bytes": {"min": 1024, "max": 65536}},
        {"name": "sql_read", "mutates_state": False, "weight": 2.0,
         "runtime_ms": {"min": 5.0, "max": 500.0},
         "cost_usd": {"min": 0.0, "max": 0.001},
         "value_bytes": {"min": 256, "max": 262144}},
        {"name": "web_read", "mutates_state": False, "weight": 1.0,
         "runtime_ms": {"min": 20.0, "max": 2000.0},
         "cost_usd": {"min": 0.0001, "max": 0.02},
         "value_bytes": {"min": 512, "max": 131072}}
    ]
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_digest(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(file, value):
    file.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merged(base, override, location="config"):
    """Merge nested objects; lists replace wholesale; reject misspelled keys."""
    require(isinstance(override, dict), f"{location} must be an object")
    result = copy.deepcopy(base)
    for key, value in override.items():
        require(key in base, f"unknown option: {location}.{key}")
        result[key] = (merged(base[key], value, f"{location}.{key}")
                       if isinstance(base[key], dict) else value)
    return result


def number(value, name, minimum=0, maximum=None, integer=False):
    require(type(value) in ((int,) if integer else (int, float)),
            f"{name} must be {'an integer' if integer else 'numeric'}")
    require(math.isfinite(value) and value >= minimum and
            (maximum is None or value <= maximum), f"{name} is out of range")


def exact_keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected),
            f"{name} requires exactly: {', '.join(expected)}")


def validate_range(value, name, minimum=0, integer=False):
    exact_keys(value, ("min", "max"), name)
    number(value["min"], name + ".min", minimum, integer=integer)
    number(value["max"], name + ".max", value["min"], integer=integer)


def phase_lengths(c):
    # Largest-remainder apportionment keeps the configured total exact.
    raw = [p["fraction"] * c["num_rollouts"] for p in c["phases"]]
    lengths = [math.floor(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - lengths[i]), i))
    for i in order[:c["num_rollouts"] - sum(lengths)]:
        lengths[i] += 1
    return lengths


def validate_config(c):
    exact_keys(c, DEFAULT_CONFIG, "config")
    number(c["seed"], "seed", integer=True)
    for key in ("num_tasks", "num_rollouts", "templates_per_task",
                "min_stateful_calls", "max_stateful_calls"):
        number(c[key], key, 1, integer=True)
    number(c["shared_prefix_length"], "shared_prefix_length", 0, integer=True)
    require(c["shared_prefix_length"] < c["min_stateful_calls"] <= c["max_stateful_calls"],
            "need shared_prefix_length < min_stateful_calls <= max_stateful_calls")
    number(c["branch_factor"], "branch_factor", 2, integer=True)
    capacity = c["branch_factor"] ** (c["min_stateful_calls"] - c["shared_prefix_length"])
    require(c["templates_per_task"] <= capacity,
            "too many templates for the shortest branching suffix")
    for key in ("hot_template_probability", "read_only_probability"):
        number(c[key], key, 0, 1)
    number(c["task_zipf_exponent"], "task_zipf_exponent")
    number(c["event_interval_ms"], "event_interval_ms", 0.000001)
    number(c["inter_rollout_gap_ms"], "inter_rollout_gap_ms")
    number(c["node_overhead_bytes"], "node_overhead_bytes", integer=True)
    require(isinstance(c["phases"], list) and c["phases"], "phases must be nonempty")
    require(c["num_rollouts"] >= len(c["phases"]), "need at least one rollout per phase")
    for i, phase in enumerate(c["phases"]):
        exact_keys(phase, ("fraction", "hot_task_fraction", "hot_task_probability",
                           "hot_task_offset"), f"phases[{i}]")
        number(phase["fraction"], "phase fraction", 0.000001, 1)
        number(phase["hot_task_fraction"], "hot_task_fraction", 0.000001, 1)
        number(phase["hot_task_probability"], "hot_task_probability", 0, 1)
        number(phase["hot_task_offset"], "hot_task_offset", 0, c["num_tasks"]-1, integer=True)
    require(math.isclose(sum(p["fraction"] for p in c["phases"]), 1, abs_tol=1e-9),
            "phase fractions must sum to 1")
    require(all(n > 0 for n in phase_lengths(c)), "each phase must receive a rollout")
    s = c["snapshot"]
    exact_keys(s, DEFAULT_CONFIG["snapshot"], "snapshot")
    require(s["mode"] in ("none", "probability", "cost_threshold"), "invalid snapshot mode")
    number(s["probability"], "snapshot probability", 0, 1)
    number(s["handle_bytes"], "handle_bytes", 1, integer=True)
    validate_range(s["size_bytes"], "snapshot.size_bytes", 1, integer=True)
    for key in ("create_ms_per_mib", "restore_ms_per_mib"):
        number(s[key], key)
    require(isinstance(c["tool_profiles"], list) and c["tool_profiles"],
            "tool_profiles must be nonempty")
    names = set()
    for t in c["tool_profiles"]:
        exact_keys(t, DEFAULT_CONFIG["tool_profiles"][0], "tool profile")
        require(isinstance(t["name"], str) and t["name"] and t["name"] not in names,
                "tool names must be unique nonempty strings")
        names.add(t["name"])
        require(type(t["mutates_state"]) is bool, "mutates_state must be boolean")
        number(t["weight"], "tool weight", 0.000001)
        validate_range(t["runtime_ms"], "runtime_ms", 0.000001)
        validate_range(t["cost_usd"], "cost_usd")
        validate_range(t["value_bytes"], "value_bytes", 1, integer=True)
    require(any(t["mutates_state"] for t in c["tool_profiles"]), "need a stateful tool")
    require(c["read_only_probability"] == 0 or
            any(not t["mutates_state"] for t in c["tool_profiles"]), "need a read-only tool")


def load_config(file=None):
    override = json.loads(Path(file).read_text(encoding="utf-8")) if file else {}
    c = merged(DEFAULT_CONFIG, override)
    validate_config(c)
    return c


def stable_rng(seed, *parts):
    digest = hashlib.sha256(canonical([seed, *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def node_id(task_id, key):
    digest = hashlib.sha256(canonical([task_id, key]).encode()).hexdigest()
    return "n_" + digest[:32]


def sample(rng, interval, integer=False):
    lo, hi = interval["min"], interval["max"]
    # Log-uniform for positive ranges, uniform for ranges starting at zero.
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if lo > 0 else rng.uniform(lo, hi)
    return max(lo, min(hi, round(v) if integer else round(v, 8)))


class History:
    """Observed graph only. Features are sampled before the current access."""
    def __init__(self):
        self.count = Counter()
        self.last, self.last_time, self.children = {}, {}, {}
        self.subtree_count = Counter()
        self.subtree_last, self.parents = {}, {}

    def features(self, node, index, time_ms):
        nid, parent = node["node_id"], node["parent_id"]
        last = self.last.get(nid)
        sub_last = self.subtree_last.get(nid)
        return {
            "access_count_before": self.count[nid],
            "last_access_index": last,
            "events_since_last_access": None if last is None else index - last,
            "time_since_last_access_ms": None if last is None else round(time_ms - self.last_time[nid], 6),
            "stateful_prefix_depth": node["depth"] - 1,
            "observed_children_before": len(self.children.get(nid, ())),
            "observed_parent_children_before": len(self.children.get(parent, ())),
            "subtree_access_count_before": self.subtree_count[nid],
            "events_since_subtree_access": None if sub_last is None else index - sub_last,
        }

    def record(self, node, index, time_ms):
        nid, parent = node["node_id"], node["parent_id"]
        self.parents[nid] = parent
        if parent is not None:
            self.children.setdefault(parent, set()).add(nid)
        self.count[nid] += 1
        self.last[nid], self.last_time[nid] = index, time_ms
        cursor = nid
        while cursor is not None:
            self.subtree_count[cursor] += 1
            self.subtree_last[cursor] = index
            cursor = self.parents[cursor]


def make_node(c, task, parent, key, call, profile, prefix_runtime, first_seen):
    nid = node_id(task, key)
    rng = stable_rng(c["seed"], "metadata", nid)
    runtime = sample(rng, profile["runtime_ms"])
    value_bytes = sample(rng, profile["value_bytes"], integer=True)
    cost = sample(rng, profile["cost_usd"])
    snapshot = None
    s = c["snapshot"]
    if call["mutates_state"] and s["mode"] != "none" and rng.random() < s["probability"]:
        external = sample(rng, s["size_bytes"], integer=True)
        create = round(external / 1048576 * s["create_ms_per_mib"], 8)
        restore = round(external / 1048576 * s["restore_ms_per_mib"], 8)
        if s["mode"] == "probability" or prefix_runtime + runtime > create + restore:
            snapshot = {"external_bytes": external, "handle_bytes": s["handle_bytes"],
                        "create_ms": create, "restore_ms": restore}
    return {
        "node_id": nid, "task_id": task, "parent_id": parent,
        "depth": len(key), "tool_call": call, "key": key,
        "runtime_ms": runtime, "tool_cost_usd": cost, "value_bytes": value_bytes,
        "cache_entry_bytes": value_bytes + c["node_overhead_bytes"] +
                             (snapshot["handle_bytes"] if snapshot else 0),
        "snapshot_candidate": snapshot, "first_seen_event": first_seen,
    }


def rollout_calls(c, task, variant):
    """Finite task templates share a stateful prefix and branch after it."""
    rng = stable_rng(c["seed"], "template", task, variant)
    length = rng.randint(c["min_stateful_calls"], c["max_stateful_calls"])
    prefix, parent, prefix_runtime = [], None, 0.0
    stateful = [t for t in c["tool_profiles"] if t["mutates_state"]]
    readonly = [t for t in c["tool_profiles"] if not t["mutates_state"]]
    for depth in range(length):
        branch_depth = depth - c["shared_prefix_length"]
        branch = 0 if branch_depth < 0 else (variant // c["branch_factor"] ** branch_depth) % c["branch_factor"]
        actions = [(True, branch)]
        if rng.random() < c["read_only_probability"]:
            actions.append((False, 0))
        for mutates, token in actions:
            options = stateful if mutates else readonly
            identity_rng = stable_rng(c["seed"], "call", task, prefix, mutates, token)
            profile = identity_rng.choices(options, weights=[t["weight"] for t in options], k=1)[0]
            call = {"tool": profile["name"], "arguments": {"branch": token},
                    "mutates_state": mutates}
            key = [*prefix, canonical(call)]
            yield parent, key, call, profile, prefix_runtime
            if mutates:
                node = make_node(c, task, parent, key, call, profile, prefix_runtime, 0)
                parent, prefix = node["node_id"], key
                prefix_runtime += node["runtime_ms"]


def generate(c, output):
    validate_config(c)
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()),
            f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arrivals = random.Random(c["seed"])
    nodes, history = {}, History()
    task_counts, tool_counts, totals = Counter(), Counter(), Counter()
    phase_counts = []
    index, rollout_index, clock = 0, 0, 0.0
    zipf = [1 / (i + 1) ** c["task_zipf_exponent"] for i in range(c["num_tasks"])]
    with (output / "nodes.jsonl").open("w", encoding="utf-8", newline="\n") as nf, \
         (output / "events.jsonl").open("w", encoding="utf-8", newline="\n") as ef:
        for phase_index, (phase, length) in enumerate(zip(c["phases"], phase_lengths(c))):
            hot_n = max(1, math.ceil(c["num_tasks"] * phase["hot_task_fraction"]))
            hot = [(phase["hot_task_offset"] + i) % c["num_tasks"] for i in range(hot_n)]
            cold = [i for i in range(c["num_tasks"]) if i not in hot]
            per_phase = Counter()
            for _ in range(length):
                clock += c["inter_rollout_gap_ms"]
                hot_draw = arrivals.random() < phase["hot_task_probability"] or not cold
                pool = hot if hot_draw else cold
                task_index = arrivals.choices(pool, weights=zipf[:len(pool)], k=1)[0]
                task = f"task_{task_index:04d}"
                variant = 0 if c["templates_per_task"] == 1 or arrivals.random() < c["hot_template_probability"] else arrivals.randrange(1, c["templates_per_task"])
                calls = list(rollout_calls(c, task, variant))
                task_counts[task] += 1
                per_phase[task] += 1
                for step, (parent, key, call, profile, prefix_runtime) in enumerate(calls):
                    nid = node_id(task, key)
                    if nid not in nodes:
                        nodes[nid] = make_node(c, task, parent, key, call, profile, prefix_runtime, index)
                        nf.write(canonical(nodes[nid]) + "\n")
                    node = nodes[nid]
                    time_ms = round(clock, 6)
                    features = history.features(node, index, time_ms)
                    event = {"event_index": index, "event": "access", "time_ms": time_ms,
                             "rollout_id": f"rollout_{rollout_index:07d}", "step": step,
                             "phase": phase_index, "task_id": task, "node_id": nid,
                             "is_last": step == len(calls) - 1, "features": features}
                    ef.write(canonical(event) + "\n")
                    tool_counts[call["tool"]] += 1
                    totals["uncached_tool_time_ms"] += node["runtime_ms"]
                    totals["uncached_tool_cost_usd"] += node["tool_cost_usd"]
                    if features["access_count_before"] > 0:
                        totals["repeat_accesses"] += 1
                        totals["repeat_tool_time_ms"] += node["runtime_ms"]
                        totals["repeat_tool_cost_usd"] += node["tool_cost_usd"]
                    history.record(node, index, time_ms)
                    index += 1
                    clock += c["event_interval_ms"]
                rollout_index += 1
            phase_counts.append({"phase": phase_index, "rollouts": length,
                                 "hot_tasks": [f"task_{i:04d}" for i in hot],
                                 "task_rollouts": dict(sorted(per_phase.items()))})
    summary = {
        "events": index, "rollouts": rollout_index, "unique_nodes": len(nodes),
        "unique_tasks_observed": len(task_counts),
        "read_only_nodes": sum(not n["tool_call"]["mutates_state"] for n in nodes.values()),
        "snapshot_candidate_nodes": sum(n["snapshot_candidate"] is not None for n in nodes.values()),
        "unique_cache_entry_bytes": sum(n["cache_entry_bytes"] for n in nodes.values()),
        "unique_external_snapshot_bytes": sum(n["snapshot_candidate"]["external_bytes"] for n in nodes.values() if n["snapshot_candidate"]),
        "repeat_fraction": round(totals["repeat_accesses"] / index, 8),
        "task_rollouts": dict(sorted(task_counts.items())),
        "tool_accesses": dict(sorted(tool_counts.items())), "phases": phase_counts,
        **{k: round(totals[k], 8) for k in ("uncached_tool_time_ms", "uncached_tool_cost_usd",
                                           "repeat_tool_time_ms", "repeat_tool_cost_usd")},
        "repeat_accesses": totals["repeat_accesses"],
    }
    dump_json(output / "summary.json", summary)
    dump_json(output / "manifest.json", {
        "format": FORMAT, "generator_version": VERSION, "synthetic": True, "config": c,
        "files_sha256": {name: file_digest(output / name)
                         for name in OUTPUT_FILES if name != "manifest.json"},
        "upstream_reference": {
            "tvcache_rust": "https://github.com/abhivijay96/tvcache-rust#readme",
            "tvclient": "https://github.com/abhivijay96/tvclient#readme",
            "basis": "Documented stateful-prefix semantics; not a native dump or wire format."
        }
    })
    return summary


def read_jsonl(file):
    with Path(file).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{file}:{line_no}: invalid JSON") from exc


def validate(output):
    """Check integrity, key/state semantics, history features and rollout boundaries."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == FORMAT and manifest["synthetic"] is True, "unexpected format")
    c = manifest["config"]
    validate_config(c)
    for name in OUTPUT_FILES[:-1]:
        actual = file_digest(output / name)
        require(manifest["files_sha256"].get(name) == actual, f"checksum mismatch: {name}")
    nodes = {}
    for node in read_jsonl(output / "nodes.jsonl"):
        nid, parent, key = node["node_id"], node["parent_id"], node["key"]
        require(nid not in nodes and nid == node_id(node["task_id"], key), "duplicate/incorrect node ID")
        require(len(key) == node["depth"] >= 1 and key[-1] == canonical(node["tool_call"]),
                f"invalid key/depth: {nid}")
        require(type(node["tool_call"]["mutates_state"]) is bool, "mutates_state must be boolean")
        if parent is None:
            require(len(key) == 1, "non-root missing parent")
        else:
            require(parent in nodes and nodes[parent]["tool_call"]["mutates_state"],
                    "parent must be an earlier stateful node")
            require(nodes[parent]["key"] == key[:-1] and nodes[parent]["task_id"] == node["task_id"],
                    "key prefix or task differs from parent")
        for field in ("runtime_ms", "tool_cost_usd"):
            number(node[field], field)
        number(node["value_bytes"], "value_bytes", 1, integer=True)
        number(node["first_seen_event"], "first_seen_event", integer=True)
        snap = node["snapshot_candidate"]
        if snap is not None:
            exact_keys(snap, ("external_bytes", "handle_bytes", "create_ms", "restore_ms"), "snapshot")
            require(node["tool_call"]["mutates_state"], "read-only node cannot own a snapshot")
            for field in ("external_bytes", "handle_bytes"):
                number(snap[field], field, 1, integer=True)
            for field in ("create_ms", "restore_ms"):
                number(snap[field], field)
        require(node["cache_entry_bytes"] == node["value_bytes"] + c["node_overhead_bytes"] +
                (snap["handle_bytes"] if snap else 0), "cache size accounting mismatch")
        nodes[nid] = node
    # Rebuild access history independently of the generator's History class.
    counts, last, last_time, children, subtree, sublast = Counter(), {}, {}, {}, Counter(), {}
    rollout_count, current_rollout, current_task, cursor = 0, None, None, None
    previous_time, previous_step, ended, event_count = -1, -1, True, 0
    repeats, total_time, total_cost, repeat_time, repeat_cost = 0, 0.0, 0.0, 0.0, 0.0
    phase_sequence = [i for i, length in enumerate(phase_lengths(c)) for _ in range(length)]
    tasks, tools, phase_tasks = Counter(), Counter(), [Counter() for _ in c["phases"]]
    for index, event in enumerate(read_jsonl(output / "events.jsonl")):
        require(event["event"] == "access" and event["event_index"] == index, "event order mismatch")
        number(event["time_ms"], "time_ms")
        require(event["time_ms"] > previous_time, "time must be strictly increasing")
        require(type(event["is_last"]) is bool, "is_last must be boolean")
        if event["rollout_id"] != current_rollout:
            require(ended and event["step"] == 0, "invalid rollout boundary")
            require(event["rollout_id"] == f"rollout_{rollout_count:07d}", "rollout order mismatch")
            current_rollout, current_task, cursor = event["rollout_id"], event["task_id"], None
            rollout_count += 1
            previous_step = -1
            tasks[current_task] += 1
            require(type(event["phase"]) is int and 0 <= event["phase"] < len(phase_tasks),
                    "invalid phase")
            phase_tasks[event["phase"]][current_task] += 1
        else:
            require(not ended, "event after terminal rollout step")
        require(rollout_count <= len(phase_sequence) and
                event["phase"] == phase_sequence[rollout_count - 1], "incorrect phase")
        require(event["task_id"] == current_task and event["step"] == previous_step + 1,
                "inconsistent rollout task/step")
        require(event["node_id"] in nodes, "unknown node")
        node = nodes[event["node_id"]]
        nid, parent = node["node_id"], node["parent_id"]
        require(node["task_id"] == current_task and parent == cursor,
                "stateful-prefix mismatch (read-only calls must not extend state)")
        require(node["first_seen_event"] <= index and
                (counts[nid] > 0 or node["first_seen_event"] == index), "wrong first-seen event")
        gap = None if nid not in last else index - last[nid]
        expected = {
            "access_count_before": counts[nid], "last_access_index": last.get(nid),
            "events_since_last_access": gap,
            "time_since_last_access_ms": None if gap is None else round(event["time_ms"] - last_time[nid], 6),
            "stateful_prefix_depth": node["depth"] - 1,
            "observed_children_before": len(children.get(nid, ())),
            "observed_parent_children_before": len(children.get(parent, ())),
            "subtree_access_count_before": subtree[nid],
            "events_since_subtree_access": None if nid not in sublast else index - sublast[nid],
        }
        require(event["features"] == expected, f"history feature mismatch at event {index}")
        total_time += node["runtime_ms"]
        total_cost += node["tool_cost_usd"]
        tools[node["tool_call"]["tool"]] += 1
        if counts[nid]:
            repeats += 1
            repeat_time += node["runtime_ms"]
            repeat_cost += node["tool_cost_usd"]
        if parent:
            children.setdefault(parent, set()).add(nid)
        counts[nid] += 1
        last[nid], last_time[nid] = index, event["time_ms"]
        ancestor = nid
        while ancestor is not None:
            subtree[ancestor] += 1
            sublast[ancestor] = index
            ancestor = nodes[ancestor]["parent_id"]
        if node["tool_call"]["mutates_state"]:
            cursor = nid
        previous_time, previous_step, ended = event["time_ms"], event["step"], event["is_last"]
        event_count += 1
    require(ended and event_count > 0, "empty/truncated event stream")
    require(rollout_count == c["num_rollouts"] and len(counts) == len(nodes), "missing rollouts/nodes")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    expected_totals = {
        "events": event_count, "unique_nodes": len(nodes), "rollouts": rollout_count,
        "repeat_accesses": repeats, "repeat_fraction": round(repeats / event_count, 8),
        "unique_tasks_observed": len(tasks),
        "read_only_nodes": sum(not n["tool_call"]["mutates_state"] for n in nodes.values()),
        "snapshot_candidate_nodes": sum(n["snapshot_candidate"] is not None for n in nodes.values()),
        "unique_cache_entry_bytes": sum(n["cache_entry_bytes"] for n in nodes.values()),
        "unique_external_snapshot_bytes": sum(n["snapshot_candidate"]["external_bytes"]
                                            for n in nodes.values() if n["snapshot_candidate"]),
        "task_rollouts": dict(tasks), "tool_accesses": dict(tools),
    }
    for field, value in expected_totals.items():
        require(summary[field] == value, f"summary mismatch: {field}")
    require(len(summary["phases"]) == len(c["phases"]), "missing phase summary")
    for phase_index, phase in enumerate(summary["phases"]):
        require(phase["phase"] == phase_index and
                phase["rollouts"] == sum(phase_tasks[phase_index].values()) and
                phase["task_rollouts"] == dict(phase_tasks[phase_index]), "phase summary mismatch")
    for field, value in (("uncached_tool_time_ms", total_time), ("uncached_tool_cost_usd", total_cost),
                         ("repeat_tool_time_ms", repeat_time), ("repeat_tool_cost_usd", repeat_cost)):
        require(math.isclose(summary[field], value, rel_tol=1e-10, abs_tol=1e-7),
                f"summary mismatch: {field}")
    return {"valid": True, "events": event_count, "unique_nodes": len(nodes), "rollouts": rollout_count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="generate a deterministic workload")
    gen.add_argument("--config", type=Path, help="JSON overrides; missing fields use defaults")
    gen.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    gen.add_argument("--seed", type=int, help="override the configured random seed")
    gen.add_argument("--rollouts", type=int, help="override the configured rollout count")
    check = commands.add_parser("validate", help="check a generated trace directory")
    check.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            c = load_config(args.config)
            if args.seed is not None:
                c["seed"] = args.seed
            if args.rollouts is not None:
                c["num_rollouts"] = args.rollouts
            result = generate(c, args.out)
            validate(args.out)
            print(json.dumps({"output": str(args.out), "validated": True,
                              **{k: result[k] for k in ("events", "rollouts", "unique_nodes", "repeat_fraction")}},
                             indent=2))
        else:
            print(json.dumps(validate(args.trace), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
