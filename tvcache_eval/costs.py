"""Explicit, replaceable estimates; there are no hardcoded real-world prices."""

from dataclasses import asdict, dataclass, field

from .trace import TraceError, number, read_json


ESTIMATE_FIELDS = {"runtime_ms", "tool_cost_usd", "snapshot_create_ms", "snapshot_restore_ms",
                   "snapshot_create_usd", "snapshot_restore_usd"}


@dataclass(frozen=True)
class CostModel:
    # Null/missing overrides fall back to synthetic node estimates, then zero for
    # snapshot dollars (which the current traces do not supply).
    defaults: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)
    runtime_scale: float = 1.0
    tool_cost_scale: float = 1.0
    snapshot_time_scale: float = 1.0
    lookup_ms: float = 0.0
    insert_ms: float = 0.0
    eviction_ms: float = 0.0
    policy_ms: float = 0.0
    lookup_usd: float = 0.0
    insert_usd: float = 0.0
    eviction_usd: float = 0.0
    policy_usd: float = 0.0
    memory_usd_per_gib_hour: float = 0.0
    snapshot_storage_usd_per_gib_hour: float = 0.0
    calibrated: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in {"defaults", "tools", "nodes"}:
                groups = {"defaults": value} if name == "defaults" else value
                if not isinstance(groups, dict):
                    raise TraceError(f"cost {name} must be an object")
                for key, overrides in groups.items():
                    if not isinstance(overrides, dict) or set(overrides) - ESTIMATE_FIELDS:
                        raise TraceError(f"unknown/invalid cost overrides in {name}.{key}")
                    for field_name, estimate in overrides.items():
                        if estimate is not None:
                            number(estimate, f"{name}.{key}.{field_name}")
            elif name == "calibrated":
                if not isinstance(value, bool):
                    raise TraceError("calibrated must be boolean")
            else:
                number(value, f"cost {name}")

    @classmethod
    def load(cls, path):
        data = read_json(path)
        if not isinstance(data, dict) or set(data) - cls.__dataclass_fields__.keys():
            raise TraceError("unknown cost-model fields")
        return cls(**data)

    def estimate(self, node, name):
        snapshot = node["snapshot_candidate"] or {}
        base = {"runtime_ms": node["runtime_ms"], "tool_cost_usd": node["tool_cost_usd"],
                "snapshot_create_ms": snapshot.get("create_ms", 0),
                "snapshot_restore_ms": snapshot.get("restore_ms", 0),
                "snapshot_create_usd": 0, "snapshot_restore_usd": 0}[name]
        for overrides in (self.defaults, self.tools.get(node["tool_call"]["tool"], {}),
                          self.nodes.get(node["node_id"], {})):
            if overrides.get(name) is not None:
                base = overrides[name]
        scale = {"runtime_ms": self.runtime_scale, "tool_cost_usd": self.tool_cost_scale,
                 "snapshot_create_ms": self.snapshot_time_scale,
                 "snapshot_restore_ms": self.snapshot_time_scale}.get(name, 1)
        result = base * scale
        number(result, f"effective {node['node_id']}.{name}")
        return result

    def as_dict(self):
        return asdict(self)
