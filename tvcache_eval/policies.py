"""Small reference adapters, replaceable by the team's Rust simulator/policies."""

from .evaluator import Decision, entry_sizes


class ReferencePolicy:
    def __init__(self, name="lru", *, snapshots=True):
        if name not in {"none", "lru", "lfu"}:
            raise ValueError(f"unknown reference policy: {name}")
        self.name = name
        self.snapshots = snapshots
        self.cache_enabled = name != "none"

    def decide(self, node, event, context):
        node_id = node["node_id"]
        if node_id in context.entries:
            return Decision(event["event_index"], node_id, hit=True)
        miss = Decision(event["event_index"], node_id, hit=False)
        if self.name == "none":
            return miss
        store_snapshot = self.snapshots and node["snapshot_candidate"] is not None
        size, external_size = entry_sizes(node, store_snapshot)
        # Decline an oversized snapshot, then still consider caching its value.
        if (context.external_budget_bytes is not None and external_size > context.external_budget_bytes
                or size > context.memory_budget_bytes):
            store_snapshot = False
            size, external_size = entry_sizes(node, False)
        if size > context.memory_budget_bytes:
            return miss
        memory, external = context.memory_bytes, context.external_bytes
        key = ((lambda x: (x.last_access, x.node_id)) if self.name == "lru"
               else (lambda x: (x.frequency, x.last_access, x.node_id)))
        victims = []
        for entry in sorted(context.entries.values(), key=key):
            if (memory + size <= context.memory_budget_bytes
                    and (context.external_budget_bytes is None
                         or external + external_size <= context.external_budget_bytes)):
                break
            victims.append(entry.node_id)
            memory -= entry.memory_bytes
            external -= entry.external_bytes
        return Decision(event["event_index"], node_id, hit=False, admit=True,
                        evict=tuple(victims), store_snapshot=store_snapshot)
