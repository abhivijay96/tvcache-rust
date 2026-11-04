"""Integration tests for graph contraction through tvclient's Contractor.

Run the server first with `cargo run`. The tvclient repo must be checked out
next to this one.
"""

import asyncio
import itertools
import json
import sys
from pathlib import Path
from uuid import uuid4

from cache_client import AsyncPersistentCacheClient
from test_embedding import HOST, PORT, check, get_node

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tvclient"))

from tvclient import (
    Contractor,
    Embedder,
    Sandbox,
    ToolCall,
    TvcacheToolExecutor,
    get_embedder,
    set_embedder,
)

LIVE: dict[str, dict[str, int]] = {}  # Live sandbox id to its state
SNAPSHOTS: dict[str, dict[str, int]] = {}  # Snapshot id to its saved state
PEAK = [0]  # The most sandboxes live at once since the last reset
_id_counter = itertools.count()


class SetCall(ToolCall):
    """Sets a key to an integer, so `x 1` and `x 01` have the same effect."""

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "SetCall":
        return cls(data["key"], data["value"])

    def embed(self) -> list[float]:
        return get_embedder().encode(f"{self.key} {self.value}")


class DictSandbox(Sandbox):
    """A sandbox whose state is a dict, restored into a fresh live id."""

    async def start(self) -> None:
        state = {} if self.env_id is None else dict(SNAPSHOTS[self.env_id])
        self.env_id = f"live-{next(_id_counter)}"
        LIVE[self.env_id] = state
        PEAK[0] = max(PEAK[0], len(LIVE))

    async def fork(self) -> "DictSandbox":
        snapshot_id = f"snapshot-{next(_id_counter)}"
        SNAPSHOTS[snapshot_id] = dict(LIVE[self.env_id])
        return DictSandbox(self.task_name, snapshot_id)

    async def execute(self, tool_call: ToolCall) -> str:
        if isinstance(tool_call, SetCall):
            LIVE[self.env_id][tool_call.key] = int(tool_call.value)
        return json.dumps(LIVE[self.env_id], sort_keys=True)

    async def equals(self, other: Sandbox) -> bool:
        return LIVE[self.env_id] == LIVE[other.env_id]

    async def stop(self) -> None:
        if LIVE.pop(self.env_id, None) is None:
            SNAPSHOTS.pop(self.env_id, None)


class StubEmbedder(Embedder):
    """Gives the texts in each group nearly identical vectors, and others their own.

    Texts are spelled like `x 1`, the same as calls(). Encoding a withheld text
    raises, so the executor inserts its call without an embedding.
    """

    DIMENSIONS = 256

    def __init__(self, *groups: list[str], withheld: tuple[str, ...] = ()):
        self.axes = {text: axis for axis, group in enumerate(groups) for text in group}
        self.offsets = {text: k for group in groups for k, text in enumerate(group)}
        self.next_axis = len(groups)
        self.withheld = set(withheld)

    def encode(self, text: str) -> list[float]:
        if text in self.withheld:
            raise RuntimeError(f"withholding the embedding of {text}")
        if text not in self.axes:
            self.axes[text] = self.next_axis
            self.next_axis += 1
        vector = [0.0] * self.DIMENSIONS
        vector[self.axes[text]] = 1.0  # An IndexError leaves the call unembedded
        vector[-1] = 0.01 * self.offsets.get(text, 0)  # Keeps grouped vectors distinct
        return vector


def calls(*texts: str) -> list[SetCall]:
    """Builds calls from texts like `x 1`."""
    return [SetCall(*text.split()) for text in texts]


def reset_workload() -> None:
    """Checks that no sandbox is live and resets the peak count."""
    check(not LIVE, f"expected no live sandboxes, got {LIVE}")
    PEAK[0] = 0


async def rollout(task_id: str, history: list[SetCall], snapshots: bool = True) -> int:
    """Runs history as one rollout and returns how many calls executed.

    Every call is snapshotted unless snapshots is False.
    """
    async with TvcacheToolExecutor(
        DictSandbox,
        SetCall,
        task_id,
        HOST,
        PORT,
        adaptive=False,
        fixed_threshold_seconds=1e-9 if snapshots else 3600,
        embed_tool_calls=True,
    ) as executor:
        for index in range(len(history)):
            await executor.lookup_or_execute(history[: index + 1])
    return executor.total_executions


async def contract(task_id: str, max_sandboxes: int = 4) -> None:
    """Runs a contraction job for task_id until it ends on its own."""
    contractor = Contractor(DictSandbox, SetCall, 0.9, max_sandboxes, HOST, PORT)
    contractor.start(task_id)
    job, _ = contractor._jobs[task_id]  # Contractor has no public wait
    done, _ = await asyncio.wait({job}, timeout=30)
    check(bool(done), f"expected the job for {task_id} to finish")
    reset_workload()


def serialized(text: str) -> str:
    """Returns the serialized call for a text like `x 1`."""
    return calls(text)[0].serialize()


async def node_at(client: AsyncPersistentCacheClient, task_id: str, *texts: str) -> dict:
    """Returns the node at the path of texts, failing if it's missing."""
    node = await get_node(client, task_id, [serialized(text) for text in texts])
    check(node is not None, f"expected a node at {texts}")
    return node


async def test_workload_builds_graph(client: AsyncPersistentCacheClient, task_id: str) -> None:
    set_embedder(StubEmbedder(["x 1", "x 01"]))
    for text in ("x 1", "x 01", "x 2"):
        check(await rollout(task_id, calls(text)) == 1, f"expected {text} executed once")
    reset_workload()
    for text, state in (("x 1", {"x": 1}), ("x 01", {"x": 1}), ("x 2", {"x": 2})):
        node = await node_at(client, task_id, text)
        check(node["embedding"] is not None, f"expected {text} embedded, got {node}")
        check(node["value"] == json.dumps(state), f"expected {state} for {text}, got {node}")
        snapshot = SNAPSHOTS.get(node["sandbox_id"])
        check(snapshot == state, f"expected a snapshot of {state} for {text}, got {node}")
    groups = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    check(len(groups) == 1 and len(groups[0]["pairs"]) == 1, f"expected one pair, got {groups}")
    paired = sorted(child["calls"][0] for child in groups[0]["children"])
    expected = sorted(call.serialize() for call in calls("x 1", "x 01"))
    check(paired == expected, f"expected x 1 paired with x 01, got {paired}")


async def test_equivalent_siblings_merge(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(StubEmbedder(["x 1", "x 01"]))
    await rollout(task_id, calls("x 1", "y 1"))
    await rollout(task_id, calls("x 01"))
    reset_workload()
    await contract(task_id)
    merged = await node_at(client, task_id, "x 01")
    calls_held = sorted(merged["tool_calls"])
    expected = sorted([serialized("x 1"), serialized("x 01")])
    check(calls_held == expected, f"expected x 1 and x 01 in one node, got {calls_held}")
    check(merged["settled"], f"expected the merged node settled, got {merged}")
    groups = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    check(groups == [], f"expected nothing left to compare, got {groups}")
    executed = await rollout(task_id, calls("x 01", "y 1"))
    check(executed == 0, f"expected the absorbed call to hit the cache, got {executed}")


async def test_unequal_siblings_turn_blue(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(StubEmbedder(["x 1", "x 2"]))
    for text in ("x 1", "x 2"):
        await rollout(task_id, calls(text))
    reset_workload()
    await contract(task_id)
    for text in ("x 1", "x 2"):
        node = await node_at(client, task_id, text)
        check(list(node["tool_calls"]) == [serialized(text)], f"expected {text} alone, got {node}")
        check(node["settled"], f"expected {text} settled, got {node}")
    groups = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    check(groups == [], f"expected nothing left to compare, got {groups}")


async def test_shared_children_merge_recursively(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(StubEmbedder(["x 1", "x 01"], ["z 1", "z 01"]))
    for history in ("x 1", "y 1"), ("x 01", "y 1"), ("x 1", "z 1"), ("x 01", "z 01"):
        await rollout(task_id, calls(*history))
    reset_workload()
    await contract(task_id)
    parent = await node_at(client, task_id, "x 1")
    children = sorted(sorted(child["tool_calls"]) for child in parent["children"])
    expected = sorted([[serialized("y 1")], sorted([serialized("z 1"), serialized("z 01")])])
    check(children == expected, f"expected one y child and one merged z child, got {children}")
    merged = await node_at(client, task_id, "x 01", "z 1")
    check(merged["settled"], f"expected the merged z node settled, got {merged}")


async def test_new_embedding_reopens_settled_node(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # The executor prints one expected warning when the embedding is withheld.
    # x 2 is similar to x 1, so the first job settles x 1.
    embedder = StubEmbedder(["x 1", "x 01", "x 2"], withheld=("x 01",))
    set_embedder(embedder)
    for text in ("x 1", "x 01", "x 2"):
        await rollout(task_id, calls(text))
    reset_workload()
    await contract(task_id)
    unembedded = await node_at(client, task_id, "x 01")
    check(unembedded["embedding"] is None, f"expected x 01 unembedded, got {unembedded}")
    check(not unembedded["settled"], f"expected the job to skip x 01, got {unembedded}")
    await client.apply_contraction(task_id, [], [], [serialized("x 01")])
    for text in ("x 1", "x 01", "x 2"):
        node = await node_at(client, task_id, text)
        check(node["settled"], f"expected {text} settled, got {node}")

    embedder.withheld.clear()
    embedding = embedder.encode("x 01")
    await client.insert(task_id, [serialized("x 01")], None, None, 0, None, embedding)
    reopened = await node_at(client, task_id, "x 01")
    check(not reopened["settled"], f"expected the new embedding to turn it red, got {reopened}")
    await contract(task_id)
    merged = await node_at(client, task_id, "x 01")
    check(serialized("x 1") in merged["tool_calls"], f"expected x 01 merged with x 1, got {merged}")
    check(merged["settled"], f"expected the merged node settled, got {merged}")
    unequal = await node_at(client, task_id, "x 2")
    check(list(unequal["tool_calls"]) == [serialized("x 2")], f"expected x 2 alone, got {unequal}")
    check(unequal["settled"], f"expected x 2 still settled, got {unequal}")


async def run_tests() -> None:
    prefix = f"contraction-{uuid4().hex}"
    async with AsyncPersistentCacheClient(HOST, PORT) as client:
        await test_workload_builds_graph(client, f"{prefix}-workload")
        await test_equivalent_siblings_merge(client, f"{prefix}-equivalent")
        await test_unequal_siblings_turn_blue(client, f"{prefix}-unequal")
        await test_shared_children_merge_recursively(client, f"{prefix}-recursive")
        await test_new_embedding_reopens_settled_node(client, f"{prefix}-reopened")
    print("all contraction tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())
