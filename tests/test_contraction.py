"""Integration tests for graph contraction through tvclient's Contractor.

Run the server first with `cargo run`. The tvclient repo must be checked out
next to this one.
"""

import asyncio
import itertools
import json
import math
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
ORIGINS: dict[str, str | None] = {}  # Live sandbox id to the snapshot it started from
COMPARED: list[tuple[dict, dict, bool]] = []  # Each equals call's states and liveness
FAIL_EQUALS: list[tuple[dict, dict]] = []  # State pairs whose equals raises
FAIL_RESTORES: set[str] = set()  # Snapshot ids whose restore raises
FAILED_RESTORES: list[str] = []  # Each restore that raised
GATED = [False]  # Whether equals waits for GATE
GATE = asyncio.Event()  # Set to let held equals calls return
IN_EQUALS = asyncio.Event()  # Set once an equals call is held
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
        if self.env_id in FAIL_RESTORES:
            FAILED_RESTORES.append(self.env_id)
            raise RuntimeError(f"failing the restore of {self.env_id}")
        state = {} if self.env_id is None else dict(SNAPSHOTS[self.env_id])
        live_id = f"live-{next(_id_counter)}"
        ORIGINS[live_id] = self.env_id
        self.env_id = live_id
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
        both_live = self.env_id in LIVE and other.env_id in LIVE
        mine, theirs = LIVE.get(self.env_id, {}), LIVE.get(other.env_id, {})
        COMPARED.append((dict(mine), dict(theirs), both_live))
        if not both_live:
            raise RuntimeError("compared a sandbox that isn't live")
        if (mine, theirs) in FAIL_EQUALS or (theirs, mine) in FAIL_EQUALS:
            raise RuntimeError(f"failing equals for {mine} and {theirs}")
        if GATED[0]:
            IN_EQUALS.set()
            await GATE.wait()
        return mine == theirs

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


class FixedEmbedder(Embedder):
    """Places each text at a chosen angle in the plane.

    A pair's score is then the cosine of the angle between them.
    """

    def __init__(self, degrees: dict[str, float]):
        self.degrees = degrees

    def encode(self, text: str) -> list[float]:
        angle = math.radians(self.degrees[text])
        return [math.cos(angle), math.sin(angle)]


def calls(*texts: str) -> list[SetCall]:
    """Builds calls from texts like `x 1`."""
    return [SetCall(*text.split()) for text in texts]


def reset_workload() -> None:
    """Checks that no sandbox is live and resets the counts and records."""
    check(not LIVE, f"expected no live sandboxes, got {LIVE}")
    PEAK[0] = 0
    ORIGINS.clear()
    COMPARED.clear()
    FAIL_EQUALS.clear()
    FAIL_RESTORES.clear()
    FAILED_RESTORES.clear()
    GATED[0] = False
    GATE.clear()
    IN_EQUALS.clear()


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


async def contract(*task_ids: str, max_sandboxes: int = 4) -> int:
    """Runs one contractor's jobs for task_ids until they end on their own.

    Returns the most sandboxes live at once since the last reset_workload.
    """
    contractor = Contractor(DictSandbox, SetCall, 0.9, max_sandboxes, HOST, PORT)
    for task_id in task_ids:
        contractor.start(task_id)
    jobs = {contractor._jobs[task_id][0] for task_id in task_ids}  # No public wait
    _, pending = await asyncio.wait(jobs, timeout=30)
    check(not pending, f"expected the jobs for {task_ids} to finish")
    check(not LIVE, f"expected no live sandboxes, got {LIVE}")
    return PEAK[0]


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


async def test_replays_without_snapshots(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(StubEmbedder(["x 1", "x 01"]))
    snapshots_before = set(SNAPSHOTS)
    for history in ("a 1", "x 1"), ("a 1", "x 01"):
        await rollout(task_id, calls(*history), snapshots=False)
    added = set(SNAPSHOTS) - snapshots_before
    check(not added, f"expected the rollouts to take no snapshots, got {added}")
    reset_workload()
    await contract(task_id)
    fresh = bool(ORIGINS) and set(ORIGINS.values()) == {None}
    check(fresh, f"expected fresh starts, got {ORIGINS}")
    states = [state for compared in COMPARED for state in compared[:2]]
    check(states == [{"a": 1, "x": 1}] * 2, f"expected replayed paths compared, got {COMPARED}")
    merged = await node_at(client, task_id, "a 1", "x 01")
    check(serialized("x 1") in merged["tool_calls"], f"expected x 1 and x 01 merged, got {merged}")
    for texts in ("a 1",), ("a 1", "x 1"):
        node = await node_at(client, task_id, *texts)
        check(node["sandbox_id"] is None, f"expected no snapshot at {texts}, got {node}")
    added = set(SNAPSHOTS) - snapshots_before
    check(not added, f"expected the contractor to take no snapshots, got {added}")


async def test_live_sandboxes_stay_under_cap(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    texts = [f"x {value}" for value in range(1, 7)]
    set_embedder(StubEmbedder(texts))
    tasks = [f"{task_id}-{k}" for k in range(4)]
    for task in tasks:
        for text in texts:
            await rollout(task, calls(text))
    runs = ((4, tasks[:2], None), (12, tasks[2:3], 6), (2, tasks[3:], 2))
    for cap, contracted, lone_peak in runs:
        reset_workload()
        peak = await contract(*contracted, max_sandboxes=cap)
        check(peak <= cap, f"expected at most {cap} live sandboxes, got {peak}")
        if lone_peak is not None:
            check(peak == lone_peak, f"expected a lone job to keep {lone_peak} live, got {peak}")
        for task in contracted:
            groups = (await client.list_sibling_groups(task, 1, 0.9))["groups"]
            check(groups == [], f"expected {task} fully compared, got {groups}")


async def test_pinned_sandbox_survives_eviction(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # Pairs score AB > EF > AC > BC > BD, and no other pair reaches the threshold.
    # EF outscores AC, but AC goes first because A is already live.
    degrees = {"x 1": 0, "x 2": 10, "x 3": -12, "x 4": 35, "x 5": 90, "x 6": 101}
    set_embedder(FixedEmbedder(degrees))
    for text in degrees:
        await rollout(task_id, calls(text))
    a, b, c, d, e, f = ({"x": value} for value in range(1, 7))
    reset_workload()
    FAIL_EQUALS.append((a, c))
    peak = await contract(task_id, max_sandboxes=2)
    check(peak == 2, f"expected a full two-sandbox pool, got {peak}")
    # After AB, A has one pair left and B two, so acquiring C would evict A if its pair
    # didn't pin it.
    pairs = [(first, second) for first, second, _ in COMPARED]
    expected = [(a, b), (a, c), (b, c), (b, d), (e, f)]
    check(pairs == expected, f"expected the ranked order, got {pairs}")
    check(all(live for _, _, live in COMPARED), f"expected both sides live, got {COMPARED}")
    for text, settled in zip(degrees, (False, True, False, True, True, True)):
        node = await node_at(client, task_id, text)
        check(node["settled"] == settled, f"expected {text} settled={settled}, got {node}")


async def stop_mid_comparison(task_id: str) -> None:
    """Starts a job, stops it during its first gated equals, and checks stop waits."""
    GATE.clear()
    IN_EQUALS.clear()
    GATED[0] = True
    contractor = Contractor(DictSandbox, SetCall, 0.9, 4, HOST, PORT)
    contractor.start(task_id)
    await asyncio.wait_for(IN_EQUALS.wait(), timeout=30)
    stopping = asyncio.create_task(contractor.stop(task_id))
    await asyncio.sleep(0.1)
    check(not stopping.done(), "expected stop to wait for the comparison in progress")
    GATED[0] = False
    GATE.set()
    await asyncio.wait_for(stopping, timeout=30)
    check(not LIVE, f"expected stop to return after cleanup, got {LIVE}")


async def test_failed_restore_releases_pins(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # Only AB and CD qualify, and B can't be restored. CD must then evict A, which a
    # pin leaked from AB would block.
    degrees = {"x 1": 0, "x 2": 5, "x 3": 90, "x 4": 96}
    set_embedder(FixedEmbedder(degrees))
    for text in degrees:
        await rollout(task_id, calls(text))
    c, d = {"x": 3}, {"x": 4}
    snapshot_a = (await node_at(client, task_id, "x 1"))["sandbox_id"]
    snapshot_b = (await node_at(client, task_id, "x 2"))["sandbox_id"]
    reset_workload()
    FAIL_RESTORES.add(snapshot_b)
    await contract(task_id, max_sandboxes=2)
    failed = FAILED_RESTORES
    check(failed == [snapshot_b], f"expected B's restore to fail once, got {failed}")
    check(snapshot_a in ORIGINS.values(), f"expected A restored before B failed, got {ORIGINS}")
    pairs = [(first, second) for first, second, _ in COMPARED]
    check(pairs == [(c, d)], f"expected CD compared, got {pairs}")


async def test_stop_applies_partial_result(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(FixedEmbedder({"x 1": 0, "x 01": 3, "x 2": 10}))
    for text in ("x 1", "x 01", "x 2"):
        await rollout(task_id, calls(text))
    reset_workload()
    await stop_mid_comparison(task_id)
    check(len(COMPARED) == 1, f"expected only the first pair compared, got {COMPARED}")
    merged = await node_at(client, task_id, "x 01")
    expected = sorted([serialized("x 1"), serialized("x 01")])
    check(sorted(merged["tool_calls"]) == expected, f"expected x 1 and x 01 merged, got {merged}")
    check(not merged["settled"], f"expected the merged node red, got {merged}")
    unmerged = await node_at(client, task_id, "x 2")
    alone = list(unmerged["tool_calls"]) == [serialized("x 2")]
    check(alone and not unmerged["settled"], f"expected x 2 alone and red, got {unmerged}")
    reset_workload()
    await contract(task_id)
    for text in ("x 1", "x 2"):
        node = await node_at(client, task_id, text)
        check(node["settled"], f"expected {text} settled by a later window, got {node}")


async def test_stopped_class_stays_red_until_decided(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # A is x 1, B is x 01 and C is x 2. AB and BC qualify, AC falls below the
    # threshold, and C starts blue.
    set_embedder(FixedEmbedder({"x 1": 0, "x 01": 12, "x 2": 26}))
    for text in ("x 1", "x 01", "x 2"):
        await rollout(task_id, calls(text))
    await client.apply_contraction(task_id, [], [], [serialized("x 2")])
    c = await node_at(client, task_id, "x 2")
    check(c["settled"], f"expected C blue before the window, got {c}")
    reset_workload()
    await stop_mid_comparison(task_id)
    merged = await node_at(client, task_id, "x 01")
    check(serialized("x 1") in merged["tool_calls"], f"expected A and B merged, got {merged}")
    check(not merged["settled"], f"expected the merged node red after the stop, got {merged}")
    c = await node_at(client, task_id, "x 2")
    alone = list(c["tool_calls"]) == [serialized("x 2")]
    check(alone and c["settled"], f"expected C alone and still blue, got {c}")
    groups = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    one_pair = len(groups) == 1 and len(groups[0]["pairs"]) == 1
    check(one_pair, f"expected the merged node and C offered, got {groups}")
    reset_workload()
    await contract(task_id)
    merged = await node_at(client, task_id, "x 01")
    check(merged["settled"], f"expected the merged node blue once decided, got {merged}")


async def test_failing_pair_tried_once_per_window(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    set_embedder(StubEmbedder(["x 1", "x 2", "x 3"]))
    for text in ("x 1", "x 2", "x 3"):
        await rollout(task_id, calls(text))
    a, b, c = ({"x": value} for value in range(1, 4))
    for _ in range(2):
        reset_workload()
        FAIL_EQUALS.append((a, b))
        await contract(task_id)
        pairs = sorted(sorted((first["x"], second["x"])) for first, second, _ in COMPARED)
        # A and B stay red, so every pair qualifies again in the second window.
        check(pairs == [[1, 2], [1, 3], [2, 3]], f"expected each pair once, got {pairs}")
        for text, settled in (("x 1", False), ("x 2", False), ("x 3", True)):
            node = await node_at(client, task_id, text)
            check(node["settled"] == settled, f"expected {text} settled={settled}, got {node}")
    reset_workload()
    await contract(task_id)
    for text in ("x 1", "x 2", "x 3"):
        node = await node_at(client, task_id, text)
        check(node["settled"], f"expected {text} settled once decided, got {node}")


async def run_tests() -> None:
    prefix = f"contraction-{uuid4().hex}"
    async with AsyncPersistentCacheClient(HOST, PORT) as client:
        await test_workload_builds_graph(client, f"{prefix}-workload")
        await test_equivalent_siblings_merge(client, f"{prefix}-equivalent")
        await test_unequal_siblings_turn_blue(client, f"{prefix}-unequal")
        await test_shared_children_merge_recursively(client, f"{prefix}-recursive")
        await test_new_embedding_reopens_settled_node(client, f"{prefix}-reopened")
        await test_replays_without_snapshots(client, f"{prefix}-replay")
        await test_live_sandboxes_stay_under_cap(client, f"{prefix}-cap")
        await test_pinned_sandbox_survives_eviction(client, f"{prefix}-pinned")
        await test_failed_restore_releases_pins(client, f"{prefix}-restore")
        await test_stop_applies_partial_result(client, f"{prefix}-stop")
        await test_stopped_class_stays_red_until_decided(client, f"{prefix}-stopped")
        await test_failing_pair_tried_once_per_window(client, f"{prefix}-failing")
    print("all contraction tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())
