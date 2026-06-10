#!/usr/bin/env python3
"""Integration test for the tvclient library against a live TVCache server.

Run the server first: cargo run
Then: python tests/test_tvclient.py
"""

import asyncio
import itertools
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tvclient import (
    AsyncStatefulExecutor,
    ForkBank,
    SandboxManager,
    ToolCall,
    ToolCallEnv,
    TVCacheClient,
)

HOST = "127.0.0.1"
PORT = 7878
FORK_THRESHOLD_SECONDS = 0.05
SLOW_CALL_SECONDS = 0.08
FAST_CALL_SECONDS = 0.001


class CommandToolCall(ToolCall):
    def __init__(self, command: str, mutates: bool = True, slow: bool = False):
        self.command = command
        self.mutates = mutates
        self.slow = slow

    def to_dict(self) -> dict:
        return {"command": self.command}

    def will_mutate_state(self) -> bool:
        return self.mutates


class MockToolCallEnv(ToolCallEnv):
    """In-memory sandbox: the state is the list of serialized calls applied."""

    registry: Dict[str, List[str]] = {}
    execute_log: List[Tuple[str, str]] = []
    stop_log: List[str] = []
    _counter = itertools.count()

    async def start(self) -> None:
        self.env_id = f"env-{next(MockToolCallEnv._counter)}"
        MockToolCallEnv.registry[self.env_id] = []

    async def fork(self) -> "MockToolCallEnv":
        child = MockToolCallEnv(task_name=self.task_name)
        child.env_id = f"env-{next(MockToolCallEnv._counter)}"
        # .get tolerates foreign parent ids left in the cache by other test runs.
        child_state = list(MockToolCallEnv.registry.get(self.env_id, []))
        MockToolCallEnv.registry[child.env_id] = child_state
        return child

    async def execute(self, tool_call: ToolCall) -> str:
        await asyncio.sleep(
            SLOW_CALL_SECONDS if getattr(tool_call, "slow", False) else FAST_CALL_SECONDS
        )
        serialized = tool_call.serialize()
        MockToolCallEnv.execute_log.append((self.env_id, serialized))
        if tool_call.will_mutate_state():
            MockToolCallEnv.registry[self.env_id].append(serialized)
        return f"result:{serialized}"

    async def stop(self) -> None:
        MockToolCallEnv.stop_log.append(self.env_id)
        MockToolCallEnv.registry.pop(self.env_id, None)


def make_executor(task_id: str) -> AsyncStatefulExecutor:
    return AsyncStatefulExecutor(
        MockToolCallEnv,
        CommandToolCall,
        task_id,
        host=HOST,
        port=PORT,
        fork_threshold_seconds=FORK_THRESHOLD_SECONDS,
    )


async def sandbox_references(client: TVCacheClient, task_id: str) -> dict:
    cache_dump = await client.dump_all()
    if task_id not in cache_dump:
        return {}
    return cache_dump[task_id]["sandbox_references"]


async def test_lease(client: TVCacheClient, prefix: str) -> None:
    task_id = f"{prefix}-lease"
    sandbox_id = f"{task_id}-sb"

    await client.insert(task_id, ["a"], "a-value", sandbox_id, runtime=1)

    async with await client.find(task_id, ["a", "b"]) as found:
        assert found.sandbox_id == sandbox_id, found.sandbox_id
        references = await sandbox_references(client, task_id)
        assert references.get(sandbox_id) == 1, references
    references = await sandbox_references(client, task_id)
    assert references == {}, f"lease exit must unref: {references}"

    # The unref must run even when the body raises.
    try:
        async with await client.find(task_id, ["a", "b"]) as found:
            raise RuntimeError("rollout crashed")
    except RuntimeError:
        pass
    references = await sandbox_references(client, task_id)
    assert references == {}, f"lease must unref on exception: {references}"

    print("lease leg passed")


async def test_executor(client: TVCacheClient, prefix: str) -> tuple:
    task_id = f"{prefix}-executor"
    c1 = CommandToolCall("setup", slow=True)
    c2 = CommandToolCall("ls", mutates=False)
    c3 = CommandToolCall("patch", slow=True)

    # Rollout A: all misses; slow calls store sandbox snapshots.
    async with make_executor(task_id) as rollout_a:
        value_1 = await rollout_a.execute([c1])
        value_2 = await rollout_a.execute([c1, c2])
        value_3 = await rollout_a.execute([c1, c2, c3])
        assert value_1 == f"result:{c1.serialize()}", value_1
        assert value_2 == f"result:{c2.serialize()}", value_2
        assert value_3 == f"result:{c3.serialize()}", value_3
        assert rollout_a.total_executions == 3, rollout_a.total_executions
        # Stateless c2 must not extend the live sandbox's history.
        assert rollout_a.sandbox_history == [c1.serialize(), c3.serialize()]

    executions_after_a = len(MockToolCallEnv.execute_log)

    # Rollout B: the same sequence is now all exact hits; nothing executes and
    # no environment is ever created.
    async with make_executor(task_id) as rollout_b:
        assert await rollout_b.execute([c1]) == value_1
        assert await rollout_b.execute([c1, c2]) == value_2
        assert await rollout_b.execute([c1, c2, c3]) == value_3
        assert rollout_b.rollout_environment is None
    assert len(MockToolCallEnv.execute_log) == executions_after_a, "hits must not execute"

    # Rollout C: shares the c1 prefix then diverges; must fork the snapshot
    # stored at [c1] and execute only the suffix.
    c5 = CommandToolCall("diverge")
    async with make_executor(task_id) as rollout_c:
        value_5 = await rollout_c.execute([c1, c5])
        assert value_5 == f"result:{c5.serialize()}", value_5
        new_executions = MockToolCallEnv.execute_log[executions_after_a:]
        assert len(new_executions) == 1, f"fork must skip the matched prefix: {new_executions}"
        forked_env_id = rollout_c.rollout_environment.get_id()
        assert MockToolCallEnv.registry[forked_env_id] == [c1.serialize(), c5.serialize()]

    # Rollout D: a hit in the middle must not discard the live sandbox; the
    # following miss fast-forwards it instead of replaying from scratch.
    c6 = CommandToolCall("fast-a")
    c7 = CommandToolCall("fast-b")
    c8 = CommandToolCall("fast-c")
    async with make_executor(task_id) as rollout_d0:
        await rollout_d0.execute([c6])
        await rollout_d0.execute([c6, c7])
        await rollout_d0.execute([c6, c7, c8])

    c9 = CommandToolCall("fast-d")
    async with make_executor(task_id) as rollout_d:
        await rollout_d.execute([c6])            # miss already cached? no: cached by d0 -> hit
        assert rollout_d.rollout_environment is None
        await rollout_d.execute([c6, CommandToolCall("d-branch")])  # miss, fresh env, replay
        live_env_id = rollout_d.rollout_environment.get_id()
        await rollout_d.execute([c6, CommandToolCall("d-branch"), c7])  # miss, fast-forward
        assert rollout_d.rollout_environment.get_id() == live_env_id, (
            "fast-forward must reuse the live sandbox"
        )
        before = len(MockToolCallEnv.execute_log)
        # A hit with a live sandbox: the sandbox is kept, just falls behind.
        d_history = [c6, CommandToolCall("d-branch"), c7]
        await client.insert(
            task_id,
            [c.serialize() for c in d_history] + [c9.serialize()],
            "pre-cached",
            None,
            runtime=1,
        )
        assert await rollout_d.execute(d_history + [c9]) == "pre-cached"
        assert rollout_d.rollout_environment.get_id() == live_env_id
        assert len(MockToolCallEnv.execute_log) == before, "hit must not execute"
        # The next miss fast-forwards through the gap (re-executes c9) and the
        # new call, still in the same environment.
        value_8 = await rollout_d.execute(d_history + [c9, c8])
        assert value_8 == f"result:{c8.serialize()}", value_8
        assert rollout_d.rollout_environment.get_id() == live_env_id
        gap_executions = MockToolCallEnv.execute_log[before:]
        assert len(gap_executions) == 2, f"gap + current expected: {gap_executions}"

    references = await sandbox_references(client, task_id)
    assert references == {}, f"all leases must be released: {references}"

    print("executor leg passed")
    return task_id, c1


async def test_fork_bank(client: TVCacheClient, executor_task_id: str, c1: CommandToolCall) -> None:
    # Bump lpm hits on the [c1] snapshot node so it shows up in the hot list.
    for _ in range(5):
        async with await client.find(executor_task_id, [c1.serialize(), "no-such-call"]):
            pass

    # Learn which sandbox the cache hands out for the [c1] prefix; that is the
    # parent whose pre-forked copy the executor must consume.
    async with await client.find(executor_task_id, [c1.serialize(), "probe"]) as probe:
        c1_sandbox_id = probe.sandbox_id
    assert c1_sandbox_id is not None

    bank = ForkBank(
        MockToolCallEnv,
        tasks=[executor_task_id],
        warm_roots_per_task=1,
        copies_per_node=1,
        host=HOST,
        port=PORT,
    )
    await bank.warm_up(top_k=10)

    root_copies = bank.shelf.get((executor_task_id, "root"), [])
    assert len(root_copies) == 1, bank.shelf
    assert MockToolCallEnv.registry[root_copies[0]] == []

    hot_copies = bank.shelf.get((executor_task_id, c1_sandbox_id), [])
    assert len(hot_copies) == 1, f"expected a pre-forked copy of {c1_sandbox_id}: {bank.shelf}"
    shelved_env_id = hot_copies[0]
    assert MockToolCallEnv.registry[shelved_env_id] == [c1.serialize()]

    references = await sandbox_references(client, executor_task_id)
    assert references == {}, f"warm_up must release its leases: {references}"

    # An executor wired to the bank consumes the pre-forked copy instead of
    # forking synchronously.
    async with make_executor(executor_task_id) as rollout:
        rollout.set_fork_bank(bank)
        await rollout.execute([c1, CommandToolCall("bank-branch")])
        assert rollout.rollout_environment.get_id() == shelved_env_id, (
            f"expected the bank copy {shelved_env_id} of {c1_sandbox_id}"
        )
    assert bank.shelf[(executor_task_id, c1_sandbox_id)] == [], "the copy must be consumed"

    await bank.stop()
    print("fork bank leg passed")


async def test_lifecycle(
    client: TVCacheClient, executor_task_id: str, c1: CommandToolCall
) -> None:
    cache_dump = await client.dump_all()
    our_sandboxes = set(_collect_sandbox_ids(cache_dump[executor_task_id]["root"]))
    assert our_sandboxes, "executor leg should have stored sandboxes"

    manager = SandboxManager(MockToolCallEnv, host=HOST, port=PORT)
    marked = await manager.evict(limit=200)
    assert marked > 0, marked

    # Marked sandboxes are never handed out again: this path returned the [c1]
    # snapshot before the eviction mark.
    found = await client.find_unmanaged(executor_task_id, [c1.serialize(), "post-evict"])
    assert found.sandbox_id is None, found.sandbox_id

    drained = await manager.drain_once()
    drained_ours = {
        sandbox.sandbox_id for sandbox in drained if sandbox.task_id == executor_task_id
    }
    assert our_sandboxes <= drained_ours, (our_sandboxes, drained_ours)
    for sandbox_id in our_sandboxes:
        assert sandbox_id in MockToolCallEnv.stop_log, f"{sandbox_id} was not deleted"
        assert sandbox_id not in MockToolCallEnv.registry

    await manager.close()
    print("lifecycle leg passed")


def _collect_sandbox_ids(node: dict) -> List[str]:
    sandbox_ids = []
    if node.get("sandbox_id") is not None:
        sandbox_ids.append(node["sandbox_id"])
    for child in node.get("children", {}).values():
        sandbox_ids.extend(_collect_sandbox_ids(child))
    return sandbox_ids


async def test_concurrency(client: TVCacheClient, prefix: str) -> None:
    task_id = f"{prefix}-concurrent"
    k1 = CommandToolCall("conc-setup", slow=True)
    k2 = CommandToolCall("conc-step")
    variants = [CommandToolCall(f"conc-variant-{i}") for i in range(4)]

    async def rollout(index: int) -> None:
        async with make_executor(task_id) as executor:
            await executor.execute([k1])
            await executor.execute([k1, k2])
            await executor.execute([k1, k2, variants[index % len(variants)]])

    await asyncio.gather(*[rollout(index) for index in range(16)])

    references = await sandbox_references(client, task_id)
    assert references == {}, f"all concurrent leases must be released: {references}"

    print("concurrency leg passed")


async def run_test() -> None:
    prefix = f"tvclient-{int(time.time())}"

    async with TVCacheClient(HOST, PORT) as client:
        await test_lease(client, prefix)
        executor_task_id, c1 = await test_executor(client, prefix)
        await test_fork_bank(client, executor_task_id, c1)
        await test_lifecycle(client, executor_task_id, c1)
        await test_concurrency(client, prefix)

    print("all tvclient tests passed")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
