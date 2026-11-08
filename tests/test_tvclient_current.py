#!/usr/bin/env python3
"""Current sibling tvclient API smoke test against a fresh running server.

PYTHONPATH=../tvclient python3 tests/test_tvclient_current.py
Uses an in-memory mock sandbox; no external tools or real snapshots required.
"""
import asyncio
import itertools
import time

from tvclient import Sandbox, ToolCall, TVCacheClient, TvcacheToolExecutor


class Call(ToolCall):
    def __init__(self, command, mutates=True):
        self.command, self.mutates = command, mutates

    def to_dict(self):
        return {"command": self.command, "mutates": self.mutates}

    def will_mutate_state(self):
        return self.mutates


class MemorySandbox(Sandbox):
    states = {}
    ids = itertools.count()
    executions = []

    async def start(self):
        state = list(self.states[self.env_id]) if self.env_id is not None else []
        self.env_id = f"live-{next(self.ids)}"
        self.states[self.env_id] = state

    async def fork(self):
        handle = f"snapshot-{next(self.ids)}"
        self.states[handle] = list(self.states[self.env_id])
        return type(self)(self.task_name, handle)

    async def execute(self, call):
        # Ensure execution exceeds the tiny fixed snapshot threshold.
        await asyncio.sleep(0.001)
        self.executions.append(call.command)
        if call.will_mutate_state():
            self.states[self.env_id].append(call.command)
        return "/".join(self.states[self.env_id]) + ":" + call.command

    async def stop(self):
        self.states.pop(self.env_id, None)


async def check(snapshots):
    task = f"current-api-{time.time_ns()}"
    options = dict(adaptive=False, fixed_threshold_seconds=0.000001 if snapshots else 1e6)
    history = [Call("a"), Call("read", False), Call("c")]
    expected = ["a:a", "a:read", "a/c:c"]
    for repeat in range(2):
        before = len(MemorySandbox.executions)
        async with TvcacheToolExecutor(MemorySandbox, Call, task, **options) as executor:
            for step, value in enumerate(expected, 1):
                assert await executor.lookup_or_execute(history[:step]) == value
        assert len(MemorySandbox.executions) - before == (3 if repeat == 0 else 0)
    # Cached stateful prefix followed by a miss: restore or reconstruct it.
    before = len(MemorySandbox.executions)
    async with TvcacheToolExecutor(MemorySandbox, Call, task, **options) as executor:
        assert await executor.lookup_or_execute(history[:1]) == "a:a"
        assert await executor.lookup_or_execute(history[:2]) == "a:read"
        assert await executor.lookup_or_execute(history[:2] + [Call("d")]) == "a/d:d"
    assert len(MemorySandbox.executions) - before == (1 if snapshots else 2)
    async with TVCacheClient() as client:
        dump = await client.dump_all()
        assert all(count == 0 for count in dump[task]["sandbox_references"].values())
        if snapshots:
            assert await client.get_fork_cost(task) > 0
    print(f"Current tvclient: hits, stateless keys, branch state, leases, snapshots={snapshots} passed")


async def main():
    await check(False)
    await check(True)


if __name__ == "__main__":
    asyncio.run(main())
