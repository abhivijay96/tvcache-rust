"""Tests that tvclient's executor embeds stateful tool calls on cache misses.

Run the server first with `cargo run`. The tvclient repo must be checked out
next to this one.
"""

import asyncio
import sys
import time
from pathlib import Path

from cache_client import AsyncPersistentCacheClient
from test_embedding import HOST, PORT, check, get_node

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tvclient"))

from tvclient import (
    Embedder,
    Sandbox,
    TestToolCall,
    ToolCall,
    TvcacheToolExecutor,
    set_embedder,
)


class CommandToolCall(ToolCall):
    def __init__(self, command: str, mutates: bool = True):
        self.command = command
        self.mutates = mutates

    def to_dict(self) -> dict:
        return {"command": self.command}

    def will_mutate_state(self) -> bool:
        return self.mutates


class EchoSandbox(Sandbox):
    """Returns each call's serialization as its result."""

    async def start(self) -> None:
        self.env_id = self.env_id or "echo"

    async def fork(self) -> "EchoSandbox":
        return EchoSandbox(self.task_name, "echo-snapshot")

    async def execute(self, tool_call: ToolCall) -> str:
        return f"ran {tool_call.serialize()}"

    async def stop(self) -> None:
        pass


class StubEmbedder(Embedder):
    """Returns a fixed vector, or raises it if it's an exception."""

    def __init__(self, result: list[float] | Exception):
        self.result = result
        self.calls = 0

    def encode(self, text: str) -> list[float]:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def run_rollout(task_id: str, embedder: StubEmbedder) -> list[str]:
    """Runs a stateful call, a stateless call, then the test. Returns their keys."""
    set_embedder(embedder)
    history = [CommandToolCall("touch file"), CommandToolCall("ls", mutates=False)]
    async with TvcacheToolExecutor(
        EchoSandbox, CommandToolCall, task_id, HOST, PORT, embed_tool_calls=True
    ) as executor:
        for index in range(len(history)):
            await executor.lookup_or_execute(history[: index + 1])
        await executor.test(history)
    return [call.serialize() for call in history] + [TestToolCall().serialize()]


async def test_embeds_stateful_misses(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    embedding = [0.5, 1.5, 2.5]
    embedder = StubEmbedder(embedding)
    stateful, stateless, test = await run_rollout(task_id, embedder)

    node = await get_node(client, task_id, [stateful])
    check(node["embedding"] == embedding, f"expected the stateful call embedded, got {node}")
    for leaf in (stateless, test):
        node = await get_node(client, task_id, [stateful, leaf])
        check(node["embedding"] is None, f"expected no embedding on {leaf}, got {node}")
    check(embedder.calls == 1, f"expected one embedding, got {embedder.calls}")

    # A second identical rollout hits the cache everywhere, so nothing is embedded.
    await run_rollout(task_id, embedder)
    check(embedder.calls == 1, f"expected no embedding on hits, got {embedder.calls}")


async def test_survives_embedder_failure(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # The executor prints one expected warning to stderr, then inserts without an embedding.
    embedder = StubEmbedder(RuntimeError("stub failure"))
    stateful = (await run_rollout(task_id, embedder))[0]
    check(embedder.calls == 1, f"expected the embedder called once, got {embedder.calls}")
    node = await get_node(client, task_id, [stateful])
    check(node["value"] is not None, f"expected the value inserted, got {node}")
    check(node["embedding"] is None, f"expected no embedding, got {node}")


async def run_tests() -> None:
    prefix = f"embedding-executor-{int(time.time())}"
    async with AsyncPersistentCacheClient(HOST, PORT) as client:
        await test_embeds_stateful_misses(client, f"{prefix}-misses")
        await test_survives_embedder_failure(client, f"{prefix}-failure")
    print("all embedding executor tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())
