"""Fills a task's graph with real tool-call embeddings to inspect in the visualizer.

Run the server first with `cargo run`, then run
`uv run --with sentence-transformers --with 'transformers<5' sample_embedding_workload.py`.
Jina's model code doesn't load on transformers 5. The tvclient repo must be
checked out next to this one, and the first run downloads the embedding model.
"""

import asyncio
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from cache_client import AsyncPersistentCacheClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tvclient"))

from tvclient import (
    Sandbox,
    SentenceTransformerEmbedder,
    ToolCall,
    TvcacheToolExecutor,
    get_embedder,
    set_embedder,
)

HOST = "127.0.0.1"
PORT = 7878
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

# Rollouts share a prefix, then branch into siblings that do the same thing written
# differently (ls -la / ls -al, pip / python -m pip) and ones that do something
# else (git status, rm -rf build).
ROLLOUTS = [
    ["cd project", "ls -la", "pip install requests"],
    ["cd project", "ls -la", "python -m pip install requests"],
    ["cd project", "ls -la", "rm -rf build"],
    ["cd project", "ls -al"],
    ["cd project", "git status"],
]


class CommandToolCall(ToolCall):
    def __init__(self, command: str):
        self.command = command

    def to_dict(self) -> dict:
        return {"command": self.command}

    def embed(self) -> list[float]:
        # Embeds only the command, since the shared JSON wrapper inflates similarity.
        return get_embedder().encode(self.command)


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


async def run_rollout(task_id: str, commands: list[str]) -> None:
    history = []
    async with TvcacheToolExecutor(
        EchoSandbox, CommandToolCall, task_id, HOST, PORT, embed_tool_calls=True
    ) as executor:
        for command in commands:
            history.append(CommandToolCall(command))
            await executor.lookup_or_execute(history)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b))


def print_sibling_similarities(node: dict[str, Any], path: list[str]) -> None:
    """Prints the similarity of every pair of embedded siblings, as stored on the server."""
    embedded = {
        json.loads(key)["command"]: child["embedding"]
        for key, child in node["children"].items()
        if child["embedding"] is not None
    }
    if len(embedded) > 1:
        print(f"\nSiblings under {' > '.join(path) or 'the root'}:")
        for (a, a_embedding), (b, b_embedding) in combinations(sorted(embedded.items()), 2):
            print(f"  {cosine_similarity(a_embedding, b_embedding):6.3f}  {a!r} vs {b!r}")

    for key, child in node["children"].items():
        print_sibling_similarities(child, path + [json.loads(key)["command"]])


async def main() -> None:
    task_id = f"embedding-sample-{int(time.time())}"
    embedder = SentenceTransformerEmbedder(MODEL_NAME, max_seq_length=1024, trust_remote_code=True)
    # Loads the model now, since the executor would only log a failed load.
    print(f"Loaded {MODEL_NAME} with dimension {len(embedder.encode('warm up'))}")
    set_embedder(embedder)

    for commands in ROLLOUTS:
        await run_rollout(task_id, commands)

    async with AsyncPersistentCacheClient(HOST, PORT) as client:
        response = await client.dump_all_response()
    if response["type"] != "json":
        raise RuntimeError(f"expected json response, got {response}")
    print_sibling_similarities(json.loads(response["json"])[task_id]["root"], [])
    print(f"\nPick task {task_id} in the visualizer to see each node's embedding.")


if __name__ == "__main__":
    asyncio.run(main())
