"""Tests that the server stores tool-call embeddings sent with Insert.

Run the server first with `cargo run`.
"""

import asyncio
import json
import time
from typing import Any

from cache_client import AsyncPersistentCacheClient, encode_insert

HOST = "127.0.0.1"
PORT = 7878


async def get_node(
    client: AsyncPersistentCacheClient, task_id: str, path: list[str]
) -> dict[str, Any] | None:
    response = await client.dump_all_response()
    if response["type"] != "json":
        raise RuntimeError(f"expected json response, got {response}")
    node = json.loads(response["json"])[task_id]["root"]
    for path_entry in path:
        node = node["children"].get(path_entry)
        if node is None:
            return None
    return node


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def test_storage(client: AsyncPersistentCacheClient, task_id: str) -> None:
    path = ["a"]
    await client.insert(task_id, path, "value", None, runtime=1)
    node = await get_node(client, task_id, path)
    check(node["embedding"] is None, f"expected no embedding, got {node}")
    size_with = node["size_in_bytes"] + 16

    embedding = [1.0, 2.0, 3.0, 4.0]
    await client.insert(task_id, path, None, None, runtime=1, embedding=embedding)
    node = await get_node(client, task_id, path)
    check(node["embedding"] == embedding, f"expected {embedding}, got {node}")
    check(node["size_in_bytes"] == size_with, f"expected size {size_with}, got {node}")

    # The first embedding is kept on re-insert with or without a new one.
    await client.insert(task_id, path, "value", None, runtime=1, embedding=[0.0] * 8)
    await client.insert(task_id, path, "value", None, runtime=1)
    node = await get_node(client, task_id, path)
    check(node["embedding"] == embedding, f"expected the first embedding kept, got {node}")
    check(node["size_in_bytes"] == size_with, f"expected size {size_with}, got {node}")


async def test_zero_dimension_is_rejected(
    client: AsyncPersistentCacheClient, task_id: str
) -> None:
    # The client refuses dimension 0, so append it to an insert by hand.
    request = encode_insert(task_id, ["a"], "value", None, 1)[:-1] + b"\x01\x00\x00"
    reader, writer = await asyncio.open_connection(HOST, PORT)
    writer.write(request)
    await writer.drain()
    reply = await asyncio.wait_for(reader.read(1), timeout=5)
    check(reply == b"", f"expected the server to close the connection, got {reply!r}")
    writer.close()
    await writer.wait_closed()

    # The server keeps serving, and the rejected insert left nothing behind.
    await client.insert(task_id, ["b"], "value", None, runtime=1)
    check(await get_node(client, task_id, ["a"]) is None, "expected the bad insert ignored")
    node = await get_node(client, task_id, ["b"])
    check(node is not None and node["value"] == "value", f"expected the server to work, got {node}")


async def run_tests() -> None:
    prefix = f"embedding-{int(time.time())}"
    async with AsyncPersistentCacheClient(HOST, PORT) as client:
        await test_storage(client, f"{prefix}-storage")
        await test_zero_dimension_is_rejected(client, f"{prefix}-malformed")
    print("all embedding storage tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())
